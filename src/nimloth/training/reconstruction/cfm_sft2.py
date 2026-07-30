"""Train a conditional flow-matching visualizer from cached SFT2 states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from nimloth.recon.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    condition_sensitivity,
    sample_euler,
)
from nimloth.recon.rcdm.image_utils import diffusion_tensor_to_pil
from nimloth.recon.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.predictor import LatentWMPredictor


@dataclass
class LoadedStateImageSplit:
    states: torch.Tensor
    images_uint8: torch.Tensor
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def _load_image_uint8(path: str | Path, image_size: int) -> torch.Tensor:
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    with Image.open(path) as source:
        image = source.convert("RGB").resize((image_size, image_size), resample)
    # NumPy copies the contiguous RGB buffer directly; materializing every
    # pixel as a Python tuple makes full-cache preload several times slower.
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def load_state_image_split(
    cache_dir: Path,
    *,
    image_size: int,
    max_items: int = -1,
    expected_latent_token_count: int | None = None,
) -> LoadedStateImageSplit:
    """Sequentially preload states and resized images for random-step training."""

    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if expected_latent_token_count is not None:
        actual_count = int(manifest.get("latent_token_count", 1))
        if actual_count != expected_latent_token_count:
            raise ValueError(
                "CFM cache latent token count mismatch: "
                f"expected={expected_latent_token_count}, actual={actual_count}"
            )
    dataset = RCDMStateCacheDataset(cache_dir)
    count = len(dataset) if max_items < 0 else min(max_items, len(dataset))
    if count < 2:
        raise ValueError(f"CFM split needs at least two items, got {count}: {cache_dir}")
    condition_dim = int(manifest["cond_dim"])
    states = torch.empty((count, condition_dim), dtype=torch.float16)
    images = torch.empty((count, 3, image_size, image_size), dtype=torch.uint8)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        states[index].copy_(item["state_emb"].reshape(-1).to(dtype=torch.float16))
        images[index].copy_(_load_image_uint8(item["current_image_path"], image_size))
        rows.append(
            {
                "id": str(item.get("id", index)),
                "record_id": str(item.get("record_id", "")),
                "step_index": int(item.get("step_index", -1)),
                "action_index": int(item["action_index"]),
                "current_image_path": str(item["current_image_path"]),
                "next_image_path": str(item["next_image_path"]),
            }
        )
        if (index + 1) % 5000 == 0:
            print(json.dumps({
                "cfm_preload": str(cache_dir),
                "items": index + 1,
                "total": count,
            }), flush=True)
    return LoadedStateImageSplit(states, images, rows, manifest)


def resolve_condition_token_shape(manifest: dict[str, Any]) -> tuple[int, int]:
    """Resolve the cache-native token layout used by the CFM condition path."""

    flat_dim = int(manifest["cond_dim"])
    representation = str(manifest.get("representation", "projected"))
    state_shape = tuple(int(value) for value in manifest.get("state_shape", ()))
    if representation in {"qwen_query_hidden", "dino_grid_state"}:
        if len(state_shape) != 2:
            raise ValueError(
                f"{representation} cache needs [K,D] state_shape, got {state_shape}"
            )
        if math.prod(state_shape) != flat_dim:
            raise ValueError(
                f"{representation} state_shape {state_shape} does not match "
                f"cond_dim={flat_dim}"
            )
        return state_shape
    return 1, flat_dim


def initialize_from_cfm(
    model: TokenConditionedFlowUNet,
    checkpoint: Path,
) -> dict[str, Any]:
    """Strictly initialize from a CFM with the identical architecture."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict):
        raise ValueError("CFM initialization checkpoint lacks invariants")
    source_config = invariants.get("cfm_config")
    current_config = model.config.to_metadata()
    if source_config != current_config:
        raise ValueError(
            "CFM initialization architecture mismatch:\n"
            + json.dumps(
                {"checkpoint": source_config, "current": current_config},
                indent=2,
            )
        )
    model.load_state_dict(payload["model"], strict=True)
    return {
        "checkpoint": str(checkpoint),
        "source_step": int(payload.get("step", -1)),
        "source_best_val": float(payload.get("best_val", float("nan"))),
        "strict_model_load": True,
    }


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _checkpoint_invariants(
    *,
    config: CFMConfig,
    train_split: LoadedStateImageSplit,
    val_split: LoadedStateImageSplit,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "cfm_config": config.to_metadata(),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
        "train_cache_fingerprint": str(train_split.manifest["fingerprint"]),
        "val_cache_fingerprint": str(val_split.manifest["fingerprint"]),
        "train_items": int(train_split.states.shape[0]),
        "val_items": int(val_split.states.shape[0]),
        "latent_token_count": int(args.latent_token_count),
        "condition_dropout": float(args.condition_dropout),
        "init_cfm_checkpoint": (
            str(args.init_cfm_checkpoint)
            if args.init_cfm_checkpoint is not None
            else None
        ),
        "skip_samples": bool(args.skip_samples),
    }


def _save_checkpoint(
    *,
    path: Path,
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val: float,
    invariants: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "best_val": float(best_val),
        "invariants": invariants,
        "metadata": metadata,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    _atomic_torch_save(payload, path)


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def _load_checkpoint(
    *,
    path: Path,
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
    invariants: dict[str, Any],
    device: torch.device,
) -> tuple[int, float]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("invariants") != invariants:
        raise ValueError(
            "CFM resume invariants mismatch:\n"
            + json.dumps({
                "checkpoint": payload.get("invariants"),
                "current": invariants,
            }, indent=2)
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states is not None:
        # map_location may move serialized RNG states onto CUDA, while PyTorch's
        # RNG restore API requires CPU ByteTensors.
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return int(payload["step"]), float(payload.get("best_val", float("inf")))


def _init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.no_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        if args.require_wandb:
            raise RuntimeError("W&B import is required for this run") from exc
        print(json.dumps({"wandb_init_skipped": str(exc)}))
        return None
    run_id_path = args.output_dir / "wandb_run_id.txt"
    run_id = args.wandb_id
    resume_requested = args.resume or args.resume_checkpoint is not None
    if run_id is None and resume_requested and run_id_path.is_file():
        run_id = run_id_path.read_text(encoding="utf-8").strip() or None
    try:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if resume_requested else None,
            config=metadata,
            dir=str(args.output_dir),
        )
    except Exception as exc:
        if args.require_wandb:
            raise RuntimeError("W&B initialization is required for this run") from exc
        print(json.dumps({"wandb_init_skipped": str(exc)}))
        return None
    if getattr(run, "id", None):
        run_id_path.write_text(str(run.id), encoding="utf-8")
    return run


def _select_sample_indices(rows: list[dict[str, Any]], num_items: int) -> list[int]:
    by_record: dict[str, int] = {}
    for index, row in enumerate(rows):
        by_record.setdefault(str(row.get("record_id", index)), index)
    candidates = list(by_record.values())
    if len(candidates) <= num_items:
        return candidates
    if num_items == 1:
        return candidates[:1]
    return [
        candidates[round(position * (len(candidates) - 1) / (num_items - 1))]
        for position in range(num_items)
    ]


def _label_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images) + label_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (offset, label_height))
        draw.text((offset + 2, 2), label, fill=(0, 0, 0))
        offset += image.width
    return output


def _contact_sheet(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = math.ceil(len(images) / columns)
    output = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


@torch.no_grad()
def _save_reconstruction_samples(
    *,
    model: TokenConditionedFlowUNet,
    split: LoadedStateImageSplit,
    wm_predictor: LatentWMPredictor,
    output_dir: Path,
    device: torch.device,
    ode_steps: list[int],
    num_items: int,
    seed: int,
) -> dict[str, str]:
    indices = _select_sample_indices(split.rows, num_items)
    current_states = split.states[indices].float()
    actions = torch.tensor(
        [split.rows[index]["action_index"] for index in indices],
        device=device,
        dtype=torch.long,
    )
    predicted_next = wm_predictor(
        current_states.to(device=device), actions
    ).detach().cpu().float()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = torch.randn(
        (len(indices), 3, model.config.image_size, model.config.image_size),
        generator=generator,
    )
    outputs: dict[str, str] = {}
    for steps in ode_steps:
        current_recon = sample_euler(
            model,
            current_states,
            initial_noise,
            steps=steps,
            device=device,
        )
        predicted_recon = sample_euler(
            model,
            predicted_next,
            initial_noise,
            steps=steps,
            device=device,
        )
        sample_dir = output_dir / f"samples_{steps}ode"
        sample_dir.mkdir(parents=True, exist_ok=True)
        strips: list[Image.Image] = []
        sample_rows: list[dict[str, Any]] = []
        for offset, index in enumerate(indices):
            current_gt = Image.fromarray(
                split.images_uint8[index].permute(1, 2, 0).numpy(), mode="RGB"
            )
            next_gt = _load_image_uint8(
                split.rows[index]["next_image_path"], model.config.image_size
            )
            next_gt_image = Image.fromarray(next_gt.permute(1, 2, 0).numpy(), mode="RGB")
            strip = _label_strip(
                [
                    current_gt,
                    diffusion_tensor_to_pil(current_recon[offset]),
                    next_gt_image,
                    diffusion_tensor_to_pil(predicted_recon[offset]),
                ],
                ["current GT", "current CFM", "next GT", "WM pred-next CFM"],
            )
            strip_path = sample_dir / f"sample_{offset:03d}_strip.png"
            strip.save(strip_path)
            strips.append(strip)
            sample_rows.append({
                "sample_index": offset,
                "record_id": split.rows[index]["record_id"],
                "step_index": split.rows[index]["step_index"],
                "action_index": split.rows[index]["action_index"],
                "strip_path": str(strip_path),
            })
        contact = _contact_sheet(strips)
        contact_path = sample_dir / "contact_sheet.png"
        contact.save(contact_path)
        (sample_dir / "samples.json").write_text(
            json.dumps(sample_rows, indent=2), encoding="utf-8"
        )
        outputs[f"{steps}ode"] = str(contact_path)
    return outputs


def train_cfm_sft2(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not 0.0 <= args.condition_dropout < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
    resume_requested = bool(args.resume or args.resume_checkpoint is not None)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not resume_requested:
        raise FileExistsError(
            f"CFM output directory is not empty: {args.output_dir}; "
            "use a new output or an explicit resume"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_split = load_state_image_split(
        args.state_cache_dir / "train",
        image_size=args.image_size,
        max_items=args.max_train_items,
        expected_latent_token_count=args.latent_token_count,
    )
    val_split = load_state_image_split(
        args.state_cache_dir / args.validation_cache_split,
        image_size=args.image_size,
        max_items=args.max_val_items,
        expected_latent_token_count=args.latent_token_count,
    )
    condition_dim = int(train_split.states.shape[1])
    if val_split.states.shape[1] != condition_dim:
        raise ValueError("CFM train/val condition dimensions differ")
    token_count, token_dim = resolve_condition_token_shape(train_split.manifest)
    val_token_shape = resolve_condition_token_shape(val_split.manifest)
    if val_token_shape != (token_count, token_dim):
        raise ValueError(
            "CFM train/val token shapes differ: "
            f"train={(token_count, token_dim)}, val={val_token_shape}"
        )
    if token_count * token_dim != condition_dim:
        raise ValueError(
            "CFM token shape does not match flat condition width: "
            f"{token_count}*{token_dim} != {condition_dim}"
        )
    config = CFMConfig(
        image_size=args.image_size,
        token_count=token_count,
        token_dim=token_dim,
        base_channels=args.base_channels,
        condition_dim=args.condition_dim,
        time_dim=args.time_dim,
    )
    model = TokenConditionedFlowUNet(config).to(device)
    initialization = None
    if args.init_cfm_checkpoint is not None:
        initialization = initialize_from_cfm(model, args.init_cfm_checkpoint)
        print(json.dumps({"cfm_initialization": initialization}), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    invariants = _checkpoint_invariants(
        config=config,
        train_split=train_split,
        val_split=val_split,
        args=args,
    )
    steps_per_epoch = math.ceil(train_split.states.shape[0] / args.batch_size)
    planned_steps = args.epochs * steps_per_epoch
    total_steps = args.max_steps if args.max_steps > 0 else planned_steps
    metadata = {
        "task": "direct_conditional_flow_matching_reconstruction",
        "source_checkpoint": str(args.source_checkpoint),
        "state_cache_dir": str(args.state_cache_dir),
        "wm_checkpoint": (
            str(args.wm_checkpoint) if args.wm_checkpoint is not None else None
        ),
        "split_semantics": (
            "same train-cache subset for explicit tiny-overfit diagnostics"
            if args.validation_cache_split == "train"
            else "strict-valid train_all for training; disjoint val_all for validation only"
        ),
        "target": (
            "current_128px_image_from_dino_grid_state"
            if train_split.manifest.get("representation") == "dino_grid_state"
            else "current_128px_image_from_current_projected_sft2_state"
        ),
        "trainable_modules": "TokenConditionedFlowUNet only",
        "frozen_modules": (
            "SFT2 Qwen, StateProjector, WM predictor; training reads only the "
            "frozen state cache"
        ),
        "invariants": invariants,
        "initialization": initialization,
        "condition_dropout": args.condition_dropout,
        "epochs": args.epochs,
        "planned_steps": planned_steps,
        "total_steps": total_steps,
        "eval_interval": args.eval_interval,
        "eval_max_items": args.eval_max_items,
        "save_interval": args.save_interval,
        "wandb": {
            "project": args.wandb_project,
            "run_name": args.wandb_run_name,
            "enabled": not args.no_wandb,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    # Initialize external logging before restoring RNG so resume continues from
    # the checkpointed stochastic training stream even if W&B uses randomness.
    wandb_run = _init_wandb(args, metadata)
    start_step = 0
    best_val = float("inf")
    resume_path = args.resume_checkpoint
    if args.resume and resume_path is None:
        resume_path = _latest_checkpoint(args.output_dir)
        if resume_path is None:
            raise FileNotFoundError("--resume requested but no checkpoint_*.pt exists")
    if resume_path is not None:
        start_step, best_val = _load_checkpoint(
            path=resume_path,
            model=model,
            optimizer=optimizer,
            invariants=invariants,
            device=device,
        )
        print(json.dumps({
            "resume_checkpoint": str(resume_path),
            "start_step": start_step,
            "best_val": best_val,
        }), flush=True)

    log_path = args.output_dir / "train_step_log.csv"
    if start_step == 0 or not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([
                "time",
                "step",
                "epoch_float",
                "train_flow_mse",
                "val_correct_flow_mse",
                "val_shuffled_flow_mse",
                "val_shuffled_over_correct",
                "best_val_correct_flow_mse",
            ])
    start_time = time.time()
    last_loss = float("nan")
    last_eval: dict[str, float] | None = None
    train_count = train_split.states.shape[0]
    for step in range(start_step + 1, total_steps + 1):
        indices = torch.randint(0, train_count, (args.batch_size,))
        condition = train_split.states[indices].to(
            device=device, dtype=torch.float32
        )
        if args.condition_dropout > 0:
            drop = torch.rand(condition.shape[0], device=device) < args.condition_dropout
            condition = condition.masked_fill(drop[:, None], 0.0)
        target = train_split.images_uint8[indices].to(
            device=device, dtype=torch.float32
        ).div(127.5).sub(1.0)
        model.train()
        loss = conditional_flow_matching_loss(model, target, condition)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        epoch_float = step * args.batch_size / train_count
        evaluate = step == 1 or step % args.eval_interval == 0 or step == total_steps
        if evaluate:
            last_eval = condition_sensitivity(
                model,
                val_split.states,
                val_split.images_uint8,
                device,
                batch_size=args.batch_size,
                max_items=args.eval_max_items,
                seed=args.seed + 10_000 + step,
            )
            if last_eval["correct_flow_mse"] < best_val:
                best_val = last_eval["correct_flow_mse"]
                _save_checkpoint(
                    path=args.output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val=best_val,
                    invariants=invariants,
                    metadata=metadata,
                )
            with log_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    time.time(),
                    step,
                    epoch_float,
                    last_loss,
                    last_eval["correct_flow_mse"],
                    last_eval["shuffled_flow_mse"],
                    last_eval["shuffled_over_correct"],
                    best_val,
                ])
            metrics = {
                "cfm/train_flow_mse": last_loss,
                "cfm/val_correct_flow_mse": last_eval["correct_flow_mse"],
                "cfm/val_shuffled_flow_mse": last_eval["shuffled_flow_mse"],
                "cfm/val_shuffled_over_correct": last_eval["shuffled_over_correct"],
                "cfm/best_val_correct_flow_mse": best_val,
                "epoch": epoch_float,
            }
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            print(json.dumps({
                "step": step,
                "epoch_float": epoch_float,
                "train_flow_mse": last_loss,
                "validation": last_eval,
                "best_val": best_val,
                "elapsed_sec": time.time() - start_time,
            }), flush=True)
        elif step % args.log_interval == 0:
            with log_path.open("a", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow([
                    time.time(), step, epoch_float, last_loss, "", "", "", best_val
                ])
            if wandb_run is not None:
                wandb_run.log({
                    "cfm/train_flow_mse": last_loss,
                    "epoch": epoch_float,
                }, step=step)
            print(json.dumps({
                "step": step,
                "epoch_float": epoch_float,
                "train_flow_mse": last_loss,
                "elapsed_sec": time.time() - start_time,
            }), flush=True)
        if args.save_interval > 0 and step % args.save_interval == 0:
            checkpoint_path = args.output_dir / f"checkpoint_{step:09d}.pt"
            _save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val=best_val,
                invariants=invariants,
                metadata=metadata,
            )

    final_path = args.output_dir / f"checkpoint_{total_steps:09d}.pt"
    _save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        step=total_steps,
        best_val=best_val,
        invariants=invariants,
        metadata=metadata,
    )
    full_val = condition_sensitivity(
        model,
        val_split.states,
        val_split.images_uint8,
        device,
        batch_size=args.batch_size,
        max_items=-1,
        seed=args.seed + 20_000 + total_steps,
    )
    best_payload = torch.load(
        args.output_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(best_payload["model"], strict=True)
    sample_paths: dict[str, str] = {}
    if not args.skip_samples:
        if args.wm_checkpoint is None:
            raise ValueError("post-train samples require --wm-checkpoint")
        wm_predictor = LatentWMPredictor.load_checkpoint(
            args.wm_checkpoint, map_location=device
        ).to(device).eval()
        for parameter in wm_predictor.parameters():
            parameter.requires_grad_(False)
        sample_paths = _save_reconstruction_samples(
            model=model,
            split=val_split,
            wm_predictor=wm_predictor,
            output_dir=args.output_dir,
            device=device,
            ode_steps=args.sample_ode_steps,
            num_items=args.sample_items,
            seed=args.seed + 30_000,
        )
    if wandb_run is not None:
        import wandb

        payload: dict[str, Any] = {
            "cfm/final_full_val_correct_flow_mse": full_val["correct_flow_mse"],
            "cfm/final_full_val_shuffled_flow_mse": full_val["shuffled_flow_mse"],
            "cfm/final_full_val_shuffled_over_correct": full_val["shuffled_over_correct"],
        }
        for key, path in sample_paths.items():
            payload[f"cfm/reconstruction_{key}"] = wandb.Image(path)
        wandb_run.log(payload, step=total_steps + 1)

    summary = {
        "status": "completed",
        "train_items": int(train_split.states.shape[0]),
        "val_items": int(val_split.states.shape[0]),
        "total_steps": total_steps,
        "last_train_flow_mse": last_loss,
        "last_val_subset": last_eval,
        "best_val_correct_flow_mse": best_val,
        "final_checkpoint_full_val": full_val,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "final_checkpoint": str(final_path),
        "sample_contact_sheets": sample_paths,
        "initialization": initialization,
        "elapsed_sec": time.time() - start_time,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train direct-state conditional flow matching for SFT2 visualization"
    )
    parser.add_argument("--state-cache-dir", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, default=None)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-token-count", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--condition-dropout", type=float, default=0.0)
    parser.add_argument("--init-cfm-checkpoint", type=Path, default=None)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-max-items", type=int, default=1024)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--max-train-items", type=int, default=-1)
    parser.add_argument("--max-val-items", type=int, default=-1)
    parser.add_argument(
        "--validation-cache-split",
        choices=("train", "val"),
        default="val",
        help="Use train only for explicit overfit diagnostics; formal validation uses val",
    )
    parser.add_argument("--sample-items", type=int, default=8)
    parser.add_argument("--sample-ode-steps", type=int, nargs="+", default=[5, 50])
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--require-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_cfm_sft2(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
