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

from nimloth.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    condition_sensitivity,
    sample_euler,
)
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil
from nimloth.rcdm.state_cache import RCDMStateCacheDataset
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
        "condition_token_count_override": int(args.condition_token_count),
        "wm_history_size_override": args.wm_history_size_override,
        "condition_dropout": float(args.condition_dropout),
        "init_legacy_cfm_checkpoint": (
            str(args.init_legacy_cfm_checkpoint)
            if args.init_legacy_cfm_checkpoint is not None
            else None
        ),
        "lr_decay_step": int(args.lr_decay_step),
        "lr_after_decay": float(args.lr_after_decay),
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


def _legacy_cfm_key(key: str) -> str:
    replacements = (
        ("cond_mlp.", "condition_mlp."),
        ("rb1.", "block1."),
        ("rb2.", "block2."),
        ("rb3.", "block3."),
        ("attn3.", "attention3."),
        ("rb4.", "block4."),
        ("attn4.", "attention4."),
        ("mid1.", "middle1."),
        ("mid_attn.", "middle_attention."),
        ("mid2.", "middle2."),
        ("urb3.", "up_block3."),
        ("uattn3.", "up_attention3."),
        ("urb2.", "up_block2."),
        ("urb1.", "up_block1."),
    )
    for old, new in replacements:
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def initialize_from_legacy_cfm(
    model: TokenConditionedFlowUNet,
    checkpoint: Path,
) -> dict[str, Any]:
    """Load every shape-compatible weight from the proven 16x512 CFM.

    Query input normalization and the first 2048->256 projection are new; the
    UNet body, time/condition MLP, later token projection, and spatial
    cross-attention weights initialize from the proven visual decoder.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    translated = {_legacy_cfm_key(key): value for key, value in payload["model"].items()}
    current = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in translated.items():
        if key in current and current[key].shape == value.shape:
            current[key] = value.detach().clone()
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(current, strict=True)
    return {
        "checkpoint": str(checkpoint),
        "loaded_keys": len(loaded),
        "skipped_keys": len(skipped),
        "skipped": skipped,
    }


def resolve_condition_token_shape(
    manifest: dict[str, Any], *, token_count_override: int = 0
) -> tuple[int, int]:
    flat_dim = int(manifest["cond_dim"])
    if token_count_override:
        if token_count_override < 1 or flat_dim % token_count_override:
            raise ValueError(
                "condition token count must divide the flat condition dimension: "
                f"{token_count_override} does not divide {flat_dim}"
            )
        return token_count_override, flat_dim // token_count_override
    representation = str(manifest.get("representation", "projected"))
    state_shape = tuple(int(value) for value in manifest.get("state_shape", []))
    if representation in {"qwen_query_hidden", "dino_grid_state"}:
        if (
            representation == "qwen_query_hidden"
            and len(state_shape) == 1
            and int(manifest.get("latent_token_count", 1)) == 1
        ):
            return 1, state_shape[0]
        if len(state_shape) != 2:
            raise ValueError(
                f"{representation} cache needs [K,D]"
                + (", or legacy-flat [D] for qwen k=1" if representation == "qwen_query_hidden" else "")
                + f"; got {state_shape}"
            )
        if state_shape[0] * state_shape[1] != flat_dim:
            raise ValueError(f"{representation} state_shape {state_shape} does not match cond_dim={flat_dim}")
        return state_shape
    return 1, flat_dim


def uses_query_positive_control(manifest: dict[str, Any]) -> bool:
    """Sampling mode follows representation semantics, not CFM token count."""
    return str(manifest.get("representation", "projected")) in {
        "qwen_query_hidden",
        "dino_grid_state",
    }


def _init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if args.no_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
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


@torch.no_grad()
def _save_positive_control_samples(
    *,
    model: TokenConditionedFlowUNet,
    split: LoadedStateImageSplit,
    positive_cache_dir: Path,
    positive_cfm_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    num_items: int,
    ode_steps: int,
    cfg_scale: float,
    seed: int,
) -> dict[str, str]:
    from nimloth.training.reconstruction.state_to_vision_tokens import (
        load_proven_cfm,
        sample_euler_cfg,
    )

    indices = _select_sample_indices(split.rows, num_items)
    query = split.states[indices].float()
    query_wrong = torch.roll(query, 1, 0)
    positive_dataset = RCDMStateCacheDataset(positive_cache_dir)
    positive_rows = [positive_dataset[index] for index in indices]
    for offset, (index, row) in enumerate(zip(indices, positive_rows, strict=True)):
        expected = split.rows[index]
        for key in ("id", "record_id", "step_index", "current_image_path"):
            if str(row.get(key, "")) != str(expected.get(key, "")):
                raise ValueError(f"positive/query sample alignment mismatch offset={offset} key={key}")
    positive = torch.stack([row["state_emb"].reshape(-1).float() for row in positive_rows])
    positive_wrong = torch.roll(positive, 1, 0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(len(indices), 3, model.config.image_size, model.config.image_size, generator=generator)
    query_correct_images = sample_euler_cfg(
        model, query, noise, device=device, steps=ode_steps, cfg_scale=cfg_scale
    )
    query_wrong_images = sample_euler_cfg(
        model, query_wrong, noise, device=device, steps=ode_steps, cfg_scale=cfg_scale
    )
    positive_model = load_proven_cfm(positive_cfm_checkpoint, device)
    positive_images = sample_euler_cfg(
        positive_model, positive, noise, device=device, steps=ode_steps, cfg_scale=cfg_scale
    )
    positive_wrong_images = sample_euler_cfg(
        positive_model, positive_wrong, noise, device=device, steps=ode_steps, cfg_scale=cfg_scale
    )
    representation = str(split.manifest.get("representation", "condition"))
    condition_label = "DINO grid" if representation == "dino_grid_state" else "query latent"
    sample_dir = output_dir / f"positive_control_samples_{ode_steps}ode_cfg{cfg_scale:g}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    strips: list[Image.Image] = []
    for offset, index in enumerate(indices):
        gt = Image.fromarray(split.images_uint8[index].permute(1, 2, 0).numpy(), mode="RGB")
        strip = _label_strip(
            [
                gt,
                diffusion_tensor_to_pil(positive_images[offset]),
                diffusion_tensor_to_pil(positive_wrong_images[offset]),
                diffusion_tensor_to_pil(query_correct_images[offset]),
                diffusion_tensor_to_pil(query_wrong_images[offset]),
            ],
            ["GT", "Qwen positive", "Qwen wrong", f"{condition_label} CFM", f"{condition_label} wrong"],
        )
        strip.save(sample_dir / f"sample_{offset:03d}_strip.png")
        strips.append(strip)
    contact = _contact_sheet(strips, columns=1)
    contact_path = sample_dir / "contact_sheet.png"
    contact.save(contact_path)
    return {f"{ode_steps}ode_cfg{cfg_scale:g}": str(contact_path)}


def train_cfm_sft2(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 <= args.condition_dropout < 1.0:
        raise ValueError("condition_dropout must be in [0, 1)")
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
    token_count, token_dim = resolve_condition_token_shape(
        train_split.manifest,
        token_count_override=args.condition_token_count,
    )
    val_token_shape = resolve_condition_token_shape(
        val_split.manifest,
        token_count_override=args.condition_token_count,
    )
    if val_token_shape != (token_count, token_dim) or token_count * token_dim != condition_dim:
        raise ValueError(
            "CFM cache token shape mismatch: "
            f"train={(token_count, token_dim)}, val={val_token_shape}, flat={condition_dim}"
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
    if args.init_legacy_cfm_checkpoint is not None:
        initialization = initialize_from_legacy_cfm(model, args.init_legacy_cfm_checkpoint)
        print(json.dumps({"legacy_cfm_initialization": initialization}), flush=True)
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
        "task": "sft2_k8_direct_conditional_flow_matching_reconstruction",
        "source_checkpoint": str(args.source_checkpoint),
        "state_cache_dir": str(args.state_cache_dir),
        "wm_checkpoint": str(args.wm_checkpoint) if args.wm_checkpoint is not None else None,
        "split_semantics": (
            "same train-cache subset for explicit tiny-overfit diagnostics"
            if args.validation_cache_split == "train"
            else "strict-valid train_all for training; disjoint val_all for validation only"
        ),
        "target": {
            "qwen_query_hidden": "current_128px_image_from_qwen_query_hidden",
            "dino_grid_state": "current_128px_image_from_dino_supervised_grid_state",
        }.get(str(train_split.manifest.get("representation")), "current_128px_image_from_projected_sft2_state"),
        "trainable_modules": "TokenConditionedFlowUNet only",
        "frozen_modules": (
            "SFT1 Qwen and shared DINO-grid projector; no WM loaded"
            if str(train_split.manifest.get("representation")) == "dino_grid_state"
            else "SFT2 Qwen, StateProjector, WM predictor; WM predictor loaded only for projected-State samples"
        ),
        "invariants": invariants,
        "initialization": initialization,
        "condition_dropout": args.condition_dropout,
        "lr_schedule": {
            "initial": args.lr,
            "decay_step": args.lr_decay_step,
            "after_decay": args.lr_after_decay,
        },
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
        learning_rate = (
            args.lr_after_decay
            if args.lr_decay_step > 0 and step > args.lr_decay_step
            else args.lr
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
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
                "cfm/lr": learning_rate,
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
                    "cfm/lr": learning_rate,
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
    if uses_query_positive_control(train_split.manifest):
        if args.positive_cache_dir is None or args.positive_cfm_checkpoint is None:
            raise ValueError(
                "Positive-control sampling requires --positive-cache-dir and "
                "--positive-cfm-checkpoint"
            )
        sample_paths = _save_positive_control_samples(
            model=model,
            split=val_split,
            positive_cache_dir=args.positive_cache_dir / "val",
            positive_cfm_checkpoint=args.positive_cfm_checkpoint,
            output_dir=args.output_dir,
            device=device,
            num_items=args.sample_items,
            ode_steps=max(args.sample_ode_steps),
            cfg_scale=args.cfg_scale,
            seed=args.seed + 30_000,
        )
    else:
        if args.wm_checkpoint is None:
            raise ValueError("Projected-State sampling requires --wm-checkpoint")
        wm_predictor = LatentWMPredictor.load_checkpoint(
            args.wm_checkpoint,
            map_location=device,
            history_size_override=args.wm_history_size_override,
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
    parser.add_argument(
        "--wm-history-size-override",
        type=int,
        choices=[1],
        default=None,
        help="Explicit legacy checkpoint migration used only for projected-State samples.",
    )
    parser.add_argument(
        "--condition-token-count",
        type=int,
        default=0,
        help=(
            "Override only the CFM view of the flat condition. For example, "
            "Projected8192 with value8 becomes eight1024-d tokens while the source "
            "cache remains one Projected State. Zero preserves the cache-native view."
        ),
    )
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
    parser.add_argument(
        "--init-legacy-cfm-checkpoint",
        type=Path,
        default=None,
        help="Initialize all shape-compatible weights from the proven 16x512 ViT-token CFM",
    )
    parser.add_argument("--lr-decay-step", type=int, default=-1)
    parser.add_argument("--lr-after-decay", type=float, default=1e-5)
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
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument(
        "--positive-cache-dir",
        type=Path,
        default=None,
        help="Aligned Qwen positive-control cache root for multi-token sample sheets",
    )
    parser.add_argument("--positive-cfm-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-id", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_cfm_sft2(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
