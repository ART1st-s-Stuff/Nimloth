"""Frozen reconstruction probe for actual and WM-predicted DINO-grid states.

The evaluator encodes real states with one SFT2 checkpoint, rolls its world
model forward with recorded actions, and visualizes both actual and predicted
grids through the same older DINO-grid CFM decoder.  The older Qwen-token and
SFT1 DINO-grid reconstructions are retained as positive and lineage controls.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.eval.cfm_k8_vs_vit import _load_current_cfm, _load_legacy_cfm, _sample_cfg
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.recon.rcdm.image_utils import (
    diffusion_tensor_to_pil,
    image_to_diffusion_tensor,
)
from nimloth.recon.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.rollout.transitions import (
    TransitionJsonlDataset,
    TransitionSample,
    transition_training_item,
)
from nimloth.training.rl.planning_loader import load_planning_world_model


ACTION_NAMES = (
    "move_forward",
    "move_backward",
    "move_right",
    "move_left",
    "turn_right",
    "turn_left",
    "look_up",
    "look_down",
)

COLUMNS = (
    "GT",
    "Qwen ViT-token CFM",
    "old SFT1 DINO-grid CFM",
    "ID56 actual grid",
    "ID56 WM-predicted grid",
)

PROTOCOL = "id56_actual_vs_autoregressive_wm_predicted_dino_grid_v1"


def _freeze(module: torch.nn.Module) -> None:
    module.eval().requires_grad_(False)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing state-cache manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache_records(
    cache_dir: Path,
    *,
    representation: str,
    state_shape: tuple[int, ...],
) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Load a cache into a strict record/step index and validate its semantics."""

    manifest = _manifest(cache_dir)
    if manifest.get("representation") != representation:
        raise ValueError(
            "cache representation mismatch: "
            f"expected={representation!r}, actual={manifest.get('representation')!r}, "
            f"path={cache_dir}"
        )
    actual_shape = tuple(int(value) for value in manifest.get("state_shape", ()))
    if actual_shape != state_shape:
        raise ValueError(
            f"cache state_shape mismatch: expected={state_shape}, "
            f"actual={actual_shape}, path={cache_dir}"
        )
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    dataset = RCDMStateCacheDataset(cache_dir)
    for index in range(len(dataset)):
        item = dataset[index]
        record_id = str(item["record_id"])
        step = int(item["step_index"])
        if step in records[record_id]:
            raise ValueError(f"duplicate cache row: {record_id} step{step}")
        state = item["state_emb"]
        if tuple(state.shape) != state_shape:
            raise ValueError(
                f"cache row shape mismatch: {record_id} step{step} "
                f"expected={state_shape}, actual={tuple(state.shape)}"
            )
        records[record_id][step] = item
    return records, manifest


def _sample_index(jsonl_path: Path) -> dict[str, dict[int, TransitionSample]]:
    records: dict[str, dict[int, TransitionSample]] = defaultdict(dict)
    for sample in TransitionJsonlDataset(
        jsonl_path,
        max_records=-1,
        success_only=False,
    ).samples:
        if sample.step_index in records[sample.record_id]:
            raise ValueError(
                f"duplicate transition sample: {sample.record_id} step{sample.step_index}"
            )
        records[sample.record_id][sample.step_index] = sample
    return records


def prepare_protocol_rows(
    *,
    selections: list[dict[str, Any]],
    current_samples: dict[str, dict[int, TransitionSample]],
    old_grid_records: dict[str, dict[int, dict[str, Any]]],
    qwen_records: dict[str, dict[int, dict[str, Any]]],
    horizon: int,
) -> tuple[list[dict[str, Any]], list[list[TransitionSample]], dict[str, torch.Tensor]]:
    """Validate all three data sources and assemble the frozen protocol rows."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    rows: list[dict[str, Any]] = []
    state_samples: list[list[TransitionSample]] = []
    old_grid_states: list[torch.Tensor] = []
    qwen_states: list[torch.Tensor] = []
    seen_runs: set[int] = set()
    for selection in selections:
        run_index = int(selection["run_index"])
        if run_index in seen_runs:
            raise ValueError(f"duplicate run_index in selections: {run_index}")
        seen_runs.add(run_index)
        record_id = str(selection["record_id"])
        expected = [int(value) for value in selection["expected_actions"]]
        if len(expected) < horizon:
            raise ValueError(
                f"selection {record_id} has {len(expected)} actions, needs {horizon}"
            )

        current = current_samples.get(record_id)
        old_grid = old_grid_records.get(record_id)
        qwen = qwen_records.get(record_id)
        if current is None or old_grid is None or qwen is None:
            missing = [
                name
                for name, value in (
                    ("current_jsonl", current),
                    ("old_dino_grid_cache", old_grid),
                    ("qwen_cache", qwen),
                )
                if value is None
            ]
            raise KeyError(f"selected record {record_id} missing from {missing}")

        needed_steps = range(horizon + 1)
        for name, source in (
            ("current_jsonl", current),
            ("old_dino_grid_cache", old_grid),
            ("qwen_cache", qwen),
        ):
            absent = [step for step in needed_steps if step not in source]
            if absent:
                raise KeyError(f"{record_id} missing steps {absent} from {name}")

        current_actions = [int(current[step].action_index) for step in range(horizon)]
        old_grid_actions = [
            int(old_grid[step]["action_index"]) for step in range(horizon)
        ]
        qwen_actions = [int(qwen[step]["action_index"]) for step in range(horizon)]
        expected_prefix = expected[:horizon]
        if not (
            current_actions == expected_prefix
            and old_grid_actions == expected_prefix
            and qwen_actions == expected_prefix
        ):
            raise ValueError(
                f"action mismatch for {record_id}: expected={expected_prefix}, "
                f"current={current_actions}, old_grid={old_grid_actions}, "
                f"qwen={qwen_actions}"
            )

        trajectory_samples = [current[step] for step in needed_steps]
        state_samples.append(trajectory_samples)
        for step in range(1, horizon + 1):
            reference_image = str(current[step].current_image_path)
            for name, source in (("old_grid", old_grid), ("qwen", qwen)):
                item = source[step]
                if str(item.get("record_id", "")) != record_id:
                    raise ValueError(f"record mismatch for {record_id} step{step} {name}")
                if int(item.get("step_index", -1)) != step:
                    raise ValueError(f"step mismatch for {record_id} step{step} {name}")
                if str(item.get("current_image_path", "")) != reference_image:
                    raise ValueError(
                        f"image mismatch for {record_id} step{step} {name}: "
                        f"{item.get('current_image_path')!r} != {reference_image!r}"
                    )
            old_grid_states.append(old_grid[step]["state_emb"].float())
            qwen_states.append(qwen[step]["state_emb"].float())
            rows.append(
                {
                    "run_index": run_index,
                    "record_id": record_id,
                    "scene_note": str(selection.get("scene_note", "")),
                    "step_index": step,
                    "horizon": step,
                    "action_index": expected[step - 1],
                    "action_name": ACTION_NAMES[expected[step - 1]],
                    "action_prefix": expected[:step],
                    "gt_image_path": reference_image,
                }
            )
    return (
        rows,
        state_samples,
        {
            "old_grid": torch.stack(old_grid_states),
            "qwen": torch.stack(qwen_states),
        },
    )


def validate_dino_cfm_lineage(
    checkpoint: Path,
    old_grid_manifest: dict[str, Any],
) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    invariants = payload.get("invariants")
    if not isinstance(invariants, dict):
        raise ValueError("DINO-grid CFM checkpoint lacks training invariants")
    expected_fingerprint = str(old_grid_manifest["fingerprint"])
    if str(invariants.get("val_cache_fingerprint")) != expected_fingerprint:
        raise ValueError(
            "DINO-grid CFM/cache fingerprint mismatch: "
            f"checkpoint={invariants.get('val_cache_fingerprint')!r}, "
            f"cache={expected_fingerprint!r}"
        )
    config = invariants.get("cfm_config", {})
    shape = (int(config.get("token_count", -1)), int(config.get("token_dim", -1)))
    if shape != (16, 1024):
        raise ValueError(f"DINO-grid CFM condition shape must be (16, 1024), got {shape}")


def _load_qwen_checkpoint(
    checkpoint: Path,
    *,
    max_pixels: int,
    latent_token_count: int,
    attn_implementation: str,
    device: torch.device,
):
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = int(max_pixels)
    added = add_special_tokens(
        processor.tokenizer,
        latent_token_count=latent_token_count,
    )
    if added:
        raise ValueError(
            f"SFT2 checkpoint tokenizer is missing {added} required latent/action tokens"
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if model.get_input_embeddings().num_embeddings != len(processor.tokenizer):
        raise ValueError(
            "SFT2 checkpoint model/tokenizer vocabulary mismatch: "
            f"model={model.get_input_embeddings().num_embeddings}, "
            f"tokenizer={len(processor.tokenizer)}"
        )
    model.to(device)
    _freeze(model)
    return processor, special_token_ids(
        processor.tokenizer,
        latent_token_count=latent_token_count,
    ), model


@torch.no_grad()
def encode_actual_and_predicted_states(
    *,
    trajectory_samples: list[list[TransitionSample]],
    sft2_checkpoint: Path,
    max_length: int,
    max_pixels: int,
    latent_token_count: int,
    attn_implementation: str,
    encode_batch_size: int,
    device: torch.device,
    progress_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every real state and autoregress from each real initial state."""

    if not trajectory_samples:
        raise ValueError("trajectory_samples must not be empty")
    horizon = len(trajectory_samples[0]) - 1
    if horizon < 1 or any(len(samples) != horizon + 1 for samples in trajectory_samples):
        raise ValueError("all selected trajectories must contain the same positive horizon")
    progress_paths: list[Path | None] = []
    cached_payloads: list[dict[str, torch.Tensor] | None] = []
    if progress_dir is not None:
        progress_dir.mkdir(parents=True, exist_ok=True)
    for trajectory_index, trajectory in enumerate(trajectory_samples):
        path = (
            progress_dir / f"trajectory_{trajectory_index:04d}.pt"
            if progress_dir is not None
            else None
        )
        progress_paths.append(path)
        payload = None
        if path is not None and path.is_file():
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            expected_record = trajectory[0].record_id
            expected_actions = [
                int(sample.action_index) for sample in trajectory[:horizon]
            ]
            if (
                str(loaded.get("record_id", "")) != expected_record
                or loaded.get("actions", torch.empty(0, dtype=torch.long)).tolist()
                != expected_actions
            ):
                raise ValueError(
                    f"trajectory progress contract mismatch: {path}"
                )
            payload = loaded
        cached_payloads.append(payload)
    if all(payload is not None for payload in cached_payloads):
        return (
            torch.cat([payload["actual"] for payload in cached_payloads if payload]),
            torch.cat([payload["predicted"] for payload in cached_payloads if payload]),
        )

    processor, token_id_map, qwen_model = _load_qwen_checkpoint(
        sft2_checkpoint,
        max_pixels=max_pixels,
        latent_token_count=latent_token_count,
        attn_implementation=attn_implementation,
        device=device,
    )
    world_model = load_planning_world_model(
        qwen_config=qwen_model.config,
        wm_checkpoint=sft2_checkpoint / "wm_predictor",
        state_proj_checkpoint=sft2_checkpoint / "state_proj.pt",
        value_head_checkpoint=sft2_checkpoint / "value_head",
        device=device,
    )

    actual_groups: list[torch.Tensor] = []
    predicted_groups: list[torch.Tensor] = []
    for trajectory_index, trajectory in enumerate(trajectory_samples):
        cached = cached_payloads[trajectory_index]
        if cached is not None:
            actual_groups.append(cached["actual"].float())
            predicted_groups.append(cached["predicted"].float())
            continue
        encoded_states: list[torch.Tensor] = []
        for start in range(0, len(trajectory), encode_batch_size):
            items = [
                transition_training_item(sample)
                for sample in trajectory[start : start + encode_batch_size]
            ]
            encoding = build_qwen_batch(
                items,
                processor,
                max_length=max_length,
                latent_token_count=latent_token_count,
            )
            hidden, _loss = extract_qwen_latents(
                qwen_model,
                encoding,
                token_id_map,
                device,
                latent_token_count=latent_token_count,
            )
            encoded_states.append(world_model.project_state(hidden).detach().cpu())
        all_states = torch.cat(encoded_states)
        actions = torch.tensor(
            [[int(sample.action_index) for sample in trajectory[:horizon]]],
            dtype=torch.long,
            device=device,
        )
        predicted = world_model.simulate_action_sequences(
            all_states[0:1].to(device=device, dtype=torch.float32).unsqueeze(1),
            actions.new_empty((1, 0)),
            actions,
        )[0].detach().cpu().float()
        actual = all_states[1:].float()
        actual_groups.append(actual)
        predicted_groups.append(predicted)
        path = progress_paths[trajectory_index]
        if path is not None:
            temporary = path.with_suffix(".pt.tmp")
            torch.save(
                {
                    "record_id": trajectory[0].record_id,
                    "actions": actions[0].detach().cpu(),
                    "actual": actual,
                    "predicted": predicted,
                },
                temporary,
            )
            temporary.replace(path)
        print(
            json.dumps(
                {
                    "trajectory_encoding": trajectory_index + 1,
                    "total": len(trajectory_samples),
                    "record_id": trajectory[0].record_id,
                }
            ),
            flush=True,
        )
    return torch.cat(actual_groups), torch.cat(predicted_groups)


def calculate_metrics(
    *,
    rows: list[dict[str, Any]],
    states: dict[str, torch.Tensor],
    images: dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics: dict[str, float] = {}
    image_l1: dict[str, torch.Tensor] = {}
    for name, image in images.items():
        values = (image - gt).abs().flatten(1).mean(1)
        image_l1[name] = values
        metrics[f"image/{name}_to_gt_l1"] = float(values.mean())
    metrics["image/predicted_to_actual_output_l1"] = float(
        (images["id56_predicted"] - images["id56_actual"]).abs().mean()
    )
    metrics["image/id56_predicted_better_frame_fraction"] = float(
        (image_l1["id56_predicted"] < image_l1["id56_actual"]).float().mean()
    )

    state_pairs = {
        "predicted_to_actual": (states["id56_predicted"], states["id56_actual"]),
        "id56_actual_to_old_sft1": (states["id56_actual"], states["old_grid"]),
        "id56_predicted_to_old_sft1": (states["id56_predicted"], states["old_grid"]),
    }
    state_rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, (left, right) in state_pairs.items():
        mse = (left - right).square().flatten(1).mean(1)
        cosine = F.cosine_similarity(left.flatten(1), right.flatten(1))
        state_rows[name] = (mse, cosine)
        metrics[f"state/{name}_mse"] = float(mse.mean())
        metrics[f"state/{name}_cos"] = float(cosine.mean())

    horizon_metrics: dict[str, dict[str, float]] = {}
    horizons = sorted({int(row["horizon"]) for row in rows})
    for horizon in horizons:
        indices = [index for index, row in enumerate(rows) if int(row["horizon"]) == horizon]
        predicted_mse, predicted_cos = state_rows["predicted_to_actual"]
        horizon_metrics[str(horizon)] = {
            "count": len(indices),
            "state_predicted_to_actual_mse": float(predicted_mse[indices].mean()),
            "state_predicted_to_actual_cos": float(predicted_cos[indices].mean()),
            "image_id56_actual_to_gt_l1": float(image_l1["id56_actual"][indices].mean()),
            "image_id56_predicted_to_gt_l1": float(
                image_l1["id56_predicted"][indices].mean()
            ),
            "image_predicted_to_actual_output_l1": float(
                (
                    images["id56_predicted"][indices]
                    - images["id56_actual"][indices]
                )
                .abs()
                .mean()
            ),
        }
    return metrics, horizon_metrics


def _strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), images[0].height + label_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    x = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (x, label_height))
        draw.text((x + 2, 2), label, fill=(0, 0, 0))
        x += image.width
    return output


def _vertical(images: list[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (images[0].width, sum(image.height for image in images)), "white")
    y = 0
    for image in images:
        output.paste(image, (0, y))
        y += image.height
    return output


def _contact(images: list[Image.Image], columns: int) -> Image.Image:
    width, height = images[0].size
    output = Image.new(
        "RGB",
        (columns * width, math.ceil(len(images) / columns) * height),
        "white",
    )
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


def _wandb_upload(
    args: argparse.Namespace,
    contact_paths: list[Path],
    metrics: dict[str, float],
    horizon_metrics: dict[str, dict[str, float]],
) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text(encoding="utf-8").strip() if id_path.is_file() else None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id,
        resume="must" if run_id else None,
        dir=str(args.output_dir),
        config={
            "protocol": PROTOCOL,
            "horizon": args.horizon,
            "steps": args.steps,
            "cfg_scale": args.cfg_scale,
            "matched_noise": True,
            "sft2_checkpoint": str(args.sft2_checkpoint),
            "dino_grid_cfm_checkpoint": str(args.dino_grid_cfm_checkpoint),
        },
    )
    id_path.write_text(run.id + "\n", encoding="utf-8")
    payload: dict[str, Any] = dict(metrics)
    for horizon, values in horizon_metrics.items():
        payload.update({f"horizon/{horizon}/{key}": value for key, value in values.items()})
    for index, path in enumerate(contact_paths):
        payload[f"wm_predicted_reconstruction/contact_{index:02d}"] = wandb.Image(str(path))
    run.log(payload)
    url = run.url
    run.finish()
    return url


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    if args.horizon != 4:
        raise ValueError(
            "this formal ID56 protocol requires horizon=4; use a separate explicitly "
            "labelled run for extrapolation beyond its T=4 training objective"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --resume only "
            "for the same run contract"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selections_payload = json.loads(args.selections.read_text(encoding="utf-8"))
    selections = list(selections_payload["selections"])
    old_grid_records, old_grid_manifest = load_cache_records(
        args.old_dino_grid_cache,
        representation="dino_grid_state",
        state_shape=(16, 1024),
    )
    qwen_records, qwen_manifest = load_cache_records(
        args.qwen_cache,
        representation="qwen_compressed_vision_positive",
        state_shape=(16, 512),
    )
    validate_dino_cfm_lineage(args.dino_grid_cfm_checkpoint, old_grid_manifest)
    rows, trajectory_samples, cached_states = prepare_protocol_rows(
        selections=selections,
        current_samples=_sample_index(args.val_jsonl),
        old_grid_records=old_grid_records,
        qwen_records=qwen_records,
        horizon=args.horizon,
    )
    contract = {
        "protocol": PROTOCOL,
        "columns": list(COLUMNS),
        "num_runs": len(selections),
        "num_rows": len(rows),
        "horizon": args.horizon,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "matched_noise": True,
        "sft2_checkpoint": str(args.sft2_checkpoint),
        "val_jsonl": str(args.val_jsonl),
        "old_dino_grid_cache": str(args.old_dino_grid_cache),
        "old_dino_grid_fingerprint": old_grid_manifest["fingerprint"],
        "qwen_cache": str(args.qwen_cache),
        "qwen_fingerprint": qwen_manifest["fingerprint"],
        "dino_grid_cfm_checkpoint": str(args.dino_grid_cfm_checkpoint),
        "qwen_cfm_checkpoint": str(args.qwen_cfm_checkpoint),
        "selections": str(args.selections),
        "backbone_weights": "online checkpoint weights; vision_ema.pt is not applied",
    }
    contract_path = args.output_dir / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("--resume contract differs from existing output contract")
    else:
        _atomic_json(contract_path, contract)

    states_path = args.output_dir / "states.pt"
    if states_path.is_file():
        state_payload = torch.load(states_path, map_location="cpu", weights_only=True)
        actual = state_payload["id56_actual"].float()
        predicted = state_payload["id56_predicted"].float()
    else:
        actual, predicted = encode_actual_and_predicted_states(
            trajectory_samples=trajectory_samples,
            sft2_checkpoint=args.sft2_checkpoint,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
            latent_token_count=16,
            attn_implementation=args.attn_implementation,
            encode_batch_size=args.encode_batch_size,
            device=device,
            progress_dir=args.output_dir / "state_progress",
        )
        temporary = states_path.with_suffix(".pt.tmp")
        torch.save({"id56_actual": actual, "id56_predicted": predicted}, temporary)
        temporary.replace(states_path)
    if actual.shape != cached_states["old_grid"].shape or predicted.shape != actual.shape:
        raise ValueError(
            "ID56/old-grid state shape mismatch: "
            f"actual={tuple(actual.shape)}, predicted={tuple(predicted.shape)}, "
            f"old={tuple(cached_states['old_grid'].shape)}"
        )
    states = {
        **cached_states,
        "id56_actual": actual,
        "id56_predicted": predicted,
    }

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dino_cfm = _load_current_cfm(args.dino_grid_cfm_checkpoint, device)
    qwen_cfm = _load_legacy_cfm(args.qwen_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    images = {
        "qwen": _sample_cfg(
            qwen_cfm,
            states["qwen"].flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        ),
        "old_dino_grid": _sample_cfg(
            dino_cfm,
            states["old_grid"].flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        ),
        "id56_actual": _sample_cfg(
            dino_cfm,
            states["id56_actual"].flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        ),
        "id56_predicted": _sample_cfg(
            dino_cfm,
            states["id56_predicted"].flatten(1),
            noise,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            chunk_size=args.chunk_size,
            device=device,
        ),
    }
    gt = torch.stack(
        [
            image_to_diffusion_tensor(row["gt_image_path"], image_size=128)
            for row in rows
        ]
    )
    metrics, horizon_metrics = calculate_metrics(
        rows=rows,
        states=states,
        images=images,
        gt=gt,
    )

    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        strip = _strip(
            [
                diffusion_tensor_to_pil(gt[index]),
                diffusion_tensor_to_pil(images["qwen"][index]),
                diffusion_tensor_to_pil(images["old_dino_grid"][index]),
                diffusion_tensor_to_pil(images["id56_actual"][index]),
                diffusion_tensor_to_pil(images["id56_predicted"][index]),
            ],
            [
                f"run{row['run_index']} t+{row['horizon']} {row['action_name']} GT",
                *COLUMNS[1:],
            ],
        )
        path = args.output_dir / (
            f"run_{int(row['run_index']):02d}_step_{int(row['step_index']):02d}.png"
        )
        strip.save(path)
        row["strip_path"] = str(path)
        run_rows[int(row["run_index"])].append(strip)

    run_sheets: list[tuple[int, Image.Image]] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        path = args.output_dir / f"run_{run_index:02d}.png"
        sheet.save(path)
        run_sheets.append((run_index, sheet))
    contacts: list[Path] = []
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start : start + args.runs_per_contact]
        path = args.output_dir / (
            f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        )
        _contact([sheet for _index, sheet in group], args.contact_columns).save(path)
        contacts.append(path)

    _atomic_json(args.output_dir / "samples.json", rows)
    wandb_url = _wandb_upload(args, contacts, metrics, horizon_metrics)
    metadata = {
        **contract,
        "status": "completed",
        "metrics": metrics,
        "horizon_metrics": horizon_metrics,
        "contact_sheets": [str(path) for path in contacts],
        "wandb_url": wandb_url,
    }
    _atomic_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct ID56 actual and autoregressive WM-predicted DINO grids"
    )
    parser.add_argument("--sft2-checkpoint", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--old-dino-grid-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--dino-grid-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--encode-batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--runs-per-contact", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
