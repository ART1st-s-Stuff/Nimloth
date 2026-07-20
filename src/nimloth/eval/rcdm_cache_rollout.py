"""Visualize multi-step WM rollouts using cached states and a trained RCDM."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.recon.rcdm.checkpoint import load_state_dict
from nimloth.recon.rcdm.config import RCDMConfig, create_model_and_diffusion
from nimloth.recon.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.recon.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.predictor import LatentWMPredictor

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


def validate_turn_window(actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise ValueError(f"selected rollout action mismatch: actual={actual}, expected={expected}")
    if len(actual) != 5:
        raise ValueError(f"rollout window must contain exactly five actions, got {len(actual)}")
    if 4 not in actual or 5 not in actual:
        raise ValueError(
            "manual rollout window must contain both turn_right (4) and turn_left (5)"
        )


def _load_rcdm_config(metadata_path: Path, timestep_respacing: str) -> RCDMConfig:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config_data = metadata.get("rcdm_config")
    if not isinstance(config_data, dict):
        raise ValueError(f"RCDM metadata has no rcdm_config: {metadata_path}")
    return replace(RCDMConfig(**config_data), timestep_respacing=timestep_respacing)


def _load_rows_by_record(cache_dir: Path) -> tuple[RCDMStateCacheDataset, dict[str, dict[int, dict[str, Any]]]]:
    dataset = RCDMStateCacheDataset(cache_dir)
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for index in range(len(dataset)):
        item = dataset[index]
        record_id = str(item.get("record_id", ""))
        step = int(item.get("step_index", -1))
        if not record_id or step < 0:
            continue
        if step in records[record_id]:
            raise ValueError(f"duplicate cached transition: record={record_id}, step={step}")
        records[record_id][step] = item
    return dataset, records


def _prepare_rollouts(
    *,
    records: dict[str, dict[int, dict[str, Any]]],
    selection_data: dict[str, Any],
    predictor: LatentWMPredictor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    oracle_conditions: list[torch.Tensor] = []
    predicted_conditions: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for run_index, selection in enumerate(selection_data["selections"]):
        record_id = str(selection["record_id"])
        start = int(selection["start_step"])
        by_step = records.get(record_id)
        if by_step is None:
            raise KeyError(f"selected record is absent from state cache: {record_id}")
        required = list(range(start, start + 6))
        missing = [step for step in required if step not in by_step]
        if missing:
            raise KeyError(f"selected record {record_id} lacks cached steps {missing}")
        actual_actions = [int(by_step[step]["action_index"]) for step in range(start, start + 5)]
        expected_actions = [int(action) for action in selection["expected_actions"]]
        validate_turn_window(actual_actions, expected_actions)
        actual_success = bool(by_step[start].get("success", False))
        if actual_success != bool(selection["success"]):
            raise ValueError(
                f"selected record success mismatch for {record_id}: "
                f"cache={actual_success}, selection={selection['success']}"
            )

        initial_state = by_step[start]["state_emb"].float().unsqueeze(0).to(device)
        actions = torch.tensor(actual_actions, dtype=torch.long, device=device).unsqueeze(0)
        predicted = predictor.rollout_states(initial_state, actions)[0].detach().cpu()
        for offset in range(1, 6):
            target_step = start + offset
            oracle_conditions.append(by_step[target_step]["state_emb"].float())
            predicted_conditions.append(predicted[offset - 1])
            action_index = actual_actions[offset - 1]
            rows.append(
                {
                    "run_index": run_index,
                    "record_id": record_id,
                    "source_success": actual_success,
                    "window_start_step": start,
                    "rollout_offset": offset,
                    "target_step_index": target_step,
                    "action_index": action_index,
                    "action_name": ACTION_NAMES[action_index],
                    "gt_image_path": str(by_step[target_step]["current_image_path"]),
                    "action_sequence": actual_actions,
                    "action_names": [ACTION_NAMES[action] for action in actual_actions],
                }
            )
    return torch.stack(oracle_conditions), torch.stack(predicted_conditions), rows


@torch.no_grad()
def _sample_conditions(
    *,
    model,
    diffusion,
    conditions: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    image_size: int,
    chunk_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, conditions.shape[0], chunk_size):
        condition = conditions[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        chunk_noise = noise[start : start + chunk_size].to(
            device=device, dtype=torch.float32
        )
        sample = diffusion.ddim_sample_loop(
            model,
            (condition.shape[0], 3, image_size, image_size),
            noise=chunk_noise,
            clip_denoised=True,
            model_kwargs={"feat": condition},
            device=device,
        )
        outputs.append(sample.detach().cpu())
    return torch.cat(outputs, dim=0)


def _label_strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 20
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images) + label_height),
        "white",
    )
    draw = ImageDraw.Draw(output)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (offset, label_height))
        draw.text((offset + 2, 3), label, fill=(0, 0, 0))
        offset += image.width
    return output


def _vertical_stack(images: list[Image.Image]) -> Image.Image:
    output = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images)),
        "white",
    )
    offset = 0
    for image in images:
        output.paste(image, (0, offset))
        offset += image.height
    return output


def _contact_sheet(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new(
        "RGB", (columns * width, math.ceil(len(images) / columns) * height), "white"
    )
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


def _maybe_upload_wandb(args: argparse.Namespace, contact_path: Path, num_rows: int) -> str | None:
    if args.no_wandb:
        return None
    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow",
            dir=str(args.output_dir),
        )
        run.log(
            {
                args.wandb_key: wandb.Image(str(contact_path)),
                "rcdm_rollout5_turns/num_sequences": num_rows // 5,
                "rcdm_rollout5_turns/num_temporal_steps": 5,
            },
            step=args.wandb_step,
        )
        url = run.url
        run.finish()
        return url
    except Exception as exc:
        print(json.dumps({"wandb_upload_skipped": str(exc)}))
        return None


@torch.no_grad()
def sample_rcdm_cache_rollout(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_data = json.loads(args.selections.read_text(encoding="utf-8"))
    dataset, records = _load_rows_by_record(args.state_cache_dir)
    manifest_data = json.loads(
        (args.state_cache_dir / "manifest.json").read_text(encoding="utf-8")
    )
    actual_k = int(manifest_data.get("latent_token_count", 1))
    if actual_k != args.latent_token_count:
        raise ValueError(
            f"state cache k mismatch: expected={args.latent_token_count}, actual={actual_k}"
        )

    predictor = LatentWMPredictor.load_checkpoint(
        args.wm_checkpoint, map_location=device
    ).to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    oracle_conditions, predicted_conditions, rows = _prepare_rollouts(
        records=records,
        selection_data=selection_data,
        predictor=predictor,
        device=device,
    )

    config = _load_rcdm_config(args.metadata, args.timestep_respacing)
    model, diffusion = create_model_and_diffusion(
        config,
        cond_dim=dataset.manifest.cond_dim,
        rcdm_root=str(args.rcdm_root) if args.rcdm_root is not None else None,
    )
    model.load_state_dict(
        load_state_dict(args.rcdm_checkpoint, map_location="cpu"), strict=True
    )
    model.to(device).eval()
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(
        (len(rows), 3, config.image_size, config.image_size), generator=generator
    )
    oracle_samples = _sample_conditions(
        model=model,
        diffusion=diffusion,
        conditions=oracle_conditions,
        noise=noise,
        device=device,
        image_size=config.image_size,
        chunk_size=args.batch_size,
    )
    predicted_samples = _sample_conditions(
        model=model,
        diffusion=diffusion,
        conditions=predicted_conditions,
        noise=noise,
        device=device,
        image_size=config.image_size,
        chunk_size=args.batch_size,
    )

    run_strips: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        gt = diffusion_tensor_to_pil(
            image_to_diffusion_tensor(row["gt_image_path"], image_size=config.image_size)
        )
        oracle = diffusion_tensor_to_pil(oracle_samples[index])
        predicted = diffusion_tensor_to_pil(predicted_samples[index])
        prefix = f"t+{row['rollout_offset']} {row['action_name']}"
        strip = _label_strip(
            [gt, oracle, predicted],
            [f"{prefix} GT", "GT-state RCDM", "pred-state RCDM"],
        )
        strip_path = args.output_dir / (
            f"run_{row['run_index']:02d}_step_{row['rollout_offset']:02d}_strip.png"
        )
        strip.save(strip_path)
        row["strip_path"] = str(strip_path)
        run_strips[int(row["run_index"])].append(strip)

    run_sheets: list[Image.Image] = []
    for run_index in sorted(run_strips):
        sheet = _vertical_stack(run_strips[run_index])
        sheet_path = args.output_dir / f"run_{run_index:02d}_rollout5.png"
        sheet.save(sheet_path)
        run_sheets.append(sheet)
    contact = _contact_sheet(run_sheets, columns=2)
    contact_path = args.output_dir / "contact_sheet_rollout5_turn_left_right.png"
    contact.save(contact_path)
    (args.output_dir / "samples.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "completed",
        "state_cache_dir": str(args.state_cache_dir),
        "state_cache_fingerprint": manifest_data.get("fingerprint"),
        "latent_token_count": actual_k,
        "wm_checkpoint": str(args.wm_checkpoint),
        "rcdm_checkpoint": str(args.rcdm_checkpoint),
        "rcdm_metadata": str(args.metadata),
        "selections": str(args.selections),
        "num_sequences": len(run_sheets),
        "temporal_steps": 5,
        "sampler": "DDIM",
        "timestep_respacing": args.timestep_respacing,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "columns": ["GT image", "GT-state RCDM", "predicted-state RCDM"],
        "contact_sheet": str(contact_path),
    }
    wandb_url = _maybe_upload_wandb(args, contact_path, len(rows))
    metadata["wandb_url"] = wandb_url
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Five-action WM rollout reconstruction from cached k=8 states"
    )
    parser.add_argument("--state-cache-dir", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--rcdm-checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rcdm-root", type=Path, default=None)
    parser.add_argument("--latent-token-count", type=int, default=8)
    parser.add_argument("--timestep-respacing", default="ddim250")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-id", default="v8xoufn6")
    parser.add_argument("--wandb-step", type=int, default=7425)
    parser.add_argument(
        "--wandb-key", default="rcdm_rollout5_turn_left_right/contact_sheet"
    )
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return sample_rcdm_cache_rollout(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
