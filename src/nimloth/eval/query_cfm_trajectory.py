"""Visualize direct 8-query CFM reconstruction along recorded action sequences."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.cfm import CFMConfig, TokenConditionedFlowUNet
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.rcdm.state_cache import RCDMStateCacheDataset

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


def _load_query_cfm(path: Path, device: torch.device) -> TokenConditionedFlowUNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = CFMConfig(**payload["invariants"]["cfm_config"])
    if (config.token_count, config.token_dim) != (8, 2048):
        raise ValueError(
            "query trajectory evaluator requires an 8x2048 CFM, got "
            f"{config.token_count}x{config.token_dim}"
        )
    model = TokenConditionedFlowUNet(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def _load_cache(cache_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("representation") != "qwen_query_hidden":
        raise ValueError(f"expected qwen_query_hidden cache: {cache_dir}")
    if [int(value) for value in manifest.get("state_shape", [])] != [8, 2048]:
        raise ValueError(f"expected state_shape [8,2048]: {cache_dir}")
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    dataset = RCDMStateCacheDataset(cache_dir)
    for index in range(len(dataset)):
        item = dataset[index]
        record_id = str(item["record_id"])
        step = int(item["step_index"])
        if step in records[record_id]:
            raise ValueError(f"duplicate query cache row: {record_id} step{step}")
        records[record_id][step] = item
    return records


def prepare_trajectory_rows(
    selections: list[dict[str, Any]],
    caches: dict[str, dict[str, dict[int, dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    """Validate five-action trajectories and return GT and wrong conditions.

    Wrong controls use the next selected run's condition at the same temporal
    offset, so correct/wrong branches share noise and horizon while changing
    scene identity.
    """

    rows: list[dict[str, Any]] = []
    conditions_by_run: list[list[torch.Tensor]] = []
    for selection in selections:
        source = str(selection["source"])
        record_id = str(selection["record_id"])
        expected = [int(action) for action in selection["expected_actions"]]
        if len(expected) != 5:
            raise ValueError(f"selection must contain five actions: {record_id}")
        if source not in caches:
            raise KeyError(f"unknown query cache source {source!r}")
        record = caches[source].get(record_id)
        if record is None:
            raise KeyError(f"record missing from {source} query cache: {record_id}")
        missing = [step for step in range(6) if step not in record]
        if missing:
            raise KeyError(f"record {record_id} missing steps {missing}")
        actual = [int(record[step]["action_index"]) for step in range(5)]
        if actual != expected:
            raise ValueError(
                f"action mismatch for {record_id}: expected={expected}, actual={actual}"
            )
        run_conditions: list[torch.Tensor] = []
        for step in range(1, 6):
            state = record[step]["state_emb"].float()
            if tuple(state.shape) != (8, 2048):
                raise ValueError(f"wrong query state shape for {record_id} step{step}: {state.shape}")
            run_conditions.append(state.reshape(-1))
            rows.append(
                {
                    "run_index": int(selection["run_index"]),
                    "source": source,
                    "record_id": record_id,
                    "step_index": step,
                    "action_index": expected[step - 1],
                    "action_name": ACTION_NAMES[expected[step - 1]],
                    "action_prefix": expected[:step],
                    "action_names": [ACTION_NAMES[action] for action in expected[:step]],
                    "gt_image_path": str(record[step]["current_image_path"]),
                }
            )
        conditions_by_run.append(run_conditions)
    correct = torch.stack([condition for run in conditions_by_run for condition in run])
    wrong = torch.stack(
        [
            condition
            for run_index in range(len(conditions_by_run))
            for condition in conditions_by_run[(run_index + 1) % len(conditions_by_run)]
        ]
    )
    return rows, correct, wrong


@torch.no_grad()
def _sample_cfg(
    model: TokenConditionedFlowUNet,
    condition: torch.Tensor,
    noise: torch.Tensor,
    *,
    steps: int,
    cfg_scale: float,
    chunk_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, condition.shape[0], chunk_size):
        cond = condition[start : start + chunk_size].to(device=device, dtype=torch.float32)
        image = noise[start : start + chunk_size].to(device=device, dtype=torch.float32).clone()
        uncond = torch.zeros_like(cond)
        delta = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (cond.shape[0],),
                (index + 0.5) / steps,
                device=device,
                dtype=torch.float32,
            )
            unconditional_velocity = model(image, time, uncond)
            conditional_velocity = model(image, time, cond)
            image = image + delta * (
                unconditional_velocity
                + cfg_scale * (conditional_velocity - unconditional_velocity)
            )
        outputs.append(image.clamp(-1, 1).cpu())
    return torch.cat(outputs)


def _strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 18
    output = Image.new("RGB", (sum(image.width for image in images), 128 + label_height), "white")
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


def _contact(images: list[Image.Image], columns: int = 2) -> Image.Image:
    width, height = images[0].size
    output = Image.new("RGB", (columns * width, math.ceil(len(images) / columns) * height), "white")
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


def _wandb_upload(args: argparse.Namespace, contact_path: Path, metrics: dict[str, float]) -> str | None:
    if args.no_wandb:
        return None
    try:
        import wandb

        id_path = args.output_dir / "wandb_run_id.txt"
        run_id = id_path.read_text(encoding="utf-8").strip() if id_path.is_file() else None
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=run_id,
            resume="allow" if run_id else None,
            dir=str(args.output_dir),
        )
        id_path.write_text(run.id, encoding="utf-8")
        run.log({args.wandb_key: wandb.Image(str(contact_path)), **metrics})
        url = run.url
        run.finish()
        return url
    except Exception as exc:
        print(json.dumps({"wandb_upload_skipped": str(exc)}), flush=True)
        return None


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection_data = json.loads(args.selections.read_text(encoding="utf-8"))
    caches = {
        "old": _load_cache(args.old_query_cache),
        "current": _load_cache(args.current_query_cache),
    }
    rows, correct_condition, wrong_condition = prepare_trajectory_rows(
        selection_data["selections"], caches
    )
    model = _load_query_cfm(args.query_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    correct_images = _sample_cfg(
        model,
        correct_condition,
        noise,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        chunk_size=args.chunk_size,
        device=device,
    )
    wrong_images = _sample_cfg(
        model,
        wrong_condition,
        noise,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        chunk_size=args.chunk_size,
        device=device,
    )
    gt = torch.stack(
        [image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows]
    )
    metrics = {
        "trajectory/correct_to_gt_l1": float((correct_images - gt).abs().mean()),
        "trajectory/wrong_to_gt_l1": float((wrong_images - gt).abs().mean()),
        "trajectory/wrong_over_correct_gt_l1": float(
            (wrong_images - gt).abs().mean() / (correct_images - gt).abs().mean()
        ),
        "trajectory/correct_wrong_output_l1": float((correct_images - wrong_images).abs().mean()),
    }
    source_metrics: dict[str, dict[str, float]] = {}
    for source in ("old", "current"):
        indices = [index for index, row in enumerate(rows) if row["source"] == source]
        if not indices:
            continue
        source_metrics[source] = {
            "correct_to_gt_l1": float((correct_images[indices] - gt[indices]).abs().mean()),
            "wrong_to_gt_l1": float((wrong_images[indices] - gt[indices]).abs().mean()),
            "correct_wrong_output_l1": float(
                (correct_images[indices] - wrong_images[indices]).abs().mean()
            ),
        }
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        gt_image = diffusion_tensor_to_pil(gt[index])
        correct_image = diffusion_tensor_to_pil(correct_images[index])
        wrong_image = diffusion_tensor_to_pil(wrong_images[index])
        action_label = row["action_name"]
        strip = _strip(
            [gt_image, correct_image, wrong_image],
            [
                f"run{row['run_index']} t{row['step_index']} {action_label} GT",
                "8-query CFM",
                "wrong-query control",
            ],
        )
        strip_path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(strip_path)
        row["strip_path"] = str(strip_path)
        run_rows[int(row["run_index"])].append(strip)
    run_sheets: list[Image.Image] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append(sheet)
    contact = _contact(run_sheets, columns=args.contact_columns)
    contact_path = args.output_dir / "contact_sheet.png"
    contact.save(contact_path)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    metadata: dict[str, Any] = {
        "status": "completed",
        "num_runs": len(run_sheets),
        "num_rows": len(rows),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "columns": ["GT", "8-query CFM", "wrong-query control"],
        "query_cfm_checkpoint": str(args.query_cfm_checkpoint),
        "selections": str(args.selections),
        "contact_sheet": str(contact_path),
        "metrics": metrics,
        "source_metrics": source_metrics,
    }
    metadata["wandb_url"] = _wandb_upload(args, contact_path, metrics)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-query-cache", type=Path, required=True)
    parser.add_argument("--current-query-cache", type=Path, required=True)
    parser.add_argument("--query-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key", default="query_cfm_trajectory/contact_sheet")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
