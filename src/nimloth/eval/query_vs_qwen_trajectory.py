"""Compare direct 8-query CFM and Qwen ViT-token CFM on diverse trajectories."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.eval.query_cfm_trajectory import (
    ACTION_NAMES,
    _load_query_cfm,
    _sample_cfg,
)
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm


def _records(cache_dir: Path, *, representation: str, state_shape: list[int]) -> dict[str, dict[int, dict[str, Any]]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("representation") != representation:
        raise ValueError(
            f"cache representation mismatch: expected={representation}, "
            f"actual={manifest.get('representation')}, path={cache_dir}"
        )
    if [int(value) for value in manifest.get("state_shape", [])] != state_shape:
        raise ValueError(f"cache state_shape mismatch: expected={state_shape}, path={cache_dir}")
    output: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    dataset = RCDMStateCacheDataset(cache_dir)
    for index in range(len(dataset)):
        item = dataset[index]
        record_id = str(item["record_id"])
        step = int(item["step_index"])
        if step in output[record_id]:
            raise ValueError(f"duplicate cache row: {record_id} step{step}")
        output[record_id][step] = item
    return output


def prepare_comparison_rows(
    selections: list[dict[str, Any]],
    query_records: dict[str, dict[int, dict[str, Any]]],
    qwen_records: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    rows: list[dict[str, Any]] = []
    query_conditions: list[torch.Tensor] = []
    qwen_conditions: list[torch.Tensor] = []
    for selection in selections:
        record_id = str(selection["record_id"])
        expected = [int(action) for action in selection["expected_actions"]]
        if len(expected) != 5:
            raise ValueError(f"selection must contain five actions: {record_id}")
        query = query_records.get(record_id)
        qwen = qwen_records.get(record_id)
        if query is None or qwen is None:
            raise KeyError(f"selected record absent from query/Qwen cache: {record_id}")
        missing = [step for step in range(6) if step not in query or step not in qwen]
        if missing:
            raise KeyError(f"selected record {record_id} misses steps {missing}")
        actual = [int(query[step]["action_index"]) for step in range(5)]
        qwen_actions = [int(qwen[step]["action_index"]) for step in range(5)]
        if actual != expected or qwen_actions != expected:
            raise ValueError(
                f"action mismatch for {record_id}: expected={expected}, "
                f"query={actual}, qwen={qwen_actions}"
            )
        for step in range(6):
            if str(query[step]["current_image_path"]) != str(qwen[step]["current_image_path"]):
                raise ValueError(f"query/Qwen image mismatch for {record_id} step{step}")
        for step in range(1, 6):
            query_state = query[step]["state_emb"].float()
            qwen_state = qwen[step]["state_emb"].float()
            if tuple(query_state.shape) != (8, 2048):
                raise ValueError(f"wrong query shape for {record_id} step{step}: {query_state.shape}")
            if tuple(qwen_state.shape) != (16, 512):
                raise ValueError(f"wrong Qwen shape for {record_id} step{step}: {qwen_state.shape}")
            query_conditions.append(query_state.reshape(-1))
            qwen_conditions.append(qwen_state.reshape(-1))
            rows.append(
                {
                    "run_index": int(selection["run_index"]),
                    "candidate_index": int(selection.get("candidate_index", -1)),
                    "scene_note": str(selection.get("scene_note", "")),
                    "record_id": record_id,
                    "step_index": step,
                    "action_index": expected[step - 1],
                    "action_name": ACTION_NAMES[expected[step - 1]],
                    "action_prefix": expected[:step],
                    "action_names": [ACTION_NAMES[action] for action in expected[:step]],
                    "gt_image_path": str(query[step]["current_image_path"]),
                }
            )
    return rows, torch.stack(query_conditions), torch.stack(qwen_conditions)


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


def _contact(images: list[Image.Image], columns: int) -> Image.Image:
    width, height = images[0].size
    output = Image.new("RGB", (columns * width, math.ceil(len(images) / columns) * height), "white")
    for index, image in enumerate(images):
        output.paste(image, ((index % columns) * width, (index // columns) * height))
    return output


def _metric_rows(
    rows: list[dict[str, Any]],
    gt: torch.Tensor,
    qwen: torch.Tensor,
    query: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    qwen_l1 = (qwen - gt).abs().flatten(1).mean(1)
    query_l1 = (query - gt).abs().flatten(1).mean(1)
    metrics = {
        "trajectory/qwen_vit_to_gt_l1": float(qwen_l1.mean()),
        "trajectory/query_to_gt_l1": float(query_l1.mean()),
        "trajectory/query_over_qwen_gt_l1": float(query_l1.mean() / qwen_l1.mean()),
        "trajectory/qwen_query_output_l1": float((qwen - query).abs().mean()),
        "trajectory/query_better_frame_fraction": float((query_l1 < qwen_l1).float().mean()),
    }
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["action_name"]].append(index)
    action_metrics: dict[str, dict[str, float]] = {}
    for action, indices in sorted(grouped.items()):
        qwen_value = qwen_l1[indices].mean()
        query_value = query_l1[indices].mean()
        action_metrics[action] = {
            "count": len(indices),
            "qwen_vit_to_gt_l1": float(qwen_value),
            "query_to_gt_l1": float(query_value),
            "query_over_qwen": float(query_value / qwen_value),
            "query_better_frame_fraction": float((query_l1[indices] < qwen_l1[indices]).float().mean()),
        }
    return metrics, action_metrics


def _wandb_upload(
    args: argparse.Namespace,
    contact_paths: list[Path],
    metrics: dict[str, float],
) -> str | None:
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
        payload: dict[str, Any] = dict(metrics)
        for index, path in enumerate(contact_paths):
            payload[f"{args.wandb_key}/group_{index:02d}"] = wandb.Image(str(path))
        run.log(payload)
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
    selections = json.loads(args.selections.read_text(encoding="utf-8"))["selections"]
    query_records = _records(
        args.query_cache,
        representation="qwen_query_hidden",
        state_shape=[8, 2048],
    )
    qwen_records = _records(
        args.qwen_cache,
        representation="qwen_compressed_vision_positive",
        state_shape=[16, 512],
    )
    rows, query_condition, qwen_condition = prepare_comparison_rows(
        selections, query_records, qwen_records
    )
    query_model = _load_query_cfm(args.query_cfm_checkpoint, device)
    qwen_model = load_proven_cfm(args.qwen_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    query_images = _sample_cfg(
        query_model,
        query_condition,
        noise,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        chunk_size=args.chunk_size,
        device=device,
    )
    qwen_images = _sample_cfg(
        qwen_model,
        qwen_condition,
        noise,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        chunk_size=args.chunk_size,
        device=device,
    )
    gt = torch.stack(
        [image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows]
    )
    metrics, action_metrics = _metric_rows(rows, gt, qwen_images, query_images)
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        strip = _strip(
            [
                diffusion_tensor_to_pil(gt[index]),
                diffusion_tensor_to_pil(qwen_images[index]),
                diffusion_tensor_to_pil(query_images[index]),
            ],
            [
                f"run{row['run_index']} t{row['step_index']} {row['action_name']} GT",
                "Qwen ViT-token CFM",
                "8-query CFM",
            ],
        )
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path)
        row["strip_path"] = str(path)
        run_rows[int(row["run_index"])].append(strip)
    run_sheets: list[tuple[int, Image.Image]] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append((run_index, sheet))
    contact_paths: list[Path] = []
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start : start + args.runs_per_contact]
        contact = _contact([sheet for _, sheet in group], columns=args.contact_columns)
        path = args.output_dir / f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        contact.save(path)
        contact_paths.append(path)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output_dir / "action_metrics.json").write_text(
        json.dumps(action_metrics, indent=2), encoding="utf-8"
    )
    metadata: dict[str, Any] = {
        "status": "completed",
        "num_runs": len(run_sheets),
        "num_rows": len(rows),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "columns": ["GT", "Qwen ViT-token CFM", "8-query CFM"],
        "query_cfm_checkpoint": str(args.query_cfm_checkpoint),
        "qwen_cfm_checkpoint": str(args.qwen_cfm_checkpoint),
        "selections": str(args.selections),
        "contact_sheets": [str(path) for path in contact_paths],
        "metrics": metrics,
        "action_metrics": action_metrics,
    }
    metadata["wandb_url"] = _wandb_upload(args, contact_paths, metrics)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--query-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--runs-per-contact", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key", default="query_vs_qwen_trajectory")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
