"""Compare DINO-supervised SFT1 grid reconstruction with the Qwen ViT control."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.eval.cfm_k8_vs_vit import _load_current_cfm
from nimloth.eval.query_cfm_trajectory import ACTION_NAMES, _sample_cfg
from nimloth.eval.query_vs_qwen_trajectory import _contact, _records, _strip, _vertical, _wandb_upload
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm

COLUMNS = ["GT", "Qwen ViT-token CFM", "DINO-grid CFM"]


def prepare_rows(selections: list[dict[str, Any]], grid_records: dict[str, dict[int, dict[str, Any]]],
                 qwen_records: dict[str, dict[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows, grid_states, qwen_states = [], [], []
    for selection in selections:
        record_id = str(selection["record_id"])
        expected = [int(value) for value in selection["expected_actions"]]
        if len(expected) != 5:
            raise ValueError(f"selection must contain five actions: {record_id}")
        grid, qwen = grid_records.get(record_id), qwen_records.get(record_id)
        if grid is None or qwen is None:
            raise KeyError(f"selected record absent from DINO-grid/Qwen cache: {record_id}")
        missing = [step for step in range(6) if step not in grid or step not in qwen]
        if missing:
            raise KeyError(f"selected record {record_id} misses steps {missing}")
        grid_actions = [int(grid[step]["action_index"]) for step in range(5)]
        qwen_actions = [int(qwen[step]["action_index"]) for step in range(5)]
        if grid_actions != expected or qwen_actions != expected:
            raise ValueError(f"action mismatch {record_id}: expected={expected}, grid={grid_actions}, qwen={qwen_actions}")
        for step in range(1, 6):
            for key in ("id", "record_id", "step_index", "current_image_path"):
                if str(grid[step].get(key, "")) != str(qwen[step].get(key, "")):
                    raise ValueError(f"alignment mismatch {record_id} step{step} key={key}")
            grid_state, qwen_state = grid[step]["state_emb"].float(), qwen[step]["state_emb"].float()
            if tuple(grid_state.shape) != (16, 1024):
                raise ValueError(f"wrong DINO grid shape for {record_id} step{step}: {tuple(grid_state.shape)}")
            if tuple(qwen_state.shape) != (16, 512):
                raise ValueError(f"wrong Qwen shape for {record_id} step{step}: {tuple(qwen_state.shape)}")
            grid_states.append(grid_state)
            qwen_states.append(qwen_state)
            rows.append({
                "run_index": int(selection["run_index"]), "record_id": record_id,
                "step_index": step, "action_index": expected[step - 1],
                "action_name": ACTION_NAMES[expected[step - 1]],
                "gt_image_path": str(grid[step]["current_image_path"]),
            })
    return rows, {"grid": torch.stack(grid_states), "qwen": torch.stack(qwen_states)}


def calculate_image_metrics(images: dict[str, torch.Tensor], gt: torch.Tensor) -> dict[str, float]:
    return {f"image/{name}_to_gt_l1": float((value - gt).abs().mean()) for name, value in images.items()}


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(args.selections.read_text(encoding="utf-8"))["selections"]
    grid_manifest = json.loads((args.dino_grid_cache / "manifest.json").read_text())
    qwen_manifest = json.loads((args.qwen_cache / "manifest.json").read_text())
    grid_records = _records(args.dino_grid_cache, representation="dino_grid_state", state_shape=[16, 1024])
    qwen_records = _records(args.qwen_cache, representation="qwen_compressed_vision_positive", state_shape=[16, 512])
    rows, states = prepare_rows(selections, grid_records, qwen_records)
    grid_model = _load_current_cfm(args.dino_grid_cfm_checkpoint, device)
    qwen_model = load_proven_cfm(args.qwen_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    images = {
        "qwen": _sample_cfg(qwen_model, states["qwen"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "dino_grid": _sample_cfg(grid_model, states["grid"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
    }
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows])
    metrics = calculate_image_metrics(images, gt)
    metrics["image/dino_grid_over_qwen_gt_l1"] = metrics["image/dino_grid_to_gt_l1"] / metrics["image/qwen_to_gt_l1"]
    grid_l1 = (images["dino_grid"] - gt).abs().flatten(1).mean(1)
    qwen_l1 = (images["qwen"] - gt).abs().flatten(1).mean(1)
    metrics["image/dino_grid_better_frame_fraction"] = float((grid_l1 < qwen_l1).float().mean())
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        strip = _strip(
            [diffusion_tensor_to_pil(gt[index]), diffusion_tensor_to_pil(images["qwen"][index]), diffusion_tensor_to_pil(images["dino_grid"][index])],
            [f"run{row['run_index']} t{row['step_index']} {row['action_name']} GT", *COLUMNS[1:]],
        )
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path); row["strip_path"] = str(path); run_rows[row["run_index"]].append(strip)
    run_sheets = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index]); sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append((run_index, sheet))
    contacts = []
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start:start + args.runs_per_contact]
        path = args.output_dir / f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        _contact([sheet for _, sheet in group], args.contact_columns).save(path); contacts.append(path)
    wandb_url = _wandb_upload(args, contacts, metrics)
    metadata = {
        "status": "completed", "protocol": "direct frozen SFT1 DINO-grid reconstruction; no WM prediction",
        "columns": COLUMNS, "num_rows": len(rows), "steps": args.steps, "cfg_scale": args.cfg_scale,
        "matched_noise": True, "metrics": metrics,
        "inputs": {
            "dino_grid_cache": str(args.dino_grid_cache), "dino_grid_fingerprint": grid_manifest["fingerprint"],
            "qwen_cache": str(args.qwen_cache), "qwen_fingerprint": qwen_manifest["fingerprint"],
            "dino_grid_cfm_checkpoint": str(args.dino_grid_cfm_checkpoint),
            "qwen_cfm_checkpoint": str(args.qwen_cfm_checkpoint), "selections": str(args.selections),
        },
        "contact_sheets": [str(path) for path in contacts], "wandb_url": wandb_url,
    }
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINO-grid versus Qwen CFM trajectory reconstruction")
    parser.add_argument("--dino-grid-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--dino-grid-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--runs-per-contact", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-key", default="dino_grid_reconstruction")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
