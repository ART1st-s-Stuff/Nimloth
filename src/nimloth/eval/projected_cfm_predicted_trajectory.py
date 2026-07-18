"""Compare native Query/Projected CFMs and visualize WM-predicted projected State."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.eval.cfm_k8_vs_vit import _load_current_cfm
from nimloth.eval.query_cfm_trajectory import _sample_cfg
from nimloth.eval.query_projected_predicted_trajectory import prepare_rows
from nimloth.eval.query_vs_qwen_trajectory import _records
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm
from nimloth.wm.predictor import LatentWMPredictor


def _strip(images: list[Image.Image], labels: list[str]) -> Image.Image:
    output = Image.new("RGB", (sum(image.width for image in images), 146), "white")
    draw = ImageDraw.Draw(output)
    x = 0
    for image, label in zip(images, labels, strict=True):
        output.paste(image.convert("RGB"), (x, 18))
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


def calculate_metrics(
    rows: list[dict[str, Any]],
    states: dict[str, torch.Tensor],
    images: dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics = {
        f"image/{name}_to_gt_l1": float((image - gt).abs().mean())
        for name, image in images.items()
    }
    state_mse = (states["predicted"] - states["projected"]).square().flatten(1).mean(1)
    state_cos = torch.nn.functional.cosine_similarity(states["predicted"], states["projected"])
    metrics["state/predicted_to_actual_mse"] = float(state_mse.mean())
    metrics["state/predicted_to_actual_cos"] = float(state_cos.mean())
    horizon: dict[str, dict[str, float]] = {}
    for step in range(1, 6):
        indices = [index for index, row in enumerate(rows) if row["horizon"] == step]
        horizon[str(step)] = {
            "count": len(indices),
            "state_mse": float(state_mse[indices].mean()),
            "state_cos": float(state_cos[indices].mean()),
            "projected_actual_image_l1": float((images["projected"][indices] - gt[indices]).abs().mean()),
            "predicted_image_l1": float((images["predicted"][indices] - gt[indices]).abs().mean()),
            "actual_predicted_output_l1": float((images["projected"][indices] - images["predicted"][indices]).abs().mean()),
        }
    return metrics, horizon


def _wandb(args: argparse.Namespace, contacts: list[Path], metrics: dict[str, float]) -> str | None:
    if args.no_wandb:
        return None
    import wandb

    id_path = args.output_dir / "wandb_run_id.txt"
    run_id = id_path.read_text().strip() if id_path.is_file() else None
    run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=run_id, resume="allow" if run_id else None, dir=str(args.output_dir))
    id_path.write_text(run.id)
    payload: dict[str, Any] = dict(metrics)
    for index, path in enumerate(contacts):
        payload[f"projected_cfm_predicted/contact_{index:02d}"] = wandb.Image(str(path))
    run.log(payload)
    url = run.url
    run.finish()
    return url


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(args.selections.read_text())["selections"]
    query_records = _records(args.query_cache, representation="qwen_query_hidden", state_shape=[8, 2048])
    projected_records = _records(args.projected_cache, representation="projected", state_shape=[8192])
    qwen_records = _records(args.qwen_cache, representation="qwen_compressed_vision_positive", state_shape=[16, 512])
    predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    rows, states = prepare_rows(selections, query_records, projected_records, qwen_records, predictor, device)
    models = {
        "qwen": load_proven_cfm(args.qwen_cfm_checkpoint, device),
        "query": _load_current_cfm(args.query_cfm_checkpoint, device),
        "projected": _load_current_cfm(args.projected_cfm_checkpoint, device),
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    conditions = {
        "qwen": states["positive"].flatten(1),
        "query": states["query"].flatten(1),
        "projected": states["projected"],
        "predicted": states["predicted"],
    }
    images = {
        "qwen": _sample_cfg(models["qwen"], conditions["qwen"], noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "query": _sample_cfg(models["query"], conditions["query"], noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "projected": _sample_cfg(models["projected"], conditions["projected"], noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "predicted": _sample_cfg(models["projected"], conditions["predicted"], noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
    }
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows])
    metrics, horizon = calculate_metrics(rows, states, images, gt)
    labels = ["GT", "Qwen ViT", "Query CFM", "Projected CFM", "WM predicted CFM"]
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        strip = _strip(
            [diffusion_tensor_to_pil(gt[index]), *[diffusion_tensor_to_pil(images[name][index]) for name in ("qwen", "query", "projected", "predicted")]],
            [f"run{row['run_index']} t{row['step_index']} {row['action_name']} GT", *labels[1:]],
        )
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path)
        row["strip_path"] = str(path)
        run_rows[row["run_index"]].append(strip)
    run_sheets = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append((run_index, sheet))
    contacts = []
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start : start + args.runs_per_contact]
        path = args.output_dir / f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        _contact([sheet for _, sheet in group], args.contact_columns).save(path)
        contacts.append(path)
    metadata: dict[str, Any] = {
        "status": "completed", "num_runs": len(run_sheets), "num_rows": len(rows),
        "columns": labels, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "metrics": metrics, "horizon_metrics": horizon,
        "contact_sheets": [str(path) for path in contacts],
    }
    metadata["wandb_url"] = _wandb(args, contacts, metrics)
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2))
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--query-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--projected-cfm-checkpoint", type=Path, required=True)
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
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
