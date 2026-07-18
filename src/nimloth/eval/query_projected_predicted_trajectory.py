"""Visualize Qwen, Query, projected State, and WM-predicted State together."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.eval.query_cfm_trajectory import ACTION_NAMES, _sample_cfg
from nimloth.eval.query_vs_qwen_trajectory import _records
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.training.reconstruction.state_to_vision_tokens import (
    StateToVisionTokens,
    VisionTokenAdapterConfig,
    load_proven_cfm,
)
from nimloth.wm.predictor import LatentWMPredictor


def _load_adapters(
    query_checkpoint: Path,
    projected_checkpoint: Path,
    device: torch.device,
) -> tuple[StateToVisionTokens, StateToVisionTokens]:
    query_payload = torch.load(query_checkpoint, map_location="cpu", weights_only=False)
    projected_payload = torch.load(projected_checkpoint, map_location="cpu", weights_only=False)
    query_config = VisionTokenAdapterConfig(**query_payload["invariants"]["query_config"])
    projected_config = VisionTokenAdapterConfig(
        **projected_payload["invariants"]["projected_config"]
    )
    query = StateToVisionTokens(query_config)
    projected = StateToVisionTokens(projected_config)
    query.load_state_dict(query_payload["query_adapter"], strict=True)
    projected.load_state_dict(projected_payload["projected_adapter"], strict=True)
    for module in (query, projected):
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return query, projected


def prepare_rows(
    selections: list[dict[str, Any]],
    query_records: dict[str, dict[int, dict[str, Any]]],
    projected_records: dict[str, dict[int, dict[str, Any]]],
    qwen_records: dict[str, dict[int, dict[str, Any]]],
    predictor: LatentWMPredictor,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    query_states: list[torch.Tensor] = []
    projected_states: list[torch.Tensor] = []
    positive_tokens: list[torch.Tensor] = []
    initial_states: list[torch.Tensor] = []
    action_sequences: list[list[int]] = []
    for selection in selections:
        record_id = str(selection["record_id"])
        expected = [int(value) for value in selection["expected_actions"]]
        caches = {
            "query": query_records.get(record_id),
            "projected": projected_records.get(record_id),
            "qwen": qwen_records.get(record_id),
        }
        if any(record is None for record in caches.values()):
            raise KeyError(f"missing selected record: {record_id}")
        assert caches["query"] is not None
        assert caches["projected"] is not None
        assert caches["qwen"] is not None
        for step in range(6):
            for cache in caches.values():
                if step not in cache:
                    raise KeyError(f"{record_id} misses step{step}")
            reference = caches["query"][step]
            for name, cache in caches.items():
                for key in ("id", "record_id", "step_index", "current_image_path"):
                    if str(cache[step].get(key, "")) != str(reference.get(key, "")):
                        raise ValueError(f"alignment mismatch {record_id} step{step} {name} {key}")
        actual_actions = [int(caches["query"][step]["action_index"]) for step in range(5)]
        if actual_actions != expected:
            raise ValueError(
                f"action mismatch {record_id}: expected={expected}, actual={actual_actions}"
            )
        initial_states.append(caches["projected"][0]["state_emb"].reshape(-1).float())
        action_sequences.append(expected)
        for step in range(1, 6):
            query_states.append(caches["query"][step]["state_emb"].float())
            projected_states.append(caches["projected"][step]["state_emb"].reshape(-1).float())
            positive_tokens.append(caches["qwen"][step]["state_emb"].float())
            rows.append(
                {
                    "run_index": int(selection["run_index"]),
                    "record_id": record_id,
                    "scene_note": str(selection.get("scene_note", "")),
                    "step_index": step,
                    "horizon": step,
                    "action_index": expected[step - 1],
                    "action_name": ACTION_NAMES[expected[step - 1]],
                    "action_prefix": expected[:step],
                    "gt_image_path": str(caches["query"][step]["current_image_path"]),
                }
            )
    initial = torch.stack(initial_states).to(device=device, dtype=torch.float32)
    actions = torch.tensor(action_sequences, device=device, dtype=torch.long)
    predicted = predictor.rollout_states(initial, actions).detach().cpu().reshape(-1, initial.shape[-1])
    return rows, {
        "query": torch.stack(query_states),
        "projected": torch.stack(projected_states),
        "predicted": predicted,
        "positive": torch.stack(positive_tokens),
    }


def _adapt(
    adapter: StateToVisionTokens,
    states: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, states.shape[0], batch_size):
        outputs.append(adapter(states[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs)


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


def _metrics(
    rows: list[dict[str, Any]],
    states: dict[str, torch.Tensor],
    tokens: dict[str, torch.Tensor],
    images: dict[str, torch.Tensor],
    gt: torch.Tensor,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    result: dict[str, float] = {}
    for name, image in images.items():
        result[f"image/{name}_to_gt_l1"] = float((image - gt).abs().mean())
    for name, token in tokens.items():
        result[f"tokens/{name}_to_qwen_mse"] = float(
            torch.nn.functional.mse_loss(token, states["positive"])
        )
        result[f"tokens/{name}_to_qwen_cos"] = float(
            torch.nn.functional.cosine_similarity(token.flatten(1), states["positive"].flatten(1)).mean()
        )
    state_mse = (states["predicted"] - states["projected"]).square().flatten(1).mean(1)
    state_cos = torch.nn.functional.cosine_similarity(states["predicted"], states["projected"])
    result["state/predicted_to_actual_mse"] = float(state_mse.mean())
    result["state/predicted_to_actual_cos"] = float(state_cos.mean())
    horizon_metrics: dict[str, dict[str, float]] = {}
    for horizon in range(1, 6):
        indices = [index for index, row in enumerate(rows) if row["horizon"] == horizon]
        horizon_metrics[str(horizon)] = {
            "count": len(indices),
            "state_mse": float(state_mse[indices].mean()),
            "state_cos": float(state_cos[indices].mean()),
            "predicted_token_mse": float(
                torch.nn.functional.mse_loss(tokens["predicted"][indices], states["positive"][indices])
            ),
            "predicted_image_l1": float((images["predicted"][indices] - gt[indices]).abs().mean()),
            "actual_projected_image_l1": float((images["projected"][indices] - gt[indices]).abs().mean()),
        }
    return result, horizon_metrics


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
        payload[f"predicted_state/contact_{index:02d}"] = wandb.Image(str(path))
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
    query_adapter, projected_adapter = _load_adapters(args.best_query_adapter, args.best_projected_adapter, device)
    tokens = {
        "qwen": states["positive"],
        "query": _adapt(query_adapter, states["query"], device, args.batch_size),
        "projected": _adapt(projected_adapter, states["projected"], device, args.batch_size),
        "predicted": _adapt(projected_adapter, states["predicted"], device, args.batch_size),
    }
    cfm = load_proven_cfm(args.cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    images = {
        name: _sample_cfg(cfm, token, noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device)
        for name, token in tokens.items()
    }
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows])
    metrics, horizon = _metrics(rows, states, tokens, images, gt)
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    labels = ["GT", "Qwen ViT", "Query actual", "Projected actual", "WM predicted"]
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
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2))
    metadata: dict[str, Any] = {
        "status": "completed", "num_runs": len(run_sheets), "num_rows": len(rows),
        "columns": labels, "steps": args.steps, "cfg_scale": args.cfg_scale,
        "metrics": metrics, "horizon_metrics": horizon,
        "contact_sheets": [str(path) for path in contacts],
    }
    metadata["wandb_url"] = _wandb(args, contacts, metrics)
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--best-query-adapter", type=Path, required=True)
    parser.add_argument("--best-projected-adapter", type=Path, required=True)
    parser.add_argument("--cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
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
