"""Render query-latent CFM and one-step teacher-forced WM predictions."""

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
from nimloth.eval.query_cfm_trajectory import ACTION_NAMES, _sample_cfg
from nimloth.eval.query_vs_qwen_trajectory import _records
from nimloth.rcdm.image_utils import diffusion_tensor_to_pil, image_to_diffusion_tensor
from nimloth.training.reconstruction.projected_query_decoder import (
    ProjectedQueryDecoder,
    validate_cache_lineage,
)
from nimloth.training.reconstruction.state_to_vision_tokens import load_proven_cfm
from nimloth.wm.predictor import LatentWMPredictor


COLUMNS = ["GT", "Qwen ViT-token CFM", "query-latent CFM", "WM pred + Decoder + CFM"]


def calculate_metrics(
    query: torch.Tensor,
    decoded: torch.Tensor,
    images: dict[str, torch.Tensor],
    gt: torch.Tensor,
    clean_decoded: torch.Tensor | None = None,
) -> dict[str, float]:
    result = {
        f"image/{name}_to_gt_l1": float((image - gt).abs().mean())
        for name, image in images.items()
    }
    result["image/query_predicted_output_l1"] = float(
        (images["query"] - images["predicted"]).abs().mean()
    )
    result["decoder/predicted_to_query_mse"] = float(
        torch.nn.functional.mse_loss(decoded.float(), query.float())
    )
    result["decoder/predicted_to_query_cos"] = float(
        torch.nn.functional.cosine_similarity(decoded.float().flatten(1), query.float().flatten(1)).mean()
    )
    if clean_decoded is not None:
        result["decoder/clean_to_query_mse"] = float(
            torch.nn.functional.mse_loss(clean_decoded.float(), query.float())
        )
        result["decoder/clean_to_query_cos"] = float(
            torch.nn.functional.cosine_similarity(clean_decoded.float().flatten(1), query.float().flatten(1)).mean()
        )
    return result


def prepare_teacher_forced_rows(
    selections: list[dict[str, Any]],
    query_records: dict[str, dict[int, dict[str, Any]]],
    projected_records: dict[str, dict[int, dict[str, Any]]],
    qwen_records: dict[str, dict[int, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    previous_projected: list[torch.Tensor] = []
    current_projected: list[torch.Tensor] = []
    previous_actions: list[int] = []
    query: list[torch.Tensor] = []
    qwen: list[torch.Tensor] = []
    for selection in selections:
        record_id = str(selection["record_id"])
        expected_actions = [int(value) for value in selection["expected_actions"]]
        caches = {
            "query": query_records.get(record_id),
            "projected": projected_records.get(record_id),
            "qwen": qwen_records.get(record_id),
        }
        if any(cache is None for cache in caches.values()):
            raise KeyError(f"missing selected record: {record_id}")
        assert all(cache is not None for cache in caches.values())
        actual_actions = [int(caches["query"][step]["action_index"]) for step in range(5)]
        if actual_actions != expected_actions:
            raise ValueError(f"action mismatch {record_id}: expected={expected_actions}, actual={actual_actions}")
        for step in range(1, 6):
            reference = caches["query"][step]
            for name, cache in caches.items():
                for key in ("id", "record_id", "step_index", "current_image_path"):
                    if str(cache[step].get(key, "")) != str(reference.get(key, "")):
                        raise ValueError(f"alignment mismatch {record_id} step{step} {name} {key}")
            previous_action = int(caches["projected"][step - 1]["action_index"])
            if previous_action != expected_actions[step - 1]:
                raise ValueError(f"previous action mismatch {record_id} step{step}")
            previous_projected.append(caches["projected"][step - 1]["state_emb"].reshape(-1).float())
            current_projected.append(caches["projected"][step]["state_emb"].reshape(-1).float())
            previous_actions.append(previous_action)
            query.append(caches["query"][step]["state_emb"].float())
            qwen.append(caches["qwen"][step]["state_emb"].float())
            rows.append({
                "run_index": int(selection["run_index"]),
                "record_id": record_id,
                "step_index": step,
                "action_index": previous_action,
                "action_name": ACTION_NAMES[previous_action],
                "gt_image_path": str(reference["current_image_path"]),
            })
    return rows, {
        "previous_projected": torch.stack(previous_projected),
        "current_projected": torch.stack(current_projected),
        "previous_actions": torch.tensor(previous_actions, dtype=torch.long),
        "query": torch.stack(query),
        "qwen": torch.stack(qwen),
    }


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


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections = json.loads(args.selections.read_text())["selections"]
    query_manifest = json.loads((args.query_cache / "manifest.json").read_text())
    projected_manifest = json.loads((args.projected_cache / "manifest.json").read_text())
    qwen_manifest = json.loads((args.qwen_cache / "manifest.json").read_text())
    validate_cache_lineage(projected_manifest, query_manifest)
    query_records = _records(args.query_cache, representation="qwen_query_hidden", state_shape=[8, 2048])
    projected_records = _records(args.projected_cache, representation="projected", state_shape=[8192])
    qwen_records = _records(args.qwen_cache, representation="qwen_compressed_vision_positive", state_shape=[16, 512])
    rows, states = prepare_teacher_forced_rows(selections, query_records, projected_records, qwen_records)
    predictor = LatentWMPredictor.load_checkpoint(
        args.wm_checkpoint,
        map_location=device,
        history_size_override=args.wm_history_size_override,
    ).to(device).eval()
    decoder_payload = torch.load(args.decoder_checkpoint, map_location="cpu", weights_only=False)
    decoder_invariants = decoder_payload.get("invariants")
    if not isinstance(decoder_invariants, dict):
        raise ValueError("decoder checkpoint lacks training invariants")
    expected_decoder_cache = {
        "val_projected_fingerprint": str(projected_manifest["fingerprint"]),
        "val_query_fingerprint": str(query_manifest["fingerprint"]),
    }
    for key, expected in expected_decoder_cache.items():
        if str(decoder_invariants.get(key)) != expected:
            raise ValueError(
                f"decoder/eval cache mismatch for {key}: "
                f"{decoder_invariants.get(key)!r} != {expected!r}"
            )
    decoder = ProjectedQueryDecoder.load_checkpoint(args.decoder_checkpoint, map_location=device).to(device).eval()
    for module in (predictor, decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    predicted_projected = []
    decoded_query = []
    clean_decoded_query = []
    for start in range(0, len(rows), args.chunk_size):
        stop = start + args.chunk_size
        predicted = predictor(
            states["previous_projected"][start:stop].to(device),
            states["previous_actions"][start:stop].to(device),
        )
        predicted_projected.append(predicted.cpu())
        decoded_query.append(decoder(predicted.float()).cpu())
        clean_decoded_query.append(
            decoder(states["current_projected"][start:stop].to(device).float()).cpu()
        )
    states["predicted_projected"] = torch.cat(predicted_projected)
    states["decoded_query"] = torch.cat(decoded_query)
    states["clean_decoded_query"] = torch.cat(clean_decoded_query)
    qwen_cfm = load_proven_cfm(args.qwen_cfm_checkpoint, device)
    query_cfm = _load_current_cfm(args.query_cfm_checkpoint, device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(len(rows), 3, 128, 128, generator=generator)
    images = {
        "qwen": _sample_cfg(qwen_cfm, states["qwen"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "query": _sample_cfg(query_cfm, states["query"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "predicted": _sample_cfg(query_cfm, states["decoded_query"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
        "decoded_clean": _sample_cfg(query_cfm, states["clean_decoded_query"].flatten(1), noise, steps=args.steps, cfg_scale=args.cfg_scale, chunk_size=args.chunk_size, device=device),
    }
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in rows])
    metrics = calculate_metrics(
        states["query"], states["decoded_query"], images, gt, states["clean_decoded_query"]
    )
    metrics["state/predicted_to_current_projected_mse"] = float(torch.nn.functional.mse_loss(states["predicted_projected"], states["current_projected"]))
    metrics["state/predicted_to_current_projected_cos"] = float(torch.nn.functional.cosine_similarity(states["predicted_projected"], states["current_projected"]).mean())
    labels = COLUMNS
    run_rows: dict[int, list[Image.Image]] = defaultdict(list)
    for index, row in enumerate(rows):
        strip = _strip(
            [diffusion_tensor_to_pil(gt[index]), diffusion_tensor_to_pil(images["qwen"][index]), diffusion_tensor_to_pil(images["query"][index]), diffusion_tensor_to_pil(images["predicted"][index])],
            [f"run{row['run_index']} t{row['step_index']} {row['action_name']} GT", *labels[1:]],
        )
        path = args.output_dir / f"run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        strip.save(path)
        gate_path = args.output_dir / f"decoder_gate_run_{row['run_index']:02d}_step_{row['step_index']:02d}.png"
        _strip(
            [diffusion_tensor_to_pil(gt[index]), diffusion_tensor_to_pil(images["query"][index]), diffusion_tensor_to_pil(images["decoded_clean"][index])],
            ["GT", "direct query", "decoded actual projected"],
        ).save(gate_path)
        row["strip_path"] = str(path)
        row["decoder_gate_path"] = str(gate_path)
        run_rows[row["run_index"]].append(strip)
    contacts: list[Path] = []
    run_sheets: list[tuple[int, Image.Image]] = []
    for run_index in sorted(run_rows):
        sheet = _vertical(run_rows[run_index])
        sheet.save(args.output_dir / f"run_{run_index:02d}.png")
        run_sheets.append((run_index, sheet))
    for start in range(0, len(run_sheets), args.runs_per_contact):
        group = run_sheets[start : start + args.runs_per_contact]
        path = args.output_dir / f"contact_runs_{group[0][0]:02d}_{group[-1][0]:02d}.png"
        _contact([sheet for _, sheet in group], args.contact_columns).save(path)
        contacts.append(path)
    wandb_url = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, dir=str(args.output_dir))
        run.log({**metrics, **{f"teacher_forced/contact_{index:02d}": wandb.Image(str(path)) for index, path in enumerate(contacts)}})
        wandb_url = run.url
        run.finish()
    metadata = {
        "status": "completed",
        "protocol": "one-step teacher-forced: WM(actual projected state_{t-1}, action_{t-1}) then Decoder to query latent_t",
        "columns": labels,
        "num_rows": len(rows),
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "metrics": metrics,
        "inputs": {
            "query_cache": str(args.query_cache),
            "query_cache_fingerprint": str(query_manifest["fingerprint"]),
            "projected_cache": str(args.projected_cache),
            "projected_cache_fingerprint": str(projected_manifest["fingerprint"]),
            "qwen_cache": str(args.qwen_cache),
            "qwen_cache_fingerprint": str(qwen_manifest["fingerprint"]),
            "wm_checkpoint": str(args.wm_checkpoint),
            "decoder_checkpoint": str(args.decoder_checkpoint),
            "query_cfm_checkpoint": str(args.query_cfm_checkpoint),
            "qwen_cfm_checkpoint": str(args.qwen_cfm_checkpoint),
            "selections": str(args.selections),
        },
        "contact_sheets": [str(path) for path in contacts],
        "wandb_url": wandb_url,
    }
    (args.output_dir / "samples.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher-forced query CFM reconstruction")
    parser.add_argument("--query-cache", type=Path, required=True)
    parser.add_argument("--projected-cache", type=Path, required=True)
    parser.add_argument("--qwen-cache", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--query-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--qwen-cfm-checkpoint", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wm-history-size-override", type=int, choices=[1], default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--contact-columns", type=int, default=2)
    parser.add_argument("--runs-per-contact", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return evaluate(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
