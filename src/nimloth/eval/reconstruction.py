"""Offline reconstruction diagnostics for Nimloth world-model checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder
from nimloth.wm.state_proj import StateProjector


def image_to_tensor(path: str | Path, *, image_size: int, device: torch.device | None = None) -> torch.Tensor:
    """Load an RGB image as a float tensor in ``[0, 1]`` with shape ``(3, H, W)``."""

    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    img = Image.open(path).convert("RGB").resize((image_size, image_size), resample)
    data = torch.tensor(list(img.getdata()), dtype=torch.float32)
    data = data.view(image_size, image_size, 3).permute(2, 0, 1).div(255.0)
    return data.to(device) if device is not None else data


def reconstruction_metrics(pred: torch.Tensor, target: torch.Tensor, *, prefix: str = "") -> dict[str, float]:
    mse = F.mse_loss(pred.float(), target.float()).detach()
    mae = F.l1_loss(pred.float(), target.float()).detach()
    psnr = -10.0 * math.log10(max(float(mse.item()), 1e-12))
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}mse": float(mse.item()),
        f"{key}mae": float(mae.item()),
        f"{key}psnr": float(psnr),
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row})
    return {k: sum(float(row.get(k, 0.0)) for row in rows) / len(rows) for k in keys}


def _save_image(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().clamp(0, 1).mul(255).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)


@torch.no_grad()
def evaluate_reconstruction(
    *,
    model,
    processor,
    token_id_map: dict[str, int],
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    decoder: WMImageDecoder,
    loader,
    device: torch.device,
    output_dir: Path,
    max_batches: int = -1,
    max_length: int = 12000,
    save_samples: int = 16,
) -> dict[str, float]:
    """Evaluate oracle/predictive/copy reconstruction on transition batches."""

    model.eval()
    state_proj.eval()
    wm_predictor.eval()
    decoder.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    rows: list[dict[str, float]] = []
    saved = 0

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = None
        for batch_idx, items in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            eligible = [item for item in items if item.get("next_messages")]
            if not eligible:
                continue

            cur_enc = build_qwen_batch(eligible, processor, max_length=max_length)
            next_items = [{"messages": item["next_messages"]} for item in eligible]
            next_enc = build_qwen_batch(next_items, processor, max_length=max_length)
            cur_hidden, _ = extract_qwen_latents(model, cur_enc, token_id_map, device)
            next_hidden, _ = extract_qwen_latents(model, next_enc, token_id_map, device)

            actions = torch.tensor([item["action_index"] for item in eligible], device=device, dtype=torch.long)
            s_cur = state_proj(cur_hidden).float()
            s_next = state_proj(next_hidden).float()
            s_pred = wm_predictor(s_cur, actions).float()

            target = torch.stack([
                image_to_tensor(item["next_image_path"], image_size=decoder.config.image_size, device=device)
                for item in eligible
            ])
            oracle = decoder(s_next)
            pred = decoder(s_pred)
            copy = decoder(s_cur)
            shuffled_actions = actions.roll(1) if actions.numel() > 1 else (actions + 1) % wm_predictor.config.action_dim
            shuffled = decoder(wm_predictor(s_cur, shuffled_actions).float())

            for i, item in enumerate(eligible):
                row: dict[str, float] = {
                    "batch": float(batch_idx),
                    "index": float(i),
                    **reconstruction_metrics(oracle[i : i + 1], target[i : i + 1], prefix="oracle"),
                    **reconstruction_metrics(pred[i : i + 1], target[i : i + 1], prefix="pred"),
                    **reconstruction_metrics(copy[i : i + 1], target[i : i + 1], prefix="copy"),
                    **reconstruction_metrics(shuffled[i : i + 1], target[i : i + 1], prefix="shuffled_action"),
                }
                row["pred_gap"] = row["pred_mse"] - row["oracle_mse"]
                row["pred_vs_copy_improvement"] = row["copy_mse"] - row["pred_mse"]
                rows.append(row)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)

                if saved < save_samples:
                    stem = f"sample_{saved:04d}_{str(item.get('id', '')).replace('/', '_').replace(':', '_')}"
                    _save_image(target[i], sample_dir / f"{stem}_gt.png")
                    _save_image(oracle[i], sample_dir / f"{stem}_oracle.png")
                    _save_image(pred[i], sample_dir / f"{stem}_pred.png")
                    _save_image(copy[i], sample_dir / f"{stem}_copy.png")
                    saved += 1

    summary = _mean_dict(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate WM reconstruction diagnostics")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--state-proj-checkpoint", type=Path, required=True)
    ap.add_argument("--wm-checkpoint", type=Path, required=True)
    ap.add_argument("--decoder-checkpoint", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-records", type=int, default=-1)
    ap.add_argument("--max-batches", type=int, default=-1)
    ap.add_argument("--max-length", type=int, default=12000)
    ap.add_argument("--max-pixels", type=int, default=602112)
    ap.add_argument("--save-samples", type=int, default=16)
    ap.add_argument("--attn-implementation", default="sdpa")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device)
    state_proj = StateProjector(model.config.hidden_size, wm_predictor.emb_dim).to(device)
    state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location=device, weights_only=True))
    decoder = WMImageDecoder.load_checkpoint(args.decoder_checkpoint, map_location=device).to(device)

    ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_records)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )
    summary = evaluate_reconstruction(
        model=model,
        processor=processor,
        token_id_map=token_id_map,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        decoder=decoder,
        loader=loader,
        device=device,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        max_length=args.max_length,
        save_samples=args.save_samples,
    )
    print(json.dumps({"reconstruction_summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
