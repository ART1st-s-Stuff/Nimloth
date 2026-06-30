"""Train a post-hoc WM image decoder with frozen Qwen and state projector."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.eval.reconstruction import evaluate_reconstruction, image_to_tensor
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.reconstruction import WMImageDecoder, WMImageDecoderConfig
from nimloth.wm.state_proj import StateProjector


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


@torch.no_grad()
def _encode_states(model, processor, token_id_map, items, state_proj, device, max_length: int) -> torch.Tensor:
    enc = build_qwen_batch(items, processor, max_length=max_length)
    hidden, _ = extract_qwen_latents(model, enc, token_id_map, device)
    return state_proj(hidden).float()


def _maybe_init_wandb(args: argparse.Namespace, meta: dict) -> object | None:
    if getattr(args, "no_wandb", False):
        return None
    try:
        import wandb
    except Exception:
        return None
    return wandb.init(
        project="nimloth",
        name=getattr(args, "wandb_run_name", None),
        config=meta,
        dir=str(args.output_dir),
    )


def train_reconstruction_decoder(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

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
    _freeze(model)

    wm_predictor = LatentWMPredictor.load_checkpoint(args.wm_checkpoint, map_location=device).to(device)
    _freeze(wm_predictor)
    state_proj = StateProjector(model.config.hidden_size, wm_predictor.emb_dim).to(device)
    state_proj.load_state_dict(torch.load(args.state_proj_checkpoint, map_location=device, weights_only=True))
    _freeze(state_proj)

    decoder = WMImageDecoder(
        WMImageDecoderConfig(
            emb_dim=wm_predictor.emb_dim,
            image_size=args.image_size,
            patch_size=args.patch_size,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
            heads=args.heads,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only)
    val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )

    meta = {
        "model": str(args.model),
        "state_proj_checkpoint": str(args.state_proj_checkpoint),
        "wm_checkpoint": str(args.wm_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "decoder_config": decoder.config.__dict__,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    wandb_run = _maybe_init_wandb(args, meta)

    log_path = args.output_dir / "train_step_log.csv"
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["time", "epoch", "step", "loss", "val_pred_mse", "val_oracle_mse"])

    step = 0
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        for items in train_loader:
            states = _encode_states(model, processor, token_id_map, items, state_proj, device, args.max_length)
            targets = torch.stack([
                image_to_tensor(item["current_image_path"], image_size=args.image_size, device=device)
                for item in items
            ])
            pred = decoder(states)
            loss = F.l1_loss(pred, targets) if args.loss == "l1" else F.mse_loss(pred, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % args.log_interval == 0:
                loss_value = float(loss.detach().item())
                with log_path.open("a", newline="") as f:
                    csv.writer(f).writerow([time.time(), epoch, step, loss_value, "", ""])
                if wandb_run is not None:
                    wandb_run.log({"reconstruction/train_loss": loss_value, "epoch": epoch}, step=step)
                print(json.dumps({"epoch": epoch, "step": step, "loss": loss_value}))

        val_metrics = evaluate_reconstruction(
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            decoder=decoder,
            loader=val_loader,
            device=device,
            output_dir=args.output_dir / f"val_epoch_{epoch:03d}",
            max_batches=args.max_val_batches,
            max_length=args.max_length,
            save_samples=args.save_samples,
        )
        val_pred = float(val_metrics.get("pred_mse", float("inf")))
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                time.time(), epoch, step, "", val_pred, val_metrics.get("oracle_mse", "")
            ])
        if wandb_run is not None:
            wandb_run.log(
                {f"reconstruction/val_{key}": value for key, value in val_metrics.items()},
                step=step,
            )
        decoder.save_checkpoint(args.output_dir / f"epoch_{epoch:03d}")
        if val_pred < best_val:
            best_val = val_pred
            decoder.save_checkpoint(args.output_dir / "best")
        print(json.dumps({"epoch": epoch, "val_metrics": val_metrics, "best_val_pred_mse": best_val}))

    decoder.save_checkpoint(args.output_dir / "final")
    if wandb_run is not None:
        wandb_run.finish()
    return 0
