#!/usr/bin/env python3
"""SFT2: align Qwen <|latent_state|> embeddings with a LeWM predictor.

Supports two modes:
  - frozen LeWM (requires --lewm-checkpoint): alignment-only after LeWM pretrain
  - --end-to-end: jointly train LeWM on pixels + Qwen alignment + LM CE
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_VAGEN_ROOT = Path(__file__).resolve().parents[2] / "external" / "VAGEN"
if _VAGEN_ROOT.is_dir() and str(_VAGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_VAGEN_ROOT))

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, extract_latent_state, find_last_latent_state_index, last_hidden_state, special_token_ids
from nimloth.latent.extraction import LatentActionTokens
from nimloth.sft2 import (
    MetricAccumulator,
    StateProjector,
    compute_combined_loss,
    compute_end_to_end_step_loss,
    compute_wm_alignment_loss,
    qwen_lr_schedule,
    set_optimizer_group_lr,
    transition_collate_for_qwen,
    wm_loss_weight_schedule,
)
from nimloth.wm import LeWMConfig, LeWMWrapper, default_image_transform
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        dist.init_process_group(backend="nccl")
        return rank, world, local, torch.device(f"cuda:{local}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class TransitionQwenDataset(Dataset):
    def __init__(self, path: Path, *, max_records: int = -1, success_only: bool = False):
        self.samples = TransitionJsonlDataset(path, max_records=max_records, success_only=success_only).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]


def collate_transition_batch(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    return transition_collate_for_qwen(batch)


def build_qwen_batch(items: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    all_images: list[list[Image.Image]] = []
    for item in items:
        texts.append(processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False))
        imgs: list[Image.Image] = []
        for msg in item["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        imgs.append(Image.open(part["image"]).convert("RGB"))
        all_images.append(imgs)

    return processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def extract_batch_latent_hidden(
    model,
    enc: dict[str, torch.Tensor],
    token_id_map: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    model_inputs = {k: v.to(device) for k, v in enc.items() if k != "labels"}
    output = model(**model_inputs, output_hidden_states=True, return_dict=True)
    hidden = last_hidden_state(output)
    tokens = LatentActionTokens()
    rows: list[torch.Tensor] = []
    for row in range(hidden.shape[0]):
        latent_index = find_last_latent_state_index(enc["input_ids"][row], token_id_map, tokens)
        rows.append(extract_latent_state(hidden[row : row + 1], latent_index))
    return torch.stack(rows, dim=0)


def load_pixel_batch(items: list[dict[str, Any]], image_transform, device: torch.device, *, current: bool) -> torch.Tensor:
    key = "current_image_path" if current else "next_image_path"
    return torch.stack(
        [image_transform(Image.open(item[key]).convert("RGB")) for item in items],
        dim=0,
    ).to(device)


def save_checkpoint(
    model,
    state_proj,
    processor,
    out_dir: Path,
    *,
    lewm: LeWMWrapper | None = None,
    optimizer=None,
    step: int = 0,
    epoch: int = 0,
    best_val: float = float("inf"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    proj = state_proj.module if hasattr(state_proj, "module") else state_proj
    torch.save(proj.state_dict(), out_dir / "state_proj.pt")
    if lewm is not None:
        lewm.save_checkpoint(out_dir / "lewm")
    state = {"step": step, "epoch": epoch, "best_val": best_val}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, out_dir / "training_state.pt")


@torch.no_grad()
def evaluate(
    model,
    state_proj,
    lewm,
    loader,
    processor,
    token_id_map,
    image_transform,
    device,
    *,
    end_to_end: bool,
    max_batches: int = -1,
    max_length: int = 20000,
) -> dict[str, float]:
    model.eval()
    state_proj.eval()
    lewm.eval()
    acc = MetricAccumulator()
    for i, batch_samples in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        items = batch_samples
        enc = build_qwen_batch(items, processor, max_length=max_length)
        latent_hidden = extract_batch_latent_hidden(model, enc, token_id_map, device)
        action_indices = torch.tensor([it["action_index"] for it in items], device=device)
        next_pixels = load_pixel_batch(items, image_transform, device, current=False)

        if end_to_end:
            current_pixels = load_pixel_batch(items, image_transform, device, current=True)
            _, metrics = lewm.pretrain_step(current_pixels, next_pixels, action_indices)
            acc.update({k: v for k, v in metrics.items() if k in {"pred_loss", "loss"}})

        _, wm_metrics = compute_wm_alignment_loss(
            qwen_hidden_at_latent=latent_hidden,
            action_indices=action_indices,
            next_pixels=next_pixels,
            state_proj=state_proj,
            lewm=lewm,
        )
        acc.update(wm_metrics)

    model.train()
    state_proj.train()
    if end_to_end:
        lewm.train()
    return acc.averages()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="SFT1 checkpoint (HF dir)")
    ap.add_argument("--lewm-checkpoint", type=Path, default=None, help="Optional; omit when using --end-to-end")
    ap.add_argument("--end-to-end", action="store_true", help="Jointly train LeWM + Qwen alignment + CE")
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr-qwen-start", type=float, default=1e-8, help="Initial Qwen LR (low); ramps up to peak")
    ap.add_argument("--lr-qwen-peak", type=float, default=5e-7, help="Peak Qwen LR after warmup ramp")
    ap.add_argument("--qwen-lr-warmup-ratio", type=float, default=0.15, help="Fraction of total steps for Qwen LR ramp")
    ap.add_argument("--state-proj-lr", type=float, default=1e-4)
    ap.add_argument("--lewm-lr", type=float, default=3e-4, help="LeWM LR in --end-to-end mode")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=20000)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-wm-start", type=float, default=0.1)
    ap.add_argument("--lambda-wm-end", type=float, default=1.0)
    ap.add_argument("--lambda-lewm", type=float, default=1.0, help="LeWM pixel loss weight in --end-to-end mode")
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.end_to_end and args.lewm_checkpoint is None:
        ap.error("either --end-to-end or --lewm-checkpoint is required")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

    train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only)
    val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_transition_batch,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(processor.tokenizer))
    model.to(device)

    if args.end_to_end:
        lewm_cfg = LeWMConfig(img_size=args.img_size, emb_dim=args.emb_dim)
        lewm = LeWMWrapper.create(lewm_cfg).to(device)
    else:
        lewm = LeWMWrapper.load_checkpoint(args.lewm_checkpoint, map_location=device).to(device)
        lewm.freeze()

    hidden_size = model.config.hidden_size
    model_dtype = next(model.parameters()).dtype
    state_proj = StateProjector(hidden_size, lewm.emb_dim).to(device=device, dtype=model_dtype)

    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        state_proj = DDP(state_proj, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if args.end_to_end:
            lewm = DDP(lewm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    lewm_core = lewm.module if hasattr(lewm, "module") else lewm
    image_transform = default_image_transform(lewm_core.config.img_size)

    param_groups = [
        {"params": model.parameters(), "lr": args.lr_qwen_start, "name": "qwen"},
        {"params": state_proj.parameters(), "lr": args.state_proj_lr, "name": "state_proj"},
    ]
    if args.end_to_end:
        param_groups.append({"params": lewm.parameters(), "lr": args.lewm_lr, "name": "lewm"})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))

    log_path = args.output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["time", "epoch", "global_step", "total_loss", "wm_mse", "lm_ce", "lewm_loss", "lambda_wm", "qwen_lr", "val_wm_mse", "val_lewm_loss"]
            )

    global_step = 0
    best_val = float("inf")

    def _unwrap(module):
        return module.module if hasattr(module, "module") else module

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum = MetricAccumulator()
        micro = 0

        for batch_samples in train_loader:
            items = batch_samples
            enc = build_qwen_batch(items, processor, args.max_length)
            latent_hidden = extract_batch_latent_hidden(model, enc, token_id_map, device)
            action_indices = torch.tensor([it["action_index"] for it in items], device=device)
            next_pixels = load_pixel_batch(items, image_transform, device, current=False)
            current_pixels = load_pixel_batch(items, image_transform, device, current=True)

            lambda_wm = wm_loss_weight_schedule(
                global_step,
                total_steps,
                start=args.lambda_wm_start,
                end=args.lambda_wm_end,
            )

            model_inputs = {k: v.to(device) for k, v in enc.items()}
            lm_out = model(**model_inputs)
            lm_loss = lm_out.loss

            if args.end_to_end:
                loss, metrics = compute_end_to_end_step_loss(
                    lewm=_unwrap(lewm),
                    qwen_hidden_at_latent=latent_hidden,
                    action_indices=action_indices,
                    current_pixels=current_pixels,
                    next_pixels=next_pixels,
                    state_proj=_unwrap(state_proj),
                    lm_loss=lm_loss,
                    lambda_wm=lambda_wm,
                    lambda_ce=args.lambda_ce,
                    lambda_lewm=args.lambda_lewm,
                    train_lewm=True,
                )
            else:
                wm_loss, _ = compute_wm_alignment_loss(
                    qwen_hidden_at_latent=latent_hidden,
                    action_indices=action_indices,
                    next_pixels=next_pixels,
                    state_proj=_unwrap(state_proj),
                    lewm=lewm,
                )
                loss, metrics = compute_combined_loss(
                    wm_loss=wm_loss,
                    lm_loss=lm_loss,
                    lambda_wm=lambda_wm,
                    lambda_ce=args.lambda_ce,
                )

            (loss / args.grad_accum).backward()
            accum.update(metrics)
            micro += 1

            if micro % args.grad_accum == 0:
                qwen_lr = qwen_lr_schedule(
                    global_step,
                    warmup_steps=qwen_warmup_steps,
                    total_steps=total_steps,
                    start_lr=args.lr_qwen_start,
                    peak_lr=args.lr_qwen_peak,
                )
                set_optimizer_group_lr(optimizer, "qwen", qwen_lr)

                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main():
                    avg = accum.averages()
                    with log_path.open("a", newline="") as f:
                        csv.writer(f).writerow(
                            [
                                time.time(),
                                epoch,
                                global_step,
                                avg.get("total_loss", ""),
                                avg.get("wm_mse", ""),
                                avg.get("lm_ce", ""),
                                avg.get("lewm_loss", ""),
                                lambda_wm,
                                qwen_lr,
                                "",
                                "",
                            ]
                        )
                    accum.reset()

        val_metrics = evaluate(
            model,
            _unwrap(state_proj),
            _unwrap(lewm),
            val_loader,
            processor,
            token_id_map,
            image_transform,
            device,
            end_to_end=args.end_to_end,
            max_batches=args.max_val_batches,
            max_length=args.max_length,
        )
        val_wm = val_metrics.get("wm_mse", float("inf"))

        if is_main():
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        time.time(),
                        epoch,
                        global_step,
                        "",
                        val_metrics.get("wm_mse", ""),
                        "",
                        val_metrics.get("loss", ""),
                        "",
                        "",
                        val_metrics.get("wm_mse", ""),
                        val_metrics.get("pred_loss", ""),
                    ]
                )
            save_checkpoint(
                model,
                state_proj,
                processor,
                args.output_dir / f"epoch_{epoch:03d}",
                lewm=_unwrap(lewm) if args.end_to_end else None,
                optimizer=optimizer,
                step=global_step,
                epoch=epoch,
                best_val=best_val,
            )
            if val_wm < best_val:
                best_val = val_wm
                save_checkpoint(
                    model,
                    state_proj,
                    processor,
                    args.output_dir / "best",
                    lewm=_unwrap(lewm) if args.end_to_end else None,
                    optimizer=optimizer,
                    step=global_step,
                    epoch=epoch,
                    best_val=best_val,
                )
            print(json.dumps({"epoch": epoch, "global_step": global_step, "val_metrics": val_metrics, "best_val_wm_mse": best_val, "end_to_end": args.end_to_end}))

    if is_main():
        save_checkpoint(
            model,
            state_proj,
            processor,
            args.output_dir / "final",
            lewm=_unwrap(lewm) if args.end_to_end else None,
            optimizer=optimizer,
            step=global_step,
            epoch=args.epochs,
            best_val=best_val,
        )
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

