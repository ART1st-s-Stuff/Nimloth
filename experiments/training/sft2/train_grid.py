#!/usr/bin/env python3
"""SFT2: train one joint WM to predict the complete next 4x4 query-state grid."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.grid_step import compute_grid_sft2_loss
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.grid import GridLatentWMPredictor, load_sft1_slot_projector
from nimloth.wm.value_head import ValueHead


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--wm-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-val-records", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wm-depth", type=int, default=4)
    parser.add_argument("--wm-heads", type=int, default=8)
    parser.add_argument("--wm-mlp-dim", type=int, default=2048)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _encode_pair(model, processor, token_ids, items, device, args):
    current = build_qwen_batch(
        items,
        processor,
        args.max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    next_items = []
    for index, item in enumerate(items):
        if item.get("next_messages") is None:
            raise ValueError(f"transition item {index} is missing next_messages")
        next_items.append({**item, "messages": item["next_messages"]})
    nxt = build_qwen_batch(
        next_items,
        processor,
        args.max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    with torch.no_grad():
        current_hidden, _ = extract_qwen_latents(
            model, current, token_ids, device, latent_token_count=16
        )
        next_hidden, _ = extract_qwen_latents(
            model, nxt, token_ids, device, latent_token_count=16
        )
    return current_hidden, next_hidden


def _save(wm, value, optimizer, output: Path, *, epoch: int, step: int, args) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _unwrap(wm).save_checkpoint(output / "grid_wm")
    _unwrap(value).save_checkpoint(output / "value_head")
    torch.save(
        {
            "epoch": int(epoch),
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "sft1_checkpoint": str(args.sft1_checkpoint.resolve()),
            "grid_tokens": 16,
            "state_dim": 1024,
            "dino_teacher_in_sft2": False,
            "wm_target": "next_prefix_query_grid_stop_gradient",
            "value_pool": "mean_over_4x4_grid",
        },
        output / "training_state.pt",
    )


def _load(wm, value, optimizer, checkpoint: Path, args) -> tuple[int, int]:
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("grid_tokens") != 16 or state.get("state_dim") != 1024:
        raise ValueError("resume checkpoint grid interface mismatch")
    if Path(state["sft1_checkpoint"]).resolve() != args.sft1_checkpoint.resolve():
        raise ValueError("resume SFT1 checkpoint mismatch")
    loaded_wm = GridLatentWMPredictor.load_checkpoint(checkpoint / "grid_wm")
    wm.load_state_dict(loaded_wm.state_dict())
    loaded_value = ValueHead.load_checkpoint(checkpoint / "value_head", emb_dim=1024)
    value.load_state_dict(loaded_value.state_dict())
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1, int(state["step"])


def _reduce(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def evaluate(model, projector, wm, value, loader, processor, token_ids, device, args):
    wm.eval()
    value.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for items in loader:
        current, nxt = _encode_pair(model, processor, token_ids, items, device, args)
        actions = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
        targets = torch.tensor([item["action_value_target"] for item in items], device=device)
        _, metrics = compute_grid_sft2_loss(
            current_query_hidden=current,
            next_query_hidden=nxt,
            action_indices=actions,
            value_targets=targets,
            slot_projector=projector,
            grid_wm=wm,
            value_head=value,
            wm_weight=args.wm_weight,
            value_weight=args.value_weight,
        )
        count = len(items)
        totals += torch.tensor(
            [metrics["grid_wm_mse"] * count, metrics["value_mse"] * count, count],
            dtype=torch.float64,
            device=device,
        )
    totals = _reduce(totals)
    count = max(float(totals[2].item()), 1.0)
    return {"grid_wm_mse": float(totals[0].item() / count), "value_mse": float(totals[1].item() / count)}


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)

    processor = AutoProcessor.from_pretrained(str(args.sft1_checkpoint), trust_remote_code=True)
    if add_special_tokens(processor.tokenizer, latent_token_count=16) != 0:
        raise ValueError("SFT1 checkpoint tokenizer is missing one or more of the 16 query tokens")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.sft1_checkpoint),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device)
    if int(getattr(model.config, "nimloth_latent_token_count", -1)) != 16:
        raise ValueError("SFT1 checkpoint does not declare latent_token_count=16")
    model.requires_grad_(False).eval()
    token_ids = special_token_ids(processor.tokenizer, latent_token_count=16)
    projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        qwen_hidden_dim=int(model.config.hidden_size),
        state_dim=1024,
        grid_tokens=16,
        map_location=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ).to(device)
    projector.requires_grad_(False).eval()

    wm = GridLatentWMPredictor(
        grid_tokens=16,
        emb_dim=1024,
        action_dim=6,
        depth=args.wm_depth,
        heads=args.wm_heads,
        mlp_dim=args.wm_mlp_dim,
    ).to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    value = ValueHead(emb_dim=1024).to(
        device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": list(wm.parameters()), "lr": args.lr},
            {"params": list(value.parameters()), "lr": args.value_lr},
        ],
        weight_decay=args.weight_decay,
    )
    start_epoch, step = 1, 0
    if args.resume is not None:
        start_epoch, step = _load(wm, value, optimizer, args.resume, args)

    train_ds = TransitionQwenDataset(args.train_jsonl, max_records=args.max_train_records)
    val_ds = TransitionQwenDataset(args.val_jsonl, max_records=args.max_val_records)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate_transition_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_transition_batch,
    )
    if world_size > 1:
        kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        wm = DDP(wm, find_unused_parameters=False, **kwargs)
        value = DDP(value, find_unused_parameters=False, **kwargs)

    if is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"train_transitions": len(train_ds), "val_transitions": len(val_ds), "world_size": world_size}))

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        wm.train()
        value.train()
        optimizer.zero_grad(set_to_none=True)
        totals = torch.zeros(4, dtype=torch.float64, device=device)
        for micro_step, items in enumerate(train_loader, start=1):
            current, nxt = _encode_pair(model, processor, token_ids, items, device, args)
            actions = torch.tensor([item["action_index"] for item in items], device=device, dtype=torch.long)
            targets = torch.tensor([item["action_value_target"] for item in items], device=device)
            loss, metrics = compute_grid_sft2_loss(
                current_query_hidden=current,
                next_query_hidden=nxt,
                action_indices=actions,
                value_targets=targets,
                slot_projector=projector,
                grid_wm=wm,
                value_head=value,
                wm_weight=args.wm_weight,
                value_weight=args.value_weight,
            )
            (loss / args.grad_accum).backward()
            if micro_step % args.grad_accum == 0 or micro_step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None], 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            count = len(items)
            totals += torch.tensor(
                [float(loss.item()) * count, metrics["grid_wm_mse"] * count, metrics["value_mse"] * count, count],
                dtype=torch.float64,
                device=device,
            )
        totals = _reduce(totals)
        val = evaluate(model, projector, wm, value, val_loader, processor, token_ids, device, args)
        if is_main():
            count = max(float(totals[3].item()), 1.0)
            print(json.dumps({
                "epoch": epoch,
                "step": step,
                "train/loss": float(totals[0].item() / count),
                "train/grid_wm_mse": float(totals[1].item() / count),
                "train/value_mse": float(totals[2].item() / count),
                "val/grid_wm_mse": val["grid_wm_mse"],
                "val/value_mse": val["value_mse"],
            }, sort_keys=True))
            epoch_dir = args.output / f"epoch_{epoch:03d}"
            _save(wm, value, optimizer, epoch_dir, epoch=epoch, step=step, args=args)
            latest = args.output / "latest"
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(epoch_dir, latest)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        _save(wm, value, optimizer, args.output / "final", epoch=args.epochs, step=step, args=args)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
