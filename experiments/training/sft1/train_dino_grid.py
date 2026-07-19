#!/usr/bin/env python3
"""Prefix-exact SFT1 with 4x4 pooled DINOv2 patch alignment.

Each transition is forwarded independently so the final 16 query states match
SFT2 inference semantics. One shared slot projector maps every Qwen hidden to a
1024-D regional state. The frozen DINO teacher is used only in this SFT1 stage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.dino import DEFAULT_DINO_MODEL, FrozenDINOEncoder
from nimloth.backbone.qwen_tuning import configure_qwen_tuning
from nimloth.latent import (
    add_special_tokens,
    initialize_extra_latent_token_embeddings,
    install_query_embedding_adapter,
    latent_state_tokens,
    special_token_ids,
)
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft1.dino_grid import compute_dino_grid_alignment_loss
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.grid import SharedSlotProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-model", default=DEFAULT_DINO_MODEL)
    parser.add_argument("--latent-token-count", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--dino-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--projector-lr", type=float, default=2e-4)
    parser.add_argument("--query-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=602112)
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-val-records", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.latent_token_count != args.grid_size**2:
        parser.error("latent-token-count must equal grid-size squared")
    if args.grid_size != 4 or args.latent_token_count != 16:
        parser.error("this experiment is fixed to a 4x4/16-token interface")
    if args.dino_loss_weight <= 0:
        parser.error("dino-loss-weight must be positive")
    # configure_qwen_tuning consumes these canonical fields.
    args.lora = True
    args.llm_tune = "lora"
    args.vision_tune = "freeze"
    return args


def _unwrap(module):
    return module.module if hasattr(module, "module") else module


def _save_checkpoint(model, projector, processor, optimizer, output: Path, *, epoch: int, step: int, args) -> None:
    output.mkdir(parents=True, exist_ok=True)
    module = _unwrap(model)
    module.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    adapter = getattr(module, "nimloth_query_embedding_adapter", None)
    if adapter is None:
        raise RuntimeError("query embedding adapter is missing at checkpoint save")
    torch.save(
        {"token_ids": adapter.token_ids.detach().cpu(), "delta": adapter.delta.detach().cpu()},
        output / "query_adapter.pt",
    )
    proj = _unwrap(projector)
    torch.save(proj.state_dict(), output / "slot_projector.pt")
    torch.save(
        {
            "epoch": int(epoch),
            "step": int(step),
            "optimizer": optimizer.state_dict(),
            "latent_token_count": 16,
            "grid_size": 4,
            "dino_model": str(args.dino_model),
            "dino_loss_weight": float(args.dino_loss_weight),
            "projector": {
                "input_dim": proj.input_dim,
                "hidden_dim": proj.hidden_dim,
                "output_dim": proj.output_dim,
                "shared_across_slots": True,
            },
            "forward_semantics": "one_transition_prefix_per_sample",
        },
        output / "training_state.pt",
    )


def _load_checkpoint(model, projector, optimizer, checkpoint: Path) -> tuple[int, int]:
    from nimloth.training.sft2.checkpoint import load_lora_adapter_state

    load_lora_adapter_state(model, checkpoint)
    adapter_state = torch.load(checkpoint / "query_adapter.pt", map_location="cpu", weights_only=True)
    adapter = getattr(model, "nimloth_query_embedding_adapter", None)
    if adapter is None:
        raise RuntimeError("query embedding adapter must be installed before resume")
    if not torch.equal(adapter.token_ids.cpu(), adapter_state["token_ids"]):
        raise ValueError("resume query token ids do not match")
    adapter.delta.data.copy_(adapter_state["delta"].to(adapter.delta))
    projector.load_state_dict(torch.load(checkpoint / "slot_projector.pt", map_location="cpu", weights_only=True))
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("latent_token_count") != 16 or state.get("grid_size") != 4:
        raise ValueError("resume checkpoint is not a 4x4/16-token SFT1 checkpoint")
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"]) + 1, int(state["step"])


def _save_merged(model, projector, processor, output: Path) -> None:
    module = _unwrap(model)
    adapter = getattr(module, "nimloth_query_embedding_adapter", None)
    if adapter is None:
        raise RuntimeError("query embedding adapter is missing before merge")
    merged = module.merge_and_unload()
    merged_adapter = getattr(merged, "nimloth_query_embedding_adapter", adapter)
    embedding = merged.get_input_embeddings()
    with torch.no_grad():
        embedding.weight.index_add_(0, merged_adapter.token_ids, merged_adapter.delta.to(embedding.weight))
    merged_adapter.enabled = False
    merged.config.nimloth_latent_token_count = 16
    merged.config.nimloth_latent_query_mode = "inject"
    merged.config.nimloth_spatial_grid_size = 4
    merged.config.nimloth_state_dim = int(_unwrap(projector).output_dim)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)
    processor.save_pretrained(output)
    torch.save(_unwrap(projector).state_dict(), output / "slot_projector.pt")
    (output / "grid_state_config.json").write_text(
        json.dumps(
            {
                "grid_size": 4,
                "grid_tokens": 16,
                "qwen_hidden_dim": _unwrap(projector).input_dim,
                "state_dim": _unwrap(projector).output_dim,
                "projector_hidden_dim": _unwrap(projector).hidden_dim,
                "shared_slot_projector": True,
                "ordering": "row_major",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _metric_reduce(values: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def evaluate(model, projector, teacher, loader, processor, token_ids, device, args) -> dict[str, float]:
    model.eval()
    projector.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for items in loader:
        enc = build_qwen_batch(
            items,
            processor,
            args.max_length,
            latent_token_count=16,
            mask_latent_query_labels=True,
        )
        hidden, ce_loss = extract_qwen_latents(
            model, enc, token_ids, device, latent_token_count=16
        )
        dino_loss, _ = compute_dino_grid_alignment_loss(
            current_query_hidden=hidden,
            items=items,
            slot_projector=projector,
            dino_encoder=teacher,
            grid_size=4,
        )
        batch = len(items)
        totals += torch.tensor(
            [float(ce_loss.item()) * batch, float(dino_loss.item()) * batch, batch],
            device=device,
            dtype=torch.float64,
        )
    totals = _metric_reduce(totals)
    count = max(float(totals[2].item()), 1.0)
    return {"ce": float(totals[0].item() / count), "dino_grid_mse": float(totals[1].item() / count)}


def main() -> None:
    args = parse_args()
    rank, local_rank, world_size = setup_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    processor = AutoProcessor.from_pretrained(str(args.model), trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = args.max_pixels
    added = add_special_tokens(processor.tokenizer, latent_token_count=16)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    if added:
        model.resize_token_embeddings(len(processor.tokenizer))
    token_ids = special_token_ids(processor.tokenizer, latent_token_count=16)
    initialize_extra_latent_token_embeddings(model, token_ids, latent_token_count=16)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = configure_qwen_tuning(model, args)
    query_token_ids = [token_ids[token] for token in latent_state_tokens(16)]
    query_adapter = install_query_embedding_adapter(model, query_token_ids)
    model.to(device)

    qwen_hidden_dim = int(model.config.hidden_size)
    projector = SharedSlotProjector(
        qwen_hidden_dim,
        1024,
        2048,
        grid_tokens=16,
    ).to(device=device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    teacher = FrozenDINOEncoder.from_pretrained(
        args.dino_model,
        device=device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if teacher.hidden_size != projector.output_dim:
        raise ValueError(
            f"DINO hidden size {teacher.hidden_size} != projected state size {projector.output_dim}"
        )

    train_ds = TransitionQwenDataset(
        args.train_jsonl,
        max_records=args.max_train_records,
        success_only=True,
    )
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

    model_groups = []
    query_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (query_params if "nimloth_query_embedding_adapter" in name else model_groups).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": model_groups, "lr": args.lr},
            {"params": query_params, "lr": args.query_lr},
            {"params": list(projector.parameters()), "lr": args.projector_lr},
        ],
        weight_decay=args.weight_decay,
    )
    start_epoch, global_step = 1, 0
    if args.resume is not None:
        start_epoch, global_step = _load_checkpoint(model, projector, optimizer, args.resume)

    if world_size > 1:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)
        projector = DDP(projector, find_unused_parameters=False, **ddp_kwargs)

    if is_main():
        args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"train_transitions": len(train_ds), "val_transitions": len(val_ds), "world_size": world_size}))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_totals = torch.zeros(4, device=device, dtype=torch.float64)
        for micro_step, items in enumerate(train_loader, start=1):
            enc = build_qwen_batch(
                items,
                processor,
                args.max_length,
                latent_token_count=16,
                mask_latent_query_labels=True,
            )
            hidden, ce_loss = extract_qwen_latents(
                model, enc, token_ids, device, latent_token_count=16
            )
            dino_loss, _ = compute_dino_grid_alignment_loss(
                current_query_hidden=hidden,
                items=items,
                slot_projector=projector,
                dino_encoder=teacher,
                grid_size=4,
            )
            loss = ce_loss + args.dino_loss_weight * dino_loss
            (loss / args.grad_accum).backward()
            should_step = micro_step % args.grad_accum == 0 or micro_step == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None],
                    1.0,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            batch = len(items)
            epoch_totals += torch.tensor(
                [float(loss.item()) * batch, float(ce_loss.item()) * batch, float(dino_loss.item()) * batch, batch],
                device=device,
                dtype=torch.float64,
            )
        epoch_totals = _metric_reduce(epoch_totals)
        val = evaluate(model, projector, teacher, val_loader, processor, token_ids, device, args)
        if is_main():
            count = max(float(epoch_totals[3].item()), 1.0)
            metrics = {
                "epoch": epoch,
                "step": global_step,
                "train/loss": float(epoch_totals[0].item() / count),
                "train/ce": float(epoch_totals[1].item() / count),
                "train/dino_grid_mse": float(epoch_totals[2].item() / count),
                "val/ce": val["ce"],
                "val/dino_grid_mse": val["dino_grid_mse"],
            }
            print(json.dumps(metrics, sort_keys=True))
            epoch_dir = args.output / f"epoch_{epoch:03d}"
            _save_checkpoint(model, projector, processor, optimizer, epoch_dir, epoch=epoch, step=global_step, args=args)
            latest = args.output / "latest"
            if latest.exists():
                shutil.rmtree(latest)
            shutil.copytree(epoch_dir, latest)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if is_main():
        _save_merged(model, projector, processor, args.output / "final" / "hf_merged")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    cleanup_dist()


if __name__ == "__main__":
    main()
