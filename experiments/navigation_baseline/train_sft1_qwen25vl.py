#!/usr/bin/env python3
"""Qwen2.5-VL supervised fine-tuning for Nimloth SFT1 records.

This is intentionally separate from verl's text-only SFTDataset so step 3 can
use the actual rollout screenshots referenced by step 2 records.
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

from vagen.envs.navigation.utils.nimloth_format import SPECIAL_TOKENS


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


def load_jsonl(path: Path, max_records: int = -1) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records


class NimlothVLSFTDataset(Dataset):
    def __init__(self, path: Path, processor: AutoProcessor, max_records: int = -1, max_images_per_record: int = -1):
        self.records = load_jsonl(path, max_records=max_records)
        self.processor = processor
        self.max_images_per_record = max_images_per_record

    def __len__(self) -> int:
        return len(self.records)

    def _content_with_images(self, text: str, image_paths_iter) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        chunks = text.split("<image>")
        for i, chunk in enumerate(chunks):
            if chunk:
                parts.append({"type": "text", "text": chunk})
            if i < len(chunks) - 1:
                try:
                    path = next(image_paths_iter)
                except StopIteration as e:
                    raise ValueError("not enough image paths for <image> placeholders") from e
                parts.append({"type": "image", "image": path})
        return parts

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_paths = list(rec.get("image_paths", []))
        if self.max_images_per_record > 0:
            # Keep deterministic prefix for memory-bounded smoke/debug runs only.
            image_paths = image_paths[: self.max_images_per_record]
        image_iter = iter(image_paths)
        messages = []
        for msg in rec["messages"]:
            content = msg["content"]
            if "<image>" in content:
                content = self._content_with_images(content, image_iter)
            messages.append({"role": msg["role"], "content": content})
        return {"id": rec["id"], "messages": messages, "num_images": len(image_paths)}


def assistant_char_spans(messages: list[dict[str, Any]], processor: AutoProcessor) -> list[tuple[int, int]]:
    """Return character spans for assistant response bodies in rendered chat."""
    spans: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prev = processor.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=False)
        prev_gen = processor.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        cur = processor.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        start = len(prev_gen)
        end = len(cur)
        if start < end:
            spans.append((start, end))
    return spans


def collate_fn(batch: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, torch.Tensor]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in batch:
        text = processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(text)
        spans_per_item.append(assistant_char_spans(item["messages"], processor))
        imgs: list[Image.Image] = []
        for msg in item["messages"]:
            content = msg["content"]
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        img = Image.open(part["image"]).convert("RGB")
                        imgs.append(img)
        all_images.append(imgs)

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offset_batch = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )["offset_mapping"]
    labels = enc["input_ids"].clone()
    labels[:] = -100
    # The processor/tokenizer input ids are expected to align for rendered Qwen2.5-VL
    # chat text. If future processor versions alter expansion, fail closed by keeping
    # labels masked rather than training on user/system tokens.
    usable = min(labels.shape[1], offset_batch.shape[1])
    for row, spans in enumerate(spans_per_item):
        for tok_idx in range(usable):
            start = int(offset_batch[row, tok_idx, 0])
            end = int(offset_batch[row, tok_idx, 1])
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                labels[row, tok_idx] = enc["input_ids"][row, tok_idx]
    enc["labels"] = labels
    if torch.all(labels == -100):
        raise ValueError("assistant-only SFT mask is empty for the whole batch")
    return enc


def save_checkpoint(
    model,
    processor,
    out_dir: Path,
    name: str,
    optimizer=None,
    scheduler=None,
    step: int = 0,
    epoch: int = 0,
    best_val: float = float("inf"),
) -> None:
    ckpt = out_dir / name
    ckpt.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
    module.save_pretrained(ckpt, safe_serialization=True)
    processor.save_pretrained(ckpt)
    state = {"step": step, "epoch": epoch, "best_val": best_val}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, ckpt / "training_state.pt")


@torch.no_grad()
def evaluate(model, loader, device: torch.device, max_batches: int = -1) -> float:
    model.eval()
    total = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device)
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss = model(**batch).loss.detach()
        total += loss
        count += 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    model.train()
    return (total / count.clamp_min(1)).item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--max-length", type=int, default=20000)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--max-images-per-record", type=int, default=-1)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=602112,
        help="Cap vision tokens per image (default ~768*28*28, lower than Qwen2.5-VL factory max).",
    )
    ap.add_argument("--min-pixels", type=int, default=3136, help="Minimum pixels per image (~4*28*28).")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = args.min_pixels
    processor.image_processor.max_pixels = args.max_pixels
    added = processor.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if is_main():
        print(f"special tokens requested={len(SPECIAL_TOKENS)} newly_added={added}")

    train_ds = NimlothVLSFTDataset(args.train_jsonl, processor, args.max_train_records, args.max_images_per_record)
    val_ds = NimlothVLSFTDataset(args.val_jsonl, processor, args.max_val_records, args.max_images_per_record)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, processor, args.max_length),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, processor, args.max_length),
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
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    log_path = args.output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["time", "epoch", "global_step", "train_loss", "val_loss", "lr"])

    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    resume_ckpt = args.output_dir / "best" / "training_state.pt"
    if args.resume and resume_ckpt.exists():
        state = torch.load(resume_ckpt, map_location="cpu")
        global_step = int(state.get("step", 0))
        best_val = float(state.get("best_val", float("inf")))
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        else:
            epoch_dirs = sorted(args.output_dir.glob("epoch_*"))
            start_epoch = int(epoch_dirs[-1].name.split("_")[-1]) + 1 if epoch_dirs else 1
        if best_val == float("inf") and log_path.exists():
            rows = list(csv.reader(log_path.open()))
            for row in reversed(rows):
                if len(row) >= 5 and row[4]:
                    try:
                        best_val = min(best_val, float(row[4]))
                    except ValueError:
                        pass
                    break
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if is_main():
            print(
                json.dumps(
                    {
                        "resume": True,
                        "resume_ckpt": str(resume_ckpt),
                        "start_epoch": start_epoch,
                        "global_step": global_step,
                        "best_val": best_val,
                    }
                )
            )

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            accum_loss += loss.detach().float().item()
            accum += 1
            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if is_main():
                    with log_path.open("a", newline="") as f:
                        csv.writer(f).writerow([time.time(), epoch, global_step, accum_loss, "", scheduler.get_last_lr()[0]])
                accum_loss = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        val_loss = evaluate(model, val_loader, device, args.max_val_batches)
        if is_main():
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([time.time(), epoch, global_step, "", val_loss, scheduler.get_last_lr()[0]])
            save_checkpoint(
                model, processor, args.output_dir, f"epoch_{epoch:03d}", optimizer, scheduler, global_step, epoch, best_val
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, processor, args.output_dir, "best", optimizer, scheduler, global_step, epoch, best_val
                )
            print(json.dumps({"epoch": epoch, "global_step": global_step, "val_loss": val_loss, "best_val": best_val}))

    if is_main():
        save_checkpoint(
            model, processor, args.output_dir, "final", optimizer, scheduler, global_step, args.epochs, best_val
        )
    cleanup_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
