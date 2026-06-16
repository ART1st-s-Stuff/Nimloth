#!/usr/bin/env python3
"""Pretrain a LeWM-style JEPA on Nimloth navigation rollout transitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nimloth.wm import LeWMConfig, LeWMWrapper, TransitionSample, default_image_transform
from nimloth.wm.dataset import TransitionJsonlDataset


class LeWMTransitionDataset(Dataset):
    def __init__(self, path: Path, *, max_records: int = -1, success_only: bool = False, img_size: int = 96):
        self.base = TransitionJsonlDataset(path, max_records=max_records, success_only=success_only)
        self.transform = default_image_transform(img_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int | str]:
        sample: TransitionSample = self.base[index]
        current = self.transform(Image.open(sample.current_image_path).convert("RGB"))
        nxt = self.transform(Image.open(sample.next_image_path).convert("RGB"))
        return {
            "id": f"{sample.record_id}:{sample.step_index}",
            "current": current,
            "next": nxt,
            "action_index": sample.action_index,
        }


def collate_lewm(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "ids": [str(item["id"]) for item in batch],
        "current": torch.stack([item["current"] for item in batch], dim=0),
        "next": torch.stack([item["next"] for item in batch], dim=0),
        "action_index": torch.tensor([item["action_index"] for item in batch], dtype=torch.long),
    }


@torch.no_grad()
def evaluate(model: LeWMWrapper, loader: DataLoader, device: torch.device, max_batches: int = -1) -> dict[str, float]:
    model.eval()
    totals = {"pred_loss": 0.0, "loss": 0.0}
    count = 0
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        current = batch["current"].to(device)
        nxt = batch["next"].to(device)
        actions = batch["action_index"].to(device)
        _, metrics = model.pretrain_step(current, nxt, actions)
        for key in totals:
            totals[key] += metrics.get(key, 0.0)
        count += 1
    model.train()
    if count == 0:
        return totals
    return {k: v / count for k, v in totals.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=96)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--max-train-records", type=int, default=-1)
    ap.add_argument("--max-val-records", type=int, default=-1)
    ap.add_argument("--max-val-batches", type=int, default=-1)
    ap.add_argument("--success-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    cfg = LeWMConfig(img_size=args.img_size, emb_dim=args.emb_dim)
    model = LeWMWrapper.create(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = LeWMTransitionDataset(args.train_jsonl, max_records=args.max_train_records, success_only=args.success_only, img_size=args.img_size)
    val_ds = LeWMTransitionDataset(args.val_jsonl, max_records=args.max_val_records, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_lewm)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_lewm)

    log_path = args.output_dir / "pretrain_step_log.csv"
    if not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(["time", "epoch", "step", "train_loss", "train_pred_loss", "val_loss", "val_pred_loss", "lr"])

    global_step = 0
    best_val = float("inf")
    steps_per_epoch = max(1, math.ceil(len(train_loader)))

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_pred = 0.0
        for batch in train_loader:
            current = batch["current"].to(device)
            nxt = batch["next"].to(device)
            actions = batch["action_index"].to(device)
            loss, metrics = model.pretrain_step(current, nxt, actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_loss += metrics["loss"]
            train_pred += metrics["pred_loss"]
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([time.time(), epoch, global_step, metrics["loss"], metrics["pred_loss"], "", "", args.lr])

        val_metrics = evaluate(model, val_loader, device, args.max_val_batches)
        avg_train_loss = train_loss / steps_per_epoch
        avg_train_pred = train_pred / steps_per_epoch
        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([time.time(), epoch, global_step, "", "", val_metrics["loss"], val_metrics["pred_loss"], args.lr])

        ckpt_dir = args.output_dir / f"epoch_{epoch:03d}"
        model.save_checkpoint(ckpt_dir)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save_checkpoint(args.output_dir / "best")

        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": avg_train_loss,
                    "train_pred_loss": avg_train_pred,
                    "val_loss": val_metrics["loss"],
                    "val_pred_loss": val_metrics["pred_loss"],
                    "best_val": best_val,
                }
            ),
            flush=True,
        )

    model.save_checkpoint(args.output_dir / "final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
