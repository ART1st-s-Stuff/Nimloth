#!/usr/bin/env python3
"""Backfill per-epoch val/eval metrics from train_step_log.csv into an existing wandb run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "flower"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    args = ap.parse_args()

    import wandb

    run = wandb.init(
        id=args.run_id,
        project=args.project,
        entity=args.entity,
        resume="must",
    )
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")

    rows = list(csv.DictReader(args.csv.open()))
    seen: set[int] = set()
    for row in rows:
        if not row.get("val_loss"):
            continue
        epoch = int(row["epoch"])
        if epoch in seen:
            continue
        seen.add(epoch)
        val_loss = float(row["val_loss"])
        fmt = float(row["format_correct_rate"]) if row.get("format_correct_rate") else 0.0
        wandb.log(
            {
                "val/loss": val_loss,
                "val/format_correct_rate": fmt,
                "eval/val_loss": val_loss,
                "eval/format_correct_rate": fmt,
                "epoch": epoch,
            },
        )
        print(f"logged epoch={epoch} val_loss={val_loss} format={fmt}")

    wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
