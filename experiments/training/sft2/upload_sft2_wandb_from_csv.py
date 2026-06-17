#!/usr/bin/env python3
"""Upload SFT2 train/val curves from train_step_log.csv to wandb."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any


def _float_or_none(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    return float(value)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def split_train_val(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows: list[dict[str, str]] = []
    val_rows: list[dict[str, str]] = []
    for row in rows:
        if row.get("total_loss", "").strip():
            train_rows.append(row)
            continue
        if row.get("val_wm_mse", "").strip():
            val_rows.append(row)
    return train_rows, val_rows


def dedupe_train_rows(train_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_step: dict[int, dict[str, str]] = {}
    for row in train_rows:
        by_step[int(row["global_step"])] = row
    return [by_step[s] for s in sorted(by_step)]


def dedupe_val_rows(val_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_epoch: dict[int, dict[str, str]] = {}
    for row in val_rows:
        epoch = int(row["epoch"])
        by_epoch[epoch] = row
    return [by_epoch[e] for e in sorted(by_epoch)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="Path to train_step_log.csv")
    ap.add_argument("--run-dir", type=Path, required=True, help="SFT2 output directory")
    ap.add_argument("--name", required=True, help="wandb run name")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "nimloth"))
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    ap.add_argument("--mode", default=os.environ.get("WANDB_MODE", "online"))
    ap.add_argument("--run-id", default=None, help="Resume an existing wandb run if set")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    train_rows, val_rows = split_train_val(rows)
    train_rows = dedupe_train_rows(train_rows)
    val_rows = dedupe_val_rows(val_rows)
    if not train_rows and not val_rows:
        raise SystemExit(f"No train/val rows found in {args.csv}")

    import wandb

    out_dir = args.run_dir / "wandb_retro"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(out_dir))

    init_kwargs: dict[str, Any] = {
        "project": args.project,
        "entity": args.entity,
        "name": args.name,
        "dir": str(out_dir),
        "mode": args.mode,
        "config": {
            "source": "retrospective_train_step_log_csv",
            "csv_path": str(args.csv),
            "run_dir": str(args.run_dir),
            "train_steps": len(train_rows),
            "val_epochs": len(val_rows),
        },
    }
    if args.run_id:
        init_kwargs["id"] = args.run_id
        init_kwargs["resume"] = "must"

    run = wandb.init(**init_kwargs)
    assert run is not None

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    for row in train_rows:
        step = int(row["global_step"])
        payload: dict[str, Any] = {
            "epoch": int(row["epoch"]),
            "train/total_loss": _float_or_none(row["total_loss"]),
            "train/wm_mse": _float_or_none(row.get("wm_mse", "")),
            "train/lm_ce": _float_or_none(row.get("lm_ce", "")),
            "train/lewm_loss": _float_or_none(row.get("lewm_loss", "")),
            "train/lambda_wm": _float_or_none(row.get("lambda_wm", "")),
            "train/qwen_lr": _float_or_none(row.get("qwen_lr", "")),
        }
        wandb.log({k: v for k, v in payload.items() if v is not None}, step=step)

    for row in val_rows:
        epoch = int(row["epoch"])
        payload = {
            "epoch": epoch,
            "val/wm_mse": _float_or_none(row.get("val_wm_mse", "")),
            "val/lewm_loss": _float_or_none(row.get("val_lewm_loss", "")),
        }
        wandb.log({k: v for k, v in payload.items() if v is not None}, commit=True)

    if train_rows:
        last = train_rows[-1]
        run.summary["final_global_step"] = int(last["global_step"])
        run.summary["final_train/total_loss"] = _float_or_none(last["total_loss"])
        run.summary["final_train/wm_mse"] = _float_or_none(last.get("wm_mse", ""))
        run.summary["final_train/lm_ce"] = _float_or_none(last.get("lm_ce", ""))
        run.summary["final_train/lewm_loss"] = _float_or_none(last.get("lewm_loss", ""))
    if val_rows:
        last_val = val_rows[-1]
        run.summary["final_val/wm_mse"] = _float_or_none(last_val.get("val_wm_mse", ""))
        run.summary["final_val/lewm_loss"] = _float_or_none(last_val.get("val_lewm_loss", ""))
        run.summary["final_epoch"] = int(last_val["epoch"])

    wandb.save(str(args.csv), base_path=str(args.run_dir))
    run.finish()

    print(
        {
            "wandb_run": run.url,
            "train_steps": len(train_rows),
            "val_epochs": len(val_rows),
            "project": args.project,
            "name": args.name,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
