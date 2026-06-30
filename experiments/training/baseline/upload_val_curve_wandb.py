#!/usr/bin/env python3
"""Append val-only metrics from a VAGEN console log to a persistent wandb run."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Reuse parser from legacy navigation baseline helper.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "navigation_baseline"))
from upload_retry2_wandb_from_log import parse_step_line  # noqa: E402

VAL_KEYS = (
    "val-aux/navigation_base/traj_success/mean@1",
    "val-aux/navigation_common/traj_success/mean@1",
    "val-core/navigation_base/reward/mean@1",
    "val-core/navigation_common/reward/mean@1",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--val-run-dir", required=True, type=Path)
    ap.add_argument("--checkpoint-step", required=True, type=int)
    ap.add_argument("--project", default="nimloth_navigation")
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--entity", default=None)
    args = ap.parse_args()

    metrics: dict[str, float] = {}
    with args.log.open("r", errors="replace") as f:
        for line in f:
            parsed = parse_step_line(line)
            if not parsed:
                continue
            _step, row = parsed
            val_row = {k: v for k, v in row.items() if k.startswith("val-")}
            if val_row:
                metrics = val_row

    if not metrics:
        print(f"No val metrics parsed from {args.log}", file=sys.stderr)
        return

    curve_path = args.val_run_dir / "val_curve.jsonl"
    payload = {"checkpoint_step": args.checkpoint_step, **metrics}
    with curve_path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")

    import wandb

    os.environ.setdefault("WANDB_DIR", str(args.val_run_dir / "wandb"))
    os.environ["WANDB_RUN_ID"] = args.run_id
    os.environ["WANDB_RESUME"] = "allow"
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        id=args.run_id,
        resume="allow",
        dir=str(args.val_run_dir / "wandb"),
    )
    assert run is not None
    wandb.define_metric("checkpoint_step")
    for key in VAL_KEYS:
        short = key.replace("val-aux/", "val/").replace("val-core/", "val/")
        wandb.define_metric(short, step_metric="checkpoint_step")
    log_payload = {
        key.replace("val-aux/", "val/").replace("val-core/", "val/"): val
        for key, val in metrics.items()
    }
    wandb.log(log_payload, step=args.checkpoint_step)
    wandb.save(str(curve_path), base_path=str(args.val_run_dir))
    run.finish()

    print(
        json.dumps(
            {
                "checkpoint_step": args.checkpoint_step,
                "metrics": metrics,
                "curve_path": str(curve_path),
                "wandb_run_id": args.run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
