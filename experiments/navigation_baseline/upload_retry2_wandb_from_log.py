#!/usr/bin/env python3
"""Create a retrospective offline wandb run from VAGEN console metrics.

This parses lines like:
  step:12 - actor/entropy:... - critic/vf_loss:... - ...
and logs numeric metrics to wandb with the original training global step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s*-\s*(.*)$")
NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def parse_value(v: str) -> float | None:
    v = v.strip().rstrip(",")
    if not NUM_RE.match(v):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    line = strip_ansi(line)
    m = STEP_RE.search(line)
    if not m:
        return None
    step = int(m.group(1))
    rest = m.group(2)
    metrics: dict[str, float] = {}
    # Metrics are emitted as "key:value - key:value". Values are scalar numbers.
    for part in rest.split(" - "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip()
        val = parse_value(v)
        if k and val is not None:
            metrics[k] = val
    return step, metrics


def parse_log(log_path: Path) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    # Merge repeated step lines; later lines update/augment earlier ones. This preserves
    # full train metrics from original step lines and resume-time val-only refreshes.
    with log_path.open("r", errors="replace") as f:
        for line in f:
            parsed = parse_step_line(line)
            if not parsed:
                continue
            step, metrics = parsed
            if not metrics:
                continue
            by_step.setdefault(step, {}).update(metrics)
    return by_step


def write_csv(path: Path, by_step: dict[int, dict[str, float]]) -> None:
    keys = sorted({k for metrics in by_step.values() for k in metrics})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", *keys])
        w.writeheader()
        for step in sorted(by_step):
            row: dict[str, Any] = {"step": step}
            row.update(by_step[step])
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--project", default="nimloth_navigation")
    ap.add_argument("--name", required=True)
    ap.add_argument("--entity", default=None)
    ap.add_argument("--mode", default="offline", choices=["offline", "online", "disabled"])
    args = ap.parse_args()

    by_step = parse_log(args.log)
    if not by_step:
        raise SystemExit(f"No step metrics parsed from {args.log}")

    out_dir = args.run_dir / "wandb_retro"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_from_console.csv"
    json_path = out_dir / "metrics_from_console.json"
    write_csv(csv_path, by_step)
    json_path.write_text(json.dumps({str(k): v for k, v in sorted(by_step.items())}, indent=2, sort_keys=True))

    import wandb

    os.environ.setdefault("WANDB_MODE", args.mode)
    os.environ.setdefault("WANDB_DIR", str(out_dir))
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        dir=str(out_dir),
        mode=args.mode,
        config={
            "source": "retrospective_console_log_parse",
            "log_path": str(args.log),
            "run_dir": str(args.run_dir),
            "num_steps_parsed": len(by_step),
            "min_step": min(by_step),
            "max_step": max(by_step),
        },
    )
    assert run is not None
    for step in sorted(by_step):
        wandb.log(by_step[step], step=step)
    # Add the CSV as an artifact-like file in the run.
    wandb.save(str(csv_path), base_path=str(out_dir))
    wandb.save(str(json_path), base_path=str(out_dir))
    run.summary["final_step"] = max(by_step)
    for k, v in by_step[max(by_step)].items():
        run.summary[f"final/{k}"] = v
    run.finish()

    latest = sorted((out_dir / "wandb").glob("offline-run-*"))
    print(f"parsed_steps={len(by_step)} min_step={min(by_step)} max_step={max(by_step)}")
    print(f"csv={csv_path}")
    print(f"json={json_path}")
    if latest:
        print(f"offline_wandb_run={latest[-1]}")


if __name__ == "__main__":
    main()
