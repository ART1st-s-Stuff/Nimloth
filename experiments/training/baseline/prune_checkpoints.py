#!/usr/bin/env python3
"""Prune VAGEN checkpoints: keep latest, every Nth step, and best validation step.

Default policy (Nimloth baseline):
- always keep latest checkpoint (for resume)
- keep one checkpoint every ``--keep-every`` steps (e.g. 10, 20, 30, ...)
- always keep best validation step (by mean traj_success if available, else mean score)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def step_from_ckpt_dir(path: Path) -> int | None:
    m = re.fullmatch(r"global_step_(\d+)", path.name)
    return int(m.group(1)) if m else None


def mean_score(jsonl_path: Path) -> float | None:
    total = 0.0
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                score = obj.get("score")
                if score is not None:
                    total += float(score)
                    count += 1
            except Exception:
                continue
    return total / count if count else None


def traj_success_metric(record: dict) -> float | None:
    base = record.get("val-aux/navigation_base/traj_success/mean@1")
    common = record.get("val-aux/navigation_common/traj_success/mean@1")
    if base is not None and common is not None:
        return (float(base) + float(common)) / 2.0
    if base is not None:
        return float(base)
    if common is not None:
        return float(common)
    return None


def best_validation_step(validation_dir: Path) -> tuple[int | None, float | None]:
    best_step = None
    best_score = None
    if not validation_dir.exists():
        return None, None
    for p in validation_dir.glob("*.jsonl"):
        try:
            step = int(p.stem)
        except ValueError:
            continue
        score = mean_score(p)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_step = step
            best_score = score
    return best_step, best_score


def best_from_val_curve(val_curve_path: Path) -> tuple[int | None, float | None]:
    best_step = None
    best_score = None
    if not val_curve_path.exists():
        return None, None
    with val_curve_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = obj.get("checkpoint_step")
            metric = traj_success_metric(obj)
            if step is None or metric is None:
                continue
            step = int(step)
            if best_score is None or metric > best_score:
                best_step = step
                best_score = metric
    return best_step, best_score


def read_latest_step(checkpoint_dir: Path) -> int | None:
    latest_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not latest_file.exists():
        return None
    try:
        return int(latest_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def compute_keep_steps(
    all_steps: list[int],
    *,
    latest: int | None,
    keep_every: int,
    best_step: int | None,
    keep_last: int | None,
    last_val_step: int | None = None,
) -> set[int]:
    keep: set[int] = set()
    if latest is not None:
        keep.add(latest)
    if keep_every > 0:
        keep.update(step for step in all_steps if step % keep_every == 0)
    if best_step is not None:
        keep.add(best_step)
    if keep_last is not None and keep_last > 0 and all_steps:
        keep.update(all_steps[-keep_last:])
    # Never prune checkpoints the val watcher has not evaluated yet.
    if last_val_step is not None:
        keep.update(step for step in all_steps if step > last_val_step)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--validation-dir", required=True, type=Path)
    ap.add_argument(
        "--keep-every",
        type=int,
        default=10,
        help="Keep checkpoints at steps divisible by this value (default: 10).",
    )
    ap.add_argument(
        "--keep-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep best validation checkpoint (default: true).",
    )
    ap.add_argument(
        "--keep-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always keep latest checkpoint for resume (default: true).",
    )
    ap.add_argument(
        "--val-curve",
        type=Path,
        default=None,
        help="Optional val_curve.jsonl from val_wandb_watcher (traj_success metric).",
    )
    ap.add_argument(
        "--last-val-step",
        type=int,
        default=None,
        help=(
            "Highest checkpoint step already val'd by val_wandb_watcher. "
            "Checkpoints with step > this are never pruned until val completes."
        ),
    )
    ap.add_argument(
        "--val-run-dir",
        type=Path,
        default=None,
        help="If set, read last_val_step.txt from this val_wandb_watcher directory.",
    )
    ap.add_argument(
        "--keep-last",
        type=int,
        default=None,
        help="Optional: also keep the last N checkpoints (legacy mode).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    last_val_step = args.last_val_step
    if last_val_step is None and args.val_run_dir is not None:
        state_file = args.val_run_dir / "last_val_step.txt"
        if state_file.exists():
            try:
                last_val_step = int(state_file.read_text(encoding="utf-8").strip())
            except ValueError:
                last_val_step = None

    ckpt_dir = args.checkpoint_dir
    ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("global_step_*") if ckpt_dir.exists() else []:
        step = step_from_ckpt_dir(p)
        if step is not None and p.is_dir():
            ckpts.append((step, p))
    ckpts.sort()

    if not ckpts:
        print(f"No checkpoint dirs found under {ckpt_dir}")
        return 0

    all_steps = [step for step, _ in ckpts]
    latest = read_latest_step(ckpt_dir) if args.keep_latest else None

    best_step = None
    best_score = None
    if args.keep_best:
        best_step, best_score = best_from_val_curve(args.val_curve) if args.val_curve else (None, None)
        val_best_step, val_best_score = best_validation_step(args.validation_dir)
        if val_best_step in all_steps and val_best_score is not None:
            if best_score is None or val_best_score > best_score:
                best_step = val_best_step
                best_score = val_best_score
    if best_step is not None and best_step not in all_steps:
        best_step = None
        best_score = None

    keep_steps = compute_keep_steps(
        all_steps,
        latest=latest,
        keep_every=args.keep_every,
        best_step=best_step,
        keep_last=args.keep_last,
        last_val_step=last_val_step,
    )

    info = {
        "policy": "keep_latest + keep_every + keep_best + keep_unvalued",
        "keep_every": args.keep_every,
        "latest_step": latest,
        "best_step": best_step,
        "best_score": best_score,
        "last_val_step": last_val_step,
        "keep_steps": sorted(keep_steps),
        "all_steps": all_steps,
    }
    print(json.dumps(info, indent=2, sort_keys=True))

    meta_path = ckpt_dir / "retained_checkpoints.json"
    if not args.dry_run:
        meta_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for step, path in ckpts:
        if step in keep_steps:
            continue
        print(f"Removing {path}")
        if not args.dry_run:
            shutil.rmtree(path, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
