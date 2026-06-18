#!/usr/bin/env python3
"""Prune VAGEN checkpoints to last N global steps plus best validation step.

Best step is selected by mean `score` across validation JSONL files named `<step>.jsonl`.
If no validation scores are found, only the last N checkpoints are kept.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--validation-dir", required=True, type=Path)
    ap.add_argument("--keep-last", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    ckpts = []
    for p in ckpt_dir.glob("global_step_*") if ckpt_dir.exists() else []:
        step = step_from_ckpt_dir(p)
        if step is not None and p.is_dir():
            ckpts.append((step, p))
    ckpts.sort()

    if not ckpts:
        print(f"No checkpoint dirs found under {ckpt_dir}")
        return 0

    last_steps = {step for step, _ in ckpts[-args.keep_last :]}
    best_step, best_score = best_validation_step(args.validation_dir)
    keep_steps = set(last_steps)
    if best_step is not None:
        keep_steps.add(best_step)

    info = {
        "last_steps": sorted(last_steps),
        "best_step": best_step,
        "best_score": best_score,
        "keep_steps": sorted(keep_steps),
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
