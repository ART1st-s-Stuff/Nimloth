#!/usr/bin/env python3
"""Print side-by-side val/test success rates from summarize_sft1_eval_rollouts JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("left", type=Path, help="summary JSON (e.g. epoch17)")
    ap.add_argument("right", type=Path, help="summary JSON (e.g. step79)")
    ap.add_argument("--left-label", default=None)
    ap.add_argument("--right-label", default=None)
    args = ap.parse_args()

    left = load(args.left)
    right = load(args.right)
    ll = args.left_label or args.left.parent.name
    rl = args.right_label or args.right.parent.name

    print(f"{'split':<6} {'metric':<14} {ll:<28} {rl:<28} {'delta':>8}")
    print("-" * 90)
    for split in ("val", "test"):
        lk = f"{split}_aggregate"
        lv = left.get(lk, {})
        rv = right.get(lk, {})
        lr = float(lv.get("success_rate", 0.0) or 0.0)
        rr = float(rv.get("success_rate", 0.0) or 0.0)
        print(f"{split:<6} {'success_rate':<14} {lr:>26.2%} {rr:>26.2%} {lr-rr:>+8.2%}")
        print(
            f"{'':6} {'n_success':<14} "
            f"{int(lv.get('success',0))}/{int(lv.get('records',0)):<24} "
            f"{int(rv.get('success',0))}/{int(rv.get('records',0)):<24}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
