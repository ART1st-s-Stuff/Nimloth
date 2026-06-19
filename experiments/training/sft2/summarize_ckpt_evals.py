#!/usr/bin/env python3
"""Aggregate greedy rollout success rates across SFT2 checkpoint eval runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-out", type=Path, required=True)
    ap.add_argument("--tag-prefix", type=str, default="sft2_ckpt")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    eval_root = args.train_out / "eval_rollouts"
    rows: list[dict] = []
    if not eval_root.is_dir():
        print(json.dumps({"train_out": str(args.train_out), "rows": [], "error": "missing eval_rollouts"}))
        return 1

    seen: set[str] = set()
    patterns = (
        f"sft2_eval_{args.tag_prefix}_*",
        f"sft1_eval_{args.tag_prefix}_*",
    )
    run_dirs: list[Path] = []
    for pattern in patterns:
        for run_dir in sorted(eval_root.glob(pattern)):
            key = str(run_dir)
            if key in seen:
                continue
            seen.add(key)
            run_dirs.append(run_dir)

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        ckpt_tag = run_dir.name
        for prefix in (f"sft2_eval_{args.tag_prefix}_", f"sft1_eval_{args.tag_prefix}_"):
            if ckpt_tag.startswith(prefix):
                ckpt_tag = ckpt_tag[len(prefix):]
                break
        summary = load_summary(run_dir / "summary_0.json")
        row = {
            "ckpt": ckpt_tag,
            "run_dir": str(run_dir),
            "summary": summary,
        }
        if summary:
            for split in ("val", "test"):
                agg = summary.get(f"{split}_aggregate")
                if agg:
                    row[f"{split}_success_rate"] = agg.get("success_rate")
                    row[f"{split}_records"] = agg.get("records")
        rows.append(row)

    out = {"train_out": str(args.train_out), "tag_prefix": args.tag_prefix, "checkpoints": rows}
    text = json.dumps(out, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
