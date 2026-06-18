#!/usr/bin/env python3
"""Summarize traj_success from SFT1 val/test rollout jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_jsonl(path: Path) -> dict:
    total = 0
    success = 0.0
    rewards: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            total += 1
            ts = float(obj.get("traj_success", 0.0) or 0.0)
            if ts >= 1.0:
                success += 1
            rewards.append(float(obj.get("reward", obj.get("score", 0.0)) or 0.0))
    return {
        "path": str(path),
        "records": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
        "mean_reward": (sum(rewards) / len(rewards)) if rewards else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="eval run root with validation/{val,test}")
    ap.add_argument("--step", type=str, required=True, help="jsonl stem, e.g. 48 or 1")
    args = ap.parse_args()

    rows = []
    for split in ("val", "test"):
        split_dir = args.root / "validation" / split
        if not split_dir.exists():
            continue
        for jsonl in sorted(split_dir.glob(f"*/{args.step}.jsonl")):
            row = summarize_jsonl(jsonl)
            row["split"] = split
            row["shard"] = jsonl.parent.name
            rows.append(row)

    out = {"root": str(args.root), "step": args.step, "splits": rows}
    if rows:
        for split in ("val", "test"):
            split_rows = [r for r in rows if r["split"] == split]
            if split_rows:
                total = sum(r["records"] for r in split_rows)
                succ = sum(r["success"] for r in split_rows)
                out[f"{split}_aggregate"] = {
                    "records": total,
                    "success": succ,
                    "success_rate": succ / total if total else 0.0,
                }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
