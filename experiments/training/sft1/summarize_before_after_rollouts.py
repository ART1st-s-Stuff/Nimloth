#!/usr/bin/env python3
"""Summarize traj_success across baseline and post-SFT rollout trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_run(root: Path, step: str) -> dict[str, dict[str, float | int]]:
    stats: dict[str, dict[str, float | int]] = {}
    if not root.is_dir():
        return stats
    for split_dir in sorted((root / "validation").glob("*")):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        total = success = 0
        for jsonl in sorted(split_dir.glob(f"*/{step}.jsonl")):
            with jsonl.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    total += 1
                    if float(obj.get("traj_success", 0.0) or 0.0) >= 1.0:
                        success += 1
        if total:
            stats[split] = {
                "success": success,
                "total": total,
                "success_rate": success / total,
            }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scriptdir", type=Path, required=True)
    args = ap.parse_args()
    runs = {
        "baseline_step48": (args.scriptdir / "runs/sft1_rollouts_vagen48_ws2_2node_externalenv", "48"),
        "sft_epoch012": (args.scriptdir / "runs/sft1_rollouts_vagen48_sft_epoch012", "1"),
        "sft_final": (args.scriptdir / "runs/sft1_rollouts_vagen48_sft_final", "1"),
    }
    out: dict[str, dict] = {}
    for name, (root, step) in runs.items():
        out[name] = summarize_run(root, step)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
