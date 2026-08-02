#!/usr/bin/env python3
"""Merge one-episode rollout workers into one strict fresh-policy batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.rollout import merge_fresh_rollout_shards
from nimloth.training.rl.evaluation import summarize_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-offset", type=int, required=True)
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--eval-sets", nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seed_offset < 1:
        raise ValueError("seed_offset must be positive")
    if args.num_episodes < 1:
        raise ValueError("num_episodes must be positive")
    if len(args.shard_manifest) != args.num_episodes:
        raise ValueError(
            "parallel rollout requires exactly one shard per episode: "
            f"shards={len(args.shard_manifest)}, episodes={args.num_episodes}"
        )
    if len(set(args.eval_sets)) != len(args.eval_sets):
        raise ValueError("parallel rollout eval sets must be unique")

    for shard_index, manifest_path in enumerate(args.shard_manifest):
        summary_path = manifest_path.parent / "rollout_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"rollout shard summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_eval_set = args.eval_sets[shard_index % len(args.eval_sets)]
        if summary.get("status") != "ALL_OK":
            raise ValueError(f"rollout shard is not complete: {summary_path}")
        if int(summary.get("num_trajectories", -1)) != 1:
            raise ValueError(f"rollout shard must contain exactly one episode: {summary_path}")
        if tuple(summary.get("eval_sets", ())) != (expected_eval_set,):
            raise ValueError(
                "rollout shard dataset does not match global round-robin order: "
                f"shard={shard_index}, actual={summary.get('eval_sets')}, "
                f"expected={expected_eval_set}"
            )

    expected_record_ids = tuple(
        f"rl_{seed:06d}"
        for seed in range(
            args.seed_offset,
            args.seed_offset + args.num_episodes,
        )
    )
    trajectories, manifest = merge_fresh_rollout_shards(
        args.shard_manifest,
        output_dir=args.output_dir,
        expected_record_ids=expected_record_ids,
    )
    grouped = {name: [] for name in args.eval_sets}
    for index, trajectory in enumerate(trajectories):
        grouped[args.eval_sets[index % len(args.eval_sets)]].append(trajectory)
    summary_path = args.output_dir / "rollout_summary.json"
    summary = {
        "status": "ALL_OK",
        "num_trajectories": len(trajectories),
        "num_transitions": sum(item.num_steps for item in trajectories),
        "jsonl": manifest.trajectory_path,
        "summary_json": str(summary_path.resolve()),
        "fresh_manifest": str(
            (args.output_dir / "fresh_policy_manifest.json").resolve()
        ),
        "processor_pixel_bounds": [
            manifest.processor_min_pixels,
            manifest.processor_max_pixels,
        ],
        "eval_sets": list(args.eval_sets),
        "seed_per_eval_set": False,
        "metrics": {
            "overall": summarize_rollouts(trajectories),
            "by_eval_set": {
                name: summarize_rollouts(items)
                for name, items in grouped.items()
            },
        },
        "shard_manifests": [str(path.resolve()) for path in args.shard_manifest],
        "expected_record_ids": list(expected_record_ids),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
