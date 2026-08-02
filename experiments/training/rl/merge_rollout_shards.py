#!/usr/bin/env python3
"""Strictly merge multi-episode rollout shards into one ordered batch."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from nimloth.rollout import merge_fresh_rollout_shards
from nimloth.training.rl.evaluation import summarize_rollouts


@dataclass(frozen=True)
class ShardSpec:
    manifest_path: Path
    eval_sets: tuple[str, ...]
    seed_offset: int
    num_episodes: int

    def expected_schedule(
        self,
        *,
        seed_per_eval_set: bool,
    ) -> tuple[tuple[str, str], ...]:
        if not self.eval_sets:
            raise ValueError("rollout shard requires at least one eval set")
        if len(set(self.eval_sets)) != len(self.eval_sets):
            raise ValueError(f"rollout shard eval sets must be unique: {self.eval_sets}")
        if self.seed_offset < 1:
            raise ValueError("rollout shard seed offset must be positive")
        if self.num_episodes < 1:
            raise ValueError("rollout shard episode count must be positive")

        schedule: list[tuple[str, str]] = []
        counters = {name: self.seed_offset for name in self.eval_sets}
        global_seed = self.seed_offset
        for episode_index in range(self.num_episodes):
            eval_set = self.eval_sets[episode_index % len(self.eval_sets)]
            if seed_per_eval_set:
                seed = counters[eval_set]
                counters[eval_set] += 1
                record_id = f"rl_{eval_set}_{seed:06d}"
            else:
                seed = global_seed
                global_seed += 1
                record_id = f"rl_{seed:06d}"
            schedule.append((record_id, eval_set))
        return tuple(schedule)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--shard-eval-sets", action="append", required=True)
    parser.add_argument("--shard-seed-offset", type=int, action="append", required=True)
    parser.add_argument("--shard-num-episodes", type=int, action="append", required=True)
    parser.add_argument("--seed-per-eval-set", action="store_true")
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _shard_specs(args: argparse.Namespace) -> tuple[ShardSpec, ...]:
    counts = {
        "manifests": len(args.shard_manifest),
        "eval_sets": len(args.shard_eval_sets),
        "seed_offsets": len(args.shard_seed_offset),
        "episode_counts": len(args.shard_num_episodes),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"parallel shard argument counts disagree: {counts}")
    return tuple(
        ShardSpec(
            manifest_path=manifest_path,
            eval_sets=tuple(
                name for name in eval_sets_csv.split(",") if name
            ),
            seed_offset=seed_offset,
            num_episodes=num_episodes,
        )
        for manifest_path, eval_sets_csv, seed_offset, num_episodes in zip(
            args.shard_manifest,
            args.shard_eval_sets,
            args.shard_seed_offset,
            args.shard_num_episodes,
            strict=True,
        )
    )


def main() -> int:
    args = parse_args()
    specs = _shard_specs(args)
    schedules = tuple(
        spec.expected_schedule(seed_per_eval_set=args.seed_per_eval_set)
        for spec in specs
    )
    for spec, schedule in zip(specs, schedules, strict=True):
        summary_path = spec.manifest_path.parent / "rollout_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"rollout shard summary is missing: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "ALL_OK":
            raise ValueError(f"rollout shard is not complete: {summary_path}")
        if int(summary.get("num_trajectories", -1)) != len(schedule):
            raise ValueError(
                "rollout shard episode count is wrong: "
                f"path={summary_path}, actual={summary.get('num_trajectories')}, "
                f"expected={len(schedule)}"
            )
        if tuple(summary.get("eval_sets", ())) != spec.eval_sets:
            raise ValueError(
                "rollout shard datasets do not match its schedule: "
                f"path={summary_path}, actual={summary.get('eval_sets')}, "
                f"expected={spec.eval_sets}"
            )
        if bool(summary.get("seed_per_eval_set")) != args.seed_per_eval_set:
            raise ValueError(
                f"rollout shard seed mode does not match: {summary_path}"
            )

    flattened_schedule = tuple(item for schedule in schedules for item in schedule)
    expected_record_ids = tuple(record_id for record_id, _ in flattened_schedule)
    trajectories, manifest = merge_fresh_rollout_shards(
        tuple(spec.manifest_path for spec in specs),
        output_dir=args.output_dir,
        expected_record_ids=expected_record_ids,
    )
    wrong_splits = sorted({item.split for item in trajectories if item.split != args.split})
    if wrong_splits:
        raise ValueError(
            f"merged rollout has records outside split={args.split}: {wrong_splits}"
        )

    eval_sets = tuple(dict.fromkeys(eval_set for _, eval_set in flattened_schedule))
    grouped = {name: [] for name in eval_sets}
    for trajectory, (_, eval_set) in zip(
        trajectories,
        flattened_schedule,
        strict=True,
    ):
        grouped[eval_set].append(trajectory)
    summary_path = args.output_dir / "rollout_summary.json"
    summary = {
        "status": "ALL_OK",
        "split": args.split,
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
        "eval_sets": list(eval_sets),
        "seed_per_eval_set": args.seed_per_eval_set,
        "metrics": {
            "overall": summarize_rollouts(trajectories),
            "by_eval_set": {
                name: summarize_rollouts(items) for name, items in grouped.items()
            },
        },
        "shards": [
            {
                "manifest": str(spec.manifest_path.resolve()),
                "eval_sets": list(spec.eval_sets),
                "seed_offset": spec.seed_offset,
                "num_episodes": spec.num_episodes,
            }
            for spec in specs
        ],
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
