#!/usr/bin/env python3
"""Aggregate parallel per-dataset SFT2 MCTS rollout evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-sets", nargs="+", required=True)
    parser.add_argument("--episodes-per-eval-set", type=int, required=True)
    parser.add_argument("--shards-per-eval-set", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=1)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def aggregate(
    output_dir: Path,
    eval_sets: tuple[str, ...],
    episodes_per_eval_set: int,
    *,
    shards_per_eval_set: int = 1,
    seed_offset: int = 1,
) -> dict[str, Any]:
    if episodes_per_eval_set < 1:
        raise ValueError("episodes_per_eval_set must be positive")
    if not eval_sets or len(set(eval_sets)) != len(eval_sets):
        raise ValueError("eval_sets must be non-empty and unique")
    if shards_per_eval_set < 1:
        raise ValueError("shards_per_eval_set must be positive")
    if episodes_per_eval_set % shards_per_eval_set != 0:
        raise ValueError(
            "episodes_per_eval_set must be divisible by shards_per_eval_set"
        )

    by_eval_set: dict[str, dict[str, float]] = {}
    common_contract: dict[str, Any] | None = None
    weighted = {"success_rate": 0.0, "avg_reward": 0.0, "avg_steps": 0.0}
    total_episodes = 0
    total_transitions = 0
    episodes_per_shard = episodes_per_eval_set // shards_per_eval_set
    for eval_set in eval_sets:
        eval_weighted = {key: 0.0 for key in weighted}
        eval_count = 0
        for shard_index in range(shards_per_eval_set):
            dataset_dir = output_dir / "eval_sets" / eval_set
            if shards_per_eval_set > 1:
                dataset_dir = dataset_dir / f"shard_{shard_index:02d}"
            summary = _read_json(dataset_dir / "rollout_summary.json")
            contract = _read_json(dataset_dir / "evaluation_contract.json")
            shard_name = f"{eval_set}/shard_{shard_index:02d}"
            if summary.get("status") != "ALL_OK":
                raise ValueError(f"{shard_name} rollout did not report ALL_OK")
            count = int(summary.get("num_trajectories", -1))
            if count != episodes_per_shard:
                raise ValueError(
                    f"{shard_name} trajectory count mismatch: "
                    f"{count} != {episodes_per_shard}"
                )
            if contract.get("eval_sets") != [eval_set]:
                raise ValueError(
                    f"{shard_name} evaluation contract has wrong eval_sets"
                )
            if int(contract.get("episodes_per_eval_set", -1)) != count:
                raise ValueError(
                    f"{shard_name} contract has wrong episode count"
                )
            expected_seed = seed_offset + shard_index * episodes_per_shard
            if int(contract.get("seed_offset_per_eval_set", -1)) != expected_seed:
                raise ValueError(
                    f"{shard_name} seed range is not the requested contiguous shard"
                )
            comparable = {
                key: value
                for key, value in contract.items()
                if key
                not in {
                    "eval_sets",
                    "episodes_per_eval_set",
                    "seed_offset_per_eval_set",
                }
            }
            if common_contract is None:
                common_contract = comparable
            elif comparable != common_contract:
                raise ValueError(
                    f"{shard_name} evaluation contract does not match peers"
                )

            metrics = summary.get("metrics", {}).get("overall")
            if not isinstance(metrics, dict):
                raise ValueError(f"{shard_name} summary has no overall metrics")
            normalized = {key: float(metrics[key]) for key in weighted}
            if not all(math.isfinite(value) for value in normalized.values()):
                raise ValueError(
                    f"{shard_name} metrics are non-finite: {normalized}"
                )
            for key, value in normalized.items():
                eval_weighted[key] += value * count
                weighted[key] += value * count
            eval_count += count
            total_episodes += count
            total_transitions += int(summary.get("num_transitions", 0))
        if eval_count != episodes_per_eval_set:
            raise AssertionError("validated shard counts did not cover eval set")
        by_eval_set[eval_set] = {
            key: value / eval_count for key, value in eval_weighted.items()
        }

    assert common_contract is not None
    overall = {key: value / total_episodes for key, value in weighted.items()}
    return {
        "status": "ALL_OK",
        "evaluation": "sft2_pre_rl_mcts_parallel_v2",
        "num_trajectories": total_episodes,
        "num_transitions": total_transitions,
        "eval_sets": list(eval_sets),
        "episodes_per_eval_set": episodes_per_eval_set,
        "shards_per_eval_set": shards_per_eval_set,
        "seed_offset_per_eval_set": seed_offset,
        "metrics": {"overall": overall, "by_eval_set": by_eval_set},
        "contract": {
            **common_contract,
            "episodes_per_eval_set": episodes_per_eval_set,
            "seed_offset_per_eval_set": seed_offset,
            "shards_per_eval_set": shards_per_eval_set,
        },
    }


def main() -> int:
    args = parse_args()
    summary = aggregate(
        args.output_dir,
        tuple(args.eval_sets),
        args.episodes_per_eval_set,
        shards_per_eval_set=args.shards_per_eval_set,
        seed_offset=args.seed_offset,
    )
    summary_path = args.output_dir / "rollout_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config=summary["contract"],
        dir=str(args.output_dir),
    )
    metrics = summary["metrics"]
    logged = {
        "eval/overall/success_rate": metrics["overall"]["success_rate"],
        "eval/overall/avg_reward": metrics["overall"]["avg_reward"],
        "eval/overall/avg_steps": metrics["overall"]["avg_steps"],
        "eval/num_trajectories": summary["num_trajectories"],
    }
    for eval_set, values in metrics["by_eval_set"].items():
        for key, value in values.items():
            logged[f"eval/{eval_set}/{key}"] = value
    run.log(logged)
    for key, value in logged.items():
        run.summary[key] = value
    run.finish(exit_code=0)
    (args.output_dir / "mcts_eval_done.flag").write_text(
        "ALL_OK\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
