#!/usr/bin/env python3
"""Validate and record one fixed-seed external RL evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nimloth.util.wandb import init_wandb_run, log_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--eval-sets", nargs="+", required=True)
    parser.add_argument("--seeds-per-eval-set", type=int, required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser.parse_args()


def _metrics(summary: dict, eval_sets: tuple[str, ...]) -> dict[str, float]:
    raw_metrics = summary["metrics"]
    metrics = {
        key: float(value) for key, value in raw_metrics["overall"].items()
    }
    for eval_set in eval_sets:
        for key, value in raw_metrics["by_eval_set"][eval_set].items():
            metrics[f"{eval_set}_{key}"] = float(value)
    return metrics


def main() -> int:
    args = parse_args()
    if args.iteration < 1:
        raise ValueError("evaluation iteration must be positive")
    if args.seeds_per_eval_set < 1:
        raise ValueError("evaluation seeds per set must be positive")
    eval_sets = tuple(args.eval_sets)
    if len(set(eval_sets)) != len(eval_sets):
        raise ValueError("evaluation datasets must be unique")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    expected_record_ids = tuple(
        f"rl_{eval_set}_{seed:06d}"
        for eval_set in eval_sets
        for seed in range(1, args.seeds_per_eval_set + 1)
    )
    if summary.get("status") != "ALL_OK" or summary.get("split") != "eval":
        raise ValueError("evaluation summary is not a complete held-out eval")
    if tuple(summary.get("eval_sets", ())) != eval_sets:
        raise ValueError("evaluation datasets do not match the fixed contract")
    if tuple(summary.get("expected_record_ids", ())) != expected_record_ids:
        raise ValueError("evaluation record IDs do not cover the fixed seed contract")
    if int(summary.get("num_trajectories", -1)) != len(expected_record_ids):
        raise ValueError("evaluation trajectory count does not match the contract")

    metrics = _metrics(summary, eval_sets)
    csv_path = args.evaluation_root / "eval_step_log.csv"
    columns = (
        "iteration",
        "num_trajectories",
        "num_transitions",
        *metrics.keys(),
        "summary_json",
    )
    existing_rows: dict[int, dict[str, str]] = {}
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8") as stream:
            existing_rows = {
                int(row["iteration"]): row for row in csv.DictReader(stream)
            }
    already_recorded = args.iteration in existing_rows
    if already_recorded:
        recorded_summary = Path(existing_rows[args.iteration]["summary_json"])
        if recorded_summary.resolve() != args.summary.resolve():
            raise ValueError(
                f"evaluation iteration {args.iteration} points to another summary"
            )
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "iteration": args.iteration,
                    "num_trajectories": summary["num_trajectories"],
                    "num_transitions": summary["num_transitions"],
                    **metrics,
                    "summary_json": str(args.summary.resolve()),
                }
            )

        try:
            run = init_wandb_run(
                rank=0,
                output_dir=args.evaluation_root,
                enabled=True,
                default_project=args.wandb_project,
                run_name=args.wandb_run_name,
                config={
                    "evaluation": {
                        "eval_sets": list(eval_sets),
                        "seeds_per_eval_set": args.seeds_per_eval_set,
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "navigation_profile": "vagen_eval",
                    }
                },
                metric_definitions=(
                    ("eval_iteration", None),
                    ("eval/*", "eval_iteration"),
                ),
            )
            log_metrics(
                run,
                namespace="eval",
                metrics=metrics,
                step=args.iteration,
                context={"eval_iteration": args.iteration},
            )
            if run is not None:
                run.finish()
        except Exception as error:  # W&B is auxiliary; the strict local record wins.
            print(
                json.dumps(
                    {"wandb": "failed", "error": f"{type(error).__name__}: {error}"}
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "status": "EVAL_RECORDED",
                "iteration": args.iteration,
                "metrics": metrics,
                "csv": str(csv_path.resolve()),
                "already_recorded": already_recorded,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
