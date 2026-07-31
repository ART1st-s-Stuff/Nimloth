#!/usr/bin/env python3
"""Validate a completed formal DINO-grid SFT2 output before downstream use."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch

from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE
from nimloth.training.sft2.checkpoint import is_trainable_checkpoint_dir
from nimloth.training.sft2.mcts_evaluation import (
    load_sft2_mcts_evaluation_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--expected-grad-accum", type=int, required=True)
    parser.add_argument("--expected-epochs", type=int, default=2)
    parser.add_argument("--expected-wandb-run-id", required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final = args.output_dir / "final"
    assert args.output_dir.joinpath("sft2_done.flag").is_file()
    assert is_trainable_checkpoint_dir(final)
    run_id = (
        args.output_dir.joinpath("wandb_run_id.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert run_id == args.expected_wandb_run_id
    state = torch.load(
        final / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    invariants = state["training_invariants"]
    assert int(state["step"]) == args.expected_step
    assert int(state["epoch"]) == args.expected_epochs
    assert state["epoch_complete"] is True
    assert invariants["world_size"] == args.expected_world_size
    assert invariants["batch_size"] == 1
    assert invariants["grad_accum"] == args.expected_grad_accum
    assert invariants["history_size"] == 1
    assert invariants["prediction_horizon"] == 4
    assert invariants["value_objective"] == SFT2_VALUE_OBJECTIVE
    assert invariants["sigreg_batch_scope"] == "global_valid_states_v1"
    contract = load_sft2_mcts_evaluation_contract(final)

    with args.output_dir.joinpath("train_step_log.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    finite_rows = 0
    seen_steps: set[int] = set()
    for row in rows:
        if row.get("global_step"):
            seen_steps.add(int(row["global_step"]))
        values = [
            row[key]
            for key in (
                "total_loss",
                "wm_mse",
                "dino_grid_mse",
                "sigreg_loss",
                "value_total",
                "value_mc_mse",
                "lm_ce",
            )
            if row.get(key)
        ]
        if values:
            assert all(math.isfinite(float(value)) for value in values)
            finite_rows += 1
    assert args.expected_step in seen_steps
    assert finite_rows > 0

    result = {
        "status": "passed",
        "checkpoint": str(final),
        "step": int(state["step"]),
        "epoch": int(state["epoch"]),
        "world_size": invariants["world_size"],
        "grad_accum": invariants["grad_accum"],
        "value_objective": invariants["value_objective"],
        "history_size": contract.history_size,
        "prediction_horizon": contract.prediction_horizon,
        "action_count": contract.action_count,
        "finite_metric_rows": finite_rows,
        "wandb_run_id": run_id,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.result_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
