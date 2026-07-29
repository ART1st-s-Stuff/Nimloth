from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.training.sft2.aggregate_mcts_rollouts import aggregate


def _write_dataset(
    root: Path,
    name: str,
    *,
    count: int = 2,
    success_rate: float = 0.5,
) -> None:
    output = root / "eval_sets" / name
    output.mkdir(parents=True)
    (output / "evaluation_contract.json").write_text(
        json.dumps(
            {
                "history_size": 1,
                "prediction_horizon": 4,
                "num_simulations": 100,
                "exploration_constant": 1.0,
                "eval_sets": [name],
            }
        )
    )
    (output / "rollout_summary.json").write_text(
        json.dumps(
            {
                "status": "ALL_OK",
                "num_trajectories": count,
                "num_transitions": count * 3,
                "metrics": {
                    "overall": {
                        "success_rate": success_rate,
                        "avg_reward": success_rate * 2,
                        "avg_steps": 3.0,
                    }
                },
            }
        )
    )


def test_aggregate_combines_parallel_eval_sets(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "base", success_rate=0.5)
    _write_dataset(tmp_path, "common_sense", success_rate=1.0)

    summary = aggregate(tmp_path, ("base", "common_sense"), 2)

    assert summary["num_trajectories"] == 4
    assert summary["num_transitions"] == 12
    assert summary["metrics"]["overall"] == {
        "success_rate": 0.75,
        "avg_reward": 1.5,
        "avg_steps": 3.0,
    }


def test_aggregate_rejects_incomplete_dataset(tmp_path: Path) -> None:
    _write_dataset(tmp_path, "base", count=1)

    with pytest.raises(ValueError, match="trajectory count mismatch"):
        aggregate(tmp_path, ("base",), 2)
