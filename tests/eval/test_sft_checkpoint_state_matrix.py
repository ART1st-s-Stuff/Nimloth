from __future__ import annotations

from collections import Counter

import numpy as np

from nimloth.eval.sft_checkpoint_state_matrix import (
    combination_metrics,
    select_early_transition_records,
)


def _record(source: str, index: int, action: int, *, steps: int = 2) -> dict:
    return {
        "id": f"{source}/{index}",
        "data_source": source,
        "action_indices": [action] * steps,
        "success": bool(index % 2),
    }


def test_select_early_transition_records_is_deterministic_and_source_balanced() -> None:
    records = [
        _record(source, index, index % 2)
        for source in ("base", "common", "long")
        for index in range(6)
    ]

    selected = select_early_transition_records(records, per_source=4)
    repeated = select_early_transition_records(list(reversed(records)), per_source=4)

    assert [row["id"] for row in selected] == [row["id"] for row in repeated]
    assert Counter(row["data_source"] for row in selected) == {
        "base": 4,
        "common": 4,
        "long": 4,
    }
    for source in ("base", "common", "long"):
        assert {row["action_indices"][0] for row in selected if row["data_source"] == source} == {0, 1}


def test_select_early_transition_records_rejects_short_trajectories() -> None:
    records = [_record("base", index, 0, steps=1) for index in range(3)]
    try:
        select_early_transition_records(records, per_source=1)
    except ValueError as error:
        assert "at least two actions" in str(error)
    else:  # pragma: no cover
        raise AssertionError("expected selection failure")


def test_combination_metrics_reports_copy_relative_and_value_calibration() -> None:
    current = np.zeros((2, 2, 3), dtype=np.float32)
    actual_next = np.ones((2, 2, 3), dtype=np.float32)
    predicted_next = actual_next.copy()
    current_dino = current.copy()
    next_dino = actual_next.copy()
    current_q = np.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=np.float32)
    actual_next_q = np.asarray([[0.5, 0.0], [0.0, 1.0]], dtype=np.float32)
    predicted_next_q = actual_next_q.copy()

    result = combination_metrics(
        current_state=current,
        actual_next_state=actual_next,
        predicted_next_state=predicted_next,
        current_dino=current_dino,
        next_dino=next_dino,
        current_q=current_q,
        actual_next_q=actual_next_q,
        predicted_next_q=predicted_next_q,
        current_actions=np.asarray([0, 1]),
        next_actions=np.asarray([0, 1]),
        current_returns=np.asarray([0.25, 0.75]),
        next_returns=np.asarray([0.5, 1.0]),
    )

    assert result["behavior_copy_rmse"] == 1.0
    assert result["behavior_predicted_rmse"] == 0.0
    assert result["behavior_predicted_vs_copy_skill"] == 1.0
    assert result["dino_predicted_vs_copy_skill"] == 1.0
    assert result["value_current_executed_rmse"] == 0.0
    assert result["value_actual_next_executed_rmse"] == 0.0
    assert result["value_predicted_next_executed_rmse"] == 0.0
