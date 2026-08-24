from __future__ import annotations

import numpy as np
import pytest

from nimloth.eval.id75_action_outcome_audit import (
    binary_auc,
    parse_step_action_success,
    stratified_outcome_metrics,
)


def _record(feedback: str) -> dict:
    return {
        "action_indices": [3],
        "observation_texts": [
            "initial",
            f"After your action.\nThe environment feedback is: {feedback}\nnext",
        ],
    }


def test_parse_step_action_success_uses_exact_environment_feedback() -> None:
    assert parse_step_action_success(
        _record("Last action is executed successfully."), 0
    )
    assert not parse_step_action_success(
        _record("Last action is not executed successfully."), 0
    )
    with pytest.raises(ValueError, match="exactly one"):
        parse_step_action_success(_record("feedback unavailable"), 0)


def test_binary_auc_is_rank_based_and_rejects_one_class() -> None:
    labels = np.asarray([False, True, False, True])
    assert binary_auc(labels, np.asarray([0.1, 0.8, 0.2, 0.9])) == pytest.approx(1.0)
    assert binary_auc(labels, np.asarray([0.8, 0.1, 0.9, 0.2])) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="both classes"):
        binary_auc(np.asarray([True, True]), np.asarray([0.1, 0.2]))


def test_stratified_metrics_expose_blocked_harm_and_success_benefit() -> None:
    current = np.zeros((4, 1, 1), dtype=np.float32)
    target = np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float32).reshape(4, 1, 1)
    prediction = np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32).reshape(4, 1, 1)
    success = np.asarray([False, False, True, True])
    result = stratified_outcome_metrics(
        current_state=current,
        actual_next_state=target,
        predicted_next_state=prediction,
        action_success=success,
        bootstrap_seed=7,
        bootstrap_draws=200,
    )
    assert result["failed"]["count"] == 2
    assert result["failed"]["copy_rmse"] == 0.0
    assert result["failed"]["prediction_rmse"] == 1.0
    assert result["failed"]["prediction_minus_copy_mse"] == pytest.approx(1.0)
    assert result["successful"]["count"] == 2
    assert result["successful"]["copy_rmse"] == 2.0
    assert result["successful"]["prediction_rmse"] == 0.0
    assert result["successful"]["copy_relative_skill"] == pytest.approx(1.0)
    assert result["predicted_change_success_auc"] == pytest.approx(1.0)
    assert result["actual_change_success_auc"] == pytest.approx(1.0)
