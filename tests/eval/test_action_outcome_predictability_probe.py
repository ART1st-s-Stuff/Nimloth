from __future__ import annotations

import numpy as np
import pytest

from nimloth.eval.action_outcome_predictability_probe import (
    _fit_epoch_selection,
    _train_final_probe,
    average_precision,
    binary_probe_metrics,
    grouped_selection_mask,
)


def test_average_precision_handles_perfect_and_reversed_ranking() -> None:
    labels = np.asarray([False, True, False, True])
    assert average_precision(labels, np.asarray([0.1, 0.8, 0.2, 0.9])) == pytest.approx(1.0)
    assert average_precision(labels, np.asarray([0.9, 0.2, 0.8, 0.1])) == pytest.approx(
        (1 / 3 + 2 / 4) / 2
    )


def test_binary_probe_metrics_report_auc_calibration_and_baseline() -> None:
    labels = np.asarray([False, True, False, True])
    logits = np.asarray([-3.0, 3.0, -2.0, 2.0])
    result = binary_probe_metrics(labels, logits, train_success_rate=0.5, ece_bins=5)
    assert result["count"] == 4
    assert result["success_rate"] == 0.5
    assert result["roc_auc"] == pytest.approx(1.0)
    assert result["pr_auc"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["brier"] < result["constant_train_rate_baseline"]["brier"]
    assert result["ece"] < 0.2


def test_linear_probe_optimization_learns_separable_cpu_signal() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(80, 6)).astype(np.float32)
    labels = features[:, 0] > 0
    selected_epoch, history = _fit_epoch_selection(
        features[:60],
        labels[:60],
        features[60:70],
        labels[60:70],
        learning_rate=0.05,
        weight_decay=0.001,
        max_epochs=80,
        patience=10,
        seed=3,
        device="cpu",
    )
    logits, weights = _train_final_probe(
        features[:70],
        labels[:70],
        features[70:],
        learning_rate=0.05,
        weight_decay=0.001,
        epochs=selected_epoch,
        seed=3,
        device="cpu",
    )
    assert history
    assert binary_probe_metrics(
        labels[70:], logits, train_success_rate=float(labels[:70].mean())
    )["roc_auc"] > 0.8
    assert set(weights) == {"weight", "bias", "feature_mean", "feature_std"}


def test_grouped_selection_mask_keeps_identical_groups_together() -> None:
    groups = np.asarray(["a", "a", "b", "c", "c", "d", "e", "f"])
    selected = grouped_selection_mask(groups, modulo=2)
    assert selected.shape == groups.shape
    for group in set(groups.tolist()):
        assert len(set(selected[groups == group].tolist())) == 1
    assert selected.any() and not selected.all()
