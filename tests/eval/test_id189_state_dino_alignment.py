import numpy as np
import pytest

from nimloth.eval.id189_state_dino_alignment import (
    minimum_cost_slot_assignment,
    state_dino_metrics,
    state_statistics,
)


def test_state_dino_metrics_separates_scale_from_centered_direction() -> None:
    base = np.linspace(-1.0, 1.0, 16 * 8, dtype=np.float32).reshape(16, 8)
    scaled = 0.5 * base + 0.25

    metrics = state_dino_metrics(scaled, base)

    assert metrics["rmse"] > 0.0
    assert metrics["cosine"] < 1.0
    assert metrics["token_centered_cosine"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["token_standardized_rmse"] == pytest.approx(0.0, abs=1e-5)


def test_state_statistics_detects_slot_collapse() -> None:
    collapsed = np.broadcast_to(
        np.arange(8, dtype=np.float32), (16, 8)
    ).copy()
    varied = collapsed.copy()
    varied[:, 0] = np.arange(16, dtype=np.float32)

    collapsed_stats = state_statistics(collapsed)
    varied_stats = state_statistics(varied)

    assert collapsed_stats["slot_deviation_rms"] == pytest.approx(0.0)
    assert varied_stats["slot_deviation_rms"] > 0.0
    assert varied_stats["std"] > collapsed_stats["std"]


def test_minimum_cost_slot_assignment_finds_fixed_permutation() -> None:
    cost = np.full((4, 4), 100.0, dtype=np.float64)
    expected = (2, 0, 3, 1)
    for state_slot, dino_slot in enumerate(expected):
        cost[state_slot, dino_slot] = float(state_slot)

    assignment, total = minimum_cost_slot_assignment(cost)

    assert assignment == expected
    assert total == pytest.approx(6.0)


def test_minimum_cost_slot_assignment_requires_square_finite_cost() -> None:
    with pytest.raises(ValueError, match="square"):
        minimum_cost_slot_assignment(np.zeros((2, 3), dtype=np.float64))
    bad = np.zeros((2, 2), dtype=np.float64)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        minimum_cost_slot_assignment(bad)
