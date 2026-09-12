import torch
import pytest
from evaluate_dino import recovery_metrics


def test_exact_recovery_beats_mean_and_cross_trajectory_shuffle():
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(9, 4, 8, generator=generator)
    ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3])
    result = recovery_metrics(target, target, ids, torch.zeros(4, 8))
    assert result["metrics"]["state_mse"]["mean"] == 0
    assert result["metrics"]["relative_mse_reduction_vs_train_mean"]["mean"] == 1
    assert result["metrics"]["shuffled_state_mse"]["mean"] > 0
    assert torch.all(ids != ids[result["shuffle_source_indices"]])
    assert result == recovery_metrics(target, target, ids, torch.zeros(4, 8))


def test_macro_weighting_does_not_reward_long_trajectory():
    target = torch.ones(6, 2, 3)
    prediction = target.clone()
    prediction[:3] = 3
    ids = torch.tensor([0, 0, 0, 1, 1, 2])
    result = recovery_metrics(prediction, target, ids, torch.zeros(2, 3))
    assert result["metrics"]["state_mse"]["mean"] == pytest.approx(4 / 3)


def test_rejects_impossible_shuffle():
    target = torch.ones(4, 2, 3)
    with pytest.raises(ValueError, match="dominates"):
        recovery_metrics(target, target, torch.tensor([0, 0, 0, 1]), torch.zeros(2, 3))
