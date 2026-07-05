import math

import torch

from nimloth.representation_ablation.metrics import (
    EncodedTransition,
    predictor_multistep_metrics,
    predictor_one_step_metrics,
    success_ranking_auc,
    value_head_metrics,
)


class AddActionPredictor(torch.nn.Module):
    def rollout_states(self, state_emb: torch.Tensor, action_sequences: torch.Tensor) -> torch.Tensor:
        states = []
        cur = state_emb.clone()
        for t in range(action_sequences.shape[1]):
            cur = cur + action_sequences[:, t : t + 1].float()
            states.append(cur)
        return torch.stack(states, dim=1)


def test_value_head_metrics_topk_and_auc() -> None:
    values = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.2, 0.1], [0.1, 0.2, 0.7]])
    actions = torch.tensor([1, 2, 2])
    targets = torch.tensor([1.0, 0.0, 1.0])
    successes = torch.tensor([True, False, True])
    metrics = value_head_metrics(values, actions, targets, successes)
    assert metrics["value_top1_action_acc"] == 2 / 3
    assert metrics["value_top2_action_acc"] == 2 / 3
    assert 0.0 <= metrics["value_success_ranking_auc"] <= 1.0
    assert "value_calib_bin0_success_rate" in metrics


def test_success_ranking_auc_nan_without_both_classes() -> None:
    auc = success_ranking_auc(torch.tensor([0.1, 0.2]), torch.tensor([True, True]))
    assert math.isnan(auc)


def test_predictor_one_step_metrics() -> None:
    pred = torch.tensor([[1.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    metrics = predictor_one_step_metrics(pred, target)
    assert metrics["predictor_1step_mse"] == 0.0
    assert metrics["predictor_1step_cosine"] == 1.0


def test_predictor_multistep_metrics_contiguous_records() -> None:
    rows = [
        EncodedTransition("r", 0, 1, 0.0, True, torch.tensor([0.0]), torch.tensor([1.0])),
        EncodedTransition("r", 1, 2, 0.0, True, torch.tensor([1.0]), torch.tensor([3.0])),
        EncodedTransition("r", 2, 1, 0.0, True, torch.tensor([3.0]), torch.tensor([4.0])),
    ]
    metrics = predictor_multistep_metrics(AddActionPredictor(), rows, [1, 2], device=torch.device("cpu"))
    assert metrics["predictor_depth1_count"] == 3.0
    assert metrics["predictor_depth2_count"] == 2.0
    assert metrics["predictor_depth2_mse"] == 0.0
