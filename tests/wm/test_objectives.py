"""阶段 objective 的 action-value 目标测试。"""

from __future__ import annotations

import torch

from nimloth.training.common import action_value_loss
from nimloth.wm.value_head import ValueHead


def _bias_only_head(bias: torch.Tensor) -> ValueHead:
    head = ValueHead(emb_dim=4, num_actions=bias.numel(), hidden_dim=4)
    with torch.no_grad():
        head.net[0].weight.zero_()
        head.net[0].bias.zero_()
        head.net[2].weight.zero_()
        head.net[2].bias.copy_(bias)
    return head


def test_value_ranking_zero_when_chosen_is_best() -> None:
    values = _bias_only_head(torch.tensor([2.0, 0.5, 0.1]))(torch.randn(1, 4))
    result = action_value_loss(
        values,
        torch.tensor([0]),
        torch.tensor([2.0]),
        ranking_margin=0.1,
        ranking_weight=1.0,
    )
    assert result.ranking.item() == 0.0
    assert result.loss.item() == result.monte_carlo_mse.item()


def test_value_ranking_positive_when_unchosen_beats_chosen() -> None:
    values = _bias_only_head(torch.tensor([0.5, 2.0, 0.1]))(torch.randn(1, 4))
    result = action_value_loss(
        values,
        torch.tensor([0]),
        torch.tensor([1.0]),
        ranking_margin=0.1,
        ranking_weight=1.0,
    )
    assert result.ranking.item() > 0.0


def test_value_loss_backprops_to_head_and_input_state() -> None:
    head = ValueHead(emb_dim=16, num_actions=8)
    state = torch.randn(3, 16, requires_grad=True)
    result = action_value_loss(
        head(state),
        torch.tensor([0, 3, 5]),
        torch.tensor([1.0, 0.0, -0.5]),
        ranking_margin=0.1,
        ranking_weight=1.0,
    )
    result.loss.backward()
    assert head.net[0].weight.grad is not None
    assert state.grad is not None


def test_value_mse_uses_only_the_executed_actions() -> None:
    action_values = torch.tensor([[1.0, 10.0], [20.0, 3.0]])
    result = action_value_loss(
        action_values,
        torch.tensor([0, 1]),
        torch.tensor([2.0, 5.0]),
        ranking_margin=0.1,
        ranking_weight=0.0,
    )

    torch.testing.assert_close(result.selected_action_values, torch.tensor([1.0, 3.0]))
    torch.testing.assert_close(result.monte_carlo_mse, torch.tensor(2.5))
    torch.testing.assert_close(result.loss, result.monte_carlo_mse)
