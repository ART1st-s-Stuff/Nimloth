"""SFT2 与 RL 共用的世界模型目标函数测试。"""

from __future__ import annotations

import torch

from nimloth.wm.objectives import compute_action_value_loss
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
    result = compute_action_value_loss(
        state=torch.randn(1, 4),
        action_indices=torch.tensor([0]),
        return_targets=torch.tensor([2.0]),
        value_head=_bias_only_head(torch.tensor([2.0, 0.5, 0.1])),
        rank_margin=0.1,
        rank_weight=1.0,
    )
    assert result.ranking.item() == 0.0
    assert result.loss.item() == result.regression.item()


def test_value_ranking_positive_when_unchosen_beats_chosen() -> None:
    result = compute_action_value_loss(
        state=torch.randn(1, 4),
        action_indices=torch.tensor([0]),
        return_targets=torch.tensor([1.0]),
        value_head=_bias_only_head(torch.tensor([0.5, 2.0, 0.1])),
    )
    assert result.ranking.item() > 0.0


def test_value_loss_backprops_to_head_and_input_state() -> None:
    head = ValueHead(emb_dim=16, num_actions=8)
    state = torch.randn(3, 16, requires_grad=True)
    result = compute_action_value_loss(
        state=state,
        action_indices=torch.tensor([0, 3, 5]),
        return_targets=torch.tensor([1.0, 0.0, -0.5]),
        value_head=head,
    )
    result.loss.backward()
    assert head.net[0].weight.grad is not None
    assert state.grad is not None
