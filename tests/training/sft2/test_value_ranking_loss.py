from __future__ import annotations

import torch

from nimloth.training.sft2.loss import compute_value_loss
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
    head = _bias_only_head(torch.tensor([2.0, 0.5, 0.1]))
    state_emb = torch.randn(1, 4)
    loss, metrics = compute_value_loss(
        state_emb=state_emb,
        action_indices=torch.tensor([0]),
        action_value_targets=torch.tensor([2.0]),
        value_head=head,
        rank_margin=0.1,
        lambda_rank=1.0,
    )
    assert metrics["value_rank"] == 0.0
    assert loss.item() == metrics["value_reg"]


def test_value_ranking_positive_when_unchosen_beats_chosen() -> None:
    head = _bias_only_head(torch.tensor([0.5, 2.0, 0.1]))
    state_emb = torch.randn(1, 4)
    _, metrics = compute_value_loss(
        state_emb=state_emb,
        action_indices=torch.tensor([0]),
        action_value_targets=torch.tensor([1.0]),
        value_head=head,
        rank_margin=0.1,
        lambda_rank=1.0,
    )
    assert metrics["value_rank"] > 0.0


def test_value_loss_backprops_to_head() -> None:
    head = ValueHead(emb_dim=16, num_actions=8)
    state_emb = torch.randn(3, 16, requires_grad=True)
    actions = torch.tensor([0, 3, 5])
    targets = torch.tensor([1.0, 0.0, -0.5])

    loss, _ = compute_value_loss(
        state_emb=state_emb,
        action_indices=actions,
        action_value_targets=targets,
        value_head=head,
    )
    loss.backward()

    assert head.net[0].weight.grad is not None
    assert state_emb.grad is not None
