from __future__ import annotations

import torch

from nimloth.training.rl.value import ppo_action_value_loss


def test_ppo_value_loss_uses_frozen_old_value_and_executed_action_only() -> None:
    action_values = torch.tensor([[1.6, 99.0]], requires_grad=True)
    old_action_values = torch.tensor([1.0], requires_grad=True)

    result = ppo_action_value_loss(
        action_values,
        torch.tensor([0]),
        torch.tensor([2.0]),
        old_action_values,
        clip_range=0.2,
    )
    result.loss.backward()

    torch.testing.assert_close(result.selected_action_values, torch.tensor([1.6]))
    torch.testing.assert_close(result.clipped_action_values, torch.tensor([1.2]))
    torch.testing.assert_close(result.unclipped_mse, torch.tensor(0.16))
    torch.testing.assert_close(result.clipped_mse, torch.tensor(0.64))
    torch.testing.assert_close(result.loss, torch.tensor(0.64))
    torch.testing.assert_close(result.clip_fraction, torch.tensor(1.0))
    torch.testing.assert_close(action_values.grad, torch.tensor([[0.0, 0.0]]))
    assert old_action_values.grad is None


def test_ppo_value_loss_backprops_unclipped_executed_value() -> None:
    action_values = torch.tensor([[1.1, -5.0]], requires_grad=True)

    result = ppo_action_value_loss(
        action_values,
        torch.tensor([0]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        clip_range=0.2,
    )
    result.loss.backward()

    torch.testing.assert_close(result.loss, torch.tensor(0.81))
    torch.testing.assert_close(result.clip_fraction, torch.tensor(0.0))
    torch.testing.assert_close(action_values.grad, torch.tensor([[-1.8, 0.0]]))
