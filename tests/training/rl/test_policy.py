from __future__ import annotations

import torch

from nimloth.training.rl.policy import ppo_action_policy_loss


def test_action_policy_ppo_uses_executed_action_ratio_and_clipping() -> None:
    logits = torch.tensor([[2.0, -2.0]], requires_grad=True)
    executed = torch.tensor([0])
    current_log_prob = torch.log_softmax(logits.detach(), dim=-1)[0, 0]
    old_log_prob = current_log_prob - torch.log(torch.tensor(2.0))

    objective = ppo_action_policy_loss(
        action_logits=logits,
        executed_actions=executed,
        old_log_probs=old_log_prob.reshape(1),
        advantages=torch.tensor([3.0]),
        temperature=1.0,
        clip_ratio=0.2,
    )

    torch.testing.assert_close(objective.probability_ratio, torch.tensor([2.0]))
    torch.testing.assert_close(objective.loss, torch.tensor(-3.6))
    torch.testing.assert_close(objective.clip_fraction, torch.tensor(1.0))


def test_action_policy_ppo_detaches_old_statistics_but_trains_logits() -> None:
    logits = torch.zeros((1, 3), requires_grad=True)
    old_log_probs = torch.tensor([-1.0], requires_grad=True)
    advantages = torch.tensor([2.0], requires_grad=True)

    objective = ppo_action_policy_loss(
        action_logits=logits,
        executed_actions=torch.tensor([1]),
        old_log_probs=old_log_probs,
        advantages=advantages,
        temperature=1.0,
        clip_ratio=0.2,
    )
    objective.loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 3
    assert old_log_probs.grad is None
    assert advantages.grad is None
