"""Shared PPO clipped-surrogate objectives for discrete policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOPolicyLoss:
    loss: torch.Tensor
    entropy: torch.Tensor
    probability_ratio: torch.Tensor
    clip_fraction: torch.Tensor
    selected_log_probs: torch.Tensor


def ppo_clipped_policy_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    entropies: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> PPOPolicyLoss:
    """Apply the PPO clipped surrogate to aligned on-policy samples."""

    shapes = {
        tuple(new_log_probs.shape),
        tuple(old_log_probs.shape),
        tuple(entropies.shape),
        tuple(advantages.shape),
    }
    if len(shapes) != 1 or new_log_probs.ndim != 1:
        raise ValueError("PPO log-probs, entropy and advantages must align")
    if not 0.0 < clip_ratio < 1.0:
        raise ValueError("PPO clip_ratio must be in (0, 1)")

    fixed_old_log_probs = old_log_probs.to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    ).detach()
    fixed_advantages = advantages.to(
        device=new_log_probs.device,
        dtype=new_log_probs.dtype,
    ).detach()
    probability_ratio = torch.exp(new_log_probs - fixed_old_log_probs)
    clipped_ratio = torch.clamp(
        probability_ratio,
        1.0 - clip_ratio,
        1.0 + clip_ratio,
    )
    loss = -torch.min(
        probability_ratio * fixed_advantages,
        clipped_ratio * fixed_advantages,
    ).mean()
    with torch.no_grad():
        clip_fraction = (
            (probability_ratio - 1.0).abs().gt(clip_ratio).float().mean()
        )
    return PPOPolicyLoss(
        loss=loss,
        entropy=entropies.mean(),
        probability_ratio=probability_ratio,
        clip_fraction=clip_fraction,
        selected_log_probs=new_log_probs,
    )


def ppo_action_policy_loss(
    *,
    action_logits: torch.Tensor,
    executed_actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float,
    clip_ratio: float,
) -> PPOPolicyLoss:
    """Apply PPO to one categorical PlannerPolicyHead distribution per state."""

    if temperature <= 0.0:
        raise ValueError("PlannerPolicyHead temperature must be positive")
    if action_logits.ndim != 2:
        raise ValueError("PlannerPolicyHead logits must have shape (B,A)")
    if executed_actions.shape != (action_logits.shape[0],):
        raise ValueError("executed actions must align with PlannerPolicyHead logits")
    log_probs = torch.log_softmax(action_logits / temperature, dim=-1)
    selected_log_probs = log_probs.gather(
        -1,
        executed_actions.unsqueeze(-1),
    ).squeeze(-1)
    probabilities = log_probs.exp()
    entropies = -(probabilities * log_probs).sum(dim=-1)
    return ppo_clipped_policy_loss(
        new_log_probs=selected_log_probs,
        old_log_probs=old_log_probs,
        entropies=entropies,
        advantages=advantages,
        clip_ratio=clip_ratio,
    )


__all__ = [
    "PPOPolicyLoss",
    "ppo_action_policy_loss",
    "ppo_clipped_policy_loss",
]
