"""RL 的 WM、value 与 PPO 目标函数。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


def normalized_monte_carlo_advantages(
    *,
    return_targets: torch.Tensor,
    predicted_values: torch.Tensor,
) -> torch.Tensor:
    """用当前 value baseline 归一化 Monte Carlo return。"""

    advantages = return_targets - predicted_values.detach()
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


class RLObjective(nn.Module):
    """集中计算 RL 数学目标，不持有 Agent 或 optimizer。"""

    def __init__(
        self,
        *,
        value_rank_margin: float,
        value_rank_weight: float,
        ppo_clip_ratio: float,
        entropy_weight: float,
    ) -> None:
        super().__init__()
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.ppo_clip_ratio = float(ppo_clip_ratio)
        self.entropy_weight = float(entropy_weight)

    def forward(
        self,
        *,
        predicted_next_state: torch.Tensor,
        target_next_state: torch.Tensor,
        action_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor | None,
        action_log_probs: torch.Tensor | None,
    ) -> RLStepOutput:
        wm_loss = F.mse_loss(predicted_next_state, target_next_state)
        value = self._value_loss(action_values, action_indices, return_targets)
        total = wm_loss + value["loss"]

        policy: dict[str, torch.Tensor] | None = None
        if new_log_probs is not None:
            if action_log_probs is None:
                raise ValueError("action_log_probs are required with new_log_probs")
            advantages = normalized_monte_carlo_advantages(
                return_targets=return_targets.to(
                    device=value["chosen_values"].device,
                    dtype=value["chosen_values"].dtype,
                ),
                predicted_values=value["chosen_values"],
            ).to(device=new_log_probs.device, dtype=new_log_probs.dtype)
            policy = self._policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs.to(
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                ),
                action_log_probs=action_log_probs,
                advantages=advantages,
            )
            total = total + policy["loss"] - self.entropy_weight * policy["entropy"]

        metrics = {
            "wm_mse": float(wm_loss.detach().item()),
            "value_loss": float(value["loss"].detach().item()),
            "total_loss": float(total.detach().item()),
            "actor_loss": float(policy["loss"].detach().item()) if policy else 0.0,
        }
        if policy is not None:
            metrics.update(
                {
                    "entropy": float(policy["entropy"].detach().item()),
                    "mean_advantage": float(policy["advantages"].mean().item()),
                    "clip_fraction": float(policy["clip_fraction"].item()),
                    "mean_ratio": float(policy["probability_ratio"].mean().item()),
                }
            )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": wm_loss,
                "value": value["loss"],
                "policy": policy["loss"] if policy else None,
            },
            metrics=metrics,
        )

    def _value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        chosen_values = all_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        targets = return_targets.to(device=all_values.device, dtype=all_values.dtype)
        regression = F.mse_loss(chosen_values, targets)
        chosen_mask = F.one_hot(
            action_indices,
            num_classes=all_values.shape[1],
        ).bool()
        max_other = all_values.masked_fill(chosen_mask, float("-inf")).max(dim=1).values
        ranking = F.relu(
            self.value_rank_margin + max_other - chosen_values
        ).mean()
        return {
            "loss": regression + self.value_rank_weight * ranking,
            "chosen_values": chosen_values,
        }

    def _policy_loss(
        self,
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        probability_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            probability_ratio,
            1.0 - self.ppo_clip_ratio,
            1.0 + self.ppo_clip_ratio,
        )
        loss = -torch.min(
            probability_ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        probabilities = action_log_probs.exp()
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * action_log_probs,
            torch.zeros_like(action_log_probs),
        )
        with torch.no_grad():
            clip_fraction = (
                (probability_ratio - 1.0)
                .abs()
                .gt(self.ppo_clip_ratio)
                .float()
                .mean()
            )
        return {
            "loss": loss,
            "entropy": -entropy_terms.sum(dim=-1).mean(),
            "advantages": advantages,
            "probability_ratio": probability_ratio,
            "clip_fraction": clip_fraction,
        }


__all__ = ["RLObjective", "RLStepOutput", "normalized_monte_carlo_advantages"]
