"""Planner RL 的 PPO-style action-value critic objective。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOActionValueLoss:
    """Planner critic 对执行动作的 clipped value loss 分解。"""

    loss: torch.Tensor
    unclipped_mse: torch.Tensor
    clipped_mse: torch.Tensor
    clip_fraction: torch.Tensor
    selected_action_values: torch.Tensor
    clipped_action_values: torch.Tensor


def ppo_action_value_loss(
    action_values: torch.Tensor,
    executed_actions: torch.Tensor,
    return_targets: torch.Tensor,
    old_action_values: torch.Tensor,
    *,
    clip_range: float,
) -> PPOActionValueLoss:
    """对执行动作应用 frozen-old-value clipped critic regression。

    ``old_action_values`` 必须来自产生当前 rollout 的 frozen ValueHead。当前值先限制在
    ``old +/- clip_range``，再对 unclipped/clipped 两个平方误差逐样本取较大者。
    第二个及后续 PPO epoch 因此不能通过一次过大的 value 更新规避 return 误差。
    未执行动作不进入该 objective。
    """

    if clip_range <= 0.0:
        raise ValueError("PPO action-value clip_range must be positive")
    selected_action_values = action_values.gather(
        -1,
        executed_actions.unsqueeze(-1),
    ).squeeze(-1)
    targets = return_targets.to(
        device=selected_action_values.device,
        dtype=selected_action_values.dtype,
    )
    old_values = old_action_values.to(
        device=selected_action_values.device,
        dtype=selected_action_values.dtype,
    ).detach()
    if old_values.shape != selected_action_values.shape:
        raise ValueError(
            "old action values must align with executed action values: "
            f"old={tuple(old_values.shape)}, "
            f"current={tuple(selected_action_values.shape)}"
        )
    if targets.shape != selected_action_values.shape:
        raise ValueError(
            "return targets must align with executed action values: "
            f"targets={tuple(targets.shape)}, "
            f"current={tuple(selected_action_values.shape)}"
        )

    value_delta = selected_action_values - old_values
    clipped_action_values = old_values + value_delta.clamp(
        min=-clip_range,
        max=clip_range,
    )
    unclipped_squared_error = (selected_action_values - targets).square()
    clipped_squared_error = (clipped_action_values - targets).square()

    return PPOActionValueLoss(
        loss=torch.maximum(unclipped_squared_error, clipped_squared_error).mean(),
        unclipped_mse=unclipped_squared_error.mean(),
        clipped_mse=clipped_squared_error.mean(),
        clip_fraction=(value_delta.abs() > clip_range).float().mean(),
        selected_action_values=selected_action_values,
        clipped_action_values=clipped_action_values,
    )


__all__ = ["PPOActionValueLoss", "ppo_action_value_loss"]
