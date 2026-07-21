"""SFT2 与 RL 共用的世界模型张量目标函数。

本模块只接收已经构造好的 state tensor，不负责 Qwen forward、cache、EMA、
DDP 对齐或 optimizer 更新。调用方必须在进入这里之前明确 stop-gradient 策略。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.value_head import ValueHead


@dataclass(frozen=True)
class DynamicsLoss:
    """一步 latent dynamics 目标及其预测值。"""

    loss: torch.Tensor
    predicted_next_state: torch.Tensor


@dataclass(frozen=True)
class ActionValueLoss:
    """动作价值回归和排序目标的完整分解。"""

    loss: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    all_values: torch.Tensor
    chosen_values: torch.Tensor


def compute_dynamics_loss(
    *,
    current_state: torch.Tensor,
    target_next_state: torch.Tensor,
    action_indices: torch.Tensor,
    predictor: LatentWMPredictor,
) -> DynamicsLoss:
    """计算 ``predictor(s_t, a_t)`` 与目标 ``s_{t+1}`` 的均方误差。"""

    predicted_next_state = predictor(current_state, action_indices)
    loss = F.mse_loss(predicted_next_state, target_next_state)
    return DynamicsLoss(loss=loss, predicted_next_state=predicted_next_state)


def compute_action_value_loss(
    *,
    state: torch.Tensor,
    action_indices: torch.Tensor,
    return_targets: torch.Tensor,
    value_head: ValueHead,
    rank_margin: float = 0.1,
    rank_weight: float = 1.0,
) -> ActionValueLoss:
    """计算已选动作回归目标和相对未选动作的 margin ranking 目标。"""

    all_values = value_head(state).float()
    chosen_values = all_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    targets = return_targets.to(device=all_values.device, dtype=all_values.dtype)
    regression = F.mse_loss(chosen_values, targets)

    chosen_mask = F.one_hot(
        action_indices,
        num_classes=all_values.shape[1],
    ).bool()
    max_other = all_values.masked_fill(chosen_mask, float("-inf")).max(dim=1).values
    ranking = F.relu(rank_margin + max_other - chosen_values).mean()
    loss = regression + rank_weight * ranking
    return ActionValueLoss(
        loss=loss,
        regression=regression,
        ranking=ranking,
        all_values=all_values,
        chosen_values=chosen_values,
    )


__all__ = [
    "ActionValueLoss",
    "DynamicsLoss",
    "compute_action_value_loss",
    "compute_dynamics_loss",
]
