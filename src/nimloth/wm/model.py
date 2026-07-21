"""Nimloth 世界模型的完整 PyTorch 模块。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DynamicsLoss:
    """一步 latent dynamics 目标及预测结果。"""

    loss: torch.Tensor
    predicted_next_state: torch.Tensor


@dataclass(frozen=True)
class ActionValueLoss:
    """动作价值回归、排序目标及模型输出。"""

    loss: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    all_values: torch.Tensor
    chosen_values: torch.Tensor


class WorldModel(nn.Module):
    """组合 state projector、WM predictor 与 value head。

    这个模块只表示可训练的世界模型，不持有 processor、optimizer、EMA 或
    distributed runtime。SFT2 和 RL 可以在各自的 objective 中决定哪里需要
    ``detach``，但不再分别拼装三个子模块。
    """

    def __init__(
        self,
        *,
        state_proj: nn.Module,
        wm_predictor: nn.Module,
        value_head: nn.Module,
    ) -> None:
        super().__init__()
        self.state_proj = state_proj
        self.wm_predictor = wm_predictor
        self.value_head = value_head

    def forward(
        self,
        qwen_hidden: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """执行一次完整的 state projection、dynamics 和 value forward。"""

        state = self.project_state(qwen_hidden)
        return {
            "state": state,
            "predicted_next_state": self.predict_next_state(state, action_indices),
            "action_values": self.predict_action_values(state),
        }

    def project_state(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """把 Qwen hidden 投影到 world-model state 空间。"""

        return self.state_proj(qwen_hidden).float()

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """预测执行动作后的下一状态。"""

        return self.wm_predictor(current_state, action_indices)

    def predict_action_values(self, state: torch.Tensor) -> torch.Tensor:
        """预测每个离散动作的 value。"""

        return self.value_head(state).float()

    def compute_dynamics_loss(
        self,
        *,
        current_state: torch.Tensor,
        target_next_state: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> DynamicsLoss:
        """计算一步 latent dynamics MSE。"""

        predicted_next_state = self.predict_next_state(
            current_state,
            action_indices,
        )
        loss = F.mse_loss(predicted_next_state, target_next_state)
        return DynamicsLoss(
            loss=loss,
            predicted_next_state=predicted_next_state,
        )

    def compute_action_value_loss(
        self,
        *,
        state: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
        rank_margin: float = 0.1,
        rank_weight: float = 1.0,
    ) -> ActionValueLoss:
        """计算已选动作的回归目标和相对未选动作的排序目标。"""

        all_values = self.predict_action_values(state)
        chosen_values = all_values.gather(
            1,
            action_indices.unsqueeze(1),
        ).squeeze(1)
        targets = return_targets.to(
            device=all_values.device,
            dtype=all_values.dtype,
        )
        regression = F.mse_loss(chosen_values, targets)

        chosen_mask = F.one_hot(
            action_indices,
            num_classes=all_values.shape[1],
        ).bool()
        max_other = all_values.masked_fill(
            chosen_mask,
            float("-inf"),
        ).max(dim=1).values
        ranking = F.relu(rank_margin + max_other - chosen_values).mean()
        loss = regression + rank_weight * ranking
        return ActionValueLoss(
            loss=loss,
            regression=regression,
            ranking=ranking,
            all_values=all_values,
            chosen_values=chosen_values,
        )


__all__ = ["ActionValueLoss", "DynamicsLoss", "WorldModel"]
