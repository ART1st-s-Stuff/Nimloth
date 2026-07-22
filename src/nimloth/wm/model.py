"""Nimloth 世界模型的完整 PyTorch 模块。"""

from __future__ import annotations

import torch
from torch import nn


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
        """执行一次完整的 state projection、WM prediction 和 value forward。"""

        state = self.project_state(qwen_hidden)
        return {
            "state": state,
            "predicted_next_state": self.predict_next_state(state, action_indices),
            "action_values": self.predict_action_values(state),
        }

    def project_state(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """把 Qwen hidden 投影到 world-model state 空间。"""

        return self.state_proj(qwen_hidden).float()

    def project_state_sequence(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """逐时间位置投影 ``(B,T,...)`` Backbone hidden 序列。"""

        if qwen_hidden.ndim < 3:
            raise ValueError(
                "state sequence hidden must have shape (B, T, ...), "
                f"got {tuple(qwen_hidden.shape)}"
            )
        batch_size, time_steps = qwen_hidden.shape[:2]
        projected = self.project_state(qwen_hidden.flatten(0, 1))
        return projected.reshape(batch_size, time_steps, *projected.shape[1:])

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> torch.Tensor:
        """预测执行动作后的下一状态。"""

        return self.wm_predictor(current_state, action_indices)

    def predict_state_sequence(
        self,
        state_context: torch.Tensor,
        action_context: torch.Tensor,
    ) -> torch.Tensor:
        """对 ``(B,T,D)`` 连续上下文的每个因果位置预测下一状态。"""

        return self.wm_predictor(state_context, action_context)

    def predict_action_values(self, state: torch.Tensor) -> torch.Tensor:
        """预测每个离散动作的 value。"""

        return self.value_head(state).float()

    def simulate_action_sequences(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """从真实历史出发模拟候选动作，不接触 environment。"""

        rollout = getattr(self.wm_predictor, "rollout_from_history", None)
        if rollout is None:
            raise TypeError(
                "wm_predictor must implement rollout_from_history() for planning"
            )
        return rollout(state_history, previous_actions, action_sequences)

__all__ = ["WorldModel"]
