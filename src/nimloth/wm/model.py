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

__all__ = ["WorldModel"]
