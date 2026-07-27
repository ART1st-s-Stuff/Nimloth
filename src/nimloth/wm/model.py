"""Nimloth 世界模型的完整 PyTorch 模块。"""

from __future__ import annotations

import torch
from torch import nn


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


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

    def project_target_state(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """用同一个 StateProjector 映射 target Backbone hidden。"""

        return self.project_state(qwen_hidden)

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

    def sigreg_state(self, state: torch.Tensor) -> torch.Tensor:
        """返回单个时间位置用于 SIGReg 的 ``(B,D)`` 表示。"""

        if state.ndim != 2:
            raise ValueError(
                f"standard latent SIGReg state must have shape (B,D), got {tuple(state.shape)}"
            )
        return state

    def sigreg_state_sequence(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """返回 SIGReg 使用的统计单位；标准 latent WM 保留完整 state。"""

        return state_sequence

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

    @property
    def trainable_modules(self) -> tuple[nn.Module, ...]:
        """SFT2 应统一切换 train/eval 的 WM 子模块。"""

        return (self.state_proj, self.wm_predictor, self.value_head)

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        """需要参与 DDP accumulation/no_sync 的实际包装模块。"""

        return self.trainable_modules

    def unwrapped(self) -> "WorldModel":
        """返回去除子模块 DDP wrapper 的同构模型视图。"""

        return WorldModel(
            state_proj=_unwrap(self.state_proj),
            wm_predictor=_unwrap(self.wm_predictor),
            value_head=_unwrap(self.value_head),
        )

__all__ = ["WorldModel"]
