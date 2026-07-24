"""Nimloth 的完整神经网络 Agent。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nimloth.backbone.base import Backbone, BackboneBatch
from nimloth.wm.model import WorldModel


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


@dataclass(frozen=True)
class AgentStateOutput:
    """Backbone 与 StateProjector 共同编码出的 Agent state。"""

    hidden: torch.Tensor
    state: torch.Tensor
    lm_loss: torch.Tensor | None


@dataclass(frozen=True)
class AgentOutput:
    """一次完整 Agent forward 的模型输出。"""

    hidden: torch.Tensor
    state: torch.Tensor
    predicted_next_state: torch.Tensor
    action_values: torch.Tensor
    lm_loss: torch.Tensor | None


class Agent(nn.Module):
    """组合可训练 Backbone 与 WorldModel 的唯一模型边界。

    ``backbone`` 负责 observation 到 latent hidden；``wm`` 负责状态投影、下一
    状态预测和动作价值。processor、cache、EMA、optimizer 与 rollout 状态均不
    属于本模块。
    """

    def __init__(self, *, backbone: Backbone, wm: WorldModel) -> None:
        super().__init__()
        self.backbone = backbone
        self.wm = wm

    def forward(
        self,
        batch: BackboneBatch,
        action_indices: torch.Tensor,
        *,
        include_lm_loss: bool = False,
    ) -> AgentOutput:
        """执行 backbone、StateProjector、WMPredictor 和 ValueHead。"""

        encoded = self.encode_state(
            batch,
            include_lm_loss=include_lm_loss,
        )
        return AgentOutput(
            hidden=encoded.hidden,
            state=encoded.state,
            predicted_next_state=self.wm.predict_next_state(
                encoded.state,
                action_indices,
            ),
            action_values=self.wm.predict_action_values(encoded.state),
            lm_loss=encoded.lm_loss,
        )

    def encode_state(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> AgentStateOutput:
        """把真实 observation batch 编码为 WM state，不执行或模拟动作。"""

        backbone_output = self.backbone(
            batch,
            include_lm_loss=include_lm_loss,
        )
        return AgentStateOutput(
            hidden=backbone_output.hidden,
            state=self.wm.project_state(backbone_output.hidden),
            lm_loss=backbone_output.lm_loss,
        )

    def forward_step_from_history(
        self,
        action_indices: torch.Tensor,
        cached_history_states: torch.Tensor,
        *,
        encoded_current: AgentStateOutput,
    ) -> AgentOutput:
        """只编码当前 step，并把 detached cache 作为最长 ``H`` 的 WM context。

        CE 与 Backbone 梯度只属于当前 ``s_t``。更老 state 必须来自它们先前
        作为 current step 时写入的在线 cache；本方法不会重新执行历史 Qwen。
        """

        if action_indices.ndim != 2:
            raise ValueError(
                "sequence action_indices must have shape (B,H), "
                f"got {tuple(action_indices.shape)}"
            )
        batch_size, history_size = action_indices.shape
        if cached_history_states.ndim != encoded_current.state.ndim + 1:
            raise ValueError(
                "cached history states must have shape (B,T-1,...state), "
                f"got {tuple(cached_history_states.shape)}"
            )
        expected_history = (batch_size, history_size - 1)
        if cached_history_states.shape[:2] != expected_history:
            raise ValueError(
                "cached history does not align with action context: "
                f"states={tuple(cached_history_states.shape[:2])}, "
                f"expected={expected_history}"
            )
        if encoded_current.state.ndim < 2 or encoded_current.state.shape[0] != batch_size:
            raise ValueError(
                "current Backbone output must have shape (B,...state), "
                f"got {tuple(encoded_current.state.shape)} for B={batch_size}"
            )
        if (
            tuple(cached_history_states.shape[2:])
            != tuple(encoded_current.state.shape[1:])
            and history_size > 1
        ):
            raise ValueError(
                "cached/current state shapes do not match: "
                f"history={tuple(cached_history_states.shape[2:])}, "
                f"current={tuple(encoded_current.state.shape[1:])}"
            )
        state_sequence = torch.cat(
            (cached_history_states.detach(), encoded_current.state.unsqueeze(1)),
            dim=1,
        )
        predicted_sequence = self.wm.predict_state_sequence(
            state_sequence,
            action_indices,
        )
        return AgentOutput(
            hidden=encoded_current.hidden,
            state=state_sequence,
            predicted_next_state=predicted_sequence[:, -1],
            action_values=self.wm.predict_action_values(state_sequence[:, -1]),
            lm_loss=encoded_current.lm_loss,
        )

    @property
    def trainable_modules(self) -> tuple[nn.Module, ...]:
        """返回需要统一切换 train/eval mode 的完整模型边界。"""

        return (
            self.backbone,
            *self.wm.trainable_modules,
        )

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        """返回可能提供 DDP/FSDP ``no_sync`` 的实际包装模块。"""

        return (
            self.backbone.model,
            *self.wm.synchronized_modules,
        )

    def unwrapped(self) -> "Agent":
        """返回解除子模块 DDP/FSDP 包装后的同结构模型视图。"""

        return Agent(
            backbone=self.backbone.with_model(_unwrap(self.backbone.model)),
            wm=self.wm.unwrapped(),
        )


__all__ = ["Agent", "AgentOutput", "AgentStateOutput"]
