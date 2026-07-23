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
        backbone_rows_per_forward: int | None = None,
        offload_backbone_chunk_activations: bool = False,
    ) -> AgentStateOutput:
        """把真实 observation batch 编码为 WM state，不执行或模拟动作。"""

        if backbone_rows_per_forward is None:
            backbone_output = self.backbone(
                batch,
                include_lm_loss=include_lm_loss,
            )
        else:
            backbone_output = self.backbone.forward_chunked(
                batch,
                max_rows=backbone_rows_per_forward,
                include_lm_loss=include_lm_loss,
                offload_saved_tensors=offload_backbone_chunk_activations,
            )
        return AgentStateOutput(
            hidden=backbone_output.hidden,
            state=self.wm.project_state(backbone_output.hidden),
            lm_loss=backbone_output.lm_loss,
        )

    def forward_step_from_history(
        self,
        batch: BackboneBatch,
        action_indices: torch.Tensor,
        *,
        include_lm_loss: bool = False,
        backbone_rows_per_forward: int | None = None,
        offload_backbone_chunk_activations: bool = False,
    ) -> AgentOutput:
        """对每个当前 step 编码最长 ``H`` 的只读历史上下文。

        CE 与 Backbone 梯度只属于每个 context 的最后一行。旧 state 作为
        WM 的显式时序 context，但在进入 predictor 前 detach。ValueHead 只读取
        当前 ``s_t``；其更老历史来自构造 ``s_t`` 的累计 Agent prompt。
        """

        if action_indices.ndim != 2:
            raise ValueError(
                "sequence action_indices must have shape (B,H), "
                f"got {tuple(action_indices.shape)}"
            )
        batch_size, history_size = action_indices.shape
        current_rows = tuple(
            range(history_size - 1, batch_size * history_size, history_size)
        )
        if (
            torch.is_grad_enabled()
            and history_size > 1
            and backbone_rows_per_forward is None
        ):
            raise ValueError(
                "training a multi-step context requires chunked Backbone forward "
                "so only the current row owns CE and Backbone gradients"
            )
        if backbone_rows_per_forward is None:
            backbone_output = self.backbone(
                batch,
                include_lm_loss=include_lm_loss,
            )
        else:
            backbone_output = self.backbone.forward_chunked(
                batch,
                max_rows=backbone_rows_per_forward,
                include_lm_loss=include_lm_loss,
                offload_saved_tensors=offload_backbone_chunk_activations,
                gradient_rows=current_rows,
                lm_loss_rows=current_rows,
            )
        hidden = backbone_output.hidden
        expected_rows = batch_size * history_size
        if hidden.shape[0] != expected_rows:
            raise ValueError(
                "sequence Backbone rows must equal B*H, "
                f"got {hidden.shape[0]} rows for B={batch_size}, H={history_size}"
            )
        hidden_sequence = hidden.reshape(
            batch_size,
            history_size,
            *hidden.shape[1:],
        )
        state_sequence = self.wm.project_state_sequence(hidden_sequence)
        if history_size > 1:
            state_sequence = torch.cat(
                (state_sequence[:, :-1].detach(), state_sequence[:, -1:]),
                dim=1,
            )
        predicted_sequence = self.wm.predict_state_sequence(
            state_sequence,
            action_indices,
        )
        return AgentOutput(
            hidden=hidden_sequence,
            state=state_sequence,
            predicted_next_state=predicted_sequence[:, -1],
            action_values=self.wm.predict_action_values(state_sequence[:, -1]),
            lm_loss=(
                backbone_output.lm_loss if include_lm_loss else None
            ),
        )

    @property
    def trainable_modules(self) -> tuple[nn.Module, ...]:
        """返回需要统一切换 train/eval mode 的完整模型边界。"""

        return (
            self.backbone,
            self.wm.state_proj,
            self.wm.wm_predictor,
            self.wm.value_head,
        )

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        """返回可能提供 DDP/FSDP ``no_sync`` 的实际包装模块。"""

        return (
            self.backbone.model,
            self.wm.state_proj,
            self.wm.wm_predictor,
            self.wm.value_head,
        )

    def unwrapped(self) -> "Agent":
        """返回解除子模块 DDP/FSDP 包装后的同结构模型视图。"""

        return Agent(
            backbone=self.backbone.with_model(_unwrap(self.backbone.model)),
            wm=WorldModel(
                state_proj=_unwrap(self.wm.state_proj),
                wm_predictor=_unwrap(self.wm.wm_predictor),
                value_head=_unwrap(self.wm.value_head),
            ),
        )


__all__ = ["Agent", "AgentOutput", "AgentStateOutput"]
