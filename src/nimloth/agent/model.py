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

        backbone_output = self.backbone(
            batch,
            include_lm_loss=include_lm_loss,
        )
        wm_output = self.wm(backbone_output.hidden, action_indices)
        return AgentOutput(
            hidden=backbone_output.hidden,
            state=wm_output["state"],
            predicted_next_state=wm_output["predicted_next_state"],
            action_values=wm_output["action_values"],
            lm_loss=backbone_output.lm_loss,
        )

    @property
    def trainable_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.backbone,
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


__all__ = ["Agent", "AgentOutput"]
