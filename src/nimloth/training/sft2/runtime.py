"""SFT2 的模型执行视图与优化运行期。"""

from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

from nimloth.agent import Agent
from nimloth.backbone import BackboneBatch, BackboneEMA
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.util.optim import (
    OptimizationRuntime,
    qwen_lr_schedule,
    set_optimizer_group_lr,
)


@dataclass(frozen=True)
class SFT2ModelRuntime:
    """封装 SFT2 的在线 Agent、下一 observation 编码与 Backbone EMA。"""

    agent: Agent
    history_cache: OnlineHistoryStateCache
    backbone_ema: BackboneEMA | None = None

    def encode_next_state(
        self,
        batch: BackboneBatch,
    ) -> torch.Tensor:
        """以固定 Backbone 与 StateProjector 编码 WM 的下一状态监督值。"""

        with torch.no_grad(), self._backbone_context():
            hidden = self.agent.backbone(
                batch,
                include_lm_loss=False,
            ).hidden.detach()
            return self.agent.wm.project_state(hidden)

    def evaluation_context(self) -> AbstractContextManager[object]:
        """让验证阶段的完整 Agent forward 使用 EMA Backbone 权重。"""

        return self._backbone_context()

    def _backbone_context(self) -> AbstractContextManager[object]:
        """按当前 runtime 的 Agent 创建 Backbone EMA 权重上下文。"""

        if self.backbone_ema is None:
            return contextlib.nullcontext()
        return self.backbone_ema.use_ema_weights(self.agent.backbone.model)

    def unwrapped(self) -> "SFT2ModelRuntime":
        """为不等长分布式验证创建不触发 wrapper collective 的模型视图。"""

        agent = self.agent.unwrapped()
        return SFT2ModelRuntime(
            agent=agent,
            history_cache=self.history_cache,
            backbone_ema=self.backbone_ema,
        )


@dataclass
class SFT2OptimizationRuntime:
    """封装 SFT2 的公共梯度更新与 Qwen 学习率策略。"""

    optimization: OptimizationRuntime
    qwen_warmup_steps: int
    total_steps: int
    qwen_start_lr: float
    qwen_peak_lr: float

    def zero_grad(self) -> None:
        self.optimization.zero_grad()

    def accumulation_context(
        self,
        *,
        sync_gradients: bool,
    ) -> AbstractContextManager[object]:
        return self.optimization.accumulation_context(
            sync_gradients=sync_gradients,
        )

    def backward(self, loss: torch.Tensor, *, grad_accum: int) -> None:
        self.optimization.backward(loss, divisor=grad_accum)

    def step(self, *, global_step: int) -> float:
        qwen_lr = qwen_lr_schedule(
            global_step,
            warmup_steps=self.qwen_warmup_steps,
            total_steps=self.total_steps,
            start_lr=self.qwen_start_lr,
            peak_lr=self.qwen_peak_lr,
        )
        set_optimizer_group_lr(
            self.optimization.optimizer,
            "qwen",
            qwen_lr,
        )
        self.optimization.step()
        return qwen_lr


__all__ = ["SFT2ModelRuntime", "SFT2OptimizationRuntime"]
