"""SFT2 的模型执行视图与优化运行期。"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass

import torch

from nimloth.agent import Agent, AgentTarget
from nimloth.util.optim import (
    OptimizationRuntime,
    qwen_lr_schedule,
    set_optimizer_group_lr,
)


@dataclass(frozen=True)
class SFT2ModelRuntime:
    """把在线 Agent 与 target-state 路径作为一个不可分割的执行契约。"""

    agent: Agent
    target: AgentTarget

    def __post_init__(self) -> None:
        if self.target.agent is not self.agent:
            raise ValueError("SFT2 target must reference the same Agent runtime")

    def unwrapped(self) -> "SFT2ModelRuntime":
        """为不等长分布式验证创建不触发 wrapper collective 的模型视图。"""

        agent = self.agent.unwrapped()
        return SFT2ModelRuntime(
            agent=agent,
            target=self.target.with_agent(agent),
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
