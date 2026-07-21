"""训练阶段的 Agent target-state 运行期。"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch

from nimloth.backbone.base import BackboneBatch

if TYPE_CHECKING:
    from nimloth.agent.model import Agent


class AgentTarget:
    """使用冻结 backbone 输出构造仍可训练的 WM target state。

    SFT2 的既有梯度语义是：下一状态 backbone 不反向传播，但 StateProjector
    同时接收当前侧和 target 侧梯度。因此 ``no_grad`` 只包住 backbone forward。
    可选的 EMA context 由训练运行期注入，本类不理解任何具体 backbone。
    """

    def __init__(
        self,
        agent: "Agent",
        *,
        backbone_context: Callable[[], AbstractContextManager[object]] | None = None,
    ) -> None:
        self.agent = agent
        self._backbone_context = backbone_context or contextlib.nullcontext

    def __call__(self, batch: BackboneBatch) -> torch.Tensor:
        with torch.no_grad(), self._backbone_context():
            hidden = self.agent.backbone(batch, include_lm_loss=False).hidden.detach()
        return self.agent.wm.project_state(hidden)

    def ema_context(self) -> AbstractContextManager[object]:
        """验证时让当前状态 forward 也使用相同 target 权重。"""

        return self._backbone_context()

    def with_agent(self, agent: "Agent") -> "AgentTarget":
        """让解除分布式包装后的 Agent 复用同一 target 权重上下文。"""

        return AgentTarget(agent, backbone_context=self._backbone_context)


__all__ = ["AgentTarget"]
