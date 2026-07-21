"""Nimloth 的完整模型组合。"""

from __future__ import annotations

import torch
from torch import nn

from nimloth.wm.model import WorldModel


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


class NimlothModel(nn.Module):
    """组合 LLM backbone 与 latent world model。

    ``llm`` 负责多模态 token forward；``wm`` 负责 state projection、latent
    dynamics 和 action value。processor、cache、EMA 和 optimizer 属于训练或
    推理 runtime，不是模型结构的一部分。
    """

    def __init__(self, *, llm: nn.Module, wm: WorldModel) -> None:
        super().__init__()
        self.llm = llm
        self.wm = wm

    def forward(self, *args, **kwargs):
        """保持与底层 LLM 相同的 forward 入口。"""

        return self.llm(*args, **kwargs)

    def forward_wm(
        self,
        qwen_hidden: torch.Tensor,
        action_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """从 LLM latent hidden 执行完整 world-model forward。"""

        return self.wm(qwen_hidden, action_indices)

    def unwrapped(self) -> "NimlothModel":
        """返回解除 DDP/FSDP 子模块包装后的同结构模型视图。"""

        return NimlothModel(
            llm=_unwrap(self.llm),
            wm=WorldModel(
                state_proj=_unwrap(self.wm.state_proj),
                wm_predictor=_unwrap(self.wm.wm_predictor),
                value_head=_unwrap(self.wm.value_head),
            ),
        )


__all__ = ["NimlothModel"]
