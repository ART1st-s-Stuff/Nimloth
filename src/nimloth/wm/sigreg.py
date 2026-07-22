"""LeWM SIGReg 的连续状态序列输入契约。"""

from __future__ import annotations

import torch
from torch import nn

from nimloth.wm._vendor_lewm import SIGReg


class SequenceSIGReg(nn.Module):
    """对 ``(B,T,D)`` 状态序列执行 LeWM SIGReg。"""

    def __init__(
        self,
        *,
        knots: int = 17,
        num_proj: int = 1024,
        regularizer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.regularizer = (
            regularizer
            if regularizer is not None
            else SIGReg(knots=knots, num_proj=num_proj)
        )

    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor | None:
        """保留 batch/time 语义；batch 小于 2 时无法估计分布并明确跳过。"""

        if state_sequence.ndim != 3:
            raise ValueError(
                "SequenceSIGReg expects states with shape (B, T, D), "
                f"got {tuple(state_sequence.shape)}"
            )
        if state_sequence.shape[1] < 2:
            raise ValueError("SequenceSIGReg requires at least two time steps")
        if state_sequence.shape[0] < 2:
            return None
        return self.regularizer(state_sequence.transpose(0, 1))

__all__ = ["SequenceSIGReg"]
