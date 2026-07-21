"""LeWM SIGReg 的单步 world-model 输入契约。"""

from __future__ import annotations

import torch
from torch import nn

from nimloth.wm._vendor_lewm import SIGReg
from nimloth.wm.predictor import ONE_STEP_WM_SEQUENCE_LENGTH


class OneStepSIGReg(nn.Module):
    """把一批 ``s_t -> s_{t+1}`` transition 组织为 ``(2, B, D)``。

    LeWM 的时间长度由 ``history_size + prediction_offset`` 决定。当前 Nimloth
    训练接口只支持一步上下文与一步预测，因此固定为 ``T=2``。这里统一持有
    vendor SIGReg。当前由 SFT2 使用；其他阶段只有在采用同一个单步契约时才能复用。
    """

    sequence_length = ONE_STEP_WM_SEQUENCE_LENGTH

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

    def forward(
        self,
        current_states: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """对 batch 轴上的 transition 分布计算 SIGReg；``B<2`` 时跳过。"""

        if current_states.ndim != 2:
            raise ValueError(
                "OneStepSIGReg expects current_states with shape (B, D), "
                f"got {tuple(current_states.shape)}"
            )
        if next_states.shape != current_states.shape:
            raise ValueError(
                "OneStepSIGReg requires matching current/next state shapes, "
                f"got {tuple(current_states.shape)} and {tuple(next_states.shape)}"
            )
        if current_states.shape[0] < 2:
            return None

        # T=2 来自一步上下文 + 一步预测；B 始终是 transition 样本轴。
        sequence = torch.stack((current_states, next_states), dim=0)
        if sequence.shape[0] != self.sequence_length:
            raise AssertionError("one-step SIGReg sequence length must be 2")
        return self.regularizer(sequence)


__all__ = ["OneStepSIGReg"]
