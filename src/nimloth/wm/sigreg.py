"""LeWM SIGReg 的连续状态序列输入契约。"""

from __future__ import annotations

import torch
from torch import nn

from nimloth.wm._vendor_lewm import SIGReg
from nimloth.wm.predictor import ONE_STEP_WM_SEQUENCE_LENGTH


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


class OneStepSIGReg(SequenceSIGReg):
    """把一批 ``s_t -> s_{t+1}`` transition 组织为 ``(2, B, D)``。

    LeWM 的时间长度由 ``history_size + prediction_offset`` 决定。当前 Nimloth
    训练接口只支持一步上下文与一步预测，因此固定为 ``T=2``。这里统一持有
    vendor SIGReg。当前由 SFT2 使用；其他阶段只有在采用同一个单步契约时才能复用。
    """

    sequence_length = ONE_STEP_WM_SEQUENCE_LENGTH

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
        sequence = torch.stack((current_states, next_states), dim=1)
        if sequence.shape[1] != self.sequence_length:
            raise AssertionError("one-step SIGReg sequence length must be 2")
        return super().forward(sequence)


__all__ = ["OneStepSIGReg", "SequenceSIGReg"]
