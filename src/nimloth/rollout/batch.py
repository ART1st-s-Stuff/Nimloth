"""从 rollout transition 构造训练 batch 的公共契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from nimloth.backbone.base import BackboneBatch


@dataclass(frozen=True)
class TransitionBatch:
    """一批 transition 的模型输入、动作监督与下一状态对齐信息。

    ``next`` 已由具体 backend 去重。``next_indices`` 把每个 transition 映射
    回去重后的下一状态；terminal 行使用 0，但会被 ``non_terminal_mask`` 排除。
    builder 必须保证全 terminal batch 仍含一个 dummy next 行，使各 DDP rank
    执行相同的模型调用结构。
    """

    current: BackboneBatch
    next: BackboneBatch
    action_indices: torch.Tensor
    value_targets: torch.Tensor
    next_indices: torch.Tensor
    non_terminal_mask: torch.Tensor
    trajectory_steps: tuple[tuple[str, int], ...]
    outcome_targets: torch.Tensor | None = None
    outcome_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if (self.outcome_targets is None) != (self.outcome_mask is None):
            raise ValueError("outcome targets and mask must be supplied together")
        if self.outcome_targets is not None:
            expected = (len(self.trajectory_steps),)
            if self.outcome_targets.shape != expected or self.outcome_mask.shape != expected:
                raise ValueError("outcome targets/mask must have one entry per transition")
            if self.outcome_mask.dtype != torch.bool:
                raise ValueError("outcome mask must be boolean")
            if not torch.all((self.outcome_targets == 0) | (self.outcome_targets == 1)):
                raise ValueError("outcome targets must be finite binary values")


class TransitionBatchBuilder(Protocol):
    """阶段 assembler 把 rollout transition 转为模型无关 batch 的协议。"""

    processor: Any

    def collate_transition_samples(self, batch: list[Any]) -> Any: ...

    def prepare(self, raw_batch: Any) -> TransitionBatch: ...


__all__ = ["TransitionBatch", "TransitionBatchBuilder"]
