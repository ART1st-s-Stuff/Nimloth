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


class TransitionBatchBuilder(Protocol):
    """阶段 assembler 把 rollout transition 转为模型无关 batch 的协议。"""

    processor: Any

    def collate_transition_samples(self, batch: list[Any]) -> Any: ...

    def collate_cached_transition_batch(self, batch: list[dict[str, Any]]) -> Any: ...

    def prepare(self, raw_batch: Any) -> TransitionBatch: ...


__all__ = ["TransitionBatch", "TransitionBatchBuilder"]
