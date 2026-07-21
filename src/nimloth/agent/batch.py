"""SFT2 与 RL 可共享的 Agent transition batch。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from nimloth.backbone.base import BackboneBatch


@dataclass(frozen=True)
class AgentBatch:
    """一次 transition 学习所需的模型输入与监督目标。

    ``next`` 已由具体 batch builder 去重。``next_indices`` 把每个 transition
    映射回去重后的下一状态；terminal 行使用 0，但会被 ``non_terminal_mask``
    排除。batch builder 必须保证即使全是 terminal，``next`` 也含一个 dummy 行，
    从而让每个 DDP rank 执行相同的模型调用结构。
    """

    current: BackboneBatch
    next: BackboneBatch
    action_indices: torch.Tensor
    value_targets: torch.Tensor
    next_indices: torch.Tensor
    non_terminal_mask: torch.Tensor
    trajectory_steps: tuple[tuple[str, int], ...]


class AgentBatchBuilder(Protocol):
    """具体 processor 向训练数据层提供的无参数 batch 构造契约。"""

    processor: Any

    def collate_transition_samples(self, batch: list[Any]) -> Any: ...

    def collate_cached_transition_batch(self, batch: list[dict[str, Any]]) -> Any: ...

    def prepare(self, raw_batch: Any) -> AgentBatch: ...


__all__ = ["AgentBatch", "AgentBatchBuilder"]
