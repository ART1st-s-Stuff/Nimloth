"""Agent 可训练 backbone 的公共接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from contextlib import AbstractContextManager
from typing import Any, Mapping, Protocol, Sequence

import torch
from torch import nn

@dataclass(frozen=True)
class BackboneBatch:
    """已经由具体 processor 构造好的模型输入。"""

    tensors: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class BackboneOutput:
    """所有 Agent 阶段都能消费的 backbone 输出。"""

    hidden: torch.Tensor
    lm_loss: torch.Tensor | None = None


class Backbone(nn.Module, ABC):
    """多模态语言模型在 Agent 中的可训练边界。

    processor、DataLoader cache 与 EMA 不属于这个模块。具体实现只负责模型
    forward、参数注册和模型 artifact 的读写。
    """

    @abstractmethod
    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        """把 processor 输出编码为 Agent 使用的 latent hidden。"""

    def forward_chunked(
        self,
        batch: BackboneBatch,
        *,
        max_rows: int,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        """按较小的输入行组执行等价 forward。

        具体 backbone 必须负责切分自己的多模态张量并保持 loss reduction 语义。
        """

        raise NotImplementedError(
            f"{type(self).__name__} does not support chunked backbone forward"
        )

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        """返回实际承载参数的底层语言模型。"""

    @abstractmethod
    def with_model(self, model: nn.Module) -> "Backbone":
        """返回配置相同但底层模型被替换的视图。"""

    @abstractmethod
    def save_pretrained(
        self,
        output_dir: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """保存 backbone 的模型 artifact。"""


class BackboneEMA(Protocol):
    """训练运行期使用的 backbone 参数 EMA 契约。"""

    shadow: dict[str, torch.Tensor]

    def update(self, model: nn.Module) -> None: ...

    def use_ema_weights(self, model: nn.Module) -> AbstractContextManager[None]: ...

    def save_checkpoint(self, path: Path) -> None: ...


class BackboneInputBuilder(Protocol):
    """把 Agent 消息和图片转换成具体 Backbone 的张量输入。"""

    processor: Any

    def build(
        self,
        messages: Sequence[Sequence[dict[str, Any]]],
        images: Sequence[Sequence[Any]],
        *,
        include_labels: bool,
    ) -> BackboneBatch: ...

    def collate_encoded(
        self,
        rows: Sequence[dict[str, torch.Tensor]],
        *,
        include_labels: bool,
    ) -> BackboneBatch: ...

    def cache_key(
        self,
        messages: Sequence[dict[str, Any]],
        images: Sequence[Any],
    ) -> str: ...


@dataclass(frozen=True)
class LoadedBackbone:
    """factory 加载模型和 processor 后返回的装配结果。"""

    backbone: Backbone
    processor: Any
    token_id_map: dict[str, int]
    added_special_token_count: int
    base_model_path: Path | str
    query_adapter: Any = None
    pair_parallel: bool = False
    resume_aux_dir: Path | None = None


__all__ = [
    "Backbone",
    "BackboneBatch",
    "BackboneEMA",
    "BackboneInputBuilder",
    "BackboneOutput",
    "LoadedBackbone",
]
