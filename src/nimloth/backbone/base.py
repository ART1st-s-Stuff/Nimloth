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

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        """返回为该 Backbone forward 提供梯度同步的包装模块。"""

        return (self.model,)


class DistributedBackbone(Backbone):
    """保留 Backbone 语义的分布式 forward 包装。

    DDP 必须直接包住产生下游 loss 所消费 tensor 的 forward 边界。
    如果只包住底层语言模型、而 Backbone 通过 forward hook 取出隐状态，
    该隐状态不是 DDP forward 返回值，reducer 无法可靠地跟踪这条反向图。
    """

    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        inner = getattr(wrapped, "module", None)
        if not isinstance(inner, Backbone):
            raise TypeError("DistributedBackbone requires a wrapped Backbone module")
        self.wrapped = wrapped

    @property
    def inner(self) -> Backbone:
        inner = getattr(self.wrapped, "module", None)
        if not isinstance(inner, Backbone):
            raise TypeError("distributed wrapper no longer contains a Backbone")
        return inner

    @property
    def model(self) -> nn.Module:
        return self.inner.model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        return self.wrapped(batch, include_lm_loss=include_lm_loss)

    def with_model(self, model: nn.Module) -> Backbone:
        """返回解除分布式包装后、替换底层模型的视图。"""

        return self.inner.with_model(model)

    def save_pretrained(
        self,
        output_dir: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.inner.save_pretrained(
            output_dir,
            metadata=metadata,
            state_dict=state_dict,
        )

    @property
    def synchronized_modules(self) -> tuple[nn.Module, ...]:
        return (self.wrapped,)


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
    "DistributedBackbone",
    "LoadedBackbone",
]
