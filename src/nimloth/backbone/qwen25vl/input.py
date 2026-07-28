"""Agent prompt 到 Qwen2.5-VL 张量输入的单一适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nimloth.agent import bind_image_placeholders
from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.batch import (
    build_qwen_batch,
    collate_qwen_encodings,
    message_cache_key,
)


@dataclass(frozen=True)
class Qwen25VLInputBuilder:
    """只负责 Qwen processor、label 和张量 collate，不解释训练目标。"""

    processor: Any
    max_length: int
    latent_token_count: int = 1
    mask_latent_query_labels: bool = True

    def build(
        self,
        messages: Sequence[Sequence[dict[str, Any]]],
        images: Sequence[Sequence[Any]],
        *,
        include_labels: bool,
    ) -> BackboneBatch:
        if len(messages) != len(images):
            raise ValueError(
                "backbone input messages/images must have equal batch size: "
                f"{len(messages)} != {len(images)}"
            )
        if not messages:
            raise ValueError("backbone input batch must not be empty")
        bound = [
            bind_image_placeholders(item_messages, item_images)
            for item_messages, item_images in zip(messages, images, strict=True)
        ]
        encoding = build_qwen_batch(
            [{"messages": item} for item in bound],
            self.processor,
            int(self.max_length),
            latent_token_count=int(self.latent_token_count),
            mask_latent_query_labels=bool(self.mask_latent_query_labels),
        )
        if not include_labels:
            encoding.pop("labels", None)
        return BackboneBatch(encoding)

    def collate_encoded(
        self,
        rows: Sequence[dict[str, torch.Tensor]],
        *,
        include_labels: bool,
    ) -> BackboneBatch:
        prepared_rows = list(rows)
        if include_labels:
            missing = [
                index
                for index, row in enumerate(prepared_rows)
                if "labels" not in row
            ]
            if missing:
                raise ValueError(
                    "supervised Qwen encoding rows must all contain labels; "
                    f"missing rows={missing}"
                )
        else:
            # Target-state forwards never consume CE labels.  Compact-cache
            # batches may legitimately mix ordinary next states (which reuse a
            # supervised current encoding) with the terminal CoT state (which
            # has no action labels), so normalize every row before collation.
            prepared_rows = [
                {key: value for key, value in row.items() if key != "labels"}
                for row in prepared_rows
            ]
        encoding = collate_qwen_encodings(
            prepared_rows,
            self.processor.tokenizer.pad_token_id,
        )
        return BackboneBatch(encoding)

    def cache_key(
        self,
        messages: Sequence[dict[str, Any]],
        images: Sequence[Any],
    ) -> str:
        return message_cache_key(
            bind_image_placeholders(messages, images)
        )


__all__ = ["Qwen25VLInputBuilder"]
