"""把在线或 cache 输入规范化为类型明确的 SFT2 batch。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nimloth.backbone.qwen25vl.batch import (
    build_qwen_batch,
    collate_qwen_encodings,
)
from nimloth.backbone.qwen25vl.transition import (
    CachedQwenNextBatch,
    QwenTransitionMessages,
    collate_next_qwen_encodings,
)


@dataclass(frozen=True)
class SFT2Transition:
    """SFT2 目标函数实际消费的一条 transition。"""

    identifier: str
    record_id: str
    step_index: int
    action_index: int
    value_target: float
    success: bool
    qwen: QwenTransitionMessages

    @property
    def trajectory_step(self) -> tuple[str, int]:
        """返回 SIGReg 分组键，并支持旧 cache 的 ``record:step`` id。"""

        record_id = self.record_id
        if not record_id and ":" in self.identifier:
            record_id = self.identifier.split(":", 1)[0]
        return record_id, self.step_index


@dataclass(frozen=True)
class SFT2Batch:
    """当前 Qwen 输入、可选下一状态 cache 和监督目标。"""

    transitions: tuple[SFT2Transition, ...]
    current_encoding: dict[str, torch.Tensor]
    cached_next: CachedQwenNextBatch | None


def _transition_from_mapping(item: dict[str, Any]) -> SFT2Transition:
    current_messages = item.get("messages")
    if not isinstance(current_messages, list):
        raise ValueError("SFT2 transition is missing current Qwen messages")
    next_messages = item.get("next_messages")
    if next_messages is not None and not isinstance(next_messages, list):
        raise ValueError("SFT2 next_messages must be a list or None")
    return SFT2Transition(
        identifier=str(item.get("id", "")),
        record_id=str(item.get("record_id", "")),
        step_index=int(item.get("step_index", 0)),
        action_index=int(item["action_index"]),
        value_target=float(item["action_value_target"]),
        success=bool(item.get("success", False)),
        qwen=QwenTransitionMessages(
            current=current_messages,
            next=next_messages,
        ),
    )


def collate_cached_transition_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    """合并 legacy cache；最终类型化在 ``prepare_sft2_batch`` 完成。"""

    items: list[dict[str, Any]] = []
    current_rows: list[dict[str, torch.Tensor]] = []
    next_rows: list[dict[str, torch.Tensor] | None] = []
    for entry in batch:
        items.append(
            {
                "id": entry["id"],
                "record_id": entry.get("record_id", ""),
                "step_index": entry.get("step_index", 0),
                "action_index": entry["action_index"],
                "action_value_target": entry["action_value_target"],
                "success": entry["success"],
                "messages": entry.get("messages"),
                "next_messages": entry.get("next_messages"),
            }
        )
        current_rows.append(entry["current_enc"])
        next_rows.append(entry.get("next_enc"))

    qwen_transitions = tuple(
        QwenTransitionMessages(
            current=item["messages"],
            next=item.get("next_messages"),
        )
        for item in items
    )
    return {
        "items": items,
        "current_enc": collate_qwen_encodings(current_rows, pad_token_id),
        "next_enc_bundle": collate_next_qwen_encodings(
            qwen_transitions,
            next_rows,
            pad_token_id=pad_token_id,
        ),
    }


def prepare_sft2_batch(
    batch: Any,
    processor: Any,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> SFT2Batch:
    """把 DataLoader 的在线/cached 输出转换为算法层唯一 batch 类型。"""

    if isinstance(batch, SFT2Batch):
        return batch
    if isinstance(batch, dict) and "current_enc" in batch:
        items = list(batch["items"])
        current_encoding = dict(batch["current_enc"])
        cached_next = batch.get("next_enc_bundle")
    else:
        items = list(batch)
        current_encoding = build_qwen_batch(
            items,
            processor,
            max_length,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
        cached_next = None

    if cached_next is not None and not isinstance(cached_next, CachedQwenNextBatch):
        # 旧 worker 进程返回的 dict 只在滚动升级时出现；立即规范化，算法层不再兼容它。
        cached_next = CachedQwenNextBatch(
            keys=tuple(cached_next.get("keys", ())),
            encoding=dict(cached_next["enc"]),
        )
    return SFT2Batch(
        transitions=tuple(_transition_from_mapping(item) for item in items),
        current_encoding=current_encoding,
        cached_next=cached_next,
    )


__all__ = [
    "SFT2Batch",
    "SFT2Transition",
    "collate_cached_transition_batch",
    "prepare_sft2_batch",
]
