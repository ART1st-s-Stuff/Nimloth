"""SFT2 transition 元数据到公共 Backbone batch 的装配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nimloth.backbone import BackboneBatch, BackboneInputBuilder
from nimloth.rollout import TransitionBatch
from nimloth.rollout.transitions import TransitionSample, transition_training_item


@dataclass(frozen=True)
class CachedNextBatch:
    """DataLoader worker 已按 prompt 去重的下一状态输入。"""

    keys: tuple[str, ...]
    batch: BackboneBatch


class SFT2BatchAssembler:
    """负责 SFT2 target 对齐；不包含任何 Qwen 模型或训练算法。"""

    def __init__(
        self,
        *,
        input_builder: BackboneInputBuilder,
        device: torch.device,
    ) -> None:
        self.input_builder = input_builder
        self.device = device

    @property
    def processor(self) -> Any:
        """预处理 cache 仍需访问 backend 的 processor。"""

        return self.input_builder.processor

    def collate_transition_samples(
        self,
        batch: list[TransitionSample],
    ) -> list[dict[str, Any]]:
        return [transition_training_item(sample) for sample in batch]

    def collate_cached_transition_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """在 DataLoader worker 内合并 legacy cache 张量。"""

        items = [self._metadata(entry) for entry in batch]
        current_rows = [entry["current_enc"] for entry in batch]
        next_rows = [entry.get("next_enc") for entry in batch]
        return {
            "items": items,
            "current": self.input_builder.collate_encoded(
                current_rows,
                include_labels=True,
            ),
            "next": self._collate_next(items, next_rows),
        }

    def prepare(self, raw_batch: Any) -> TransitionBatch:
        """构造 current/next 模型输入和对齐后的 transition target。"""

        if isinstance(raw_batch, TransitionBatch):
            return raw_batch
        if isinstance(raw_batch, dict) and "current" in raw_batch:
            items = [self._metadata(item) for item in raw_batch["items"]]
            current = raw_batch["current"]
            cached_next = raw_batch.get("next")
        elif isinstance(raw_batch, dict) and "current_enc_rows" in raw_batch:
            # compact cache 只在 worker 内恢复 mmap row，统一的输入 builder
            # 负责最后的 backend collate。
            items = [self._metadata(item) for item in raw_batch["items"]]
            current = self.input_builder.collate_encoded(
                raw_batch["current_enc_rows"],
                include_labels=True,
            )
            cached_next = self._collate_next(
                items,
                raw_batch["next_enc_rows"],
            )
        else:
            items = [self._metadata(item) for item in raw_batch]
            current = self.input_builder.build(
                [item["messages"] for item in items],
                [() for _ in items],
                include_labels=True,
            )
            cached_next = None

        unique_keys, key_to_row = self._next_prompt_index(items)
        if unique_keys:
            next_batch = self._next_batch(
                items,
                unique_keys,
                key_to_row,
                cached_next,
            )
        else:
            # 全 terminal batch 仍执行一次 target forward，保证多卡调用结构一致。
            next_batch = self.input_builder.build(
                [items[0]["messages"]],
                [()],
                include_labels=False,
            )

        non_terminal = torch.tensor(
            [item.get("next_messages") is not None for item in items],
            dtype=torch.bool,
            device=self.device,
        )
        next_indices = torch.tensor(
            [
                key_to_row[self._prompt_key(item["next_messages"])]
                if item.get("next_messages") is not None
                else 0
                for item in items
            ],
            dtype=torch.long,
            device=self.device,
        )
        return TransitionBatch(
            current=current,
            next=next_batch,
            action_indices=torch.tensor(
                [item["action_index"] for item in items],
                dtype=torch.long,
                device=self.device,
            ),
            value_targets=torch.tensor(
                [item["action_value_target"] for item in items],
                dtype=torch.float32,
                device=self.device,
            ),
            next_indices=next_indices,
            non_terminal_mask=non_terminal,
            trajectory_steps=tuple(self._trajectory_step(item) for item in items),
        )

    def _collate_next(
        self,
        items: Sequence[dict[str, Any]],
        rows: Sequence[dict[str, torch.Tensor] | None],
    ) -> CachedNextBatch | None:
        unique_rows: list[dict[str, torch.Tensor]] = []
        unique_keys: list[str] = []
        seen: set[str] = set()
        for item, row in zip(items, rows, strict=True):
            messages = item.get("next_messages")
            if messages is None or row is None:
                continue
            key = self._prompt_key(messages)
            if key in seen:
                continue
            seen.add(key)
            unique_keys.append(key)
            unique_rows.append(row)
        if not unique_rows:
            return None
        return CachedNextBatch(
            keys=tuple(unique_keys),
            batch=self.input_builder.collate_encoded(
                unique_rows,
                include_labels=False,
            ),
        )

    def _next_batch(
        self,
        items: Sequence[dict[str, Any]],
        unique_keys: Sequence[str],
        key_to_row: dict[str, int],
        cached: CachedNextBatch | dict[str, Any] | None,
    ) -> BackboneBatch:
        if cached is not None:
            if isinstance(cached, dict):
                keys = tuple(cached.get("keys", ()))
                batch = cached.get("batch")
                if not isinstance(batch, BackboneBatch):
                    encoding = cached.get("encoding", cached.get("enc"))
                    if not isinstance(encoding, dict):
                        raise ValueError("cached next-state batch is missing encoding")
                    batch = BackboneBatch(encoding)
                cached = CachedNextBatch(keys=keys, batch=batch)
            if cached.keys != tuple(unique_keys):
                raise ValueError(
                    "cached next-state order does not match transition prompts"
                )
            return cached.batch

        unique_messages: list[list[dict[str, Any]] | None] = [None] * len(unique_keys)
        for item in items:
            messages = item.get("next_messages")
            if messages is not None:
                unique_messages[key_to_row[self._prompt_key(messages)]] = messages
        return self.input_builder.build(
            [messages for messages in unique_messages if messages is not None],
            [() for _ in unique_keys],
            include_labels=False,
        )

    def _prompt_key(self, messages: Sequence[dict[str, Any]]) -> str:
        return self.input_builder.cache_key(messages, ())

    def _next_prompt_index(
        self,
        items: Sequence[dict[str, Any]],
    ) -> tuple[list[str], dict[str, int]]:
        unique_keys: list[str] = []
        key_to_row: dict[str, int] = {}
        for item in items:
            messages = item.get("next_messages")
            if messages is None:
                continue
            key = self._prompt_key(messages)
            if key not in key_to_row:
                key_to_row[key] = len(unique_keys)
                unique_keys.append(key)
        return unique_keys, key_to_row

    @staticmethod
    def _metadata(item: dict[str, Any]) -> dict[str, Any]:
        current_messages = item.get("messages")
        if not isinstance(current_messages, list):
            raise ValueError("transition is missing current Agent messages")
        next_messages = item.get("next_messages")
        if next_messages is not None and not isinstance(next_messages, list):
            raise ValueError("next_messages must be a list or None")
        return {
            "id": str(item.get("id", "")),
            "record_id": str(item.get("record_id", "")),
            "step_index": int(item.get("step_index", 0)),
            "action_index": int(item["action_index"]),
            "action_value_target": float(item["action_value_target"]),
            "success": bool(item.get("success", False)),
            "messages": current_messages,
            "next_messages": next_messages,
        }

    @staticmethod
    def _trajectory_step(item: dict[str, Any]) -> tuple[str, int]:
        record_id = item["record_id"]
        if not record_id and ":" in item["id"]:
            record_id = item["id"].split(":", 1)[0]
        return record_id, item["step_index"]


__all__ = ["CachedNextBatch", "SFT2BatchAssembler"]
