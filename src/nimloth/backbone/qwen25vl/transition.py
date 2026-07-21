"""World-model transition 到 Qwen2.5-VL 输入与 latent state 的适配。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image

from nimloth.agent import bind_image_placeholders
from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.batch import (
    build_qwen_batch,
    collate_qwen_encodings,
    message_cache_key,
)
from nimloth.rollout.batch import TransitionBatch
from nimloth.rollout.transitions import TransitionSample

# Compatibility name for existing callers. Agent owns the message/image contract.
messages_with_image_paths = bind_image_placeholders


@dataclass(frozen=True)
class QwenTransitionMessages:
    """一个 transition 对应的当前和下一状态 Qwen prompt。"""

    current: list[dict[str, Any]]
    next: list[dict[str, Any]] | None


@dataclass(frozen=True)
class CachedQwenNextBatch:
    """DataLoader worker 已去重并合并的下一状态 Qwen 输入。"""

    keys: tuple[str, ...]
    encoding: dict[str, torch.Tensor]


def collate_next_qwen_encodings(
    transitions: Sequence[QwenTransitionMessages],
    rows: Sequence[dict[str, torch.Tensor] | None],
    *,
    pad_token_id: int,
) -> CachedQwenNextBatch | None:
    """按 prompt key 去重下一状态 cache，并在 worker 内提前组成 batch。"""

    unique_rows: list[dict[str, torch.Tensor]] = []
    unique_keys: list[str] = []
    seen: set[str] = set()
    for transition, row in zip(transitions, rows, strict=True):
        if transition.next is None or row is None:
            continue
        key = message_cache_key(transition.next)
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
        unique_rows.append(row)
    if not unique_rows:
        return None
    return CachedQwenNextBatch(
        keys=tuple(unique_keys),
        encoding=collate_qwen_encodings(unique_rows, pad_token_id),
    )


class Qwen25VLBatchBuilder:
    """把 DataLoader 输出转换为模型无关的 ``TransitionBatch``。"""

    def __init__(
        self,
        *,
        processor: Any,
        device: torch.device,
        max_length: int,
        latent_token_count: int = 1,
        mask_latent_query_labels: bool = True,
    ) -> None:
        self.processor = processor
        self.device = device
        self.max_length = int(max_length)
        self.latent_token_count = int(latent_token_count)
        self.mask_latent_query_labels = bool(mask_latent_query_labels)

    def collate_transition_samples(self, batch: list[TransitionSample]) -> list[dict[str, Any]]:
        return transition_collate_for_qwen(batch)

    def collate_cached_transition_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """合并 legacy cache，保留 prompt metadata 供 terminal dummy 使用。"""

        items: list[dict[str, Any]] = []
        current_rows: list[dict[str, torch.Tensor]] = []
        next_rows: list[dict[str, torch.Tensor] | None] = []
        for entry in batch:
            items.append(self._metadata(entry))
            current_rows.append(entry["current_enc"])
            next_rows.append(entry.get("next_enc"))
        transitions = tuple(
            QwenTransitionMessages(
                current=item["messages"],
                next=item.get("next_messages"),
            )
            for item in items
        )
        return {
            "items": items,
            "current_enc": collate_qwen_encodings(
                current_rows,
                self.processor.tokenizer.pad_token_id,
            ),
            "next_enc_bundle": collate_next_qwen_encodings(
                transitions,
                next_rows,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            ),
        }

    def prepare(self, raw_batch: Any) -> TransitionBatch:
        """构造 current/next 模型输入和对齐后的 transition target。"""

        if isinstance(raw_batch, TransitionBatch):
            return raw_batch
        if isinstance(raw_batch, dict) and "current_enc" in raw_batch:
            items = [self._metadata(item) for item in raw_batch["items"]]
            current_encoding = dict(raw_batch["current_enc"])
            cached_next = raw_batch.get("next_enc_bundle")
        else:
            items = [self._metadata(item) for item in raw_batch]
            current_encoding = build_qwen_batch(
                items,
                self.processor,
                self.max_length,
                latent_token_count=self.latent_token_count,
                mask_latent_query_labels=self.mask_latent_query_labels,
            )
            cached_next = None

        unique_keys, key_to_row = self._next_prompt_index(items)
        if unique_keys:
            next_encoding = self._next_encoding(
                items,
                unique_keys,
                key_to_row,
                cached_next,
            )
        else:
            # 全 terminal batch 仍执行一次 target backbone forward，保证多卡调用结构一致。
            next_encoding = build_qwen_batch(
                [{"messages": items[0]["messages"]}],
                self.processor,
                self.max_length,
                latent_token_count=self.latent_token_count,
            )
        next_encoding.pop("labels", None)

        non_terminal = torch.tensor(
            [item.get("next_messages") is not None for item in items],
            dtype=torch.bool,
            device=self.device,
        )
        next_indices = torch.tensor(
            [
                key_to_row[message_cache_key(item["next_messages"])]
                if item.get("next_messages") is not None
                else 0
                for item in items
            ],
            dtype=torch.long,
            device=self.device,
        )
        return TransitionBatch(
            current=BackboneBatch(current_encoding),
            next=BackboneBatch(next_encoding),
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

    @staticmethod
    def _metadata(item: dict[str, Any]) -> dict[str, Any]:
        current_messages = item.get("messages")
        if not isinstance(current_messages, list):
            raise ValueError("transition is missing current Qwen messages")
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

    @staticmethod
    def _next_prompt_index(
        items: Sequence[dict[str, Any]],
    ) -> tuple[list[str], dict[str, int]]:
        unique_keys: list[str] = []
        key_to_row: dict[str, int] = {}
        for item in items:
            messages = item.get("next_messages")
            if messages is None:
                continue
            key = message_cache_key(messages)
            if key not in key_to_row:
                key_to_row[key] = len(unique_keys)
                unique_keys.append(key)
        return unique_keys, key_to_row

    def _next_encoding(
        self,
        items: Sequence[dict[str, Any]],
        unique_keys: Sequence[str],
        key_to_row: dict[str, int],
        cached: CachedQwenNextBatch | dict[str, Any] | None,
    ) -> dict[str, torch.Tensor]:
        if cached is not None:
            if isinstance(cached, dict):
                # 兼容滚动升级期间旧 worker 返回的 ``enc``，也接受字段名直译后的
                # ``encoding``；进入 builder 后统一收敛为 CachedQwenNextBatch。
                encoding = cached.get("encoding", cached.get("enc"))
                if not isinstance(encoding, dict):
                    raise ValueError("cached next-state batch is missing encoding")
                cached = CachedQwenNextBatch(
                    keys=tuple(cached.get("keys", ())),
                    encoding=dict(encoding),
                )
            if cached.keys != tuple(unique_keys):
                raise ValueError("cached next-state order does not match transition prompts")
            return dict(cached.encoding)
        unique_messages: list[list[dict[str, Any]] | None] = [None] * len(unique_keys)
        for item in items:
            messages = item.get("next_messages")
            if messages is not None:
                unique_messages[key_to_row[message_cache_key(messages)]] = messages
        return build_qwen_batch(
            [{"messages": messages} for messages in unique_messages],
            self.processor,
            self.max_length,
            latent_token_count=self.latent_token_count,
        )

def prefix_messages_with_images(sample: TransitionSample) -> list[dict[str, Any]]:
    return messages_with_image_paths(sample.prefix_messages, sample.prefix_image_paths)


def load_images_for_prefix(sample: TransitionSample) -> list[Image.Image]:
    msgs = prefix_messages_with_images(sample)
    images: list[Image.Image] = []
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    images.append(Image.open(part["image"]).convert("RGB"))
    return images


def transition_collate_for_qwen(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    """Prepare per-sample dicts for Qwen processor (messages + metadata)."""

    items: list[dict[str, Any]] = []
    for sample in batch:
        item = {
            "id": f"{sample.record_id}:{sample.step_index}",
            "record_id": sample.record_id,
            "step_index": sample.step_index,
            "messages": prefix_messages_with_images(sample),
            "action_index": sample.action_index,
            "action_value_target": sample.action_value_target,
            "success": sample.success,
            "next_image_path": sample.next_image_path,
            "current_image_path": sample.current_image_path,
            "next_messages": None,
        }
        if sample.next_prefix_messages is not None and sample.next_prefix_image_paths is not None:
            item["next_messages"] = messages_with_image_paths(
                sample.next_prefix_messages,
                sample.next_prefix_image_paths,
            )
        items.append(item)
    return items
