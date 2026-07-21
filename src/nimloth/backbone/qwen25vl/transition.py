"""World-model transition 到 Qwen2.5-VL 输入与 latent state 的适配。"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from PIL import Image

from nimloth.agent import bind_image_placeholders
from nimloth.backbone.qwen25vl.batch import (
    build_qwen_batch,
    collate_qwen_encodings,
    message_cache_key,
)
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA
from nimloth.rollout.transitions import TransitionSample

# Compatibility name for existing callers. Agent owns the message/image contract.
messages_with_image_paths = bind_image_placeholders


@dataclass(frozen=True)
class QwenTransitionMessages:
    """一个 transition 在 Qwen 中使用的当前和下一状态 prompt。"""

    current: list[dict[str, Any]]
    next: list[dict[str, Any]] | None


@dataclass(frozen=True)
class CachedQwenNextBatch:
    """DataLoader worker 已经去重并合并的下一状态 Qwen 输入。"""

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


@dataclass(frozen=True)
class QwenTransitionEncoder:
    """集中管理 SFT2 当前/下一 prefix 的 Qwen 运行期配置。"""

    processor: Any
    token_id_map: dict[str, int]
    device: torch.device
    max_length: int
    pad_token_id: int
    latent_token_count: int = 1
    mask_latent_query_labels: bool = True
    vision_ema: VisionEncoderEMA | None = None

    def encode_current(
        self,
        model: torch.nn.Module,
        encoding: dict[str, torch.Tensor],
        *,
        include_lm_loss: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """编码当前 prefix；只有训练模式保留 CE labels。"""

        model_encoding = dict(encoding)
        if not include_lm_loss:
            model_encoding.pop("labels", None)
        latent, lm_loss = extract_qwen_latents(
            model,
            model_encoding,
            self.token_id_map,
            self.device,
            latent_token_count=self.latent_token_count,
        )
        return latent, lm_loss if include_lm_loss else None

    def encode_next(
        self,
        model: torch.nn.Module,
        transitions: Sequence[QwenTransitionMessages],
        indices: Sequence[int],
        *,
        cached: CachedQwenNextBatch | None,
        use_vision_ema: bool,
    ) -> torch.Tensor:
        """去重下一状态 prompt，执行无梯度 target forward，再恢复原 batch 顺序。"""

        if not indices:
            raise ValueError("next-state indices must not be empty")

        unique_keys: list[str] = []
        key_to_row: dict[str, int] = {}
        for index in indices:
            messages = transitions[index].next
            if messages is None:
                raise ValueError("next-state index points to a terminal transition")
            key = message_cache_key(messages)
            if key not in key_to_row:
                key_to_row[key] = len(unique_keys)
                unique_keys.append(key)

        if cached is not None:
            if cached.keys != tuple(unique_keys):
                raise ValueError(
                    "cached next-state order does not match transition prompts: "
                    f"{len(cached.keys)} != {len(unique_keys)}"
                )
            next_encoding = dict(cached.encoding)
        else:
            unique_indices = [indices[key_to_row[key]] for key in unique_keys]
            next_encoding = build_qwen_batch(
                [
                    {"messages": transitions[index].next}
                    for index in unique_indices
                ],
                self.processor,
                self.max_length,
                latent_token_count=self.latent_token_count,
            )
        next_encoding.pop("labels", None)

        ema_context = self._vision_ema_context(model, use_vision_ema)
        with torch.no_grad(), ema_context:
            unique_latents, _ = extract_qwen_latents(
                model,
                next_encoding,
                self.token_id_map,
                self.device,
                latent_token_count=self.latent_token_count,
            )

        restored_rows = []
        for index in indices:
            messages = transitions[index].next
            assert messages is not None
            row = key_to_row[message_cache_key(messages)]
            restored_rows.append(unique_latents[row : row + 1])
        return torch.cat(restored_rows, dim=0)

    def encode_ddp_dummy_target(
        self,
        model: torch.nn.Module,
        messages: list[dict[str, Any]],
        *,
        use_vision_ema: bool,
    ) -> torch.Tensor:
        """terminal-only rank 仍执行一次 target Qwen forward，以对齐 DDP 调用次数。"""

        encoding = build_qwen_batch(
            [{"messages": messages}],
            self.processor,
            self.max_length,
            latent_token_count=self.latent_token_count,
        )
        encoding.pop("labels", None)
        with torch.no_grad(), self._vision_ema_context(model, use_vision_ema):
            latent, _ = extract_qwen_latents(
                model,
                encoding,
                self.token_id_map,
                self.device,
                latent_token_count=self.latent_token_count,
            )
        return latent

    def _vision_ema_context(self, model: torch.nn.Module, enabled: bool):
        if enabled and self.vision_ema is not None:
            return self.vision_ema.use_ema_weights(model)
        return contextlib.nullcontext()


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
