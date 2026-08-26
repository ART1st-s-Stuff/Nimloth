"""Qwen2.5-VL batching with assistant-span CE labels."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor

from nimloth.latent import (
    LatentActionTokens,
    latent_state_tokens,
    normalize_latent_state_blocks,
)


def message_cache_key(messages: list[dict[str, Any]]) -> str:
    """Stable key for repeated trajectory prefixes within/across epochs."""

    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=8192)
def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class _TemplateCache:
    """Small per-processor chat-template cache.

    Agent training repeatedly visits overlapping prefixes from the same trajectory.
    Caching rendered text avoids re-running Jinja chat templates for every
    current/next prefix while keeping processor ownership explicit.
    """

    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor

    @lru_cache(maxsize=131072)
    def render(self, cache_key: str, add_generation_prompt: bool) -> str:
        messages = json.loads(cache_key)
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class _OffsetCache:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @lru_cache(maxsize=131072)
    def offsets(self, text: str, max_length: int) -> tuple[tuple[int, int], ...]:
        mapping = self.tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )["offset_mapping"]
        return tuple((int(start), int(end)) for start, end in mapping)


_TEMPLATE_CACHES: dict[int, _TemplateCache] = {}
_OFFSET_CACHES: dict[int, _OffsetCache] = {}


def _template_cache(processor: AutoProcessor) -> _TemplateCache:
    key = id(processor)
    cache = _TEMPLATE_CACHES.get(key)
    if cache is None or cache.processor is not processor:
        cache = _TemplateCache(processor)
        _TEMPLATE_CACHES[key] = cache
    return cache


def _offset_cache(processor: AutoProcessor) -> _OffsetCache:
    tokenizer = processor.tokenizer
    key = id(tokenizer)
    cache = _OFFSET_CACHES.get(key)
    if cache is None or cache.tokenizer is not tokenizer:
        cache = _OffsetCache(tokenizer)
        _OFFSET_CACHES[key] = cache
    return cache


def render_messages(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    *,
    add_generation_prompt: bool,
    latent_token_count: int = 1,
) -> str:
    cache = _template_cache(processor)
    cache_key = message_cache_key(messages)
    text = cache.render(cache_key, add_generation_prompt)
    return normalize_latent_state_blocks(text, latent_token_count)


def assistant_char_spans(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    *,
    latent_token_count: int = 1,
) -> list[tuple[int, int]]:
    """Return the current transition's assistant span for CE supervision.

    Transition training expands one trajectory into many prefixes. Supervising every
    assistant span in each prefix would repeatedly train early turns.  The CE
    auxiliary loss should therefore cover only the final assistant message in
    the prefix, i.e. the action/response for the current transition.
    """

    last_assistant_index = next(
        (i for i in range(len(messages) - 1, -1, -1) if messages[i]["role"] == "assistant"),
        None,
    )
    if last_assistant_index is None:
        return []

    prev_gen = render_messages(
        messages[:last_assistant_index],
        processor,
        add_generation_prompt=True,
        latent_token_count=latent_token_count,
    )
    cur = render_messages(
        messages[: last_assistant_index + 1],
        processor,
        add_generation_prompt=False,
        latent_token_count=latent_token_count,
    )
    start = len(prev_gen)
    end = len(cur)
    return [(start, end)] if start < end else []


def collect_message_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    """Load bound message images in prompt order for processor parity."""

    imgs: list[Image.Image] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "image":
                    # Return a copy so downstream processors may safely mutate
                    # without corrupting the cached decoded RGB image.
                    imgs.append(_load_rgb_image(str(part["image"])).copy())
    return imgs


# Internal historical name retained for adjacent callers; new code uses the
# public owner above rather than reaching into a private helper.
_collect_message_images = collect_message_images


def _mask_latent_query_labels(
    labels: torch.Tensor,
    enc_input_ids: torch.Tensor,
    processor: AutoProcessor,
    *,
    latent_token_count: int,
) -> torch.Tensor:
    if not hasattr(processor.tokenizer, "convert_tokens_to_ids"):
        return labels
    latent_ids: list[int] = []
    for token in latent_state_tokens(latent_token_count, LatentActionTokens()):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(processor.tokenizer, "unk_token_id", None)
        if token_id is None or token_id == unk_id:
            continue
        latent_ids.append(int(token_id))
    for token_id in latent_ids:
        labels = labels.masked_fill(enc_input_ids == token_id, -100)
    return labels


def labels_for_text_rows(
    processor: AutoProcessor,
    enc_input_ids: torch.Tensor,
    texts: list[str],
    spans_per_item: list[list[tuple[int, int]]],
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> torch.Tensor:
    offset_cache = _offset_cache(processor)
    offset_rows = [offset_cache.offsets(text, max_length) for text in texts]
    labels = enc_input_ids.clone()
    labels[:] = -100
    for row, spans in enumerate(spans_per_item):
        usable = min(labels.shape[1], len(offset_rows[row]))
        for tok_idx in range(usable):
            start, end = offset_rows[row][tok_idx]
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                labels[row, tok_idx] = enc_input_ids[row, tok_idx]
    if mask_latent_query_labels:
        labels = _mask_latent_query_labels(
            labels,
            enc_input_ids,
            processor,
            latent_token_count=latent_token_count,
        )
    return labels


def encode_qwen_item(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    include_labels: bool = True,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    """Encode one prefix with the same semantics as ``build_qwen_batch``."""

    text = render_messages(
        messages,
        processor,
        add_generation_prompt=False,
        latent_token_count=latent_token_count,
    )
    images = _collect_message_images(messages)
    enc = processor(
        text=[text],
        images=[images] if images else None,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out: dict[str, Any] = {}
    for key, value in enc.items():
        if hasattr(value, "squeeze"):
            squeezed = value.squeeze(0)
            if key == "image_grid_thw" and hasattr(squeezed, "ndim") and squeezed.ndim == 1:
                squeezed = squeezed.unsqueeze(0)
            if hasattr(squeezed, "contiguous"):
                out[key] = squeezed.contiguous()
            else:
                out[key] = squeezed
        else:
            out[key] = value
    if include_labels:
        labels = labels_for_text_rows(
            processor,
            enc["input_ids"],
            [text],
            [assistant_char_spans(messages, processor, latent_token_count=latent_token_count)],
            max_length,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
        out["labels"] = labels.squeeze(0).contiguous()
    return out


def batch_single_encoding(enc: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add a batch dimension to a single encoded Qwen2.5-VL prefix."""

    return {
        key: value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim == 1 else value
        for key, value in enc.items()
    }


def collate_qwen_encodings(
    rows: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """合并预处理后的 Qwen prefix，并保持视觉 token 的原始拼接顺序。"""

    if not rows:
        raise ValueError("Qwen encoding rows must not be empty")
    label_presence = ["labels" in row for row in rows]
    if any(label_presence) and not all(label_presence):
        raise ValueError(
            "Qwen encoding rows must either all contain labels or all omit labels"
        )
    if len(rows) == 1:
        return {
            key: value.unsqueeze(0) if value.ndim == 1 else value
            for key, value in rows[0].items()
        }

    batch: dict[str, torch.Tensor] = {}
    if "input_ids" in rows[0]:
        batch["input_ids"] = pad_sequence(
            [row["input_ids"] for row in rows],
            batch_first=True,
            padding_value=pad_token_id,
        )
    if "attention_mask" in rows[0]:
        batch["attention_mask"] = pad_sequence(
            [row["attention_mask"] for row in rows],
            batch_first=True,
            padding_value=0,
        )
    if all(label_presence):
        batch["labels"] = pad_sequence(
            [row["labels"] for row in rows],
            batch_first=True,
            padding_value=-100,
        )
    for key in ("pixel_values", "image_grid_thw"):
        if key not in rows[0]:
            continue
        tensors = [
            row[key].unsqueeze(0) if row[key].ndim == 1 else row[key]
            for row in rows
        ]
        batch[key] = torch.cat(tensors, dim=0)
    return batch


def build_qwen_batch(
    items: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in items:
        text = render_messages(
            item["messages"],
            processor,
            add_generation_prompt=False,
            latent_token_count=latent_token_count,
        )
        texts.append(text)
        spans_per_item.append(
            assistant_char_spans(
                item["messages"],
                processor,
                latent_token_count=latent_token_count,
            )
        )
        all_images.append(_collect_message_images(item["messages"]))

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = labels_for_text_rows(
        processor,
        enc["input_ids"],
        texts,
        spans_per_item,
        max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    return enc
