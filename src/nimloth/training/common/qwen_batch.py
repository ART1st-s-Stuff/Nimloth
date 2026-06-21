"""Qwen2.5-VL batching with assistant-span CE labels."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor


def _message_cache_key(messages: list[dict[str, Any]]) -> str:
    """Stable key for repeated trajectory prefixes within/across epochs."""

    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=8192)
def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class _TemplateCache:
    """Small per-processor chat-template cache.

    Qwen SFT2 repeatedly visits overlapping prefixes from the same trajectory.
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


def assistant_char_spans(messages: list[dict[str, Any]], processor: AutoProcessor) -> list[tuple[int, int]]:
    """Return the current transition's assistant span for CE supervision.

    SFT2 expands one trajectory into many prefix transitions.  Supervising every
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

    cache = _template_cache(processor)
    prev_key = _message_cache_key(messages[:last_assistant_index])
    cur_key = _message_cache_key(messages[: last_assistant_index + 1])
    prev_gen = cache.render(prev_key, True)
    cur = cache.render(cur_key, False)
    start = len(prev_gen)
    end = len(cur)
    return [(start, end)] if start < end else []


def _collect_message_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
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


def _labels_for_text_rows(
    processor: AutoProcessor,
    enc_input_ids: torch.Tensor,
    texts: list[str],
    spans_per_item: list[list[tuple[int, int]]],
    max_length: int,
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
    return labels


def encode_qwen_item(
    messages: list[dict[str, Any]],
    processor: AutoProcessor,
    max_length: int,
    *,
    include_labels: bool = True,
) -> dict[str, Any]:
    """Encode one prefix with the same semantics as ``build_qwen_batch``."""

    cache = _template_cache(processor)
    cache_key = _message_cache_key(messages)
    text = cache.render(cache_key, False)
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
        labels = _labels_for_text_rows(
            processor,
            enc["input_ids"],
            [text],
            [assistant_char_spans(messages, processor)],
            max_length,
        )
        out["labels"] = labels.squeeze(0).contiguous()
    return out


def build_qwen_batch(items: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, Any]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    cache = _template_cache(processor)
    for item in items:
        cache_key = _message_cache_key(item["messages"])
        text = cache.render(cache_key, False)
        texts.append(text)
        spans_per_item.append(assistant_char_spans(item["messages"], processor))
        all_images.append(_collect_message_images(item["messages"]))

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = _labels_for_text_rows(processor, enc["input_ids"], texts, spans_per_item, max_length)
    return enc
