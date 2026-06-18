"""Qwen2.5-VL batching with assistant-span CE labels."""

from __future__ import annotations

from typing import Any

from PIL import Image
from transformers import AutoProcessor


def assistant_char_spans(messages: list[dict[str, Any]], processor: AutoProcessor) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        prev_gen = processor.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        cur = processor.apply_chat_template(messages[: i + 1], tokenize=False, add_generation_prompt=False)
        start = len(prev_gen)
        end = len(cur)
        if start < end:
            spans.append((start, end))
    return spans


def build_qwen_batch(items: list[dict[str, Any]], processor: AutoProcessor, max_length: int) -> dict[str, Any]:
    texts: list[str] = []
    spans_per_item: list[list[tuple[int, int]]] = []
    all_images: list[list[Image.Image]] = []
    for item in items:
        text = processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(text)
        spans_per_item.append(assistant_char_spans(item["messages"], processor))
        imgs: list[Image.Image] = []
        for msg in item["messages"]:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image":
                        imgs.append(Image.open(part["image"]).convert("RGB"))
        all_images.append(imgs)

    enc = processor(
        text=texts,
        images=all_images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    offset_batch = processor.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )["offset_mapping"]
    labels = enc["input_ids"].clone()
    labels[:] = -100
    usable = min(labels.shape[1], offset_batch.shape[1])
    for row, spans in enumerate(spans_per_item):
        for tok_idx in range(usable):
            start = int(offset_batch[row, tok_idx, 0])
            end = int(offset_batch[row, tok_idx, 1])
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                labels[row, tok_idx] = enc["input_ids"][row, tok_idx]
    enc["labels"] = labels
    return enc
