"""Keep Qwen image expansion and assistant character coordinates aligned."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def expand_image_text(
    text: str, grids: torch.Tensor, processor: Any,
    spans: Sequence[tuple[int, int]] = (),
) -> tuple[str, list[tuple[int, int]]]:
    grids = grids.reshape(-1, 3)
    if grids.numel() == 0:
        token = getattr(processor, "image_token", None)
        if token and token in text:
            raise ValueError("image placeholders have no corresponding grids")
        return text, list(spans)
    token = str(processor.image_token)
    merge = int(processor.image_processor.merge_size) ** 2
    if text.count(token) != len(grids):
        raise ValueError("image placeholder count does not match cached image grids")
    pieces = []
    replacements = []
    cursor = 0
    for grid in grids:
        patches = int(grid.prod().item())
        if patches <= 0 or patches % merge:
            raise ValueError("image grid must contain a positive divisible number of patches")
        start = text.index(token, cursor)
        end = start + len(token)
        replacement = token * (patches // merge)
        pieces.extend((text[cursor:start], replacement))
        replacements.append((start, end, len(replacement) - len(token)))
        cursor = end
    pieces.append(text[cursor:])
    def remap(position: int) -> int:
        if not 0 <= position <= len(text):
            raise ValueError("assistant span is outside rendered text")
        if any(start < position < end for start, end, _ in replacements):
            raise ValueError("assistant span boundary splits an image placeholder")
        return position + sum(delta for _, end, delta in replacements if end <= position)
    return "".join(pieces), [(remap(start), remap(end)) for start, end in spans]


def expand_image_rows(
    texts: Sequence[str], spans_per_item: Sequence[Sequence[tuple[int, int]]],
    grids: torch.Tensor | None, processor: Any,
) -> tuple[list[str], list[list[tuple[int, int]]]]:
    """Split processor-concatenated grids by each rendered row's placeholders."""
    grids = torch.empty((0, 3), dtype=torch.long) if grids is None else grids.reshape(-1, 3)
    image_token = getattr(processor, "image_token", None)
    expanded, mapped = [], []
    cursor = 0
    for text, spans in zip(texts, spans_per_item, strict=True):
        count = text.count(image_token) if image_token else 0
        row, row_spans = expand_image_text(text, grids[cursor:cursor + count], processor, spans)
        expanded.append(row)
        mapped.append(row_spans)
        cursor += count
    if cursor != len(grids):
        raise ValueError("unused image grids after splitting rendered rows")
    return expanded, mapped


def require_complete_length(encoding: Mapping[str, torch.Tensor], max_length: int) -> None:
    """Reject over-budget complete encodings; never retain a truncated prefix."""
    ids = encoding["input_ids"]
    mask = encoding.get("attention_mask")
    if ids.ndim not in (1, 2):
        raise ValueError("Qwen input IDs must be a single sequence or a batch")
    if mask is not None and mask.shape != ids.shape:
        raise ValueError("Qwen attention mask and input IDs must have the same shape")
    lengths = mask.sum(-1) if mask is not None else torch.tensor(ids.shape[-1])
    longest = int(lengths.max().item())
    if longest > max_length:
        raise ValueError(f"complete Qwen prefix exceeds max_length: required={longest}, max_length={max_length}; truncation is not allowed")


def validate_image_encoding(encoding: Mapping[str, torch.Tensor], processor: Any) -> None:
    """Reject stale/corrupt cached image-token or pixel/grid alignments per row."""
    token = getattr(processor, "image_token", None)
    grids = encoding.get("image_grid_thw")
    if token is None:
        if grids is not None and grids.numel():
            raise ValueError("image-bearing encoding requires processor image-token identity")
        return
    token_id = processor.tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(processor.tokenizer, "unk_token_id", None):
        raise ValueError("processor image token is not registered")
    actual = int((encoding["input_ids"] == token_id).sum().item())
    expected = 0
    patches = 0
    if grids is not None and grids.numel():
        products = grids.reshape(-1, 3).prod(-1)
        merge = int(processor.image_processor.merge_size) ** 2
        if torch.any(products <= 0) or torch.any(products % merge):
            raise ValueError("invalid image grid patch counts")
        patches = int(products.sum().item())
        expected = patches // merge
    if actual != expected:
        raise ValueError(f"Qwen image token/grid mismatch: tokens={actual}, features={expected}; rebuild complete encodings")
    pixels = encoding.get("pixel_values")
    if pixels is not None and pixels.shape[0] != patches:
        raise ValueError(f"Qwen pixel/grid mismatch: pixel_rows={pixels.shape[0]}, grid_patches={patches}")
