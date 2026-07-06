"""Qwen vision-encoder token extraction for representation ablations.

This module extracts the output of Qwen2.5-VL's visual encoder directly. It
intentionally does not use LLM hidden states at ``<|image_pad|>`` positions.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


@lru_cache(maxsize=8192)
def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def build_qwen_vision_batch(
    image_paths: Sequence[str | Path],
    processor,
    *,
    max_pixels: int | None = None,
) -> dict[str, torch.Tensor]:
    """Processor batch for direct Qwen vision-encoder extraction.

    The text contains exactly one image placeholder per row. The returned batch
    is suitable for ``model.visual(pixel_values, grid_thw=image_grid_thw)``.
    """

    if not image_paths:
        raise ValueError("image_paths must be non-empty")
    if max_pixels is not None:
        processor.image_processor.max_pixels = max_pixels
    images = [[_load_rgb_image(str(path)).copy()] for path in image_paths]
    text = ["<|vision_start|><|image_pad|><|vision_end|>" for _ in image_paths]
    return processor(text=text, images=images, padding=True, return_tensors="pt")


def vision_token_lengths(image_grid_thw: torch.Tensor, *, spatial_merge_unit: int) -> list[int]:
    """Return Qwen visual-output token count for each image row."""

    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape (B, 3), got {tuple(image_grid_thw.shape)}")
    lengths = []
    for row in image_grid_thw.detach().cpu():
        t, h, w = [int(x) for x in row.tolist()]
        merged = t * h * w
        if merged % spatial_merge_unit != 0:
            raise ValueError(f"grid {row.tolist()} is not divisible by spatial_merge_unit={spatial_merge_unit}")
        lengths.append(merged // spatial_merge_unit)
    return lengths


@torch.no_grad()
def extract_qwen_vision_tokens(
    model,
    processor,
    image_paths: Sequence[str | Path],
    *,
    device: torch.device,
    max_pixels: int | None = None,
    expected_num_tokens: int | None = None,
) -> torch.Tensor:
    """Extract direct Qwen vision-encoder tokens for a batch of images.

    Returns ``(B, K, D)``. This function requires every image in the batch to
    produce the same K, because downstream predictors/RCDM flattening use fixed
    shapes. It fails loudly instead of padding/truncating.
    """

    enc = build_qwen_vision_batch(image_paths, processor, max_pixels=max_pixels)
    if "pixel_values" not in enc or "image_grid_thw" not in enc:
        raise ValueError("processor output must contain pixel_values and image_grid_thw")
    pixel_values = enc["pixel_values"].to(device=device, dtype=model.visual.dtype)
    grid = enc["image_grid_thw"].to(device=device)
    tokens = model.visual(pixel_values, grid_thw=grid)
    spatial_merge_unit = int(getattr(model.visual, "spatial_merge_unit", 1))
    lengths = vision_token_lengths(grid, spatial_merge_unit=spatial_merge_unit)
    if sum(lengths) != tokens.shape[0]:
        raise ValueError(f"vision token split mismatch: lengths={lengths}, tokens={tuple(tokens.shape)}")
    if len(set(lengths)) != 1:
        raise ValueError(f"all images must produce the same number of vision tokens, got {lengths}")
    if expected_num_tokens is not None and lengths[0] != expected_num_tokens:
        raise ValueError(f"expected {expected_num_tokens} vision tokens, got {lengths[0]}")
    return torch.stack(list(tokens.split(lengths, dim=0)), dim=0)


def flatten_vision_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Direct concatenation of vision tokens into an RCDM condition vector."""

    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape (B, K, D), got {tuple(tokens.shape)}")
    return tokens.flatten(1)
