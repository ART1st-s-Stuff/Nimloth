"""Frozen DINO-family encoder used as semantic supervision for query states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn


DEFAULT_DINO_MODEL = "facebook/dinov2-large"


class FrozenDINOEncoder(nn.Module):
    """Extract detached global (CLS) features from RGB observations.

    The encoder owns no trainable path: the explicitly selected DINO-family
    model remains in eval mode and its weights are frozen. Nimloth directly
    aligns the projected query state to the final CLS token, so no trainable
    alignment head can absorb the supervision.
    """

    def __init__(self, *, model: nn.Module, image_processor: Any, source: str) -> None:
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.source = str(source)
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DINO model config must define hidden_size")
        self.hidden_size = int(hidden_size)
        self.model.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path = DEFAULT_DINO_MODEL,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FrozenDINOEncoder:
        """Load the explicitly requested DINO checkpoint and processor.

        Loading failures are intentionally propagated; the encoder never
        silently substitutes another model or DINO generation.
        """

        from transformers import AutoImageProcessor, AutoModel

        source_str = str(source)
        image_processor = AutoImageProcessor.from_pretrained(source_str, trust_remote_code=True)
        model = AutoModel.from_pretrained(source_str, trust_remote_code=True, torch_dtype=dtype)
        model.to(device=device, dtype=dtype)
        return cls(model=model, image_processor=image_processor, source=source_str)

    def train(self, mode: bool = True) -> FrozenDINOEncoder:
        """Keep the frozen teacher in eval mode even when its parent trains."""

        super().train(False)
        return self

    @torch.no_grad()
    def encode_image_paths(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return final-layer CLS features with shape ``(B, hidden_size)``."""

        if not paths:
            raise ValueError("DINO alignment requires at least one image path")
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        processed = self.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, non_blocking=True)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = pixel_values.dtype
        if pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=model_dtype)

        outputs = self.model(pixel_values=pixel_values)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            shape = None if hidden is None else tuple(hidden.shape)
            raise ValueError(
                "DINO model must return last_hidden_state with shape "
                f"(B, tokens, {self.hidden_size}), got {shape}"
            )
        return hidden[:, 0, :].detach().float()
