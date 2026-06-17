"""Exponential moving average for Qwen2.5-VL vision encoder weights."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn

from nimloth.training.common.qwen_tuning import TuneMode, is_vision_param


def vision_is_trainable(vision_tune: TuneMode) -> bool:
    return vision_tune in ("full", "lora")


def resolve_vision_ema(args, vision_tune: TuneMode) -> bool:
    """Default on when vision is trainable (per sft2_exp.md), unless explicitly disabled."""

    if getattr(args, "no_vision_ema", False):
        return False
    if getattr(args, "vision_ema", None) is True:
        return True
    if getattr(args, "vision_ema", None) is False:
        return False
    return vision_is_trainable(vision_tune)


def _unwrap_module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def iter_trainable_vision_params(model: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    module = _unwrap_module(model)
    for name, param in module.named_parameters():
        if is_vision_param(name) and param.requires_grad:
            yield name, param


class VisionEncoderEMA:
    """Track EMA of trainable vision-encoder parameters on a Qwen2.5-VL model."""

    def __init__(self, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        self.shadow.clear()
        for name, param in iter_trainable_vision_params(model):
            self.shadow[name] = param.data.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in iter_trainable_vision_params(model):
            tensor = param.data.detach()
            if name not in self.shadow:
                self.shadow[name] = tensor.clone()
                continue
            self.shadow[name].mul_(self.decay).add_(tensor, alpha=1.0 - self.decay)

    @contextmanager
    def use_ema_weights(self, model: nn.Module):
        """Temporarily swap trainable vision weights to EMA copies for forward."""

        backups: dict[str, torch.Tensor] = {}
        module = _unwrap_module(model)
        try:
            for name, param in iter_trainable_vision_params(module):
                if name not in self.shadow:
                    continue
                backups[name] = param.data.detach().clone()
                param.data.copy_(self.shadow[name])
            yield
        finally:
            for name, param in iter_trainable_vision_params(module):
                if name in backups:
                    param.data.copy_(backups[name])

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"decay": self.decay, "shadow": self.shadow}, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "VisionEncoderEMA":
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        ema = cls(decay=float(payload.get("decay", 0.999)))
        ema.shadow = {k: v.detach().clone() for k, v in payload["shadow"].items()}
        return ema
