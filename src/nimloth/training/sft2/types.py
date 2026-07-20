"""Typed contracts shared by SFT2 training and validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SFT2StepOutput:
    """Loss tensors and detached scalar metrics produced by one SFT2 forward."""

    lm_loss: torch.Tensor | None
    wm_loss: torch.Tensor | None
    sigreg_loss: torch.Tensor | None
    value_loss: torch.Tensor
    metrics: dict[str, float]
