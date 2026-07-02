"""Checkpoint helpers for Nimloth-trained RCDM models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when ``model`` is wrapped by DDP."""

    return getattr(model, "module", model)


@torch.no_grad()
def init_ema_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Create an EMA state dict on the same device as the model tensors."""

    return {k: v.detach().float().clone() for k, v in unwrap_model(model).state_dict().items()}


@torch.no_grad()
def update_ema_state(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    """Update ``ema`` in-place from ``model`` using ``decay``."""

    state = unwrap_model(model).state_dict()
    for key, value in state.items():
        if key not in ema:
            ema[key] = value.detach().float().clone()
        else:
            ema[key].mul_(decay).add_(value.detach().float(), alpha=1.0 - decay)


def save_training_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    step_in_epoch: int,
    metadata: dict[str, Any],
    ema_states: dict[float, dict[str, torch.Tensor]] | None = None,
) -> None:
    """Save raw model, optimizer/training state, metadata, and optional EMA states."""

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), output_dir / f"model_{step:09d}.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "epoch": int(epoch),
            "step_in_epoch": int(step_in_epoch),
            "metadata": metadata,
        },
        output_dir / f"training_state_{step:09d}.pt",
    )
    if ema_states:
        for rate, state in ema_states.items():
            torch.save(state, output_dir / f"ema_{rate}_{step:09d}.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_state_dict(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Load a model or EMA state dict from disk."""

    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {path} did not contain a state dict")
    return state


def parse_ema_rates(value: str | Iterable[float]) -> list[float]:
    """Parse comma-separated EMA rates, e.g. ``'0.999,0.9999'``."""

    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return [float(p) for p in parts]
    return [float(v) for v in value]
