"""Qwen2.5-VL-specific checkpoint adapters shared by training phases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class AdapterLoadReport:
    missing_keys: int
    unexpected_keys: int
    vision_full_state_loaded: bool


def find_visual_module(model: torch.nn.Module) -> torch.nn.Module:
    """Locate the visual tower through supported Qwen/PEFT wrapper layouts."""

    root = model.module if hasattr(model, "module") else model
    for path in (
        "base_model.model.model.visual",
        "base_model.model.visual",
        "model.visual",
        "visual",
    ):
        current = root
        for name in path.split("."):
            current = getattr(current, name, None)
            if current is None:
                break
        if isinstance(current, torch.nn.Module):
            return current
    raise RuntimeError(f"could not locate Qwen visual module in {type(root)}")


def load_adapter_state(model: torch.nn.Module, adapter_dir: Path) -> AdapterLoadReport:
    """Load PEFT weights plus the optional fully tuned Qwen visual tower."""

    adapter_file = adapter_dir / "adapter_model.safetensors"
    if adapter_file.is_file():
        from safetensors.torch import load_file

        state = load_file(str(adapter_file))
    else:
        bin_file = adapter_dir / "adapter_model.bin"
        if not bin_file.is_file():
            raise FileNotFoundError(f"missing adapter weights in {adapter_dir}")
        state = torch.load(bin_file, map_location="cpu", weights_only=True)

    incompatible = model.load_state_dict(state, strict=False)
    vision_full_path = adapter_dir / "vision_full_state.pt"
    vision_loaded = vision_full_path.is_file()
    if vision_loaded:
        find_visual_module(model).load_state_dict(
            torch.load(vision_full_path, map_location="cpu", weights_only=True)
        )
    return AdapterLoadReport(
        missing_keys=len(incompatible.missing_keys),
        unexpected_keys=len(incompatible.unexpected_keys),
        vision_full_state_loaded=vision_loaded,
    )


def save_full_vision_state(model: torch.nn.Module, path: Path) -> None:
    """Save the fully tuned Qwen visual tower next to a PEFT adapter."""

    torch.save(find_visual_module(model).state_dict(), path)
