"""Frozen Query encoder and exact vector/token views for WM-head ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


class FrozenQueryStateEncoder(nn.Module):
    """Load only the trained 8×2048→8×1024 encoder from a probe checkpoint."""

    def __init__(self, input_dim: int, output_dim: int, *, source_step: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.source_step = int(source_step)
        self.encoder_norm = nn.LayerNorm(input_dim)
        self.encoder = nn.Linear(input_dim, output_dim)
        self.bottleneck_norm = nn.LayerNorm(output_dim)

    @classmethod
    def from_probe_checkpoint(cls, path: Path) -> "FrozenQueryStateEncoder":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        input_shape = payload["invariants"]["input_shape"]
        output_shape = payload["invariants"]["bottleneck_shape"]
        module = cls(input_shape[1], output_shape[1], source_step=payload["step"])
        module.load_state_dict(cls._encoder_state(payload["model"]), strict=True)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    @staticmethod
    def _encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        prefixes = ("encoder_norm.", "encoder.", "bottleneck_norm.")
        return {key: value for key, value in state.items() if key.startswith(prefixes)}

    def forward(self, query_hidden: torch.Tensor) -> torch.Tensor:
        expected = (8, self.input_dim)
        if query_hidden.ndim != 3 or tuple(query_hidden.shape[1:]) != expected:
            raise ValueError(f"expected Query shape (B,8,{self.input_dim}), got {tuple(query_hidden.shape)}")
        hidden = self.encoder_norm(query_hidden.float())
        return self.bottleneck_norm(self.encoder(hidden))


@dataclass(frozen=True)
class StateViews:
    """Two shape views over one contiguous token-state tensor."""

    tokens: torch.Tensor
    vector: torch.Tensor

    @classmethod
    def from_tokens(cls, tokens: torch.Tensor) -> "StateViews":
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B,K,D), got {tuple(tokens.shape)}")
        if not tokens.is_contiguous():
            raise ValueError("tokens must be contiguous so vector and token views share storage")
        vector = tokens.view(tokens.shape[0], 1, -1)
        return cls(tokens=tokens, vector=vector)
