"""Qwen sampled-token critic used by token-level PPO credit."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class TokenValueHead(nn.Module):
    """Map the hidden state before each sampled token to a scalar value."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("token value head dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2 or hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                "token value hidden states must have shape (N, input_dim), "
                f"got {tuple(hidden_states.shape)}"
            )
        weight = self.net[0].weight
        return self.net(hidden_states.to(dtype=weight.dtype)).squeeze(-1)

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "token_value_head.pt")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "format": "nimloth_token_value_head_v1",
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "TokenValueHead":
        path = Path(path)
        config_path = path / "config.json"
        state_path = path / "token_value_head.pt"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"incomplete TokenValueHead checkpoint: {path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("format") != "nimloth_token_value_head_v1":
            raise ValueError("unsupported TokenValueHead checkpoint format")
        module = cls(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
        )
        module.load_state_dict(
            torch.load(state_path, map_location=map_location, weights_only=True)
        )
        return module


__all__ = ["TokenValueHead"]
