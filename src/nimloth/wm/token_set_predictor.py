"""Action-conditioned token-set predictor used by legacy representation ablations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class TokenSetPredictorConfig:
    num_tokens: int = 16
    emb_dim: int = 512
    action_dim: int = 8
    hidden_dim: int = 1024
    depth: int = 4
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.num_tokens < 1 or self.emb_dim < 1 or self.hidden_dim < 1:
            raise ValueError("token-set predictor dimensions must be positive")
        if self.hidden_dim % self.heads:
            raise ValueError("hidden_dim must be divisible by heads")


class TokenSetWMPredictor(nn.Module):
    """Predict the next ``(K, D)`` state from tokens and one discrete action."""

    def __init__(self, config: TokenSetPredictorConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.emb_dim, config.hidden_dim)
        self.action_embed = nn.Embedding(config.action_dim, config.hidden_dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_tokens + 1, config.hidden_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.heads,
            dim_feedforward=config.hidden_dim * config.mlp_ratio,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.emb_dim)

    def predict_next_emb(
        self, state_tokens: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        if state_tokens.ndim != 3:
            raise ValueError(f"state_tokens must be (B,K,D), got {tuple(state_tokens.shape)}")
        if state_tokens.shape[1:] != (self.config.num_tokens, self.config.emb_dim):
            raise ValueError(
                "state token shape mismatch: "
                f"expected (*,{self.config.num_tokens},{self.config.emb_dim}), "
                f"got {tuple(state_tokens.shape)}"
            )
        action = self.action_embed(action_indices.long()).unsqueeze(1)
        state = self.input_proj(state_tokens)
        hidden = torch.cat([action, state], dim=1) + self.pos_embed.to(
            dtype=state.dtype, device=state.device
        )
        hidden = self.encoder(hidden)
        return self.output_proj(self.norm(hidden[:, 1:, :]))

    def rollout_states(
        self, state_tokens: torch.Tensor, action_sequences: torch.Tensor
    ) -> torch.Tensor:
        current = state_tokens
        predictions: list[torch.Tensor] = []
        for step in range(action_sequences.shape[1]):
            current = self.predict_next_emb(current, action_sequences[:, step])
            predictions.append(current)
        if not predictions:
            return state_tokens.new_empty(state_tokens.shape[0], 0, *state_tokens.shape[1:])
        return torch.stack(predictions, dim=1)

    def forward(
        self, state_tokens: torch.Tensor, action_indices: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_next_emb(state_tokens, action_indices)

    @classmethod
    def load_checkpoint(
        cls, path: Path, map_location: str | torch.device = "cpu"
    ) -> "TokenSetWMPredictor":
        config = TokenSetPredictorConfig(
            **json.loads((path / "config.json").read_text(encoding="utf-8"))
        )
        module = cls(config)
        module.load_state_dict(
            torch.load(path / "predictor.pt", map_location=map_location, weights_only=True)
        )
        return module
