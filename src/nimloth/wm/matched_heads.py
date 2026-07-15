"""Matched-budget vector and token WM heads over one frozen State tensor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.token_set_predictor import (
    TokenSetPredictorConfig,
    TokenSetWMPredictor,
)


@dataclass(frozen=True)
class MatchedHeadSpec:
    state_tokens: int = 8
    token_dim: int = 1024
    vector_hidden_dim: int = 896
    token_hidden_dim: int = 1024
    depth: int = 4
    heads: int = 8
    mlp_ratio: int = 4
    action_dim: int = 8

    @property
    def flat_dim(self) -> int:
        return self.state_tokens * self.token_dim


class MatchedWMHeads(nn.Module):
    """Two independent predictors that differ only in State topology."""

    def __init__(
        self,
        spec: MatchedHeadSpec,
        vector: TokenSetWMPredictor,
        token: TokenSetWMPredictor,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.vector = vector
        self.token = token

    @classmethod
    def create(cls, spec: MatchedHeadSpec) -> "MatchedWMHeads":
        vector = TokenSetWMPredictor(cls._head_config(spec, vector=True))
        token = TokenSetWMPredictor(cls._head_config(spec, vector=False))
        return cls(spec, vector, token)

    @staticmethod
    def _head_config(spec: MatchedHeadSpec, *, vector: bool) -> TokenSetPredictorConfig:
        return TokenSetPredictorConfig(
            num_tokens=1 if vector else spec.state_tokens,
            emb_dim=spec.flat_dim if vector else spec.token_dim,
            action_dim=spec.action_dim,
            hidden_dim=spec.vector_hidden_dim if vector else spec.token_hidden_dim,
            depth=spec.depth,
            heads=spec.heads,
            mlp_ratio=spec.mlp_ratio,
        )

    def predict_next(
        self, state: StateViews, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vector(state.vector, actions), self.token(state.tokens, actions)

    def rollout(
        self, state: StateViews, action_sequences: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vector = self.vector.rollout_states(state.vector, action_sequences)
        token = self.token.rollout_states(state.tokens, action_sequences)
        return vector, token

    def parameter_counts(self) -> dict[str, int]:
        return {
            "vector": sum(parameter.numel() for parameter in self.vector.parameters()),
            "token": sum(parameter.numel() for parameter in self.token.parameters()),
        }

    def save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "matched_heads.pt")
        (path / "config.json").write_text(
            json.dumps(asdict(self.spec), indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_checkpoint(
        cls, path: Path, map_location: str | torch.device = "cpu"
    ) -> "MatchedWMHeads":
        spec = MatchedHeadSpec(**json.loads((path / "config.json").read_text()))
        module = cls.create(spec)
        state = torch.load(path / "matched_heads.pt", map_location=map_location, weights_only=True)
        module.load_state_dict(state, strict=True)
        return module
