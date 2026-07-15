"""Matched-budget vector and token WM heads over one frozen State tensor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.token_set_predictor import TokenSetPredictorConfig, TokenSetWMPredictor


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


def _head_config(spec: MatchedHeadSpec, vector: bool) -> TokenSetPredictorConfig:
    return TokenSetPredictorConfig(
        num_tokens=1 if vector else spec.state_tokens,
        emb_dim=spec.flat_dim if vector else spec.token_dim,
        action_dim=spec.action_dim,
        hidden_dim=spec.vector_hidden_dim if vector else spec.token_hidden_dim,
        depth=spec.depth,
        heads=spec.heads,
        mlp_ratio=spec.mlp_ratio,
    )


def _create(spec: MatchedHeadSpec) -> "MatchedWMHeads":
    vector = TokenSetWMPredictor(_head_config(spec, True))
    token = TokenSetWMPredictor(_head_config(spec, False))
    return MatchedWMHeads(spec, vector, token)


def _save(module: "MatchedWMHeads", path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path / "matched_heads.pt")
    config = json.dumps(asdict(module.spec), indent=2) + "\n"
    (path / "config.json").write_text(config, encoding="utf-8")


def _load(path: Path, map_location: str | torch.device) -> "MatchedWMHeads":
    spec = MatchedHeadSpec(**json.loads((path / "config.json").read_text()))
    module = _create(spec)
    state = torch.load(path / "matched_heads.pt", map_location=map_location, weights_only=True)
    module.load_state_dict(state, strict=True)
    return module


class MatchedWMHeads(nn.Module):
    """Two independent predictors that differ only in State topology."""

    def __init__(self, spec: MatchedHeadSpec, vector: nn.Module, token: nn.Module) -> None:
        super().__init__()
        self.spec, self.vector, self.token = spec, vector, token

    @classmethod
    def create(cls, spec: MatchedHeadSpec) -> "MatchedWMHeads":
        return _create(spec)

    def predict_next(self, state: StateViews, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vector(state.vector, actions), self.token(state.tokens, actions)

    def rollout(self, state: StateViews, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.vector.rollout_states(state.vector, actions), self.token.rollout_states(state.tokens, actions)

    def parameter_counts(self) -> dict[str, int]:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        return {"vector": count(self.vector), "token": count(self.token)}

    def save_checkpoint(self, path: Path) -> None:
        _save(self, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "MatchedWMHeads":
        return _load(path, map_location)
