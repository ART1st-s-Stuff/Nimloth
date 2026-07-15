"""Paired LeWM predictors with 8192 and 2048 dynamics embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor


@dataclass(frozen=True)
class DynamicsDimHeadSpec:
    external_dim: int = 8192
    full_dynamics_dim: int = 8192
    factorized_dynamics_dim: int = 2048
    predictor_hidden_dim: int = 1024
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 4096
    history_size: int = 4
    action_dim: int = 8


def _config(spec: DynamicsDimHeadSpec, dynamics_dim: int) -> LeWMConfig:
    return LeWMConfig(
        emb_dim=spec.external_dim,
        dynamics_dim=dynamics_dim,
        action_dim=spec.action_dim,
        predictor_depth=spec.predictor_depth,
        predictor_heads=spec.predictor_heads,
        predictor_mlp_dim=spec.predictor_mlp_dim,
        predictor_hidden_dim=spec.predictor_hidden_dim,
        history_size=spec.history_size,
    )


def _create(spec: DynamicsDimHeadSpec) -> "DynamicsDimWMHeads":
    full = LatentWMPredictor.create(_config(spec, spec.full_dynamics_dim))
    factorized = LatentWMPredictor.create(_config(spec, spec.factorized_dynamics_dim))
    return DynamicsDimWMHeads(spec, full, factorized)


def _save(module: "DynamicsDimWMHeads", path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path / "dynamics_dim_heads.pt")
    (path / "config.json").write_text(json.dumps(asdict(module.spec), indent=2) + "\n", encoding="utf-8")


def _load(path: Path, map_location: str | torch.device) -> "DynamicsDimWMHeads":
    spec = DynamicsDimHeadSpec(**json.loads((path / "config.json").read_text()))
    module = _create(spec)
    state = torch.load(path / "dynamics_dim_heads.pt", map_location=map_location, weights_only=True)
    module.load_state_dict(state, strict=True)
    return module


class DynamicsDimWMHeads(nn.Module):
    def __init__(self, spec: DynamicsDimHeadSpec, full: nn.Module, factorized: nn.Module) -> None:
        super().__init__()
        self.spec, self.full, self.factorized = spec, full, factorized

    @classmethod
    def create(cls, spec: DynamicsDimHeadSpec) -> "DynamicsDimWMHeads":
        return _create(spec)

    def predict_next(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.full(state, actions), self.factorized(state, actions)

    def rollout(self, state: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.full.rollout_states(state, actions), self.factorized.rollout_states(state, actions)

    def parameter_counts(self) -> dict[str, int]:
        count = lambda module: sum(parameter.numel() for parameter in module.parameters())
        return {"full": count(self.full), "factorized": count(self.factorized)}

    def save_checkpoint(self, path: Path) -> None:
        _save(self, path)

    @classmethod
    def load_checkpoint(cls, path: Path, map_location: str | torch.device = "cpu") -> "DynamicsDimWMHeads":
        return _load(path, map_location)


def parameter_counts_meta(spec: DynamicsDimHeadSpec) -> dict[str, int]:
    with torch.device("meta"):
        return _create(spec).parameter_counts()
