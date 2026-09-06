"""Low-capacity state-interface calibration canary primitives.

The module never owns an actor or SFT1 projector.  It adds a bounded residual
from the frozen same-generation K16 actor hidden to the frozen SFT1 state, and
uses goal/outcome heads only as training supervision and diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class StateInterfaceCanaryConfig:
    hidden_dim: int = 2048
    state_dim: int = 1024
    grid_tokens: int = 16
    adapter_rank: int = 64
    goal_classes: int = 1
    movement_actions: tuple[int, ...] = (0, 2, 3)
    max_residual_fraction: float = 0.1

    def validate(self) -> None:
        for name in ("hidden_dim", "state_dim", "grid_tokens", "adapter_rank", "goal_classes"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if not self.movement_actions or len(set(self.movement_actions)) != len(self.movement_actions):
            raise ValueError("movement_actions must be unique and non-empty")
        if not 0.0 < float(self.max_residual_fraction) <= 0.25:
            raise ValueError("max_residual_fraction must be in (0, 0.25]")


class ResidualStateInterfaceCanary(nn.Module):
    """Bounded residual state adapter plus training-only unified-state heads."""

    def __init__(self, config: StateInterfaceCanaryConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.adapter = nn.Sequential(
            nn.LayerNorm(config.hidden_dim, elementwise_affine=False),
            nn.Linear(config.hidden_dim, config.adapter_rank, bias=False),
            nn.GELU(),
            nn.Linear(config.adapter_rank, config.state_dim, bias=True),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.goal_head = nn.Linear(config.state_dim, config.goal_classes)
        self.outcome_head = nn.Linear(
            config.grid_tokens * config.state_dim,
            len(config.movement_actions),
        )
        self._action_column = {
            int(action): index for index, action in enumerate(config.movement_actions)
        }

    def _validate_state_inputs(
        self,
        hidden: torch.Tensor,
        baseline_state: torch.Tensor,
    ) -> None:
        expected_hidden = (self.config.grid_tokens, self.config.hidden_dim)
        expected_state = (self.config.grid_tokens, self.config.state_dim)
        if hidden.ndim != 3 or tuple(hidden.shape[1:]) != expected_hidden:
            raise ValueError(
                f"hidden must have shape (B,{expected_hidden[0]},{expected_hidden[1]})"
            )
        if baseline_state.ndim != 3 or tuple(baseline_state.shape[1:]) != expected_state:
            raise ValueError(
                f"baseline_state must have shape (B,{expected_state[0]},{expected_state[1]})"
            )
        if len(hidden) != len(baseline_state):
            raise ValueError("hidden and baseline state batches differ")

    def calibrated_state(
        self,
        hidden: torch.Tensor,
        baseline_state: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_state_inputs(hidden, baseline_state)
        target_dtype = next(self.adapter.parameters()).dtype
        hidden = hidden.to(dtype=target_dtype)
        baseline = baseline_state.to(dtype=target_dtype)
        raw_delta = self.adapter(hidden)
        raw_norm = raw_delta.flatten(1).norm(dim=1, keepdim=True)
        baseline_norm = baseline.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-6)
        maximum = float(self.config.max_residual_fraction) * baseline_norm
        scale = torch.clamp(maximum / raw_norm.clamp_min(1e-12), max=1.0)
        delta = raw_delta * scale.view(-1, 1, 1)
        return baseline + delta

    def goal_logits(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or tuple(state.shape[1:]) != (
            self.config.grid_tokens,
            self.config.state_dim,
        ):
            raise ValueError("goal head requires the unified K16 state")
        return self.goal_head(state.mean(dim=1))

    def outcome_logits(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or tuple(state.shape[1:]) != (
            self.config.grid_tokens,
            self.config.state_dim,
        ):
            raise ValueError("outcome head requires the unified K16 state")
        action_values = [int(value) for value in actions.detach().cpu().tolist()]
        unsupported = sorted(set(action_values) - set(self._action_column))
        if unsupported:
            raise ValueError(f"unsupported movement action(s): {unsupported}")
        columns = torch.tensor(
            [self._action_column[action] for action in action_values],
            device=state.device,
            dtype=torch.long,
        )
        logits = self.outcome_head(state.flatten(1))
        return logits.gather(1, columns[:, None]).squeeze(1)


def grouped_record_selection(
    group_keys: np.ndarray,
    labels: np.ndarray,
    *,
    modulo: int = 5,
) -> np.ndarray:
    """Select whole image groups while retaining every goal class in fit."""

    groups = np.asarray(group_keys).astype(str)
    goals = np.asarray(labels).astype(str)
    if groups.ndim != 1 or goals.shape != groups.shape or modulo < 2:
        raise ValueError("invalid grouped record selection inputs")
    choices = {
        group: int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % modulo == 0
        for group in sorted(set(groups.tolist()))
    }
    selected = np.asarray([choices[group] for group in groups], dtype=np.bool_)
    for goal in sorted(set(goals.tolist())):
        if not np.any((goals == goal) & ~selected):
            affected_groups = set(groups[goals == goal].tolist())
            selected[np.isin(groups, sorted(affected_groups))] = False
    if not selected.any() or selected.all():
        raise ValueError("grouped state-interface selection split is empty")
    return selected


def visual_state_metrics(
    candidate_state: np.ndarray,
    dino_state: np.ndarray,
    baseline_state: np.ndarray,
) -> dict[str, float]:
    candidate = np.asarray(candidate_state, dtype=np.float64)
    dino = np.asarray(dino_state, dtype=np.float64)
    baseline = np.asarray(baseline_state, dtype=np.float64)
    if candidate.shape != dino.shape or candidate.shape != baseline.shape or candidate.ndim != 3:
        raise ValueError("candidate, DINO and baseline states must align")
    if not (np.isfinite(candidate).all() and np.isfinite(dino).all() and np.isfinite(baseline).all()):
        raise ValueError("visual metric states must be finite")

    def cosine(left: np.ndarray, right: np.ndarray) -> float:
        numerator = np.sum(left * right, axis=-1)
        denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
        return float(np.mean(numerator / np.maximum(denominator, 1e-12)))

    residual = candidate - baseline
    candidate_norm = np.linalg.norm(candidate.reshape(len(candidate), -1), axis=1)
    baseline_norm = np.linalg.norm(baseline.reshape(len(baseline), -1), axis=1)
    residual_fraction = np.linalg.norm(residual.reshape(len(residual), -1), axis=1) / np.maximum(
        baseline_norm, 1e-12
    )
    return {
        "count": int(len(candidate)),
        "candidate_dino_rmse": float(np.sqrt(np.mean(np.square(candidate - dino)))),
        "baseline_dino_rmse": float(np.sqrt(np.mean(np.square(baseline - dino)))),
        "candidate_dino_cosine": cosine(candidate, dino),
        "baseline_dino_cosine": cosine(baseline, dino),
        "candidate_state_std": float(candidate.std()),
        "baseline_state_std": float(baseline.std()),
        "candidate_baseline_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_fraction_mean": float(residual_fraction.mean()),
        "residual_fraction_max": float(residual_fraction.max()),
        "candidate_norm_mean": float(candidate_norm.mean()),
    }


def canary_gate(
    *,
    visual_metrics: dict[str, Any],
    goal_gate_passed: bool,
    outcome_checks: dict[str, bool],
    hidden_probe_supports_calibration: bool,
) -> dict[str, Any]:
    checks = {
        "hidden_probe_supports_calibration": bool(hidden_probe_supports_calibration),
        "visual_rmse_no_worse": float(visual_metrics["candidate_dino_rmse"])
        <= float(visual_metrics["baseline_dino_rmse"]),
        "visual_cosine_no_worse": float(visual_metrics["candidate_dino_cosine"])
        >= float(visual_metrics["baseline_dino_cosine"]),
        "residual_bound_respected": float(visual_metrics["residual_fraction_max"]) <= 0.10001,
        "goal_gate_passed": bool(goal_gate_passed),
        **{f"outcome_{name}": bool(value) for name, value in outcome_checks.items()},
    }
    return {**checks, "passed": bool(all(checks.values()))}


def save_canary_checkpoint(model: ResidualStateInterfaceCanary, path: Path) -> None:
    """Atomically save diagnostic adapter weights without optimizer state."""

    destination = Path(path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=False)
    torch.save(
        {key: value.detach().float().cpu() for key, value in model.adapter.state_dict().items()},
        temporary / "diagnostic_adapter.pt",
    )
    metadata = {
        "schema": "nimloth_state_interface_canary_adapter_v1",
        "config": asdict(model.config),
        "optimizer_state_present": False,
        "downstream_use_authorized": False,
        "state_formula": "z_calibrated = z_sft1 + bounded_zero_initialized_adapter(h_same_generation)",
    }
    (temporary / "canary_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def normalized_multitask_loss(
    *,
    model: ResidualStateInterfaceCanary,
    visual_hidden: torch.Tensor,
    visual_baseline: torch.Tensor,
    visual_dino: torch.Tensor,
    goal_hidden: torch.Tensor,
    goal_baseline: torch.Tensor,
    goal_labels: torch.Tensor,
    outcome_hidden: torch.Tensor,
    outcome_baseline: torch.Tensor,
    outcome_actions: torch.Tensor,
    outcome_labels: torch.Tensor,
    visual_reference_loss: float,
    goal_reference_loss: float,
    outcome_reference_loss: float,
    anchor_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dimensionless balanced visual/goal/outcome objective on one state interface."""

    visual_state = model.calibrated_state(visual_hidden, visual_baseline)
    goal_state = model.calibrated_state(goal_hidden, goal_baseline)
    outcome_state = model.calibrated_state(outcome_hidden, outcome_baseline)
    visual_loss = 1.0 - F.cosine_similarity(visual_state, visual_dino, dim=-1).mean()
    goal_loss = F.cross_entropy(model.goal_logits(goal_state), goal_labels)
    outcome_loss = F.binary_cross_entropy_with_logits(
        model.outcome_logits(outcome_state, outcome_actions),
        outcome_labels.to(dtype=torch.float32),
    )
    residual = visual_state - visual_baseline
    residual_energy = residual.square().mean()
    baseline_energy = visual_baseline.square().mean().clamp_min(1e-8)
    anchor = residual_energy / (
        float(model.config.max_residual_fraction) ** 2 * baseline_energy
    )
    components = {
        "visual": visual_loss / max(float(visual_reference_loss), 1e-6),
        "goal": goal_loss / max(float(goal_reference_loss), 1e-6),
        "outcome": outcome_loss / max(float(outcome_reference_loss), 1e-6),
        "anchor": anchor,
    }
    total = components["visual"] + components["goal"] + components["outcome"]
    total = total + float(anchor_weight) * components["anchor"]
    return total, components


__all__ = [
    "ResidualStateInterfaceCanary",
    "StateInterfaceCanaryConfig",
    "canary_gate",
    "grouped_record_selection",
    "normalized_multitask_loss",
    "save_canary_checkpoint",
    "visual_state_metrics",
]
