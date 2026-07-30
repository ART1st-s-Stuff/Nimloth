"""Strict SFT2 checkpoint contract for pre-RL MCTS rollout evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE


@dataclass(frozen=True)
class SFT2MCTSEvaluationContract:
    checkpoint: Path
    history_size: int
    prediction_horizon: int
    action_count: int
    step: int
    epoch: int

    @property
    def wm_checkpoint(self) -> Path:
        return self.checkpoint / "wm_predictor"

    @property
    def state_proj_checkpoint(self) -> Path:
        return self.checkpoint / "state_proj.pt"

    @property
    def value_head_checkpoint(self) -> Path:
        return self.checkpoint / "value_head"


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"SFT2 checkpoint {name} must be a mapping")
    return value


def load_sft2_mcts_evaluation_contract(
    checkpoint: Path,
) -> SFT2MCTSEvaluationContract:
    """Read H/K from one complete SFT2 checkpoint and reject other semantics."""

    checkpoint = Path(checkpoint).resolve()
    required = (
        checkpoint / "config.json",
        checkpoint / "training_state.pt",
        checkpoint / "state_proj.pt",
        checkpoint / "wm_predictor" / "config.json",
        checkpoint / "wm_predictor" / "predictor.pt",
        checkpoint / "value_head" / "value_head.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "incomplete full-HF SFT2 checkpoint for MCTS evaluation; "
            f"missing={missing}"
        )

    training_state = _require_mapping(
        torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
            mmap=True,
        ),
        name="training_state",
    )
    if not bool(training_state.get("epoch_complete", False)):
        raise ValueError("pre-RL evaluation requires an epoch-complete SFT2 checkpoint")
    invariants = _require_mapping(
        training_state.get("training_invariants"),
        name="training_invariants",
    )
    if invariants.get("objective") != "dino_grid":
        raise ValueError(
            "MCTS evaluation requires the DINO-grid SFT2 objective; "
            f"got {invariants.get('objective')!r}"
        )
    if invariants.get("value_objective") != SFT2_VALUE_OBJECTIVE:
        raise ValueError(
            "MCTS evaluation requires the decision-state executed-action "
            f"value objective {SFT2_VALUE_OBJECTIVE!r}; "
            f"got {invariants.get('value_objective')!r}"
        )

    history_size = int(invariants.get("history_size", 0))
    prediction_horizon = int(invariants.get("prediction_horizon", 0))
    if history_size != 1:
        raise ValueError(
            "pre-RL MCTS evaluation requires SFT2 history_size=1; "
            f"got H={history_size}"
        )
    if prediction_horizon < 1:
        raise ValueError(
            "SFT2 checkpoint prediction_horizon must be positive; "
            f"got K={prediction_horizon}"
        )

    predictor_config = _require_mapping(
        json.loads(
            (checkpoint / "wm_predictor" / "config.json").read_text(
                encoding="utf-8"
            )
        ),
        name="wm_predictor config",
    )
    predictor_history = int(predictor_config.get("history_size", 0))
    if predictor_history != history_size:
        raise ValueError(
            "SFT2 training invariants and WM predictor history_size disagree: "
            f"invariants={history_size}, predictor={predictor_history}"
        )
    action_count = int(predictor_config.get("action_dim", 0))
    if action_count < 2:
        raise ValueError(
            f"SFT2 WM predictor action_dim must be at least 2, got {action_count}"
        )
    value_head_state = _require_mapping(
        torch.load(
            checkpoint / "value_head" / "value_head.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        ),
        name="value_head state",
    )
    output_weight = value_head_state.get("net.2.weight")
    if not isinstance(output_weight, torch.Tensor) or output_weight.ndim != 2:
        raise ValueError(
            "SFT2 ValueHead checkpoint is missing the final action projection"
        )
    value_action_count = int(output_weight.shape[0])
    if value_action_count != action_count:
        raise ValueError(
            "SFT2 WM predictor and ValueHead action counts disagree: "
            f"predictor={action_count}, value_head={value_action_count}"
        )

    return SFT2MCTSEvaluationContract(
        checkpoint=checkpoint,
        history_size=history_size,
        prediction_horizon=prediction_horizon,
        action_count=action_count,
        step=int(training_state.get("step", -1)),
        epoch=int(training_state.get("epoch", -1)),
    )


__all__ = [
    "SFT2MCTSEvaluationContract",
    "load_sft2_mcts_evaluation_contract",
]
