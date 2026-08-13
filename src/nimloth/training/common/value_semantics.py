"""Lightweight checkpoint contract for outgoing action-value semantics."""

from __future__ import annotations

from pathlib import Path

import torch

SFT2_VALUE_OBJECTIVE = "decision_state_executed_action_mc_v3"
PLANNER_TRAINING_OBJECTIVE = "receding_horizon_decision_state_ppo_value_v1"
PLANNER_POLICY_TRAINING_OBJECTIVE = "receding_horizon_planner_policy_ppo_v1"


def validate_planning_value_semantics(
    *,
    wm_checkpoint: Path,
    state_proj_checkpoint: Path,
    value_head_checkpoint: Path,
) -> Path:
    """Reject component sets whose action slots were trained on successor states."""

    wm_checkpoint = Path(wm_checkpoint).resolve()
    state_proj_checkpoint = Path(state_proj_checkpoint).resolve()
    value_head_checkpoint = Path(value_head_checkpoint).resolve()
    roots = {
        wm_checkpoint.parent,
        state_proj_checkpoint.parent,
        value_head_checkpoint.parent,
    }
    if len(roots) != 1:
        raise ValueError(
            "planning components must share one checkpoint root so value semantics "
            "can be verified"
        )
    root = roots.pop()
    sft2_state_path = root / "training_state.pt"
    rl_state_path = root / "rl_state.pt"
    if sft2_state_path.is_file() and rl_state_path.is_file():
        raise ValueError(
            "planning component root has ambiguous value semantics: both "
            f"{sft2_state_path.name} and {rl_state_path.name} exist"
        )
    if sft2_state_path.is_file():
        state = torch.load(
            sft2_state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        invariants = state.get("training_invariants") if isinstance(state, dict) else None
        actual = invariants.get("value_objective") if isinstance(invariants, dict) else None
        if actual != SFT2_VALUE_OBJECTIVE:
            raise ValueError(
                "planning checkpoint uses an incompatible SFT2 value objective: "
                f"expected={SFT2_VALUE_OBJECTIVE!r}, actual={actual!r}"
            )
        return root
    if rl_state_path.is_file():
        state = torch.load(
            rl_state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        actual = state.get("planner_training_objective") if isinstance(state, dict) else None
        compatible_objectives = {
            PLANNER_TRAINING_OBJECTIVE,
            PLANNER_POLICY_TRAINING_OBJECTIVE,
        }
        if actual not in compatible_objectives:
            raise ValueError(
                "planning checkpoint uses an incompatible RL value objective: "
                f"expected one of {sorted(compatible_objectives)!r}, actual={actual!r}"
            )
        return root
    raise FileNotFoundError(
        "planning component root has no training_state.pt or rl_state.pt; "
        f"cannot verify outgoing Q(s,a) semantics: {root}"
    )


__all__ = [
    "PLANNER_POLICY_TRAINING_OBJECTIVE",
    "PLANNER_TRAINING_OBJECTIVE",
    "SFT2_VALUE_OBJECTIVE",
    "validate_planning_value_semantics",
]
