"""Prepare behavior-matched complete-objective rows for Planner VERL/FSDP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from nimloth.training.rl.episodes import ExecutedTransition
from nimloth.training.rl.planner_verl_adapter import build_planner_update_dataproto
from nimloth.wm import PlannerPolicyHead, ValueHead


@dataclass(frozen=True)
class PreparedPlannerRow:
    """One real transition with frozen behavior-checkpoint statistics."""

    transition: ExecutedTransition
    return_target: torch.Tensor
    old_action_value: torch.Tensor
    old_policy_log_prob: torch.Tensor
    policy_advantage: torch.Tensor
    behavior_matched: bool
    diagnostic_only: bool

    def __post_init__(self) -> None:
        if self.behavior_matched == self.diagnostic_only:
            raise ValueError(
                "planner row must be either behavior-matched training data or "
                "an explicit nonbehavior diagnostic"
            )


def load_planner_behavior_heads(
    *,
    value_head_checkpoint: Path,
    planner_policy_head_checkpoint: Path,
    emb_dim: int,
) -> tuple[ValueHead, PlannerPolicyHead]:
    """Load only the small frozen heads needed to reconstruct old statistics."""

    if emb_dim < 1:
        raise ValueError("planner behavior head emb_dim must be positive")
    value_head = ValueHead.load_checkpoint(
        value_head_checkpoint,
        emb_dim=emb_dim,
        map_location="cpu",
    ).requires_grad_(False).eval()
    policy_head = PlannerPolicyHead.load_checkpoint(
        planner_policy_head_checkpoint,
        emb_dim=emb_dim,
        map_location="cpu",
    ).requires_grad_(False).eval()
    return value_head, policy_head


@torch.no_grad()
def _prepare_planner_head_row(
    transition: ExecutedTransition,
    *,
    return_target: torch.Tensor,
    value_head: ValueHead,
    planner_policy_head: PlannerPolicyHead,
    temperature: float,
    require_behavior_match: bool,
) -> PreparedPlannerRow:
    if temperature <= 0.0:
        raise ValueError("planner policy temperature must be positive")
    state = transition.rollout_decision_state().unsqueeze(0)
    if state.ndim != 3:
        raise ValueError(
            "planner behavior state must have shape (1, slots, emb_dim), "
            f"got {tuple(state.shape)}"
        )
    pooled = state.mean(dim=-2)
    action_values = value_head(pooled).float().squeeze(0)
    logits = planner_policy_head(pooled).float().squeeze(0)
    old_log_probs = torch.log_softmax(logits / temperature, dim=-1)
    if require_behavior_match:
        stored_log_probs = transition.behavior_action_log_probs().to(
            dtype=old_log_probs.dtype
        )
        if old_log_probs.shape != stored_log_probs.shape or not torch.allclose(
            old_log_probs,
            stored_log_probs,
            rtol=1e-5,
            atol=1e-6,
        ):
            maximum_error = (
                float((old_log_probs - stored_log_probs).abs().max().item())
                if old_log_probs.shape == stored_log_probs.shape
                else float("inf")
            )
            raise ValueError(
                "fresh rollout PlannerPolicyHead log-probs do not match the "
                f"behavior checkpoint: max_error={maximum_error}"
            )
    selected = int(transition.action_index)
    state_value = (old_log_probs.exp() * action_values).sum()
    target = return_target.detach().reshape(()).float().cpu()
    if not torch.isfinite(target):
        raise ValueError("planner return target must be finite")
    return PreparedPlannerRow(
        transition=transition,
        return_target=target,
        old_action_value=action_values[selected].detach().cpu(),
        old_policy_log_prob=old_log_probs[selected].detach().cpu(),
        policy_advantage=(target - state_value.detach().cpu()),
        behavior_matched=require_behavior_match,
        diagnostic_only=not require_behavior_match,
    )


def prepare_planner_behavior_row(
    transition: ExecutedTransition,
    *,
    return_target: torch.Tensor,
    value_head: ValueHead,
    planner_policy_head: PlannerPolicyHead,
    temperature: float,
) -> PreparedPlannerRow:
    """Reconstruct statistics and require the persisted behavior policy to match."""

    return _prepare_planner_head_row(
        transition,
        return_target=return_target,
        value_head=value_head,
        planner_policy_head=planner_policy_head,
        temperature=temperature,
        require_behavior_match=True,
    )


def prepare_planner_nonbehavior_diagnostic_row(
    transition: ExecutedTransition,
    *,
    return_target: torch.Tensor,
    value_head: ValueHead,
    planner_policy_head: PlannerPolicyHead,
    temperature: float,
) -> PreparedPlannerRow:
    """Build current-head targets for a real prefix used only as a memory probe.

    The transition remains actual recorded data, including its real CoT and state.
    Its behavior checkpoint differs from the initialized policy, so this row must
    never be consumed, checkpointed, or represented as a valid PPO training row.
    """

    return _prepare_planner_head_row(
        transition,
        return_target=return_target,
        value_head=value_head,
        planner_policy_head=planner_policy_head,
        temperature=temperature,
        require_behavior_match=False,
    )


def build_planner_rank_rounds(
    rows: Sequence[PreparedPlannerRow],
    *,
    token_counts: Sequence[int],
    dino_grid_targets: Sequence[torch.Tensor],
    world_size: int,
    provisional_update_id: str,
) -> tuple[tuple[object, ...], ...]:
    """Round-robin real rows into equal one-row FSDP rounds with zero padding."""

    prepared = tuple(rows)
    tokens = tuple(int(value) for value in token_counts)
    dino = tuple(dino_grid_targets)
    if not prepared:
        raise ValueError("planner rank rounds require real transitions")
    if any(not row.behavior_matched or row.diagnostic_only for row in prepared):
        raise ValueError(
            "transactional planner rank rounds reject nonbehavior diagnostics"
        )
    if world_size < 2:
        raise ValueError("planner rank rounds require distributed world_size >= 2")
    if len(tokens) != len(prepared) or len(dino) != len(prepared):
        raise ValueError("planner token/DINO rows must align with transitions")
    if any(value < 1 for value in tokens):
        raise ValueError("planner token counts must be positive")
    if any(not isinstance(value, torch.Tensor) for value in dino):
        raise ValueError("planner DINO rows must be tensors")

    rounds: list[tuple[object, ...]] = []
    total = len(prepared)
    for offset in range(0, total, world_size):
        rank_batches: list[object] = []
        for rank in range(world_size):
            index = offset + rank
            is_padding = index >= total
            source_index = 0 if is_padding else index
            row = prepared[source_index]
            rank_batches.append(
                build_planner_update_dataproto(
                    transitions=(row.transition,),
                    return_targets=(row.return_target,),
                    old_action_values=(row.old_action_value,),
                    old_policy_log_probs=(row.old_policy_log_prob,),
                    policy_advantages=(row.policy_advantage,),
                    loss_weights=(0.0 if is_padding else float(world_size),),
                    token_counts=(tokens[source_index],),
                    total_transitions=total,
                    update_id=provisional_update_id,
                    dino_grid_targets=(dino[source_index],),
                    behavior_matched=True,
                    diagnostic_only=False,
                )
            )
        rounds.append(tuple(rank_batches))
    return tuple(rounds)


def build_replicated_planner_gate_round(
    row: PreparedPlannerRow,
    *,
    token_count: int,
    dino_grid_target: torch.Tensor,
    world_size: int,
    provisional_update_id: str,
    allow_nonbehavior_diagnostic: bool = False,
) -> tuple[tuple[object, ...], ...]:
    """Replicate one real row so every nested FSDP rank executes the same graph."""

    if world_size < 2:
        raise ValueError("planner FSDP gate requires world_size >= 2")
    if row.diagnostic_only and not allow_nonbehavior_diagnostic:
        raise ValueError(
            "nonbehavior row requires an explicit diagnostic gate allowance"
        )
    batches = tuple(
        build_planner_update_dataproto(
            transitions=(row.transition,),
            return_targets=(row.return_target,),
            old_action_values=(row.old_action_value,),
            old_policy_log_probs=(row.old_policy_log_prob,),
            policy_advantages=(row.policy_advantage,),
            loss_weights=(1.0,),
            token_counts=(int(token_count),),
            total_transitions=1,
            update_id=provisional_update_id,
            dino_grid_targets=(dino_grid_target,),
            behavior_matched=row.behavior_matched,
            diagnostic_only=row.diagnostic_only,
        )
        for _ in range(world_size)
    )
    return (batches,)


__all__ = [
    "PreparedPlannerRow",
    "build_planner_rank_rounds",
    "build_replicated_planner_gate_round",
    "load_planner_behavior_heads",
    "prepare_planner_behavior_row",
    "prepare_planner_nonbehavior_diagnostic_row",
]
