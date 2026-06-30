"""Action selection via WM predictor + value head planning.

Algorithms selectable via config:
- ``greedy``: argmax of ValueHead(state_emb) — no search.
- ``beam_search``: K-beam rollout using predictor + value head scoring.
- ``mcts`` (future): MCTS with predictor dynamics and value head leaf eval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nimloth.wm.lewm import LeWMConfig
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.value_head import ValueHead


# ---------------------------------------------------------------------------
# Planner config
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    algorithm: str = "greedy"  # "greedy" | "beam_search" | "mcts"
    beam_width: int = 4
    rollout_depth: int = 4
    num_simulations: int = 100  # MCTS only (future)
    num_actions: int = 8
    discount: float = 0.99

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PlannerConfig":
        if d is None:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Greedy planner
# ---------------------------------------------------------------------------


def greedy_action(
    state_emb: torch.Tensor,
    value_head: ValueHead,
    *,
    num_actions: int = 8,
) -> torch.Tensor:
    """Select the action with the highest predicted value.

    Args:
        state_emb: (B, emb_dim) — WM latent state.
        value_head: WM value head.
        num_actions: number of discrete actions.

    Returns:
        (B,) int64 — selected action indices.
    """
    values = value_head(state_emb).float()  # (B, num_actions)
    return values.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Beam search planner
# ---------------------------------------------------------------------------


def beam_search_action(
    state_emb: torch.Tensor,
    predictor: LatentWMPredictor,
    value_head: ValueHead,
    *,
    beam_width: int = 4,
    rollout_depth: int = 4,
    num_actions: int = 8,
    discount: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-beam search in WM latent space.

    At each step, expand the current beam by all actions, score resulting
    states, and keep the top-K.

    Args:
        state_emb: (B, emb_dim) — initial WM latent state.
        predictor: WM autoregressive predictor.
        value_head: WM value head.
        beam_width: number of sequences to keep (K).
        rollout_depth: number of steps to look ahead.
        num_actions: number of discrete actions.

    Returns:
        (action, score) where:
          - action: (B,) int64 — first action of the best sequence.
          - score:  (B,) float32 — total value of the best sequence.
    """
    # Handle (emb_dim,) single-state input gracefully.
    single = state_emb.ndim == 1
    if single:
        state_emb = state_emb.unsqueeze(0)  # (1, emb_dim)

    B = state_emb.shape[0]
    device = state_emb.device

    # Beam state: each element is (state, action_sequence, score)
    # Start with empty sequences, all at the initial state.
    beam_states = state_emb.unsqueeze(1).expand(-1, beam_width, -1).reshape(B * beam_width, -1)
    beam_seqs = torch.zeros(B * beam_width, 0, dtype=torch.long, device=device)
    beam_scores = torch.zeros(B * beam_width, device=device)

    # Action vocabulary: every possible action at each expansion.
    all_actions = torch.arange(num_actions, device=device, dtype=torch.long)  # (num_actions,)

    for step in range(rollout_depth):
        K = beam_states.shape[0] // B  # current beam width per batch item
        # Expand each beam element by all actions.
        # beam_states: (B*K, emb_dim) → repeat to (B*K, num_actions, emb_dim)
        expanded_states = beam_states.unsqueeze(1).expand(-1, num_actions, -1).reshape(B * K * num_actions, -1)
        expanded_actions = all_actions.unsqueeze(0).expand(B * K, -1).reshape(-1)  # (B*K*num_actions,)
        expanded_seqs = (
            beam_seqs.unsqueeze(1)
            .expand(-1, num_actions, -1)
            .reshape(B * K * num_actions, -1)
        )  # (B*K*num_actions, step) — existing sequences
        expanded_scores = beam_scores.unsqueeze(1).expand(-1, num_actions).reshape(-1)  # (B*K*num_actions,)

        # Predict next states and score them.
        next_states = predictor.predict_next_emb(expanded_states, expanded_actions)  # (B*K*num_actions, emb_dim)
        step_values = value_head(next_states).float()  # (B*K*num_actions, num_actions)
        # Score = value of the *taken* action at this step.
        step_scores = step_values.gather(1, expanded_actions.unsqueeze(1)).squeeze(1)  # (B*K*num_actions,)
        # Discount future values: step 0 = undiscounted, later steps progressively discounted.
        step_scores = step_scores * (discount ** (step + 1))

        total_scores = expanded_scores + step_scores  # (B*K*num_actions,)

        # Build new sequences: old seq + this step's action.
        new_seqs = torch.cat([expanded_seqs, expanded_actions.unsqueeze(1)], dim=1)  # (B*K*num_actions, step+1)

        # Reshape per batch item and keep top-K.
        total_scores = total_scores.reshape(B, K * num_actions)
        new_seqs = new_seqs.reshape(B, K * num_actions, step + 1)
        next_states = next_states.reshape(B, K * num_actions, -1)

        topk_scores, topk_idx = total_scores.topk(beam_width, dim=-1)  # (B, K), (B, K)

        # Gather top-K states, sequences, and scores.
        beam_states = torch.gather(
            next_states, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, next_states.shape[-1]),
        ).reshape(B * beam_width, -1)
        beam_seqs = torch.gather(
            new_seqs, 1,
            topk_idx.unsqueeze(-1).expand(-1, -1, step + 1),
        ).reshape(B * beam_width, step + 1)
        beam_scores = topk_scores.reshape(B * beam_width)

    # Return first action and total score of the best beam.
    best_scores, best_idx = beam_scores.reshape(B, beam_width).max(dim=-1)  # (B,)
    best_actions = beam_seqs.reshape(B, beam_width, rollout_depth)[
        torch.arange(B, device=device), best_idx, 0
    ]  # (B,)

    if single:
        best_actions = best_actions.squeeze(0)
        best_scores = best_scores.squeeze(0)
    return best_actions, best_scores


# ---------------------------------------------------------------------------
# Planner dispatch
# ---------------------------------------------------------------------------


class Planner:
    """Unified action-selection interface, config-driven.

    Usage::

        planner = Planner(config, predictor=predictor, value_head=value_head)
        action = planner.select_action(state_emb)  # (B,) int64
    """

    def __init__(
        self,
        config: PlannerConfig | dict[str, Any] | None = None,
        *,
        predictor: LatentWMPredictor | None = None,
        value_head: ValueHead | None = None,
    ) -> None:
        if config is None:
            config = PlannerConfig()
        if isinstance(config, dict):
            config = PlannerConfig.from_dict(config)

        if config.algorithm not in ("greedy", "beam_search", "mcts"):
            raise ValueError(f"Unknown planner algorithm: {config.algorithm!r}")
        if value_head is None:
            raise ValueError("Planner.value_head is required for all algorithms")
        if config.algorithm == "beam_search" and predictor is None:
            raise ValueError("Planner.predictor is required for beam_search")
        if config.algorithm == "mcts":
            raise NotImplementedError("MCTS planner is not yet implemented")

        self._cfg = config
        self._predictor = predictor
        self._value_head = value_head

    def select_action(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Select action(s) for the given WM state(s).

        Args:
            state_emb: (B, emb_dim) — WM latent state.

        Returns:
            (B,) int64 — selected action indices.
        """
        if self._cfg.algorithm == "beam_search":
            actions, _scores = beam_search_action(
                state_emb,
                self._predictor,  # type: ignore[arg-type]  # validated in __init__
                self._value_head,  # type: ignore[arg-type]
                beam_width=self._cfg.beam_width,
                rollout_depth=self._cfg.rollout_depth,
                num_actions=self._cfg.num_actions,
                discount=self._cfg.discount,
            )
            return actions
        # greedy (default)
        return greedy_action(
            state_emb,
            self._value_head,  # type: ignore[arg-type]
            num_actions=self._cfg.num_actions,
        )
