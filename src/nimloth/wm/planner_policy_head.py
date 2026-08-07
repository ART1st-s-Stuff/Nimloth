"""Planner policy head: WM state embedding -> per-action logits."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from nimloth.wm.lewm import DEFAULT_ACTION_COUNT


class PlannerPolicyHead(nn.Module):
    """Map WM state embeddings to logits for every navigation action."""

    def __init__(
        self,
        emb_dim: int,
        num_actions: int = DEFAULT_ACTION_COUNT,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        weight = self.net[0].weight
        return self.net(state_emb.to(dtype=weight.dtype))

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "planner_policy_head.pt")

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        emb_dim: int,
        num_actions: int = DEFAULT_ACTION_COUNT,
        hidden_dim: int | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "PlannerPolicyHead":
        path = Path(path)
        state_path = path / "planner_policy_head.pt"
        if not state_path.is_file():
            raise FileNotFoundError(
                f"missing PlannerPolicyHead checkpoint: {state_path}"
            )
        state = torch.load(state_path, map_location=map_location, weights_only=True)
        first = state.get("net.0.weight")
        last = state.get("net.2.weight")
        if (
            first is None
            or last is None
            or first.ndim != 2
            or last.ndim != 2
            or int(first.shape[1]) != emb_dim
            or int(last.shape[0]) != num_actions
            or int(last.shape[1]) != int(first.shape[0])
        ):
            raise ValueError("PlannerPolicyHead checkpoint architecture mismatch")
        inferred_hidden_dim = int(first.shape[0])
        if hidden_dim is not None and hidden_dim != inferred_hidden_dim:
            raise ValueError("PlannerPolicyHead checkpoint hidden_dim mismatch")
        module = cls(
            emb_dim=emb_dim,
            num_actions=num_actions,
            hidden_dim=inferred_hidden_dim,
        )
        module.load_state_dict(state)
        return module


__all__ = ["PlannerPolicyHead"]
