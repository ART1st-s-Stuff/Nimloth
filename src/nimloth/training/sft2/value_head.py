"""Value head for SFT2: state embedding -> per-action values."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS


class ValueHead(nn.Module):
    """Map WM state embeddings to scalar values for every navigation action."""

    def __init__(
        self,
        emb_dim: int,
        num_actions: int = NUM_NAVIGATION_ACTIONS,
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
        torch.save(self.state_dict(), path / "value_head.pt")

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        emb_dim: int,
        num_actions: int = NUM_NAVIGATION_ACTIONS,
        hidden_dim: int | None = None,
        map_location: str | torch.device = "cpu",
    ) -> "ValueHead":
        path = Path(path)
        module = cls(emb_dim=emb_dim, num_actions=num_actions, hidden_dim=hidden_dim)
        state_path = path / "value_head.pt"
        if state_path.is_file():
            module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module
