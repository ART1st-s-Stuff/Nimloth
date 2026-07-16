"""Value head: WM state embedding -> per-action values."""

from __future__ import annotations

import json
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
        config_path = path / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            saved_emb_dim = int(config.get("emb_dim", emb_dim))
            saved_num_actions = int(config.get("num_actions", num_actions))
            saved_hidden_dim = config.get("hidden_dim")
            if saved_emb_dim != int(emb_dim) or saved_num_actions != int(num_actions):
                raise ValueError(
                    "ValueHead checkpoint shape metadata mismatch: "
                    f"emb_dim={saved_emb_dim}/{emb_dim}, "
                    f"num_actions={saved_num_actions}/{num_actions}"
                )
            if hidden_dim is None and saved_hidden_dim is not None:
                hidden_dim = int(saved_hidden_dim)
            elif (
                hidden_dim is not None
                and saved_hidden_dim is not None
                and int(hidden_dim) != int(saved_hidden_dim)
            ):
                raise ValueError(
                    "ValueHead hidden_dim mismatch: "
                    f"checkpoint={saved_hidden_dim}, requested={hidden_dim}"
                )
        module = cls(emb_dim=emb_dim, num_actions=num_actions, hidden_dim=hidden_dim)
        state_path = path / "value_head.pt"
        if state_path.is_file():
            module.load_state_dict(torch.load(state_path, map_location=map_location, weights_only=True))
        return module
