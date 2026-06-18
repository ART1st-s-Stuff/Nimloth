"""Map Qwen hidden states into WM predictor embedding space."""

from __future__ import annotations

from torch import nn


class StateProjector(nn.Module):
    """Linear (or MLP) bridge: Qwen latent dim -> LeWM emb dim."""

    def __init__(self, qwen_hidden_dim: int, lewm_emb_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or lewm_emb_dim
        if hidden == lewm_emb_dim:
            self.net = nn.Linear(qwen_hidden_dim, lewm_emb_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(qwen_hidden_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, lewm_emb_dim),
            )

    def forward(self, hidden):
        weight = self.net.weight if hasattr(self.net, "weight") else self.net[0].weight
        return self.net(hidden.to(dtype=weight.dtype))
