"""Action execution outcome readout from the existing predicted grid."""
from __future__ import annotations

import torch
from torch import nn


class ActionOutcomeHead(nn.Module):
    """Map (..., K, D) action-conditioned states to (...) success logits."""

    schema = "action_execution_success_bce_v1"

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.norm = nn.LayerNorm(self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, 1)

    def forward(self, predicted_grid: torch.Tensor) -> torch.Tensor:
        if predicted_grid.ndim < 3 or predicted_grid.shape[-1] != self.emb_dim:
            raise ValueError("outcome head expects (..., K, emb_dim) predicted grids")
        return self.linear(self.norm(predicted_grid.float()).mean(dim=-2)).squeeze(-1)
