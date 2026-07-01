"""Map Qwen hidden states into WM predictor embedding space.

Uses LeWM MLP projection (Linear → LayerNorm → GELU → Linear) for a
stable, non-linear bridge from Qwen's hidden state to the WM embedding.
LayerNorm avoids inplace running-buffer conflicts that BatchNorm causes
when called multiple times before backward.
"""

from __future__ import annotations

import torch
from torch import nn

from nimloth.wm._vendor_lewm import MLP


class StateProjector(nn.Module):
    """LeWM-style MLP bridge: Qwen latent dim -> WM emb dim.

    Uses LeWM MLP with BatchNorm1d, matching the LeWM paper's projection
    structure.  Default ``projector_hidden_dim=2048`` provides reasonable
    capacity for the qwen_hidden_dim→emb_dim mapping.
    """

    def __init__(
        self,
        qwen_hidden_dim: int,
        lewm_emb_dim: int,
        projector_hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        # LayerNorm avoids inplace running-buffer conflicts when state_proj is
        # called multiple times before backward (e.g. WM + value losses).
        self.net = MLP(
            qwen_hidden_dim,
            projector_hidden_dim,
            lewm_emb_dim,
            norm_fn=nn.LayerNorm,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project Qwen hidden state to WM embedding space.

        Args:
            hidden: (B, qwen_hidden_dim) in any dtype.

        Returns:
            (B, lewm_emb_dim) in network weight dtype.
        """
        target_dtype = next(self.parameters()).dtype
        return self.net(hidden.to(dtype=target_dtype))
