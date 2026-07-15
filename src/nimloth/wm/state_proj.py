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
    """LeWM-style MLP bridge from Qwen latent query states to WM emb.

    ``latent_token_count`` controls how many Qwen hidden vectors form one raw
    state. For ``latent_token_count > 1`` the input ``(B, k, H)`` is flattened to
    ``(B, k*H)`` before projection. Both the projector hidden width and downstream
    WM State width are explicit so capacity ablations do not retain an accidental
    narrow hidden bottleneck.
    """

    def __init__(
        self,
        qwen_hidden_dim: int,
        lewm_emb_dim: int,
        projector_hidden_dim: int = 2048,
        *,
        latent_token_count: int = 1,
    ) -> None:
        super().__init__()
        if latent_token_count < 1:
            raise ValueError(f"latent_token_count must be >= 1, got {latent_token_count}")
        self.qwen_hidden_dim = int(qwen_hidden_dim)
        self.latent_token_count = int(latent_token_count)
        self.input_dim = self.qwen_hidden_dim * self.latent_token_count
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.output_dim = int(lewm_emb_dim)
        # LayerNorm avoids inplace running-buffer conflicts when state_proj is
        # called multiple times before backward (e.g. WM + value losses).
        self.net = MLP(
            self.input_dim,
            self.projector_hidden_dim,
            self.output_dim,
            norm_fn=nn.LayerNorm,
        )

    def _flatten_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 2:
            if hidden.shape[-1] != self.input_dim:
                if self.latent_token_count == 1 and hidden.shape[-1] == self.qwen_hidden_dim:
                    return hidden
                raise ValueError(
                    f"StateProjector expected last dim {self.input_dim}, got {tuple(hidden.shape)}"
                )
            return hidden
        if hidden.ndim == 3:
            if hidden.shape[1] != self.latent_token_count or hidden.shape[2] != self.qwen_hidden_dim:
                raise ValueError(
                    "StateProjector expected hidden shape "
                    f"(B, {self.latent_token_count}, {self.qwen_hidden_dim}), got {tuple(hidden.shape)}"
                )
            return hidden.reshape(hidden.shape[0], self.input_dim)
        raise ValueError(f"hidden must have shape (B, D) or (B, k, D), got {tuple(hidden.shape)}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project Qwen hidden state(s) to WM embedding space.

        Args:
            hidden: ``(B, qwen_hidden_dim)`` for k=1, or
                ``(B, latent_token_count, qwen_hidden_dim)`` for k>1.

        Returns:
            ``(B, lewm_emb_dim)`` in network weight dtype.
        """
        hidden = self._flatten_hidden(hidden)
        target_dtype = next(self.parameters()).dtype
        return self.net(hidden.to(dtype=target_dtype))
