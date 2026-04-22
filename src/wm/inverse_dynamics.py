"""逆动力学模型。"""

from __future__ import annotations

import torch
from torch import nn


class InverseDynamicsModel(nn.Module):
    """从历史latent序列预测当前动作。"""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        history_len: int,
        num_patches: int,
        token_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.history_len = history_len
        self.num_patches = int(num_patches)
        self.token_dim = int(token_dim)
        expected_latent_dim = self.num_patches * self.token_dim
        if int(self.latent_dim) != int(expected_latent_dim):
            raise ValueError(f"latent_dim 与 patch 配置不一致: {latent_dim} != {expected_latent_dim}")
        self.patch_token_proj = nn.Linear(self.token_dim, hidden_dim)
        self.patch_pool = nn.Linear(hidden_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, history_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def _validate_inputs(self, z_history: torch.Tensor) -> None:
        if z_history.dim() != 4:
            raise ValueError(f"z_history 形状不合法，期望 [B,H,P,D]，实际 {tuple(z_history.shape)}")
        if z_history.size(1) != self.history_len:
            raise ValueError(f"history_len 不一致: {z_history.size(1)} != {self.history_len}")
        if z_history.size(2) != self.num_patches:
            raise ValueError(f"num_patches 不一致: {z_history.size(2)} != {self.num_patches}")
        if z_history.size(3) != self.token_dim:
            raise ValueError(f"token_dim 不一致: {z_history.size(3)} != {self.token_dim}")

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(z_history=z_history)
        patch_hidden = self.patch_token_proj(z_history)
        pooled = self.patch_pool(patch_hidden).mean(dim=2)
        x = pooled + self.pos_embedding[:, : pooled.size(1), :]
        hidden = self.encoder(x)
        return self.head(hidden[:, -1, :])
