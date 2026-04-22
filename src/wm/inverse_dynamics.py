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
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
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

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(z_history) + self.pos_embedding[:, : z_history.size(1), :]
        hidden = self.encoder(x)
        return self.head(hidden[:, -1, :])
