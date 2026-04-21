"""CFM 世界模型最小实现。"""

from __future__ import annotations

import torch
from torch import nn


class CFMWorldModel(nn.Module):
    """输入 (z_t, a_t) 输出速度场 v。"""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, action], dim=-1)
        return self.net(x)

