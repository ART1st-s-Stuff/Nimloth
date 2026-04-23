"""动作映射层：将 IDM 动作空间映射到数据动作空间。"""

from __future__ import annotations

import torch
from torch import nn


class ActionMapper(nn.Module):
    """三层 MLP 映射器，带 LayerNorm 与 GELU。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_action_mapper(input_dim: int, output_dim: int, hidden_dim: int) -> ActionMapper:
    return ActionMapper(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)

