"""WM 损失函数。"""

from __future__ import annotations

import torch
from torch import nn


def wm_cfm_loss(velocity: torch.Tensor, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    target_velocity = z_next - z_t
    return nn.functional.mse_loss(velocity, target_velocity)

