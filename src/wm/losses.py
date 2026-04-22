"""WM 损失函数。"""

from __future__ import annotations

import torch
from torch import nn


def wm_cfm_loss(velocity: torch.Tensor, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    target_velocity = z_next - z_t
    return nn.functional.mse_loss(velocity, target_velocity)


def wm_reconstruction_loss(pred_z_next: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred_z_next, z_next)


def action_supervision_loss(pred_action: torch.Tensor, gt_action: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred_action, gt_action)

