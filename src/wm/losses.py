"""WM 损失函数。"""

from __future__ import annotations

import torch
from torch import nn


def wm_cfm_loss(pred_velocity: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    """Flow matching 速度场监督损失。"""
    return nn.functional.mse_loss(pred_velocity, target_velocity)


def wm_reconstruction_loss(pred_z_next: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred_z_next, z_next)


def action_supervision_loss(pred_action: torch.Tensor, gt_action: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(pred_action, gt_action)


def sigreg_loss(
    latents: torch.Tensor,
    *,
    num_projections: int,
    num_quadrature_points: int,
    t_min: float,
    t_max: float,
    kernel_sigma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """基于随机投影 + Epps-Pulley 统计量的 SIGReg 近似实现。"""
    if latents.dim() < 2:
        raise ValueError("latents 至少需要二维张量。")
    feature_dim = int(latents.size(-1))
    flat = latents.reshape(-1, feature_dim)
    if flat.size(0) <= 1:
        return torch.zeros((), device=latents.device, dtype=latents.dtype)
    projections = max(1, int(num_projections))
    quadrature_points = max(2, int(num_quadrature_points))
    t_low = float(t_min)
    t_high = float(t_max)
    sigma = max(float(kernel_sigma), eps)

    directions = torch.randn(feature_dim, projections, device=latents.device, dtype=latents.dtype)
    directions = directions / torch.clamp(directions.norm(dim=0, keepdim=True), min=eps)
    projected = flat @ directions  # [N, M]

    t_grid = torch.linspace(t_low, t_high, quadrature_points, device=latents.device, dtype=latents.dtype)
    phase = projected.unsqueeze(0) * t_grid.view(-1, 1, 1)  # [T, N, M]
    phi_real = torch.cos(phase).mean(dim=1)
    phi_imag = torch.sin(phase).mean(dim=1)
    phi_standard = torch.exp(-0.5 * t_grid.square()).view(-1, 1)
    weights = torch.exp(-t_grid.square() / (2.0 * sigma * sigma)).view(-1, 1)
    diff_sq = (phi_real - phi_standard).square() + phi_imag.square()
    integrand = weights * diff_sq
    score_per_projection = torch.trapz(integrand, t_grid, dim=0)
    return score_per_projection.mean()

