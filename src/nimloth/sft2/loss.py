"""Loss functions for SFT2 (WM MSE + optional LM CE)."""

from __future__ import annotations

import math

import torch
from torch import nn

from nimloth.wm.lewm import LeWMWrapper


class StateProjector(nn.Module):
    """Map Qwen hidden states into LeWM embedding space."""

    def __init__(self, qwen_hidden_dim: int, lewm_emb_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = hidden_dim or lewm_emb_dim
        if hidden == lewm_emb_dim:
            self.net = nn.Linear(qwen_hidden_dim, lewm_emb_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(qwen_hidden_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, lewm_emb_dim),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


def compute_wm_alignment_loss(
    *,
    qwen_hidden_at_latent: torch.Tensor,
    action_indices: torch.Tensor,
    next_pixels: torch.Tensor,
    state_proj: StateProjector,
    lewm: LeWMWrapper,
) -> tuple[torch.Tensor, dict[str, float]]:
    """WM predictor MSE with gradients through state_proj and qwen_hidden."""

    state_emb = state_proj(qwen_hidden_at_latent)
    loss, metrics = lewm.alignment_loss(state_emb, action_indices, next_pixels)
    return loss, metrics


def wm_loss_weight_schedule(global_step: int, total_steps: int, start: float = 0.1, end: float = 1.0, warmup_fraction: float = 0.3) -> float:
    """Cosine ramp for λ_wm over the first warmup_fraction of training."""

    if total_steps <= 0:
        return end
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if global_step >= warmup_steps:
        return end
    progress = global_step / warmup_steps
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


def compute_combined_loss(
    *,
    wm_loss: torch.Tensor,
    lm_loss: torch.Tensor | None,
    lambda_wm: float,
    lambda_ce: float = 1.0,
    lewm_loss: torch.Tensor | None = None,
    lambda_lewm: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = lambda_wm * wm_loss
    metrics = {
        "wm_mse": float(wm_loss.detach().item()),
        "lambda_wm": float(lambda_wm),
        "lambda_ce": float(lambda_ce),
        "lambda_lewm": float(lambda_lewm),
    }
    if lewm_loss is not None and lambda_lewm > 0:
        total = total + lambda_lewm * lewm_loss
        metrics["lewm_loss"] = float(lewm_loss.detach().item())
    if lm_loss is not None:
        total = total + lambda_ce * lm_loss
        metrics["lm_ce"] = float(lm_loss.detach().item())
    metrics["total_loss"] = float(total.detach().item())
    return total, metrics


def compute_end_to_end_step_loss(
    *,
    lewm: LeWMWrapper,
    qwen_hidden_at_latent: torch.Tensor,
    action_indices: torch.Tensor,
    current_pixels: torch.Tensor,
    next_pixels: torch.Tensor,
    state_proj: StateProjector,
    lm_loss: torch.Tensor,
    lambda_wm: float,
    lambda_ce: float,
    lambda_lewm: float,
    train_lewm: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint LeWM pixel loss + Qwen alignment + LM CE in one step."""

    lewm_loss = None
    lewm_metrics: dict[str, float] = {}
    if train_lewm and lambda_lewm > 0:
        lewm_loss, lewm_metrics = lewm.pretrain_step(current_pixels, next_pixels, action_indices)

    wm_loss, wm_metrics = compute_wm_alignment_loss(
        qwen_hidden_at_latent=qwen_hidden_at_latent,
        action_indices=action_indices,
        next_pixels=next_pixels,
        state_proj=state_proj,
        lewm=lewm,
    )
    total, metrics = compute_combined_loss(
        wm_loss=wm_loss,
        lm_loss=lm_loss,
        lambda_wm=lambda_wm,
        lambda_ce=lambda_ce,
        lewm_loss=lewm_loss,
        lambda_lewm=lambda_lewm if train_lewm else 0.0,
    )
    metrics.update({k: v for k, v in lewm_metrics.items() if k != "loss"})
    metrics.update(wm_metrics)
    return total, metrics
