"""Loss functions for SFT2 (WM latent MSE + value head + optional LM CE)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.training.sft2.predictor import LatentWMPredictor
from nimloth.training.sft2.value_head import ValueHead


class StateProjector(nn.Module):
    """Map Qwen hidden states into WM predictor embedding space."""

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
        weight = self.net.weight if hasattr(self.net, "weight") else self.net[0].weight
        return self.net(hidden.to(dtype=weight.dtype))


def compute_wm_latent_loss(
    *,
    qwen_hidden_at_latent: torch.Tensor,
    qwen_hidden_at_next_latent: torch.Tensor,
    action_indices: torch.Tensor,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """WM predictor MSE: project Qwen latents, predict next latent from current + action."""

    state_emb = state_proj(qwen_hidden_at_latent).float()
    with torch.no_grad():
        target_emb = state_proj(qwen_hidden_at_next_latent).float()
    pred = wm_predictor.predict_next_emb(state_emb, action_indices)
    mse = F.mse_loss(pred, target_emb)
    return mse, {"wm_mse": float(mse.detach().item())}


def compute_value_loss(
    *,
    state_emb: torch.Tensor,
    action_indices: torch.Tensor,
    action_value_targets: torch.Tensor,
    value_head: ValueHead,
    rank_margin: float = 0.1,
    lambda_rank: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression on chosen-action value + margin ranking vs unchosen actions."""

    values = value_head(state_emb).float()
    chosen_values = values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    targets = action_value_targets.to(device=values.device, dtype=values.dtype)
    reg_loss = F.mse_loss(chosen_values, targets)

    mask = F.one_hot(action_indices, num_classes=values.shape[1]).bool()
    other_values = values.masked_fill(mask, float("-inf"))
    max_other = other_values.max(dim=1).values
    rank_loss = F.relu(rank_margin + max_other - chosen_values).mean()

    total = reg_loss + lambda_rank * rank_loss
    metrics = {
        "value_reg": float(reg_loss.detach().item()),
        "value_rank": float(rank_loss.detach().item()),
        "value_total": float(total.detach().item()),
    }
    return total, metrics


def wm_loss_weight_schedule(
    global_step: int,
    total_steps: int,
    start: float = 0.1,
    end: float = 1.0,
    warmup_fraction: float = 0.3,
) -> float:
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
    wm_loss: torch.Tensor | None,
    value_loss: torch.Tensor | None,
    lm_loss: torch.Tensor | None,
    lambda_wm: float,
    lambda_value: float = 1.0,
    lambda_ce: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {
        "lambda_wm": float(lambda_wm),
        "lambda_value": float(lambda_value),
        "lambda_ce": float(lambda_ce),
    }
    device = None
    for candidate in (wm_loss, value_loss, lm_loss):
        if candidate is not None:
            device = candidate.device
            break
    total = torch.zeros((), device=device or "cpu")

    if wm_loss is not None:
        total = total + lambda_wm * wm_loss
        metrics["wm_mse"] = float(wm_loss.detach().item())
    if value_loss is not None:
        total = total + lambda_value * value_loss
        metrics["value_total"] = float(value_loss.detach().item())
    if lm_loss is not None:
        total = total + lambda_ce * lm_loss
        metrics["lm_ce"] = float(lm_loss.detach().item())
    metrics["total_loss"] = float(total.detach().item())
    return total, metrics
