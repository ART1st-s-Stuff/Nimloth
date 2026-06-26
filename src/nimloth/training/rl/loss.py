"""RL loss functions: WM predictor MSE + value-head regression with optional ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead

__all__ = [
    "compute_predictor_loss",
    "compute_value_loss",
]


def compute_predictor_loss(
    *,
    qwen_hidden_current: torch.Tensor,
    qwen_hidden_next: torch.Tensor,
    action_indices: torch.Tensor,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """WM predictor MSE: project Qwen latents, predict next latent from current + action.

    ``qwen_hidden_next`` is used only as a target (no gradient through it).
    """

    state_emb = state_proj(qwen_hidden_current).float()
    with torch.no_grad():
        target_emb = state_proj(qwen_hidden_next).float()
    pred = wm_predictor(state_emb, action_indices)
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
    """Regression + margin ranking: value_head(s_t)[a_t] ≈ discounted MC return.

    Args:
        state_emb:            (B, emb_dim) WM state at step t.
        action_indices:       (B,) int64 taken action.
        action_value_targets: (B,) discounted return target.
        value_head:           ValueHead module.
        rank_margin:          Margin for ranking loss.
        lambda_rank:          Weight of ranking loss term (0 = regression only).
    """

    values = value_head(state_emb).float()
    chosen = values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    targets = action_value_targets.to(device=values.device, dtype=values.dtype)
    reg_loss = F.mse_loss(chosen, targets)

    if lambda_rank > 0:
        mask = F.one_hot(action_indices, num_classes=values.shape[1]).bool()
        other_values = values.masked_fill(mask, float("-inf"))
        max_other = other_values.max(dim=1).values
        rank_loss = F.relu(rank_margin + max_other - chosen).mean()
        total = reg_loss + lambda_rank * rank_loss
        return total, {
            "value_reg": float(reg_loss.detach().item()),
            "value_rank": float(rank_loss.detach().item()),
            "value_total": float(total.detach().item()),
        }

    return reg_loss, {"value_loss": float(reg_loss.detach().item())}
