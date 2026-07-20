"""RL loss functions: WM predictor MSE + value-head regression with optional ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead

__all__ = [
    "compute_predictor_loss",
    "compute_multistep_predictor_loss",
    "compute_value_loss",
    "compute_advantages",
    "compute_actor_loss",
    "compute_action_entropy",
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


def compute_multistep_predictor_loss(
    *,
    qwen_hidden_current: torch.Tensor,
    qwen_hidden_targets: torch.Tensor,
    action_sequences: torch.Tensor,
    valid_mask: torch.Tensor,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    loss_decay: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Recursive multi-step dynamics loss over contiguous trajectory windows.

    Step one starts from a Qwen-GT state.  Every later step consumes the
    predictor's previous output; real future states are targets only.  The
    leading shapes are ``(B, T, ...)`` for targets and ``(B, T)`` for actions
    and mask.  ``...`` is either ``H`` for k=1 or ``(k, H)`` for k>1.
    """

    if action_sequences.ndim != 2 or valid_mask.shape != action_sequences.shape:
        raise ValueError(
            "action_sequences and valid_mask must both have shape (B, T), got "
            f"{tuple(action_sequences.shape)} and {tuple(valid_mask.shape)}"
        )
    if qwen_hidden_targets.shape[:2] != action_sequences.shape:
        raise ValueError(
            "qwen_hidden_targets leading shape must match actions; got "
            f"{tuple(qwen_hidden_targets.shape[:2])} vs {tuple(action_sequences.shape)}"
        )
    if loss_decay <= 0:
        raise ValueError(f"loss_decay must be positive, got {loss_decay}")
    mask = valid_mask.to(dtype=torch.bool, device=action_sequences.device)
    if not bool(mask.any()):
        raise ValueError("multi-step predictor loss requires at least one valid target")

    batch_size, num_steps = action_sequences.shape
    state = state_proj(qwen_hidden_current).float()
    flat_targets = qwen_hidden_targets.reshape(
        batch_size * num_steps, *qwen_hidden_targets.shape[2:]
    )
    with torch.no_grad():
        target_states = state_proj(flat_targets).float().reshape(batch_size, num_steps, -1)

    weighted_loss = state.sum() * 0.0
    weight_total = torch.zeros((), dtype=state.dtype, device=state.device)
    metrics: dict[str, float] = {}
    for step in range(num_steps):
        # Preserve the SFT2/current-RL one-step predictor contract at h=1,
        # then recursively feed its own prediction for later horizons.
        state = wm_predictor(state, action_sequences[:, step])
        per_item = F.mse_loss(
            state, target_states[:, step], reduction="none"
        ).mean(dim=-1)
        step_mask = mask[:, step].to(device=per_item.device)
        if bool(step_mask.any()):
            step_loss = per_item[step_mask].mean()
            metrics[f"wm_mse_h{step + 1}"] = float(step_loss.detach().item())
            weight = float(loss_decay ** step)
            weighted_loss = weighted_loss + weight * per_item[step_mask].sum()
            weight_total = weight_total + weight * step_mask.sum().to(dtype=state.dtype)

    total = weighted_loss / weight_total.clamp_min(1.0)
    metrics["wm_mse"] = float(total.detach().item())
    metrics["wm_rollout_steps"] = float(num_steps)
    return total, metrics


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


def compute_advantages(
    *,
    value_targets: torch.Tensor,
    predicted_values: torch.Tensor,
) -> torch.Tensor:
    """TD residual advantages: A = G_t - V(s_t, a_t), normalized to mean=0 std=1.

    Returns advantages detached from the computation graph.
    """
    advantages = value_targets - predicted_values.detach()
    # unbiased=False 避免 batch size=1 时 std 产生 NaN（单样本下 unbiased std 分母为 0）
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return advantages


def compute_actor_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped policy gradient for per-step discrete actions.

    Args:
        new_log_probs: (B,) log-prob of taken actions under current policy.
        old_log_probs:  (B,) log-prob of taken actions under rollout policy.
        advantages:     (B,) detached advantages.
        clip_ratio:      PPO clipping epsilon.

    Returns:
        (loss, metrics_dict)
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = -torch.min(ratio * advantages, clipped_ratio * advantages)
    loss = surrogate.mean()

    with torch.no_grad():
        clip_frac = (ratio.abs() - 1.0).abs().gt(clip_ratio).float().mean()

    return loss, {
        "actor_loss": float(loss.detach().item()),
        "clip_fraction": float(clip_frac.item()),
        "mean_ratio": float(ratio.mean().item()),
    }


def compute_action_entropy(action_logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy over 8 action tokens.

    Returns a scalar tensor (0 to ~2.08 for 8 actions).
    """
    probs = torch.softmax(action_logits.float(), dim=-1)
    log_probs = torch.log_softmax(action_logits.float(), dim=-1)
    terms = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
    entropy = -terms.sum(dim=-1).mean()
    return entropy
