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
    "compute_advantages",
    "compute_actor_loss",
    "compute_action_entropy",
    "compute_kl_penalty",
    "compute_masked_gae_advantage_return",
    "compute_clipped_token_value_loss",
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


def compute_masked_gae_advantage_return(
    *,
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VAGEN ``masked_gae`` over only loss-masked response tokens."""

    if token_level_rewards.shape != values.shape or values.shape != loss_mask.shape:
        raise ValueError(
            "masked GAE tensors must share shape: "
            f"rewards={tuple(token_level_rewards.shape)}, "
            f"values={tuple(values.shape)}, mask={tuple(loss_mask.shape)}"
        )
    if token_level_rewards.ndim != 2:
        raise ValueError(
            f"masked GAE expects [batch, sequence], got {token_level_rewards.shape}"
        )
    mask = loss_mask.to(dtype=torch.bool)
    with torch.no_grad():
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        for row in range(token_level_rewards.shape[0]):
            valid = mask[row].nonzero(as_tuple=True)[0]
            last_gae = torch.zeros(
                (), device=values.device, dtype=values.dtype
            )
            for index in range(len(valid) - 1, -1, -1):
                position = valid[index]
                next_value = (
                    values[row, valid[index + 1]]
                    if index + 1 < len(valid)
                    else torch.zeros((), device=values.device, dtype=values.dtype)
                )
                delta = (
                    token_level_rewards[row, position]
                    + float(gamma) * next_value
                    - values[row, position]
                )
                last_gae = delta + float(gamma) * float(lam) * last_gae
                advantages[row, position] = last_gae
                returns[row, position] = last_gae + values[row, position]
        valid_advantages = advantages[mask]
        if valid_advantages.numel() == 0:
            raise ValueError("masked GAE requires at least one loss token")
        mean = valid_advantages.mean()
        std = valid_advantages.std(unbiased=False)
        advantages = torch.where(
            mask,
            (advantages - mean) / (std + 1e-8),
            torch.zeros_like(advantages),
        )
    return advantages, returns


def compute_clipped_token_value_loss(
    *,
    predicted_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    cliprange_value: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VAGEN clipped token-critic regression loss."""

    if not (
        predicted_values.shape == old_values.shape == returns.shape
    ):
        raise ValueError("token value tensors must share shape")
    clipped = torch.clamp(
        predicted_values,
        old_values - float(cliprange_value),
        old_values + float(cliprange_value),
    )
    losses = (predicted_values - returns).square()
    clipped_losses = (clipped - returns).square()
    loss = 0.5 * torch.maximum(losses, clipped_losses).mean()
    with torch.no_grad():
        clip_fraction = (clipped_losses > losses).float().mean()
    return loss, {
        "critic_loss": float(loss.detach().item()),
        "value_clip_fraction": float(clip_fraction.item()),
        "critic_value_mean": float(predicted_values.detach().mean().item()),
    }


def compute_actor_loss(
    *,
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """PPO clipped policy gradient over every stochastic response token.

    The tensors are flattened across the VAGEN-style response loss mask. For
    Nimloth inject mode this includes generated thought tokens and the sampled
    action token, while deterministic query/scaffold tokens remain masked out.

    Args:
        new_log_probs: log-probs under the current policy.
        old_log_probs: log-probs under the rollout policy.
        advantages: detached turn advantages broadcast to response tokens.
        clip_ratio:      PPO clipping epsilon.

    Returns:
        (loss, metrics_dict)
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_losses = -ratio * advantages
    clipped_losses = -clipped_ratio * advantages
    loss = torch.maximum(pg_losses, clipped_losses).mean()

    with torch.no_grad():
        clip_frac = (clipped_losses > pg_losses).float().mean()
        ppo_kl = (old_log_probs - new_log_probs).mean()

    return loss, {
        "actor_loss": float(loss.detach().item()),
        "clip_fraction": float(clip_frac.item()),
        "mean_ratio": float(ratio.mean().item()),
        "ppo_kl": float(ppo_kl.item()),
    }


def compute_kl_penalty(
    log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    *,
    penalty_type: str = "low_var_kl",
) -> torch.Tensor:
    """Match VAGEN/VERL's sampled-token KL penalty implementations."""

    if log_probs.shape != reference_log_probs.shape:
        raise ValueError(
            "policy/reference log-prob shape mismatch: "
            f"{tuple(log_probs.shape)} != {tuple(reference_log_probs.shape)}"
        )
    delta = log_probs - reference_log_probs
    if penalty_type == "kl":
        return delta
    if penalty_type == "abs":
        return delta.abs()
    if penalty_type == "mse":
        return 0.5 * delta.square()
    if penalty_type == "low_var_kl":
        reverse_delta = -delta
        penalty = torch.exp(reverse_delta) - reverse_delta - 1.0
        return torch.clamp(penalty, min=-10.0, max=10.0)
    raise ValueError(f"unsupported KL penalty type: {penalty_type!r}")


def compute_action_entropy(action_logits: torch.Tensor) -> torch.Tensor:
    """Mean categorical entropy over 8 action tokens.

    Returns a scalar tensor (0 to ~2.08 for 8 actions).
    """
    probs = torch.softmax(action_logits.float(), dim=-1)
    log_probs = torch.log_softmax(action_logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    return entropy
