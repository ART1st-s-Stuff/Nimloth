"""SFT2 step for one joint world model over a complete spatial token grid."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_grid_sft2_loss(
    *,
    current_query_hidden: torch.Tensor,
    next_query_hidden: torch.Tensor,
    action_indices: torch.Tensor,
    value_targets: torch.Tensor,
    slot_projector: torch.nn.Module,
    grid_wm: torch.nn.Module,
    value_head: torch.nn.Module,
    wm_weight: float = 1.0,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train WM/value while keeping the SFT1 representation interface frozen."""

    if current_query_hidden.shape != next_query_hidden.shape or current_query_hidden.ndim != 3:
        raise ValueError(
            "current and next query hidden must share shape (B, grid_tokens, hidden), got "
            f"{tuple(current_query_hidden.shape)} and {tuple(next_query_hidden.shape)}"
        )
    with torch.no_grad():
        current_grid = slot_projector(current_query_hidden).float()
        target_grid = slot_projector(next_query_hidden).float()
    prediction = grid_wm(current_grid, action_indices)
    if prediction.shape != target_grid.shape:
        raise ValueError(
            f"grid WM output shape {tuple(prediction.shape)} != target {tuple(target_grid.shape)}"
        )
    wm_loss = F.mse_loss(prediction.float(), target_grid.detach())
    # A scalar action-value is global to the image; mean pooling preserves the
    # state dimension while exposing every equally sized 4x4 region.
    all_action_values = value_head(current_grid.mean(dim=1))
    if all_action_values.ndim != 2 or all_action_values.shape[0] != action_indices.shape[0]:
        raise ValueError(f"value head must return (B, actions), got {tuple(all_action_values.shape)}")
    value_prediction = all_action_values.gather(1, action_indices.reshape(-1, 1)).reshape(-1)
    target_values = value_targets.float().reshape(-1)
    if value_prediction.shape != target_values.shape:
        raise ValueError(
            f"value prediction/target shapes differ: {tuple(value_prediction.shape)} != {tuple(target_values.shape)}"
        )
    value_loss = F.mse_loss(value_prediction.float(), target_values)
    total = float(wm_weight) * wm_loss + float(value_weight) * value_loss
    return total, {
        "grid_wm_mse": float(wm_loss.detach().item()),
        "value_mse": float(value_loss.detach().item()),
        "total": float(total.detach().item()),
    }
