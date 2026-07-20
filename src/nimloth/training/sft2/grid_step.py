"""SFT2 loss for LeWM-style encoding, spatial prediction, and DINO decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from nimloth.wm.lewm import SIGReg


def compute_grid_sft2_loss(
    *,
    current_query_hidden: torch.Tensor,
    next_query_hidden: torch.Tensor,
    dino_target_grid: torch.Tensor,
    action_indices: torch.Tensor,
    value_targets: torch.Tensor,
    slot_projector: torch.nn.Module,
    online_encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    grid_wm: torch.nn.Module,
    decoder: torch.nn.Module,
    value_head: torch.nn.Module,
    latent_weight: float = 1.0,
    dino_weight: float = 0.5,
    sigreg_weight: float = 0.1,
    value_weight: float = 1.0,
    sigreg_num_proj: int = 1024,
    sigreg_knots: int = 17,
    sigreg_module: torch.nn.Module | None = None,
    latent_indices: torch.Tensor | list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train SFT2 auxiliaries while SFT1 Qwen/projector and DINO stay frozen."""

    if current_query_hidden.ndim != 3 or next_query_hidden.ndim != 3:
        raise ValueError("current and next query hidden must have shape (B, grid_tokens, hidden)")
    if current_query_hidden.shape[1:] != next_query_hidden.shape[1:]:
        raise ValueError(
            "current and next query hidden grid dimensions differ: "
            f"{tuple(current_query_hidden.shape)} and {tuple(next_query_hidden.shape)}"
        )
    if latent_indices is None:
        latent_indices = torch.arange(current_query_hidden.shape[0], device=current_query_hidden.device)
    else:
        latent_indices = torch.as_tensor(latent_indices, device=current_query_hidden.device, dtype=torch.long)
    if latent_indices.numel() != next_query_hidden.shape[0]:
        raise ValueError(
            f"latent target count {next_query_hidden.shape[0]} != index count {latent_indices.numel()}"
        )
    with torch.no_grad():
        current_query_grid = slot_projector(current_query_hidden).float()
        if next_query_hidden.shape[0]:
            next_query_grid = slot_projector(next_query_hidden).float()
        else:
            next_query_grid = current_query_grid[:0]

    current_latent = online_encoder(current_query_grid)
    with torch.no_grad():
        target_latent = (
            target_encoder(next_query_grid).float()
            if next_query_grid.shape[0]
            else next_query_grid
        )
    predicted_latent = grid_wm(current_latent, action_indices)
    predicted_for_latent = predicted_latent.index_select(0, latent_indices)
    if predicted_for_latent.shape != target_latent.shape:
        raise ValueError(
            f"WM latent-target shape {tuple(predicted_for_latent.shape)} != {tuple(target_latent.shape)}"
        )
    if target_latent.numel():
        latent_loss = F.mse_loss(predicted_for_latent.float(), target_latent.detach())
    else:
        latent_loss = predicted_latent.sum() * 0.0

    decoded_prediction = decoder(predicted_latent)
    if decoded_prediction.shape != dino_target_grid.shape:
        raise ValueError(
            "decoded prediction and DINO target must match exactly, got "
            f"{tuple(decoded_prediction.shape)} and {tuple(dino_target_grid.shape)}"
        )
    dino_loss = F.mse_loss(decoded_prediction.float(), dino_target_grid.detach().float())

    regularizer = sigreg_module or SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj)
    regularizer = regularizer.to(device=current_latent.device)
    sigreg_loss = regularizer(current_latent.transpose(0, 1).float())

    all_action_values = value_head(current_latent.mean(dim=1))
    if all_action_values.ndim != 2 or all_action_values.shape[0] != action_indices.shape[0]:
        raise ValueError(f"value head must return (B, actions), got {tuple(all_action_values.shape)}")
    value_prediction = all_action_values.gather(1, action_indices.reshape(-1, 1)).reshape(-1)
    target_values = value_targets.float().reshape(-1)
    if value_prediction.shape != target_values.shape:
        raise ValueError(
            f"value prediction/target shapes differ: {tuple(value_prediction.shape)} != {tuple(target_values.shape)}"
        )
    value_loss = F.mse_loss(value_prediction.float(), target_values)

    total = (
        float(latent_weight) * latent_loss
        + float(dino_weight) * dino_loss
        + float(sigreg_weight) * sigreg_loss
        + float(value_weight) * value_loss
    )
    return total, {
        "latent_mse": float(latent_loss.detach().item()),
        "dino_grid_mse": float(dino_loss.detach().item()),
        "sigreg": float(sigreg_loss.detach().item()),
        "value_mse": float(value_loss.detach().item()),
        "total": float(total.detach().item()),
    }
