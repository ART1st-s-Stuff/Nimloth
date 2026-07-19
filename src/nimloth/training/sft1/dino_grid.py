"""SFT1 losses for aligning spatial query slots to pooled DINO patches."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nimloth.wm.grid import SharedSlotProjector


def compute_dino_grid_alignment_loss(
    *,
    current_query_hidden: torch.Tensor,
    items: list[dict[str, Any]],
    slot_projector: SharedSlotProjector,
    dino_encoder: Any,
    grid_size: int = 4,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align each current query slot with its row-major pooled DINO region."""

    if current_query_hidden.ndim != 3:
        raise ValueError(
            "DINO grid alignment requires current query hidden shape (B, slots, hidden), "
            f"got {tuple(current_query_hidden.shape)}"
        )
    expected_slots = int(grid_size) ** 2
    if current_query_hidden.shape[0] != len(items) or current_query_hidden.shape[1] != expected_slots:
        raise ValueError(
            "DINO grid alignment batch/grid mismatch: "
            f"hidden={tuple(current_query_hidden.shape)}, items={len(items)}, grid={grid_size}x{grid_size}"
        )
    paths = []
    for index, item in enumerate(items):
        path = item.get("current_image_path")
        if not path:
            raise ValueError(f"DINO grid alignment item {index} is missing current_image_path")
        paths.append(str(path))

    state_grid = slot_projector(current_query_hidden).float()
    target_grid = dino_encoder.encode_image_paths_grid(
        paths,
        device=state_grid.device,
        grid_size=int(grid_size),
    ).float()
    if state_grid.shape != target_grid.shape:
        raise ValueError(
            "projected query grid and DINO target grid must match exactly, "
            f"got state={tuple(state_grid.shape)}, target={tuple(target_grid.shape)}"
        )
    loss = F.mse_loss(state_grid, target_grid.detach())
    return loss, {"dino_grid_mse": float(loss.detach().item())}


def compute_grid_wm_loss(
    *,
    current_query_hidden: torch.Tensor,
    next_query_hidden: torch.Tensor,
    action_indices: torch.Tensor,
    slot_projector: SharedSlotProjector,
    grid_wm: torch.nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Predict the full next query-state grid with one joint world model."""

    if current_query_hidden.shape != next_query_hidden.shape:
        raise ValueError(
            f"current/next query grid shapes differ: {tuple(current_query_hidden.shape)} != "
            f"{tuple(next_query_hidden.shape)}"
        )
    current_grid = slot_projector(current_query_hidden).float()
    with torch.no_grad():
        target_grid = slot_projector(next_query_hidden).float()
    predicted_grid = grid_wm(current_grid, action_indices)
    if predicted_grid.shape != target_grid.shape:
        raise ValueError(
            f"grid WM prediction/target shapes differ: {tuple(predicted_grid.shape)} != "
            f"{tuple(target_grid.shape)}"
        )
    loss = F.mse_loss(predicted_grid.float(), target_grid.detach())
    return loss, {"grid_wm_mse": float(loss.detach().item())}
