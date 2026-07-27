"""SFT2 与 RL 共用的 world-model 监督。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WorldModelLoss:
    """WM state 预测的分项 loss 与加权和。"""

    loss: torch.Tensor
    state_mse: torch.Tensor
    dino_grid_mse: torch.Tensor | None


def world_model_loss(
    predicted_state: torch.Tensor,
    expected_state: torch.Tensor,
    *,
    state_weight: float,
    dino_grid_target: torch.Tensor | None = None,
    dino_grid_weight: float = 0.0,
) -> WorldModelLoss:
    """计算预测 state 与真实下一 observation state 的 MSE，并可选加入 DINO 监督。"""

    state_mse = F.mse_loss(predicted_state, expected_state)
    total = float(state_weight) * state_mse
    dino_grid_mse = None
    if dino_grid_target is not None:
        dino_grid_mse = F.mse_loss(
            predicted_state.float(),
            dino_grid_target.detach().float(),
        )
        total = total + float(dino_grid_weight) * dino_grid_mse
    elif dino_grid_weight != 0.0:
        raise ValueError("positive DINO-grid weight requires a DINO-grid target")
    return WorldModelLoss(
        loss=total,
        state_mse=state_mse,
        dino_grid_mse=dino_grid_mse,
    )


__all__ = ["WorldModelLoss", "world_model_loss"]
