"""WM 不确定度估计。"""

from __future__ import annotations

import numpy as np
import torch

from src.wm.model import CFMWorldModel


@torch.no_grad()
def estimate_divergence(
    model: CFMWorldModel,
    z_t: torch.Tensor,
    action: torch.Tensor,
    noise_scale: float,
    num_samples: int,
) -> torch.Tensor:
    """用输入扰动近似散度，返回 batch 级不确定度。"""
    base = model(z_t, action)
    diffs = []
    for _ in range(num_samples):
        noise = torch.randn_like(z_t) * noise_scale
        perturbed = model(z_t + noise, action)
        diffs.append(torch.norm(perturbed - base, dim=-1))
    return torch.stack(diffs, dim=0).mean(dim=0)


def percentile_threshold(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("无法计算阈值：输入为空。")
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))

