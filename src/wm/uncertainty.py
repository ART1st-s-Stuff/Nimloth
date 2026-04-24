"""WM 不确定度估计。"""

from __future__ import annotations

import numpy as np
import torch

@torch.no_grad()
def estimate_divergence(
    model: torch.nn.Module,
    z_history: torch.Tensor,
    action_history: torch.Tensor,
    noise_scale: float,
    num_samples: int,
    solver: str | None = None,
    num_steps: int | None = None,
) -> torch.Tensor:
    """用输入扰动近似散度，返回 batch 级不确定度。"""
    kwargs: dict[str, object] = {}
    if solver is not None:
        kwargs["solver"] = solver
    if num_steps is not None:
        kwargs["num_steps"] = num_steps

    try:
        base = model.predict_next(z_history, action_history, **kwargs)
    except TypeError:
        # 部分模型（如 LeWM）不支持 solver/num_steps 参数，回退到通用签名。
        base = model.predict_next(z_history, action_history)
    diffs = []
    for _ in range(num_samples):
        noise = torch.randn_like(z_history) * noise_scale
        try:
            perturbed = model.predict_next(
                z_history + noise,
                action_history,
                **kwargs,
            )
        except TypeError:
            perturbed = model.predict_next(z_history + noise, action_history)
        diff = (perturbed - base).reshape(perturbed.size(0), -1)
        diffs.append(torch.norm(diff, dim=-1))
    return torch.stack(diffs, dim=0).mean(dim=0)


def percentile_threshold(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("无法计算阈值：输入为空。")
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))

