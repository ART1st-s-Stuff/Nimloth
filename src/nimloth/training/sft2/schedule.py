"""SFT2 阶段专用的训练权重调度。"""

from __future__ import annotations

import math


def wm_loss_weight_schedule(
    global_step: int,
    total_steps: int,
    *,
    start: float = 0.1,
    end: float = 1.0,
    warmup_fraction: float = 0.3,
) -> float:
    """在训练前段用 cosine ramp 增加 WM loss 权重。"""

    if total_steps <= 0:
        return end
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if global_step >= warmup_steps:
        return end
    progress = global_step / warmup_steps
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


__all__ = ["wm_loss_weight_schedule"]
