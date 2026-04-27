"""训练配置解析与通用学习率调度函数。"""

from __future__ import annotations

import math


def parse_temporal_stride(value: object) -> int | tuple[int, int]:
    """解析 temporal_stride，支持 int 或 [min, max]。"""
    if isinstance(value, int):
        return max(1, int(value))
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("pipeline.train.temporal_stride 区间必须包含两个整数 [min, max]。")
        low = max(1, int(value[0]))
        high = max(low, int(value[1]))
        return (low, high)
    return 1


def linear_warmup_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step + 1) / float(warmup_steps)))


def cosine_annealing_lambda(
    step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> float:
    """warmup + cosine annealing 调度系数。"""
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
