"""各优化阶段共享的 optimizer 与学习率工具。"""

from __future__ import annotations

import math

import torch


def qwen_lr_schedule(
    global_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    start_lr: float,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """先将 Qwen 学习率从 start_lr 升至 peak_lr，再做余弦衰减。"""

    if warmup_steps <= 0:
        warmup_steps = 1
    if global_step < warmup_steps:
        progress = (global_step + 1) / warmup_steps
        return start_lr + (peak_lr - start_lr) * progress

    min_lr = peak_lr * min_lr_ratio
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (global_step - warmup_steps) / decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def set_optimizer_group_lr(
    optimizer: torch.optim.Optimizer,
    group_name: str,
    lr: float,
) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            group["lr"] = lr
            return
    raise KeyError(f"optimizer param group {group_name!r} not found")
