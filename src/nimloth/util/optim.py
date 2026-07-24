"""各优化阶段共享的 optimizer 与学习率工具。"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass

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


@dataclass
class OptimizationRuntime:
    """训练阶段共享的梯度同步、裁剪和 optimizer 生命周期。"""

    optimizer: torch.optim.Optimizer
    synchronized_modules: tuple[torch.nn.Module, ...] = ()
    max_grad_norm: float = 1.0
    enable_no_sync: bool = False
    after_step: Callable[[], None] | None = None

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def accumulation_context(
        self,
        *,
        sync_gradients: bool,
    ) -> AbstractContextManager[object]:
        """非 accumulation 边界时统一进入所有分布式模块的 ``no_sync``。"""

        if sync_gradients or not self.enable_no_sync:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        for module in self.synchronized_modules:
            no_sync = getattr(module, "no_sync", None)
            if no_sync is not None:
                stack.enter_context(no_sync())
        return stack

    def backward(self, loss: torch.Tensor, *, divisor: int = 1) -> None:
        if divisor < 1:
            raise ValueError(f"backward divisor must be positive, got {divisor}")
        (loss / divisor).backward()

    def step(self) -> None:
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.optimizer.step()
        if self.after_step is not None:
            self.after_step()
        self.zero_grad()
