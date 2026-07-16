"""Explicit gradient synchronization for multi-device modules."""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch import nn


def average_module_gradients(module: nn.Module) -> None:
    """Average every trainable gradient in deterministic parameter order."""
    if not dist.is_available() or not dist.is_initialized():
        return
    world = dist.get_world_size()
    if world <= 1:
        return
    params = [param for param in module.parameters() if param.requires_grad]
    if not params:
        return
    missing = [index for index, param in enumerate(params) if param.grad is None]
    missing_count = torch.tensor([len(missing)], device=params[0].device)
    dist.all_reduce(missing_count)
    if missing_count.item():
        raise RuntimeError(f"missing gradient for trainable parameter indices: {missing[:8]}")
    for param in params:
        dist.all_reduce(param.grad)
        param.grad.div_(float(world))
