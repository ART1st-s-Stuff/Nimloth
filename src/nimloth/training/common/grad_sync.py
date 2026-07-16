"""Explicit gradient synchronization for multi-device modules."""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch import nn


def average_module_gradients(
    module: nn.Module,
    *,
    group=None,
    cpu: bool = False,
) -> None:
    """Average every trainable gradient in deterministic parameter order."""
    if not dist.is_available() or not dist.is_initialized():
        return
    world = dist.get_world_size(group)
    if world <= 1:
        return
    params = [param for param in module.parameters() if param.requires_grad]
    if not params:
        return
    missing = [index for index, param in enumerate(params) if param.grad is None]
    sync_device = torch.device("cpu") if cpu else params[0].device
    missing_count = torch.tensor([len(missing)], device=sync_device)
    dist.all_reduce(missing_count, group=group)
    if missing_count.item():
        raise RuntimeError(f"missing gradient for trainable parameter indices: {missing[:8]}")
    for param in params:
        grad = param.grad.float().cpu() if cpu else param.grad
        dist.all_reduce(grad, group=group)
        grad.div_(float(world))
        if cpu:
            param.grad.copy_(grad.to(device=param.grad.device, dtype=param.grad.dtype))
