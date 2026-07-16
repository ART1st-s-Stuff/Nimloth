"""Explicit gradient synchronization for multi-device modules."""

from __future__ import annotations

import torch
from torch import distributed as dist
from torch import nn


def _sync_cpu_bucket(bucket, *, group, world: int) -> None:
    flat = torch.cat([cpu_grad.reshape(-1) for _, cpu_grad in bucket])
    dist.all_reduce(flat, group=group)
    flat.div_(float(world))
    offset = 0
    for device_grad, cpu_grad in bucket:
        count = cpu_grad.numel()
        synced = flat[offset : offset + count].view_as(cpu_grad)
        device_grad.copy_(synced.to(device=device_grad.device, dtype=device_grad.dtype))
        offset += count


def _average_cpu_buckets(params, *, group, world: int, bucket_numel: int) -> None:
    bucket = []
    count = 0
    for param in params:
        cpu_grad = param.grad.detach().float().cpu()
        if bucket and count + cpu_grad.numel() > bucket_numel:
            _sync_cpu_bucket(bucket, group=group, world=world)
            bucket, count = [], 0
        bucket.append((param.grad, cpu_grad))
        count += cpu_grad.numel()
    if bucket:
        _sync_cpu_bucket(bucket, group=group, world=world)


def average_module_gradients(
    module: nn.Module,
    *,
    group=None,
    cpu: bool = False,
    cpu_bucket_numel: int = 16_000_000,
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
    if cpu:
        _average_cpu_buckets(params, group=group, world=world, bucket_numel=cpu_bucket_numel)
        return
    for param in params:
        dist.all_reduce(param.grad, group=group)
        param.grad.div_(float(world))
