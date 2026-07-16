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


def _sync_device_bucket(bucket, *, group, world: int) -> None:
    flat = torch.cat([grad.reshape(-1) for grad in bucket])
    dist.all_reduce(flat, group=group)
    flat.div_(float(world))
    offset = 0
    for grad in bucket:
        count = grad.numel()
        grad.copy_(flat[offset : offset + count].view_as(grad))
        offset += count


def _average_device_buckets(params, *, group, world: int, bucket_numel: int) -> None:
    bucket = []
    count = 0
    for param in params:
        grad = param.grad
        incompatible = bucket and (count + grad.numel() > bucket_numel or grad.dtype != bucket[0].dtype)
        if incompatible:
            _sync_device_bucket(bucket, group=group, world=world)
            bucket, count = [], 0
        bucket.append(grad)
        count += grad.numel()
    if bucket:
        _sync_device_bucket(bucket, group=group, world=world)


def relative_trainable_placement(
    module: nn.Module, *, primary_device: int, stride: int
) -> tuple[tuple[str, int, tuple[int, ...]], ...]:
    rows = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.device.type != "cuda" or param.device.index is None:
            raise RuntimeError(f"trainable parameter {name} is not on CUDA: {param.device}")
        slot = int(param.device.index) - primary_device
        if not 0 <= slot < stride:
            raise RuntimeError(f"trainable parameter {name} is outside pair: {param.device}")
        rows.append((name, slot, tuple(param.shape)))
    return tuple(rows)


def assert_consistent_relative_placement(
    module: nn.Module, *, primary_device: int, stride: int, group
):
    local = relative_trainable_placement(module, primary_device=primary_device, stride=stride)
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    if any(item != local for item in gathered):
        raise RuntimeError("Qwen trainable parameter placement mismatch across ranks")
    return local


def average_partitioned_module_gradients(
    module: nn.Module,
    *,
    groups,
    primary_device: int,
    bucket_numel: int = 16_000_000,
) -> None:
    placement = relative_trainable_placement(
        module, primary_device=primary_device, stride=len(groups)
    )
    slots = {name: slot for name, slot, _shape in placement}
    named = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    for slot, group in enumerate(groups):
        params = [param for name, param in named if slots[name] == slot]
        if not params:
            raise RuntimeError(f"Qwen gradient device slot {slot} has no trainable parameters")
        missing = sum(param.grad is None for param in params)
        missing_count = torch.tensor([missing], device=params[0].device)
        dist.all_reduce(missing_count, group=group)
        if missing_count.item():
            raise RuntimeError(f"missing Qwen gradients in device slot {slot}")
        _average_device_buckets(
            params, group=group, world=dist.get_world_size(group), bucket_numel=bucket_numel
        )


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
