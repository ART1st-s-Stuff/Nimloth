"""Distributed training helpers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_dist() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        gpu_stride = int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
        primary = local * gpu_stride
        torch.cuda.set_device(primary)
        dist.init_process_group(backend="nccl")
        return rank, world, local, torch.device(f"cuda:{primary}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
