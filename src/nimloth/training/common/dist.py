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
        device = torch.device(f"cuda:{primary}")
        torch.cuda.set_device(primary)
        # Passing the concrete device removes ProcessGroupNCCL's rank-to-device
        # guess at the first barrier.  The guess is ambiguous when Slurm starts
        # one node-level launcher that spawns a non-uniform number of local
        # workers on each node.
        dist.init_process_group(backend="nccl", device_id=device)
        return rank, world, local, device
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
