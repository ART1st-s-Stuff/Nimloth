"""训练和评估共享的分布式运行工具。"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup_dist(
    *,
    gpu_stride: int | None = None,
) -> tuple[int, int, int, torch.device]:
    if gpu_stride is None:
        gpu_stride = int(os.environ.get("NIMLOTH_DDP_GPU_STRIDE", "1"))
    if gpu_stride < 1:
        raise ValueError(f"gpu_stride must be positive, got {gpu_stride}")
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        primary = local * gpu_stride
        if torch.cuda.is_available() and primary + gpu_stride > torch.cuda.device_count():
            raise RuntimeError(
                "distributed rank GPU group exceeds visible devices: "
                f"local_rank={local}, gpu_stride={gpu_stride}, "
                f"visible={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(primary)
        dist.init_process_group(backend="nccl")
        return rank, world, local, torch.device(f"cuda:{primary}")
    if torch.cuda.is_available():
        if gpu_stride > torch.cuda.device_count():
            raise RuntimeError(
                f"gpu_stride={gpu_stride} exceeds visible GPUs={torch.cuda.device_count()}"
            )
        torch.cuda.set_device(0)
        return 0, 1, 0, torch.device("cuda:0")
    return 0, 1, 0, torch.device("cpu")


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def broadcast_module_state(module: torch.nn.Module, *, source_rank: int = 0) -> None:
    """把小型 replicated module 的参数与 buffer 同步到所有 rank。"""

    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=source_rank)
