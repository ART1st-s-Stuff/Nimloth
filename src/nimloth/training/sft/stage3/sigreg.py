"""SIGReg 的跨 rank 有效样本汇聚、可微通信与同步随机投影。"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.distributed as dist


class _DifferentiableAllGather(torch.autograd.Function):
    """汇聚小型 state，并在 backward 时把全局梯度送回来源 rank。

    每个 rank 都计算同一个全局 loss；backward 先合计所有 rank 对 global state
    的梯度，再取回本 rank 的 slice。随后 DDP 对模型参数做平均，正好得到一次全局
    batch loss 对共享参数的梯度。
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        if not (dist.is_available() and dist.is_initialized()):
            ctx.world_size = 1
            ctx.rank = 0
            ctx.local_rows = value.shape[0]
            return value
        ctx.world_size = dist.get_world_size()
        ctx.rank = dist.get_rank()
        ctx.local_rows = value.shape[0]
        gathered = [torch.empty_like(value) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, value)
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, global_gradient: torch.Tensor) -> tuple[torch.Tensor]:
        if ctx.world_size == 1:
            return (global_gradient,)
        summed = global_gradient.contiguous().clone()
        dist.all_reduce(summed, op=dist.ReduceOp.SUM)
        start = ctx.rank * ctx.local_rows
        return (summed.narrow(0, start, ctx.local_rows),)


def gather_global_sigreg_states(
    detached_current_state: torch.Tensor,
    next_state: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """构造跨 rank 的有效 ``(s_t,s_{t+1})`` SIGReg batch。

    current state 始终无梯度；next state 使用可微 gather。不同 rank 的本地 B 可以
    不同，collective 前会补齐到最大 B，再由 global valid mask 删除补齐与 sampler
    padding 行。
    """

    if detached_current_state.requires_grad:
        raise ValueError("SFT2 SIGReg current_state must be detached")
    if detached_current_state.ndim != 2 or next_state.ndim != 2:
        raise ValueError("SFT2 global SIGReg states must have shape (B,D)")
    if detached_current_state.shape != next_state.shape:
        raise ValueError("SFT2 global SIGReg local state shapes do not match")
    if valid_mask.shape != (next_state.shape[0],):
        raise ValueError(
            "SFT2 global SIGReg valid mask must have shape (B,), "
            f"got {tuple(valid_mask.shape)} for B={next_state.shape[0]}"
        )

    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    local_size = torch.tensor(
        [next_state.shape[0]],
        device=next_state.device,
        dtype=torch.int64,
    )
    sizes = [torch.empty_like(local_size) for _ in range(world_size)]
    if distributed:
        dist.all_gather(sizes, local_size)
    else:
        sizes[0].copy_(local_size)
    max_size = max(int(size.item()) for size in sizes)

    def pad_rows(value: torch.Tensor) -> torch.Tensor:
        missing = max_size - value.shape[0]
        if missing <= 0:
            return value.contiguous()
        return torch.cat(
            (value, value.new_zeros((missing, *value.shape[1:]))),
            dim=0,
        )

    current_padded = pad_rows(detached_current_state.detach())
    next_padded = pad_rows(next_state)
    valid_padded = torch.zeros(max_size, device=next_state.device, dtype=torch.int64)
    valid_padded[: valid_mask.shape[0]] = valid_mask.to(
        device=next_state.device,
        dtype=torch.int64,
    )

    gathered_current = [torch.empty_like(current_padded) for _ in range(world_size)]
    gathered_valid = [torch.empty_like(valid_padded) for _ in range(world_size)]
    if distributed:
        dist.all_gather(gathered_current, current_padded)
        dist.all_gather(gathered_valid, valid_padded)
    else:
        gathered_current[0].copy_(current_padded)
        gathered_valid[0].copy_(valid_padded)
    global_current = torch.cat(gathered_current, dim=0)
    global_next = _DifferentiableAllGather.apply(next_padded)
    global_valid = torch.cat(gathered_valid, dim=0).bool()
    valid_count = int(global_valid.sum().item())
    return (
        global_current[global_valid],
        global_next[global_valid],
        valid_count,
    )


@contextlib.contextmanager
def shared_sigreg_rng(seed: int, device: torch.device) -> Iterator[None]:
    """只在 SIGReg 内使用跨 rank 相同的随机投影，并恢复各 rank 原 RNG。"""

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        yield
