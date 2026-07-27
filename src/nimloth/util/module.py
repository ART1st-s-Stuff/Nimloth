"""训练和评估共用的 PyTorch module 状态工具。"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import torch


DeviceTarget = torch.device | torch.nn.Module | torch.Tensor


def move_to_device(
    tensor: torch.Tensor,
    target: DeviceTarget,
) -> torch.Tensor:
    """把 tensor 移到 target 所在设备；浮点 tensor 同时采用目标 dtype。"""

    if isinstance(target, torch.nn.Module):
        target = next(target.parameters())
    if isinstance(target, torch.Tensor):
        dtype = target.dtype if tensor.is_floating_point() else tensor.dtype
        return tensor.to(
            device=target.device,
            dtype=dtype,
            non_blocking=True,
        )
    return tensor.to(device=target, non_blocking=True)


@contextmanager
def evaluating(module: torch.nn.Module) -> Iterator[None]:
    """临时关闭 dropout 等训练行为，退出时恢复原始模式。"""

    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)
