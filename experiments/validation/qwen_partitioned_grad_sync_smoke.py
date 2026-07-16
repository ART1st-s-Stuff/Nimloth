"""Validate two-device GPU NCCL gradient averaging across heterogeneous ranks."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from nimloth.training.common.grad_sync import (
    assert_consistent_relative_placement,
    average_partitioned_module_gradients,
)


def _devices(rank: int) -> tuple[int, int]:
    if os.environ.get("NIMLOTH_SMOKE_LOCAL_PAIR") == "1":
        return 0, 1
    return (0, 2, 4, 0)[rank], (1, 3, 5, 1)[rank]


class TwoDeviceParams(nn.Module):
    def __init__(self, primary: int, secondary: int) -> None:
        super().__init__()
        self.primary = nn.Parameter(torch.arange(12, device=f"cuda:{primary}").reshape(3, 4).float())
        self.secondary = nn.Parameter(torch.arange(10, device=f"cuda:{secondary}").reshape(2, 5).float())


def main() -> int:
    rank = int(os.environ["RANK"])
    primary, secondary = _devices(rank)
    torch.cuda.set_device(primary)
    dist.init_process_group("nccl")
    dist.barrier(device_ids=[primary])
    dist.new_group(backend="nccl")  # mirror trainer's auxiliary group creation
    placement_group = dist.new_group(backend="gloo")
    grad_groups = tuple(dist.new_group(backend="nccl") for _ in range(2))
    module = TwoDeviceParams(primary, secondary)
    placement = assert_consistent_relative_placement(
        module, primary_device=primary, stride=2, group=placement_group
    )
    for param in module.parameters():
        param.grad = torch.full_like(param, rank + 1.0)
    average_partitioned_module_gradients(module, groups=grad_groups, primary_device=primary)
    assert all(torch.equal(param.grad, torch.full_like(param, 2.5)) for param in module.parameters())
    print({"rank": rank, "devices": (primary, secondary), "placement": placement, "status": "PASS"})
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
