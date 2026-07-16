"""Validate separate NCCL/Gloo groups for mixed per-rank device placement."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    layouts = {
        3: ((0, 2, 4), (1, 2, 4)),
        4: ((0, 2, 4, 0), (1, 2, 4, 1)),
    }
    primary_devices, auxiliary_devices = layouts[world]
    primary = primary_devices[rank]
    auxiliary = auxiliary_devices[rank]
    torch.cuda.set_device(primary)
    dist.init_process_group("nccl")
    dist.barrier(device_ids=[primary])
    aux_group = dist.new_group(backend="nccl")
    cpu_group = dist.new_group(backend="gloo")

    aux_value = torch.tensor([rank + 1.0], device=f"cuda:{auxiliary}")
    dist.all_reduce(aux_value, group=aux_group)
    expected = world * (world + 1) / 2
    assert aux_value.item() == expected

    cpu_value = torch.tensor([rank + 1.0])
    dist.all_reduce(cpu_value, group=cpu_group)
    assert cpu_value.item() == expected
    print({"rank": rank, "primary": primary, "auxiliary": auxiliary, "sum": cpu_value.item()})
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
