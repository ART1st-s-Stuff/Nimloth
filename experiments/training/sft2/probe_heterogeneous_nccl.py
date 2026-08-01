#!/usr/bin/env python3
"""Minimal NCCL probe for heterogeneous 8+4+4 torchrun agents."""

from __future__ import annotations

import json
import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        value = torch.tensor(float(rank + 1), device="cuda")
        dist.all_reduce(value)
        expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
        assert value.item() == expected
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": dist.get_world_size(),
                    "local_rank": local_rank,
                    "local_world_size": int(os.environ["LOCAL_WORLD_SIZE"]),
                    "cuda_device_count": torch.cuda.device_count(),
                    "host": socket.gethostname(),
                    "sum": value.item(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
