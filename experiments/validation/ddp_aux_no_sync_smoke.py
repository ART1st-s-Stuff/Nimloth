"""Exercise non-static auxiliary DDP with GA-boundary synchronization."""

from __future__ import annotations

import contextlib
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _devices(rank: int) -> tuple[int, int]:
    primary = (0, 2, 4, 0)[rank]
    auxiliary = (1, 2, 4, 1)[rank]
    return primary, auxiliary


def _assert_synced(model: DDP, group) -> None:
    weight = model.module.weight.detach()
    gathered = [torch.empty_like(weight) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, weight, group=group)
    assert max((item - gathered[0]).abs().max().item() for item in gathered) == 0.0


def _train(model: DDP, group, rank: int, auxiliary: int) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(8):
            ctx = model.no_sync() if micro < 7 else contextlib.nullcontext()
            with ctx:
                value = rank + step + micro / 10
                x = torch.full((2, 4), value, device=f"cuda:{auxiliary}")
                target = torch.full((2, 4), rank / 3, device=x.device)
                loss = ((model(x) - target).square().mean() + (model(x / 2) - target).square().mean()) / 8
                loss.backward()
        optimizer.step()
        _assert_synced(model, group)


def main() -> int:
    rank = int(os.environ["RANK"])
    primary, auxiliary = _devices(rank)
    torch.cuda.set_device(primary)
    dist.init_process_group("nccl")
    dist.barrier(device_ids=[primary])
    aux_group = dist.new_group(backend="nccl")
    torch.manual_seed(7)
    module = torch.nn.Linear(4, 4, bias=False).to(f"cuda:{auxiliary}")
    model = DDP(module, process_group=aux_group, device_ids=[auxiliary], static_graph=False)
    _train(model, aux_group, rank, auxiliary)
    print({"rank": rank, "primary": primary, "auxiliary": auxiliary, "status": "PASS"})
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
