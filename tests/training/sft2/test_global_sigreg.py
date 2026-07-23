from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.training.sft2.algorithm import (
    gather_global_sigreg_states,
    shared_sigreg_rng,
)
from nimloth.wm import SequenceSIGReg


def _global_sigreg_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        encoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            encoder.weight.fill_(1.0)
        encoder = DDP(encoder, static_graph=True)

        if rank == 0:
            local_input = torch.tensor([[1.0], [2.0]])
            valid = torch.tensor([True, True])
        else:
            # 物理 B 和 rank0 不同，且该 rank 模拟 sampler 的整批 padding。
            local_input = torch.tensor([[3.0]])
            valid = torch.tensor([False])
        local_next = encoder(local_input)
        local_current = (local_input + 10.0).detach()

        global_current, global_next, valid_count = gather_global_sigreg_states(
            local_current,
            local_next,
            valid,
        )
        assert valid_count == 2
        torch.testing.assert_close(global_current, torch.tensor([[11.0], [12.0]]))
        torch.testing.assert_close(global_next.detach(), torch.tensor([[1.0], [2.0]]))

        # 所有 rank 使用同一个 global loss。gather backward 与 DDP average 合并后，
        # 应等于 d mean((w*[1,2])**2) / dw = 5。
        global_next.square().mean().backward()
        torch.testing.assert_close(encoder.module.weight.grad, torch.tensor([[5.0]]))

        # rank-local RNG 起点不同，但 SIGReg 内的随机投影必须完全相同。
        torch.manual_seed(100 + rank)
        states = torch.stack((global_current, global_next.detach()), dim=1)
        with shared_sigreg_rng(77, states.device):
            sigreg_loss = SequenceSIGReg(knots=3, num_proj=8)(states)
        assert sigreg_loss is not None
        gathered_losses = [torch.empty_like(sigreg_loss) for _ in range(2)]
        dist.all_gather(gathered_losses, sigreg_loss)
        torch.testing.assert_close(gathered_losses[0], gathered_losses[1])
    finally:
        dist.destroy_process_group()


def test_global_sigreg_gather_matches_ddp_global_batch_gradient(tmp_path: Path) -> None:
    mp.spawn(
        _global_sigreg_worker,
        args=(str(tmp_path / "gloo-global-sigreg"),),
        nprocs=2,
        join=True,
    )
