"""SFT2 autoregressive WM rollout 的真实 DDP forward/backward 回归。"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.wm.grid import (
    GridPredictorConfig,
    GridWorldModel,
    TemporalSpatialGridPredictor,
)


def _rollout_ddp_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        torch.manual_seed(7)
        predictor = TemporalSpatialGridPredictor(
            GridPredictorConfig(
                grid_tokens=2,
                emb_dim=4,
                action_dim=3,
                history_size=1,
                depth=1,
                heads=1,
                dim_head=4,
                mlp_dim=8,
                dropout=0.0,
            )
        )
        wrapped_predictor = DDP(
            predictor,
            device_ids=None,
            find_unused_parameters=False,
            static_graph=True,
        )
        world_model = GridWorldModel(
            state_proj=torch.nn.Identity(),
            wm_predictor=wrapped_predictor,
            value_head=torch.nn.Identity(),
        )
        optimizer = torch.optim.SGD(wrapped_predictor.parameters(), lr=1e-3)

        for iteration in range(2):
            optimizer.zero_grad(set_to_none=True)
            state_history = (
                torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 4)
                + rank
                + iteration * 0.25
            )
            future_actions = torch.tensor(
                [[rank % 3, (rank + 1) % 3, 2, iteration % 3]],
                dtype=torch.long,
            )
            predicted = world_model.simulate_action_sequences(
                state_history,
                torch.empty((1, 0), dtype=torch.long),
                future_actions,
            )
            assert predicted.shape == (1, 4, 2, 4)
            target = torch.full_like(predicted, float(rank + iteration))
            predicted.sub(target).square().mean().backward()

            gradients = torch.cat(
                [
                    parameter.grad.reshape(-1)
                    for parameter in predictor.parameters()
                    if parameter.grad is not None
                ]
            )
            assert torch.count_nonzero(gradients) > 0
            gathered = [torch.empty_like(gradients) for _ in range(2)]
            dist.all_gather(gathered, gradients)
            torch.testing.assert_close(gathered[0], gathered[1])
            optimizer.step()
    finally:
        dist.destroy_process_group()


def test_t4_grid_rollout_uses_ddp_forward_and_synchronizes_gradients(
    tmp_path: Path,
) -> None:
    mp.spawn(
        _rollout_ddp_worker,
        args=(str(tmp_path / "gloo-wm-rollout"),),
        nprocs=2,
        join=True,
    )
