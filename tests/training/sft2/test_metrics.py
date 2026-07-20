from __future__ import annotations

from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from nimloth.training.common.metrics import MetricAccumulator
from nimloth.training.sft2.metrics import batch_step_success_rate
from nimloth.training.sft2.dataset import DistributedEvalSampler
from nimloth.training.sft2.evaluate import (
    distributed_metric_averages,
    merge_metric_accumulators,
)


def _distributed_metric_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        accumulator = MetricAccumulator()
        accumulator.update({"wm_mse": 1.0 + 2.0 * rank}, count=rank + 1)
        assert distributed_metric_averages(accumulator) == {"wm_mse": 7.0 / 3.0}
    finally:
        dist.destroy_process_group()


def test_batch_step_success_rate() -> None:
    items = [{"success": True}, {"success": False}]
    assert batch_step_success_rate(items) == 0.5


def test_distributed_eval_sampler_partitions_without_duplicates() -> None:
    dataset = list(range(7))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank))
        for rank in range(3)
    ]

    assert sorted(index for shard in shards for index in shard) == list(range(7))
    assert sum(map(len, shards)) == 7


def test_merge_metric_accumulators_uses_all_rank_sums_and_counts() -> None:
    metrics = merge_metric_accumulators(
        [
            ({"wm_mse": 2.0}, {"wm_mse": 2}),
            ({"wm_mse": 9.0, "value_total": 4.0}, {"wm_mse": 3, "value_total": 2}),
        ]
    )

    assert metrics == {"wm_mse": 2.2, "value_total": 2.0}


def test_distributed_metric_averages_gathers_two_ranks(tmp_path: Path) -> None:
    mp.spawn(
        _distributed_metric_worker,
        args=(str(tmp_path / "gloo-init"),),
        nprocs=2,
        join=True,
    )
