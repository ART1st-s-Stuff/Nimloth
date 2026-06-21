"""Tests for trajectory-grouped batching used by packed forward."""

from __future__ import annotations

from nimloth.training.sft2.trajectory_batching import (
    TrajectoryDistributedBatchSampler,
    assert_packed_batch,
    build_record_trajectory_batches,
    build_trajectory_batch_indices,
)
from nimloth.wm.dataset import TransitionSample


def _sample(record_id: str, step_index: int) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step_index,
        prefix_messages=[],
        prefix_image_paths=[],
        action_index=0,
        current_image_path="a.png",
        next_image_path="b.png",
    )


def test_build_trajectory_batch_indices_respects_record_boundary() -> None:
    samples = [_sample("a", i) for i in range(19)] + [_sample("b", 0), _sample("b", 1)]
    batches = build_trajectory_batch_indices(samples, batch_size=2)
    assert [18] in batches
    assert [19, 20] in batches
    assert all(len({samples[i].record_id for i in batch}) == 1 for batch in batches)


def test_build_record_trajectory_batches_one_per_record() -> None:
    samples = [_sample("a", 0), _sample("a", 1), _sample("b", 0)]
    batches = build_record_trajectory_batches(samples)
    assert batches == [[0, 1], [2]]


def test_assert_packed_batch_accepts_consecutive_steps() -> None:
    steps = [_sample("r1", 0), _sample("r1", 1)]
    assert_packed_batch(steps)


def test_distributed_batch_sampler_shards_batches() -> None:
    samples = [_sample("a", i) for i in range(6)]
    sampler = TrajectoryDistributedBatchSampler(
        samples,
        2,
        rank=1,
        world_size=2,
        shuffle=False,
        seed=0,
    )
    assert list(sampler) == [[2, 3]]
