"""Every eligible trajectory is owned once, with complete-window zero-weight padding."""
from dataclasses import replace

import pytest

from nimloth.training.sft.stage3.data.samplers import TrajectoryBatchSampler
from nimloth.training.sft.stage3.data.trajectory import TrajectoryDataset


def test_sampler_owns_whole_records_and_all_windows_once(trajectory_factory):
    samples = []
    for i, length in enumerate((4, 5, 6, 7, 8)):
        samples.extend(trajectory_factory(record=str(i), length=length)[1].samples)
    dataset = TrajectoryDataset(samples, prediction_horizon=4)
    samplers = [TrajectoryBatchSampler(dataset, batch_size=2, num_replicas=3, rank=rank)
                for rank in range(3)]
    assert len({len(sampler) for sampler in samplers}) == 1
    owned = [item.index for sampler in samplers for batch in sampler for item in batch if item.loss_weight]
    assert sorted(owned) == list(range(5))
    assert sum(sum(sampler.current_steps_per_batch) for sampler in samplers) == dataset.window_count == 15
    assert sum(sum(sampler.trajectories_per_batch) for sampler in samplers) == 5
    for sampler in samplers:
        for batch in sampler:
            for index in batch:
                item = dataset[index]
                assert [sample.step_index for sample in item.samples] == list(range(len(item.samples)))


def test_distributed_padding_never_supervises_duplicate_records(trajectory_factory):
    samples = list(trajectory_factory(length=5)[1].samples)
    dataset = TrajectoryDataset(samples, prediction_horizon=4)
    left = TrajectoryBatchSampler(dataset, batch_size=1, num_replicas=2, rank=0)
    right = TrajectoryBatchSampler(dataset, batch_size=1, num_replicas=2, rank=1)
    assert left.trajectories_per_batch == (1,)
    assert right.trajectories_per_batch == (0,)
    assert right.padding_batch_count == 1
    assert dataset[next(iter(right))[0]].loss_weight == 0


def test_epoch_shuffle_preserves_trajectory_membership_and_reproducible_cursor(trajectory_factory):
    samples = [sample for i in range(12) for sample in trajectory_factory(str(i))[1].samples]
    dataset = TrajectoryDataset(samples, prediction_horizon=4)
    sampler = TrajectoryBatchSampler(dataset, batch_size=2, seed=32)
    before = list(sampler)
    sampler.set_epoch(3)
    after = list(sampler)
    assert before != after
    assert list(sampler) == after
    assert sorted(index.index for batch in before for index in batch) == sorted(index.index for batch in after for index in batch)


def test_dataset_rejects_gaps_duplicates_and_missing_terminal(trajectory_factory):
    samples = list(trajectory_factory()[1].samples)
    with pytest.raises(ValueError, match="missing or duplicate"):
        TrajectoryDataset(samples[:2] + samples[3:], prediction_horizon=4)
    with pytest.raises(ValueError, match="missing or duplicate"):
        TrajectoryDataset(samples + [samples[0]], prediction_horizon=4)
    samples[-1] = replace(samples[-1], next_prefix_messages=None)
    with pytest.raises(ValueError, match="next-state"):
        TrajectoryDataset(samples, prediction_horizon=4)


def test_short_trajectories_omitted_with_explicit_counts(trajectory_factory):
    long = list(trajectory_factory(length=6)[1].samples)
    short = [replace(sample, record_id="short") for sample in long[:2]]
    dataset = TrajectoryDataset(long + short, prediction_horizon=4)
    assert len(dataset) == 1
    assert dataset.window_count == 3
    assert dataset.omitted_short_trajectories == 1
