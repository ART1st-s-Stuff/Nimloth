from nimloth.training.sft2.trajectory_sampler import TrajectoryAwareBatchSampler
from nimloth.wm.dataset import TransitionSample


def _sample(record_id: str, step: int) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[],
        action_index=0,
        current_image_path="",
        next_image_path="",
    )


def test_trajectory_aware_sampler_groups_consecutive_steps() -> None:
    samples = [_sample("a", 0), _sample("a", 1), _sample("a", 2), _sample("b", 0)]
    sampler = TrajectoryAwareBatchSampler(samples, batch_size=2, shuffle=False)

    assert list(sampler) == [[0, 1], [2], [3]]


def test_trajectory_aware_sampler_partitions_batches_across_ranks() -> None:
    samples = [_sample("a", 0), _sample("a", 1), _sample("b", 0), _sample("b", 1), _sample("c", 0)]

    rank0 = TrajectoryAwareBatchSampler(
        samples, batch_size=2, num_replicas=2, rank=0, shuffle=False
    )
    rank1 = TrajectoryAwareBatchSampler(
        samples, batch_size=2, num_replicas=2, rank=1, shuffle=False
    )

    assert list(rank0) == [[0, 1], [4]]
    assert list(rank1) == [[2, 3], [0, 1]]
    assert len(rank0) == len(rank1) == 2


def test_full_trajectory_sampler_each_record_is_one_batch() -> None:
    """When full_trajectory=True, every batch contains ALL transitions for one record."""
    samples = [
        _sample("a", 0), _sample("a", 1), _sample("a", 2),
        _sample("b", 0), _sample("b", 1),
        _sample("c", 0),
    ]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=2, shuffle=False, full_trajectory=True,
    )
    batches = list(sampler)
    assert len(batches) == 3
    # Each batch contains all steps of one record, sorted by step_index.
    record_ids_per_batch = [
        sorted({samples[i].record_id for i in batch}) for batch in batches
    ]
    assert record_ids_per_batch == [["a"], ["b"], ["c"]]
    # Verify step counts.
    assert [len(b) for b in batches] == [3, 2, 1]


def test_full_trajectory_sampler_ddp_partitions_evenly() -> None:
    """DDP: each rank gets same number of micro-batches; different trajectories."""
    samples = [
        _sample("a", 0), _sample("a", 1),
        _sample("b", 0),
        _sample("c", 0), _sample("c", 1),
        _sample("d", 0),
    ]
    rank0 = TrajectoryAwareBatchSampler(
        samples, batch_size=1,
        num_replicas=2, rank=0, shuffle=False, full_trajectory=True,
    )
    rank1 = TrajectoryAwareBatchSampler(
        samples, batch_size=1,
        num_replicas=2, rank=1, shuffle=False, full_trajectory=True,
    )
    assert len(rank0) == len(rank1) == 2
    # Ranks have disjoint record sets.
    r0_records = {samples[b[0]].record_id for b in rank0}
    r1_records = {samples[b[0]].record_id for b in rank1}
    assert r0_records.isdisjoint(r1_records)
    assert r0_records | r1_records == {"a", "b", "c", "d"}


def test_full_trajectory_sampler_ignores_batch_size() -> None:
    """batch_size is irrelevant when full_trajectory=True."""
    samples = [_sample("a", 0), _sample("a", 1)]
    # batch_size=1 should not truncate the trajectory.
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
    )
    assert len(list(sampler)[0]) == 2  # both steps in one batch
