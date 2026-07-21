from nimloth.training.sft2.data.samplers import TrajectoryAwareBatchSampler
from nimloth.rollout.transitions import TransitionSample


def _sample(record_id: str, step: int) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[""] * (step + 1),
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
    record_ids_per_batch = [
        sorted({samples[i].record_id for i in batch}) for batch in batches
    ]
    assert record_ids_per_batch == [["a"], ["b"], ["c"]]
    assert [len(batch) for batch in batches] == [3, 2, 1]


def test_full_trajectory_sampler_ddp_partitions_evenly() -> None:
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
    r0_records = {samples[batch[0]].record_id for batch in rank0}
    r1_records = {samples[batch[0]].record_id for batch in rank1}
    assert r0_records.isdisjoint(r1_records)
    assert r0_records | r1_records == {"a", "b", "c", "d"}


def test_full_trajectory_sampler_ignores_batch_size() -> None:
    samples = [_sample("a", 0), _sample("a", 1)]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
    )
    assert len(list(sampler)[0]) == 2


def test_full_trajectory_chunks_by_image_count() -> None:
    samples = [_sample("a", index) for index in range(8)]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
        max_images_per_batch=32,
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert [len(batch) for batch in batches] == [7, 1]


def test_full_trajectory_single_prefix_can_exceed_image_budget() -> None:
    samples = [_sample("a", index) for index in range(5)]
    sampler = TrajectoryAwareBatchSampler(
        samples,
        batch_size=1,
        shuffle=False,
        full_trajectory=True,
        max_images_per_batch=3,
    )
    assert list(sampler) == [[0, 1], [2], [3], [4]]


def test_full_trajectory_hard_step_ceiling() -> None:
    samples = [_sample("a", index) for index in range(20)]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
        max_images_per_batch=1000,
        max_steps_per_trajectory=6,
    )
    batches = list(sampler)
    assert len(batches) == 4
    assert [len(batch) for batch in batches] == [6, 6, 6, 2]
