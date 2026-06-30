from nimloth.training.sft2.trajectory_sampler import TrajectoryAwareBatchSampler
from nimloth.wm.dataset import TransitionSample


def _sample(record_id: str, step: int) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[""] * (step + 1),  # cumulative prefix images
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
    """With default max_images_per_batch=32, short records span a single batch."""
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
    """batch_size irrelevant when full_trajectory=True; max_images_per_batch controls chunking."""
    samples = [_sample("a", 0), _sample("a", 1)]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
    )
    assert len(list(sampler)[0]) == 2  # both steps in one batch (below max_images=32)


def test_full_trajectory_chunks_by_image_count() -> None:
    """Records with cumulative images > max_images_per_batch are split by image ceiling."""
    # 5 steps: prefix images = 1+2+3+4+5 = 15 (≤32) → one batch
    # 8 steps: prefix images = 1+2+...+8 = 36 (>32) → must split
    #   first 7 steps = 28 images → batch1
    #   step 7 = 8 images → batch2
    samples = [_sample("a", i) for i in range(8)]
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
        max_images_per_batch=32,
    )
    batches = list(sampler)
    assert len(batches) == 2
    assert [len(b) for b in batches] == [7, 1]


def test_full_trajectory_hard_step_ceiling() -> None:
    """max_steps_per_trajectory acts as a hard ceiling even if images are below limit."""
    samples = [_sample("a", i) for i in range(20)]  # step 0..19, each has only step_index+1 images
    sampler = TrajectoryAwareBatchSampler(
        samples, batch_size=1, shuffle=False, full_trajectory=True,
        max_images_per_batch=1000,  # effectively no image limit
        max_steps_per_trajectory=6,
    )
    batches = list(sampler)
    # 20 steps / 6 = 4 batches: 6 + 6 + 6 + 2
    assert len(batches) == 4
    assert [len(b) for b in batches] == [6, 6, 6, 2]
