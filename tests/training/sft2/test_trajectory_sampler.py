from nimloth.rollout.transitions import TransitionSample
from nimloth.training.sft2.data.samplers import TrajectoryWindowBatchSampler


def _sample_indices(batches):
    return [[row.sample_index for row in batch] for batch in batches]


def _sample(
    record_id: str,
    step: int,
    *,
    has_next: bool = True,
) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[""] * (step + 1),
        action_index=0,
        current_image_path="",
        next_image_path="",
        next_prefix_messages=[] if has_next else None,
        next_prefix_image_paths=[""] * (step + 2) if has_next else None,
    )


def test_sampler_assigns_each_step_one_real_context() -> None:
    samples = [
        _sample("a", 0),
        _sample("a", 1),
        _sample("a", 2),
        _sample("b", 0),
        _sample("b", 1),
    ]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=2,
        shuffle=False,
    )

    batches = list(sampler)
    assert _sample_indices(batches) == [[0, 3], [0, 1, 1, 2], [3, 4]]
    assert sampler.window_count == 5
    current_indices = [
        row.sample_index
        for batch in batches
        for row in batch
        if row.is_current_step
    ]
    assert current_indices == [0, 3, 1, 2, 4]


def test_sampler_restarts_context_at_gaps_and_skips_only_missing_target_step() -> None:
    samples = [
        _sample("a", 0),
        _sample("a", 2),
        _sample("b", 0),
        _sample("b", 1, has_next=False),
        _sample("c", 0),
        _sample("c", 1),
    ]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
    )

    assert _sample_indices(list(sampler)) == [[0, 1, 2, 4], [4, 5]]


def test_training_sampler_pads_batch_count_across_ranks() -> None:
    samples = [_sample(record, step) for record in "abcde" for step in (0, 1)]
    rank0 = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=2,
        num_replicas=2,
        rank=0,
        shuffle=False,
    )
    rank1 = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=2,
        num_replicas=2,
        rank=1,
        shuffle=False,
    )

    assert _sample_indices(list(rank0)) == [[0, 2], [8], [4, 5, 6, 7]]
    assert _sample_indices(list(rank1)) == [[4, 6], [0, 1, 2, 3], [8, 9]]
    assert len(rank0) == len(rank1) == 3


def test_validation_sampler_partitions_without_duplication() -> None:
    samples = [_sample(record, step) for record in "abcde" for step in (0, 1)]
    rank_batches = [
        list(
            TrajectoryWindowBatchSampler(
                samples,
                history_size=2,
                batch_size=1,
                num_replicas=3,
                rank=rank,
                shuffle=False,
                pad_to_equal_batches=False,
            )
        )
        for rank in range(3)
    ]

    flattened = [
        tuple(row.sample_index for row in batch)
        for batches in rank_batches
        for batch in batches
    ]
    assert sorted(flattened) == [
        (0,),
        (0, 1),
        (2,),
        (2, 3),
        (4,),
        (4, 5),
        (6,),
        (6, 7),
        (8,),
        (8, 9),
    ]


def test_image_budget_uses_peak_row_cost_for_chunked_forward() -> None:
    samples = [_sample("a", step) for step in range(4)]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
        max_images_per_batch=5,
        backbone_rows_per_forward=1,
    )

    assert _sample_indices(list(sampler)) == [[0], [0, 1, 1, 2, 2, 3]]


def test_transition_row_budget_never_splits_a_window() -> None:
    samples = [_sample("a", step) for step in range(5)]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
        max_transition_rows_per_batch=3,
    )

    assert _sample_indices(list(sampler)) == [
        [0],
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
    ]


def test_random_mode_shuffles_complete_windows_before_batching() -> None:
    samples = [_sample(record, step) for record in "abcd" for step in (0, 1)]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=2,
        shuffle=True,
        shuffle_windows=True,
        seed=7,
    )

    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    windows = []
    for batch in epoch_zero:
        context_length = batch[0].context_length
        assert all(row.context_length == context_length for row in batch)
        windows.extend(
            tuple(row.sample_index for row in batch[index : index + context_length])
            for index in range(0, len(batch), context_length)
        )
    windows.sort()
    assert windows == [
        (0,),
        (0, 1),
        (2,),
        (2, 3),
        (4,),
        (4, 5),
        (6,),
        (6, 7),
    ]
