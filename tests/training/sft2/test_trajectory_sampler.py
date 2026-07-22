from nimloth.rollout.transitions import TransitionSample
from nimloth.training.sft2.data.samplers import TrajectoryWindowBatchSampler


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


def test_sampler_builds_sliding_h_step_windows() -> None:
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

    assert list(sampler) == [[0, 1, 1, 2], [3, 4]]
    assert sampler.window_count == 3


def test_sampler_skips_gaps_and_missing_next_states() -> None:
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

    assert list(sampler) == [[4, 5]]


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

    assert list(rank0) == [[0, 1, 2, 3], [8, 9]]
    assert list(rank1) == [[4, 5, 6, 7], [0, 1, 2, 3]]
    assert len(rank0) == len(rank1) == 2


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

    flattened = [tuple(batch) for batches in rank_batches for batch in batches]
    assert sorted(flattened) == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
    ]


def test_image_budget_uses_complete_window_cost_and_keeps_oversized_window() -> None:
    samples = [_sample("a", step) for step in range(4)]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
        max_images_per_batch=20,
    )

    # 三条 forward 顺序执行，成本取其中最大值；预算不会拆开一个 H 窗口。
    assert list(sampler) == [[0, 1, 1, 2], [2, 3]]


def test_transition_row_budget_never_splits_a_window() -> None:
    samples = [_sample("a", step) for step in range(5)]
    sampler = TrajectoryWindowBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
        max_transition_rows_per_batch=3,
    )

    assert list(sampler) == [[0, 1], [1, 2], [2, 3], [3, 4]]


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
    windows = sorted(
        tuple(batch[index : index + 2])
        for batch in epoch_zero
        for index in range(0, len(batch), 2)
    )
    assert windows == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
