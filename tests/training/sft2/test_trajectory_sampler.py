from nimloth.rollout.transitions import TransitionSample
from nimloth.training.sft2.data.samplers import (
    FutureRolloutBatchSampler,
    OnlineHistoryBatchSampler,
)


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


def _assert_cache_order(
    batches,
    samples: list[TransitionSample],
) -> set[int]:
    seen: set[int] = set()
    for batch in batches:
        assert batch
        context_length = batch[0].context_length
        assert all(row.context_length == context_length for row in batch)
        assert len(batch) % context_length == 0
        for start in range(0, len(batch), context_length):
            window = batch[start : start + context_length]
            assert [row.is_current_step for row in window] == [
                *([False] * (context_length - 1)),
                True,
            ]
            if window[-1].loss_weight == 0.0:
                assert all(row.loss_weight == 0.0 for row in window)
                continue
            for history in window[:-1]:
                assert history.sample_index in seen
            current = window[-1].sample_index
            assert current not in seen
            seen.add(current)
            trajectory = [samples[row.sample_index] for row in window]
            assert len({sample.record_id for sample in trajectory}) == 1
            assert [sample.step_index for sample in trajectory] == list(
                range(
                    trajectory[0].step_index,
                    trajectory[-1].step_index + 1,
                )
            )
    return seen


def test_online_sampler_emits_each_current_once_after_its_history() -> None:
    samples = [
        *[_sample("a", step) for step in range(4)],
        *[_sample("b", step) for step in range(3)],
        *[_sample("c", step) for step in range(2)],
    ]
    sampler = OnlineHistoryBatchSampler(
        samples,
        history_size=3,
        batch_size=2,
        shuffle=False,
        pad_to_equal_batches=False,
    )
    batches = list(sampler)

    assert _assert_cache_order(batches, samples) == set(range(len(samples)))
    assert sampler.window_count == len(samples)
    assert max(sampler.current_steps_per_batch) == 2


def test_online_sampler_restarts_context_at_gaps() -> None:
    samples = [
        _sample("a", 0),
        _sample("a", 2),
        _sample("b", 0),
        _sample("b", 1, has_next=False),
        _sample("c", 0),
        _sample("c", 1),
    ]
    sampler = OnlineHistoryBatchSampler(
        samples,
        history_size=2,
        batch_size=4,
        shuffle=False,
        pad_to_equal_batches=False,
    )

    assert _assert_cache_order(list(sampler), samples) == {0, 1, 2, 4, 5}


def test_distributed_sampler_owns_trajectories_and_pads_only_with_zero_loss() -> None:
    samples = [
        *[_sample("a", step) for step in range(4)],
        *[_sample("b", step) for step in range(4)],
        *[_sample("c", step) for step in range(3)],
        *[_sample("d", step) for step in range(2)],
        _sample("e", 0),
    ]
    samplers = [
        OnlineHistoryBatchSampler(
            samples,
            history_size=3,
            batch_size=2,
            num_replicas=2,
            rank=rank,
            shuffle=False,
            pad_to_equal_batches=True,
        )
        for rank in range(2)
    ]
    rank_batches = [list(sampler) for sampler in samplers]

    assert len(rank_batches[0]) == len(rank_batches[1])
    rank_seen = [
        _assert_cache_order(batches, samples)
        for batches in rank_batches
    ]
    assert rank_seen[0].isdisjoint(rank_seen[1])
    assert rank_seen[0] | rank_seen[1] == set(range(len(samples)))
    assert sum(sampler.padding_batch_count for sampler in samplers) > 0


def test_validation_sampler_partitions_without_duplication_or_padding() -> None:
    samples = [_sample(record, step) for record in "abcde" for step in (0, 1)]
    samplers = [
        OnlineHistoryBatchSampler(
            samples,
            history_size=2,
            batch_size=1,
            num_replicas=3,
            rank=rank,
            shuffle=False,
            pad_to_equal_batches=False,
        )
        for rank in range(3)
    ]
    rank_seen = [_assert_cache_order(list(sampler), samples) for sampler in samplers]

    assert set().union(*rank_seen) == set(range(len(samples)))
    assert sum(len(indices) for indices in rank_seen) == len(samples)
    assert all(sampler.padding_batch_count == 0 for sampler in samplers)


def test_epoch_shuffle_changes_group_order_but_not_trajectory_time_order() -> None:
    samples = [_sample(record, step) for record in "abcdef" for step in range(3)]
    sampler = OnlineHistoryBatchSampler(
        samples,
        history_size=2,
        batch_size=2,
        shuffle=True,
        seed=7,
        pad_to_equal_batches=False,
    )

    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert _sample_indices(epoch_zero) != _sample_indices(epoch_one)
    assert _assert_cache_order(epoch_zero, samples) == set(range(len(samples)))
    assert _assert_cache_order(epoch_one, samples) == set(range(len(samples)))


def test_future_rollout_sampler_emits_sliding_recorded_t4_windows() -> None:
    samples = [
        *[_sample("a", step) for step in range(6)],
        *[_sample("b", step) for step in range(3)],
    ]
    sampler = FutureRolloutBatchSampler(
        samples,
        prediction_horizon=4,
        batch_size=1,
        shuffle=False,
        pad_to_equal_batches=False,
    )

    batches = list(sampler)
    windows = [
        tuple(row.sample_index for row in batch)
        for batch in batches
    ]

    assert windows == [
        (0, 1, 2, 3),
        (1, 2, 3, 4),
        (2, 3, 4, 5),
    ]
    assert sampler.window_count == 3
    for batch in batches:
        assert [row.rollout_position for row in batch] == [0, 1, 2, 3]
        assert all(row.prediction_horizon == 4 for row in batch)


def test_future_rollout_sampler_never_crosses_gap_or_record_boundary() -> None:
    samples = [
        *[_sample("a", step) for step in (0, 1, 3, 4, 5, 6)],
        *[_sample("b", step) for step in range(4)],
    ]
    sampler = FutureRolloutBatchSampler(
        samples,
        prediction_horizon=4,
        batch_size=2,
        shuffle=False,
        pad_to_equal_batches=False,
    )

    windows = []
    for batch in sampler:
        for start in range(0, len(batch), 4):
            window = batch[start : start + 4]
            trajectory = [samples[row.sample_index] for row in window]
            windows.append(
                (trajectory[0].record_id, tuple(sample.step_index for sample in trajectory))
            )

    assert set(windows) == {("a", (3, 4, 5, 6)), ("b", (0, 1, 2, 3))}
