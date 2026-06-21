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
