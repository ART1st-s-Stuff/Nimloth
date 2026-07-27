"""SFT2 单步 ownership 与真实短历史的 DataLoader batch sampler。"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from nimloth.rollout.transitions import (
    TransitionContextIndex,
    TransitionRolloutIndex,
    TransitionSample,
)


class OnlineHistoryBatchSampler(Sampler[list[TransitionContextIndex]]):
    """按 rank 固定 trajectory lane，保证 detached history cache 先写后读。

    一个 lane group 最多含 ``batch_size`` 条 trajectory segment。每次只推进各
    trajectory 的一个 current step，完整 group 在同一 rank 上按时间顺序执行；
    因此后续 step 的 H-1 个历史 state 一定已经由更早 microbatch 写入 cache。
    训练时用零权重的完整 padding batch 对齐 rank 迭代数，不重复任何真实 loss。
    """

    def __init__(
        self,
        samples: Sequence[TransitionSample],
        *,
        history_size: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        pad_to_equal_batches: bool = True,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(
                f"rank {rank} out of range for num_replicas={num_replicas}"
            )
        self.samples = samples
        self.history_size = int(history_size)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pad_to_equal_batches = bool(pad_to_equal_batches)
        self.epoch = 0

        sequences = self._build_sequences(samples, self.history_size)
        self.window_count = sum(len(sequence) for sequence in sequences)
        groups = self._lane_groups(sequences, self.batch_size, self.seed)
        if groups and self.pad_to_equal_batches and len(groups) < self.num_replicas:
            raise ValueError(
                "online history training requires at least one trajectory lane "
                f"group per rank, got groups={len(groups)}, ranks={self.num_replicas}"
            )
        self._rank_groups = self._assign_groups(groups, self.num_replicas)
        real_counts = [
            sum(max(len(sequence) for sequence in group) for group in rank_groups)
            for rank_groups in self._rank_groups
        ]
        self._real_batch_count = real_counts[self.rank]
        self.num_batches = (
            max(real_counts, default=0)
            if self.pad_to_equal_batches
            else self._real_batch_count
        )

    @staticmethod
    def _build_sequences(
        samples: Sequence[TransitionSample],
        history_size: int,
    ) -> list[tuple[tuple[int, ...], ...]]:
        by_record: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            by_record[sample.record_id].append(index)

        sequences: list[tuple[tuple[int, ...], ...]] = []
        for indices in by_record.values():
            indices.sort(key=lambda index: samples[index].step_index)
            consecutive: list[int] = []
            windows: list[tuple[int, ...]] = []
            for index in indices:
                if consecutive and (
                    samples[index].step_index
                    != samples[consecutive[-1]].step_index + 1
                ):
                    if windows:
                        sequences.append(tuple(windows))
                    consecutive = []
                    windows = []
                consecutive.append(index)
                sample = samples[index]
                if (
                    sample.next_prefix_messages is None
                    or sample.next_prefix_image_paths is None
                ):
                    if windows:
                        sequences.append(tuple(windows))
                    consecutive = []
                    windows = []
                    continue
                windows.append(tuple(consecutive[-history_size:]))
            if windows:
                sequences.append(tuple(windows))
        return sequences

    @staticmethod
    def _lane_groups(
        sequences: Sequence[tuple[tuple[int, ...], ...]],
        batch_size: int,
        seed: int,
    ) -> list[tuple[tuple[tuple[int, ...], ...], ...]]:
        by_length: dict[int, list[tuple[tuple[int, ...], ...]]] = defaultdict(list)
        for sequence in sequences:
            by_length[len(sequence)].append(sequence)
        rng = random.Random(seed)
        groups: list[tuple[tuple[tuple[int, ...], ...], ...]] = []
        leftovers: list[tuple[tuple[int, ...], ...]] = []
        for length in sorted(by_length, reverse=True):
            bucket = list(by_length[length])
            rng.shuffle(bucket)
            full = len(bucket) - (len(bucket) % batch_size)
            groups.extend(
                tuple(bucket[start : start + batch_size])
                for start in range(0, full, batch_size)
            )
            leftovers.extend(bucket[full:])
        rng.shuffle(leftovers)
        groups.extend(
            tuple(leftovers[start : start + batch_size])
            for start in range(0, len(leftovers), batch_size)
        )
        return groups

    @staticmethod
    def _assign_groups(
        groups: Sequence[tuple[tuple[tuple[int, ...], ...], ...]],
        num_replicas: int,
    ) -> tuple[tuple[tuple[tuple[tuple[int, ...], ...], ...], ...], ...]:
        assigned: list[list[tuple[tuple[tuple[int, ...], ...], ...]]] = [
            [] for _ in range(num_replicas)
        ]
        loads = [0] * num_replicas
        ordered = sorted(
            groups,
            key=lambda group: max(len(sequence) for sequence in group),
            reverse=True,
        )
        for group in ordered:
            target = min(range(num_replicas), key=lambda rank: (loads[rank], rank))
            assigned[target].append(group)
            loads[target] += max(len(sequence) for sequence in group)
        return tuple(tuple(rank_groups) for rank_groups in assigned)

    def _real_batches(self) -> list[list[TransitionContextIndex]]:
        groups = list(self._rank_groups[self.rank])
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        batches: list[list[TransitionContextIndex]] = []
        for group in groups:
            for position in range(max(len(sequence) for sequence in group)):
                windows = [
                    sequence[position]
                    for sequence in group
                    if position < len(sequence)
                ]
                rows: list[TransitionContextIndex] = []
                for window in windows:
                    rows.extend(
                        TransitionContextIndex(
                            sample_index=index,
                            context_length=len(window),
                            is_current_step=offset == len(window) - 1,
                        )
                        for offset, index in enumerate(window)
                    )
                batches.append(rows)
        return batches

    @staticmethod
    def _padding_batch(
        batches: Sequence[list[TransitionContextIndex]],
    ) -> list[TransitionContextIndex]:
        template = next(
            (batch for batch in batches if batch and batch[0].context_length == 1),
            None,
        )
        if template is None:
            raise ValueError("online history sampler cannot build an empty-rank padding batch")
        return [
            TransitionContextIndex(
                sample_index=row.sample_index,
                context_length=row.context_length,
                is_current_step=row.is_current_step,
                loss_weight=0.0,
            )
            for row in template
        ]

    @property
    def current_steps_per_batch(self) -> tuple[int, ...]:
        return tuple(
            sum(
                1
                for row in batch
                if row.is_current_step and row.loss_weight > 0.0
            )
            for batch in self._batches()
        )

    @property
    def padding_batch_count(self) -> int:
        return self.num_batches - self._real_batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[TransitionContextIndex]]:
        batches = self._real_batches()
        if self.pad_to_equal_batches and len(batches) < self.num_batches:
            padding = self._padding_batch(batches)
            batches.extend(
                [list(padding) for _ in range(self.num_batches - len(batches))]
            )
        return batches

    def __iter__(self) -> Iterator[list[TransitionContextIndex]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return self.num_batches


class FutureRolloutBatchSampler(Sampler[list[TransitionRolloutIndex]]):
    """Build fixed-length sliding future windows from recorded trajectories.

    Every window contains exactly ``T`` consecutive executed transitions from
    one rollout.  The first row owns the real current-state forward; all rows
    provide recorded actions and aligned next-state/value targets.
    """

    def __init__(
        self,
        samples: Sequence[TransitionSample],
        *,
        prediction_horizon: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        pad_to_equal_batches: bool = True,
    ) -> None:
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed sampler rank/world size")
        self.samples = samples
        self.prediction_horizon = int(prediction_horizon)
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.pad_to_equal_batches = bool(pad_to_equal_batches)
        self.epoch = 0

        sequences = self._build_sequences(samples, self.prediction_horizon)
        self.window_count = sum(len(sequence) for sequence in sequences)
        groups = OnlineHistoryBatchSampler._lane_groups(
            sequences,
            self.batch_size,
            self.seed,
        )
        if groups and self.pad_to_equal_batches and len(groups) < self.num_replicas:
            raise ValueError(
                "future rollout training requires at least one trajectory lane "
                f"group per rank, got groups={len(groups)}, ranks={self.num_replicas}"
            )
        self._rank_groups = OnlineHistoryBatchSampler._assign_groups(
            groups,
            self.num_replicas,
        )
        real_counts = [
            sum(max(len(sequence) for sequence in group) for group in rank_groups)
            for rank_groups in self._rank_groups
        ]
        self._real_batch_count = real_counts[self.rank]
        self.num_batches = (
            max(real_counts, default=0)
            if self.pad_to_equal_batches
            else self._real_batch_count
        )

    @staticmethod
    def _build_sequences(
        samples: Sequence[TransitionSample],
        prediction_horizon: int,
    ) -> list[tuple[tuple[int, ...], ...]]:
        by_record: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            by_record[sample.record_id].append(index)

        sequences: list[tuple[tuple[int, ...], ...]] = []

        def append_segment(segment: list[int]) -> None:
            if len(segment) < prediction_horizon:
                return
            sequences.append(
                tuple(
                    tuple(segment[start : start + prediction_horizon])
                    for start in range(len(segment) - prediction_horizon + 1)
                )
            )

        for indices in by_record.values():
            indices.sort(key=lambda index: samples[index].step_index)
            consecutive: list[int] = []
            for index in indices:
                sample = samples[index]
                if consecutive and (
                    sample.step_index
                    != samples[consecutive[-1]].step_index + 1
                ):
                    append_segment(consecutive)
                    consecutive = []
                if (
                    sample.next_prefix_messages is None
                    or sample.next_prefix_image_paths is None
                ):
                    append_segment(consecutive)
                    consecutive = []
                    continue
                consecutive.append(index)
            append_segment(consecutive)
        return sequences

    def _real_batches(self) -> list[list[TransitionRolloutIndex]]:
        groups = list(self._rank_groups[self.rank])
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(groups)
        batches: list[list[TransitionRolloutIndex]] = []
        for group in groups:
            for position in range(max(len(sequence) for sequence in group)):
                windows = [
                    sequence[position]
                    for sequence in group
                    if position < len(sequence)
                ]
                rows: list[TransitionRolloutIndex] = []
                for window in windows:
                    rows.extend(
                        TransitionRolloutIndex(
                            sample_index=index,
                            prediction_horizon=self.prediction_horizon,
                            rollout_position=offset,
                        )
                        for offset, index in enumerate(window)
                    )
                batches.append(rows)
        return batches

    @staticmethod
    def _padding_batch(
        batches: Sequence[list[TransitionRolloutIndex]],
    ) -> list[TransitionRolloutIndex]:
        template = next((batch for batch in batches if batch), None)
        if template is None:
            raise ValueError("future rollout sampler cannot build an empty-rank padding batch")
        return [
            TransitionRolloutIndex(
                sample_index=row.sample_index,
                prediction_horizon=row.prediction_horizon,
                rollout_position=row.rollout_position,
                loss_weight=0.0,
            )
            for row in template
        ]

    @property
    def current_steps_per_batch(self) -> tuple[int, ...]:
        return tuple(
            sum(
                1
                for row in batch
                if row.rollout_position == 0 and row.loss_weight > 0.0
            )
            for batch in self._batches()
        )

    @property
    def padding_batch_count(self) -> int:
        return self.num_batches - self._real_batch_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _batches(self) -> list[list[TransitionRolloutIndex]]:
        batches = self._real_batches()
        if self.pad_to_equal_batches and len(batches) < self.num_batches:
            padding = self._padding_batch(batches)
            batches.extend(
                [list(padding) for _ in range(self.num_batches - len(batches))]
            )
        return batches

    def __iter__(self) -> Iterator[list[TransitionRolloutIndex]]:
        return iter(self._batches())

    def __len__(self) -> int:
        return self.num_batches


__all__ = ["FutureRolloutBatchSampler", "OnlineHistoryBatchSampler"]
