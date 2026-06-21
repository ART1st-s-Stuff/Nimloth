"""Batching helpers for P4 packed trajectory forward."""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch.utils.data import Sampler

from nimloth.wm.dataset import TransitionSample


def build_trajectory_batch_indices(
    samples: list[TransitionSample],
    batch_size: int,
) -> list[list[int]]:
    """Chunk indices within one record; never merge across ``record_id`` boundaries."""

    if not samples or batch_size <= 0:
        return []
    batches: list[list[int]] = []
    batch: list[int] = []
    for index, sample in enumerate(samples):
        if batch:
            prev = samples[batch[-1]]
            if (
                sample.record_id != prev.record_id
                or sample.step_index != prev.step_index + 1
            ):
                batches.append(batch)
                batch = []
        batch.append(index)
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def build_record_trajectory_batches(samples: list[TransitionSample]) -> list[list[int]]:
    """One batch per trajectory (contiguous indices with the same ``record_id``)."""

    if not samples:
        return []
    batches: list[list[int]] = []
    start = 0
    for index in range(1, len(samples)):
        prev = samples[index - 1]
        cur = samples[index]
        if cur.record_id != prev.record_id or cur.step_index != prev.step_index + 1:
            batches.append(list(range(start, index)))
            start = index
    batches.append(list(range(start, len(samples))))
    return batches


def assert_packed_batch(steps: list[TransitionSample]) -> None:
    if not steps:
        raise ValueError("packed-forward batch is empty")
    record_id = steps[0].record_id
    expected_step = steps[0].step_index
    for sample in steps:
        if sample.record_id != record_id:
            raise ValueError(
                f"packed-forward requires one trajectory per batch; got {record_id!r} and {sample.record_id!r}"
            )
        if sample.step_index != expected_step:
            raise ValueError(
                f"packed-forward requires consecutive steps; expected {expected_step}, got {sample.step_index}"
            )
        expected_step += 1


class TrajectoryDistributedBatchSampler(Sampler[list[int]]):
    """Yield index lists; optionally shuffle batches and shard for DDP."""

    def __init__(
        self,
        samples: list[TransitionSample],
        batch_size: int,
        *,
        rank: int,
        world_size: int,
        shuffle: bool,
        seed: int,
        record_level: bool = False,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0 and not record_level:
            raise ValueError("batch_size must be positive unless record_level=True")
        self.samples = samples
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.record_level = record_level
        self.drop_last = drop_last
        self.epoch = 0
        if record_level:
            batches = build_record_trajectory_batches(samples)
        else:
            batches = build_trajectory_batch_indices(samples, batch_size)
        if drop_last and batches and len(batches[-1]) < batch_size:
            batches = batches[:-1]
        self._batches = batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self._batches)
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[i] for i in order]
        for batch_idx in range(self.rank, len(batches), self.world_size):
            yield batches[batch_idx]

    def __len__(self) -> int:
        n = len(self._batches)
        if n == 0:
            return 0
        return math.ceil((n - self.rank) / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
