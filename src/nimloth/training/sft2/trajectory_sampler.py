"""Semantic-preserving trajectory-aware batch sampler for SFT2.

This sampler only changes which independent prefix samples share a micro-batch.
It does **not** pack a trajectory into one sequence and therefore preserves the
legacy per-prefix Qwen forward semantics.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from nimloth.wm.dataset import TransitionSample


class TrajectoryAwareBatchSampler(Sampler[list[int]]):
    """Yield batches of consecutive steps from the same trajectory record.

    Batches contain normal dataset indices.  DataLoader still collates them as
    independent samples, so Qwen sees the same per-prefix batch rows as before.

    Two modes:
    - **Chunked** (``full_trajectory=False``, default): each batch is up to
      ``batch_size`` consecutive steps from one record.  Keeps micro-batch size
      bounded but does not guarantee a complete trajectory in one batch.
    - **Full trajectory** (``full_trajectory=True``): each batch contains at
      most ``max_steps_per_trajectory`` consecutive steps from one record.
      Longer records are chunked.  SIGReg still sees long contiguous runs
      while GPU memory stays bounded.  Default max is 8 steps.

    For distributed training, batches are partitioned by batch index after
    optional deterministic shuffling.  When ``drop_last`` is false, shorter
    shards repeat from the front so every rank executes the same number of
    micro-batches, matching ``DistributedSampler`` behavior.
    """

    def __init__(
        self,
        samples: Sequence[TransitionSample],
        *,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        full_trajectory: bool = False,
        max_steps_per_trajectory: int = 8,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas={num_replicas}")
        if not full_trajectory and batch_size <= 0:
            raise ValueError("batch_size must be positive (ignored when full_trajectory=True)")
        if full_trajectory and max_steps_per_trajectory <= 0:
            raise ValueError("max_steps_per_trajectory must be positive when full_trajectory=True")
        self.samples = samples
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.full_trajectory = full_trajectory
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.epoch = 0
        self._base_batches = self._build_base_batches(
            samples, batch_size, drop_last=drop_last, full_trajectory=full_trajectory,
            max_steps_per_trajectory=max_steps_per_trajectory,
        )
        if drop_last:
            self.num_batches = len(self._base_batches) // num_replicas
        else:
            self.num_batches = math.ceil(len(self._base_batches) / num_replicas) if self._base_batches else 0
        self.total_size = self.num_batches * num_replicas

    @staticmethod
    def _build_base_batches(
        samples: Sequence[TransitionSample],
        batch_size: int,
        *,
        drop_last: bool,
        full_trajectory: bool = False,
        max_steps_per_trajectory: int = 8,
    ) -> list[list[int]]:
        by_record: dict[str, list[int]] = defaultdict(list)
        for idx, sample in enumerate(samples):
            by_record[sample.record_id].append(idx)

        batches: list[list[int]] = []
        for _record_id, indices in by_record.items():
            indices.sort(key=lambda i: samples[i].step_index)
            if full_trajectory:
                # Chunk into slices of at most max_steps_per_trajectory.
                for start in range(0, len(indices), max_steps_per_trajectory):
                    chunk = indices[start : start + max_steps_per_trajectory]
                    if chunk and (not drop_last or len(chunk) >= max_steps_per_trajectory):
                        batches.append(chunk)
            else:
                for start in range(0, len(indices), batch_size):
                    batch = indices[start : start + batch_size]
                    if len(batch) == batch_size or (batch and not drop_last):
                        batches.append(batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self._base_batches)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)

        if self.total_size > len(batches):
            if not batches:
                return iter(())
            batches.extend(batches[: self.total_size - len(batches)])
        elif self.total_size < len(batches):
            batches = batches[: self.total_size]

        rank_batches = batches[self.rank : self.total_size : self.num_replicas]
        return iter(rank_batches)

    def __len__(self) -> int:
        return self.num_batches
