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

from torch.utils.data import Dataset, Sampler

from nimloth.rollout.transitions import TransitionSample


class TrajectoryAwareBatchSampler(Sampler[list[int]]):
    """Yield batches of consecutive steps from the same trajectory record.

    Batches contain normal dataset indices. DataLoader still collates them as
    independent samples, so Qwen sees the same per-prefix batch rows as before.

    ``full_trajectory=False`` 时，每批最多包含 ``batch_size`` 个连续 step。
    ``full_trajectory=True`` 时，图片预算和 step 上限负责切分轨迹。无论哪种
    模式，每一行仍是一个独立 transition；它不定义 SIGReg 的时间轴。

    For distributed training, batches are partitioned by batch index after
    optional deterministic shuffling. When ``drop_last`` is false, shorter
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
        max_images_per_batch: int = 32,
        max_steps_per_trajectory: int = 16,
    ) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas={num_replicas}")
        if not full_trajectory and batch_size <= 0:
            raise ValueError("batch_size must be positive (ignored when full_trajectory=True)")
        if full_trajectory:
            if max_images_per_batch <= 0:
                raise ValueError("max_images_per_batch must be positive when full_trajectory=True")
            if max_steps_per_trajectory <= 0:
                raise ValueError("max_steps_per_trajectory must be positive when full_trajectory=True")
        self.samples = samples
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.full_trajectory = full_trajectory
        self.max_images_per_batch = max_images_per_batch
        self.max_steps_per_trajectory = max_steps_per_trajectory
        self.epoch = 0
        self._base_batches = self._build_base_batches(
            samples,
            batch_size,
            drop_last=drop_last,
            full_trajectory=full_trajectory,
            max_images_per_batch=max_images_per_batch,
            max_steps_per_trajectory=max_steps_per_trajectory,
        )
        if drop_last:
            self.num_batches = len(self._base_batches) // num_replicas
        else:
            self.num_batches = (
                math.ceil(len(self._base_batches) / num_replicas)
                if self._base_batches
                else 0
            )
        self.total_size = self.num_batches * num_replicas

    @staticmethod
    def _build_base_batches(
        samples: Sequence[TransitionSample],
        batch_size: int,
        *,
        drop_last: bool,
        full_trajectory: bool = False,
        max_images_per_batch: int = 32,
        max_steps_per_trajectory: int = 16,
    ) -> list[list[int]]:
        by_record: dict[str, list[int]] = defaultdict(list)
        for index, sample in enumerate(samples):
            by_record[sample.record_id].append(index)

        batches: list[list[int]] = []
        for indices in by_record.values():
            indices.sort(key=lambda index: samples[index].step_index)
            if full_trajectory:
                start = 0
                while start < len(indices):
                    image_count = 0
                    end = start
                    while end < len(indices):
                        prefix_images = samples[indices[end]].step_index + 1
                        if image_count + prefix_images > max_images_per_batch:
                            break
                        image_count += prefix_images
                        end += 1
                        if end - start >= max_steps_per_trajectory:
                            break
                    if end == start:
                        # 单个 late prefix 也可能超过总预算；仍需输出一行以继续遍历。
                        end = start + 1
                    batches.append(indices[start:end])
                    start = end
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


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation rows across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset, *, num_replicas: int, rank: int) -> None:
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} out of range for num_replicas={num_replicas}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.num_replicas - 1) // self.num_replicas)
