"""Epoch-shuffled complete trajectories with zero-weight distributed padding."""
from __future__ import annotations

import math
import random

from torch.utils.data import Sampler

from .trajectory import TrajectoryDataset, TrajectoryIndex


class TrajectoryBatchSampler(Sampler[list[TrajectoryIndex]]):
    """Assign each trajectory to exactly one rank; never split its windows."""
    def __init__(self, dataset: TrajectoryDataset, *, batch_size: int, num_replicas: int = 1,
                 rank: int = 0, shuffle: bool = True, seed: int = 0,
                 pad_to_equal_batches: bool = True):
        if batch_size < 1 or num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("invalid trajectory batch size or distributed rank")
        if not len(dataset):
            raise ValueError("no trajectory contains a complete prediction window")
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.pad_to_equal_batches = pad_to_equal_batches
        self.window_count = dataset.window_count
        self.trajectory_count = len(dataset)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _batches(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        return [order[start:start + self.batch_size] for start in range(0, len(order), self.batch_size)]

    def __len__(self):
        global_count = math.ceil(len(self.dataset) / self.batch_size)
        if self.pad_to_equal_batches:
            return math.ceil(global_count / self.num_replicas)
        return len(range(self.rank, global_count, self.num_replicas))

    @property
    def padding_batch_count(self):
        return len(self) - len(self._batches()[self.rank::self.num_replicas])

    @property
    def trajectories_per_batch(self):
        return tuple(sum(bool(item.loss_weight) for item in batch) for batch in self)

    @property
    def current_steps_per_batch(self):
        return tuple(sum(self.dataset.window_counts[item.index] for item in batch if item.loss_weight)
                     for batch in self)

    def __iter__(self):
        all_batches = self._batches()
        local = all_batches[self.rank::self.num_replicas]
        for batch in local:
            yield [TrajectoryIndex(index) for index in batch]
        for _ in range(len(self) - len(local)):
            yield [TrajectoryIndex(index, 0.0) for index in all_batches[0]]
