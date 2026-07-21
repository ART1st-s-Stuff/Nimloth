"""Small runtime helpers shared by SFT2 training and evaluation."""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterable, Iterator

import torch


def unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def training_micro_seed(base_seed: int, epoch: int, micro_step: int, rank: int) -> int:
    """Derive a replayable rank-local seed for one consumed micro-batch."""

    return int((base_seed + epoch * 1_000_003 + micro_step * 10_007 + rank) % (2**63 - 1))


def seed_training_micro_step(base_seed: int, epoch: int, micro_step: int, rank: int) -> int:
    seed = training_micro_seed(base_seed, epoch, micro_step, rank)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@contextlib.contextmanager
def preserve_module_modes(
    modules: Iterable[torch.nn.Module],
    *,
    training: bool,
) -> Iterator[None]:
    """Temporarily set module modes and restore every caller-owned state."""

    module_list = list(modules)
    previous = [module.training for module in module_list]
    try:
        for module in module_list:
            module.train(training)
        yield
    finally:
        for module, was_training in zip(module_list, previous, strict=True):
            module.train(was_training)
