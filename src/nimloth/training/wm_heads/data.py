"""Deterministic transition and sampling views over a frozen State cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from nimloth.rcdm.state_cache import RCDMStateCacheDataset


def _is_successor(current: dict[str, Any], following: dict[str, Any]) -> bool:
    same_record = current.get("record_id") == following.get("record_id")
    next_step = int(following.get("step_index", -1)) == int(current.get("step_index", -1)) + 1
    return same_record and next_step


def _preload(cache_dir: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    source = RCDMStateCacheDataset(cache_dir)
    rows, states = [], []
    for index in range(len(source)):
        item = source[index]
        states.append(item["state_emb"])
        rows.append({key: value for key, value in item.items() if key != "state_emb"})
    return torch.stack(states), rows


def _transition_pairs(rows: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    return tuple((index - 1, index) for index in range(1, len(rows)) if _is_successor(rows[index - 1], rows[index]))


class FrozenStateTransitions(Dataset):
    """Preload and pair each non-terminal State row with its successor."""

    def __init__(self, cache_dir: Path) -> None:
        self.states, self.rows = _preload(cache_dir)
        self.pairs = _transition_pairs(self.rows)
        self.transition_ids = tuple(str(self.rows[left]["id"]) for left, _ in self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        left, right = self.pairs[index]
        return {"state": self.states[left], "next_state": self.states[right], "action": int(self.rows[left]["action_index"]), "id": str(self.rows[left]["id"])}


def _next_indices(stream: "DeterministicBatchStream") -> torch.Tensor:
    parts: list[torch.Tensor] = []
    remaining = stream.batch_size
    while remaining:
        if stream.position == stream.size:
            stream.order = torch.randperm(stream.size, generator=stream.generator)
            stream.position = 0
        count = min(remaining, stream.size - stream.position)
        parts.append(stream.order[stream.position : stream.position + count])
        stream.position += count
        remaining -= count
    return torch.cat(parts)


class DeterministicBatchStream:
    """Infinite shuffled batches with an exact resumable generator position."""

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 1 or batch_size < 1:
            raise ValueError("size and batch_size must be positive")
        self.size, self.batch_size = int(size), int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(self.size, generator=self.generator)
        self.position = 0

    def next_indices(self) -> torch.Tensor:
        return _next_indices(self)

    def state_dict(self) -> dict[str, Any]:
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order.clone(), "position": self.position, "generator_state": self.generator.get_state()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("batch stream shape mismatch")
        self.order, self.position = state["order"].clone(), int(state["position"])
        self.generator.set_state(state["generator_state"])
