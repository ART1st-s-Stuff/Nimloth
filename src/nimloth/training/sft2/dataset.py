"""SFT2 transition dataset for Qwen training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from nimloth.wm.collate import transition_collate_for_qwen
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample


class TransitionQwenDataset(Dataset):
    def __init__(
        self,
        path: Path,
        *,
        max_records: int = -1,
        success_only: bool = False,
        value_gamma: float = 1.0,
    ):
        self.samples = TransitionJsonlDataset(
            path,
            max_records=max_records,
            success_only=success_only,
            value_gamma=value_gamma,
        ).samples

    @classmethod
    def from_samples(cls, samples: list) -> TransitionQwenDataset:
        ds = cls.__new__(cls)
        ds.samples = samples
        return ds

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]


class TrajectoryRecordDataset(Dataset):
    """One dataset item = full trajectory (all transitions for one record)."""

    def __init__(self, samples: list[TransitionSample]) -> None:
        from nimloth.training.sft2.trajectory_batching import build_record_trajectory_batches

        self.samples = samples
        self._record_index_lists = build_record_trajectory_batches(samples)

    def __len__(self) -> int:
        return len(self._record_index_lists)

    def __getitem__(self, index: int) -> dict:
        steps = [self.samples[i] for i in self._record_index_lists[index]]
        return collate_packed_trajectory_batch(steps)


def collate_transition_batch(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    return transition_collate_for_qwen(batch)


def collate_packed_trajectory_batch(batch: list[TransitionSample]) -> dict:
    return {"transition_samples": batch, "items": transition_collate_for_qwen(batch)}


def collate_trajectory_record_batch(batch: list[dict]) -> dict:
    if len(batch) != 1:
        raise ValueError("trajectory record batch size must be 1")
    return batch[0]
