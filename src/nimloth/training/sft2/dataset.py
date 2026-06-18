"""SFT2 transition dataset for Qwen training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from nimloth.wm.collate import transition_collate_for_qwen
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample


class TransitionQwenDataset(Dataset):
    def __init__(self, path: Path, *, max_records: int = -1, success_only: bool = False):
        self.samples = TransitionJsonlDataset(path, max_records=max_records, success_only=success_only).samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]


def collate_transition_batch(batch: list[TransitionSample]) -> list[dict[str, Any]]:
    return transition_collate_for_qwen(batch)
