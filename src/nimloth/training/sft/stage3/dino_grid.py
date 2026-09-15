"""SFT2 DINO-grid target 装配。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from nimloth.backbone.dino_grid import CachedDINOGridTargets
from nimloth.training.sft.stage3.batch import (
    Stage3BatchAssembler,
    Stage3TrajectoryBatch,
)


class DINOGridBatchAssembler:
    """Load every observed state once; index future grids for diagnostics only."""

    def __init__(
        self,
        base: Stage3BatchAssembler,
        targets: CachedDINOGridTargets,
    ) -> None:
        if targets.grid_size < 1 or targets.identity.hidden_size != 1024:
            raise ValueError(
                "SFT2 DINO supervision requires a positive grid size with hidden size 1024"
            )
        self.base = base
        self.targets = targets

    @property
    def input_builder(self):
        return self.base.input_builder

    @property
    def processor(self) -> Any:
        return self.base.processor

    @property
    def device(self) -> torch.device:
        return self.base.device

    def supervision_counts(self, raw_batch: Any) -> tuple[int, int]:
        return self.base.supervision_counts(raw_batch)

    def outcome_count(self, raw_batch: Any) -> int:
        return self.base.outcome_count(raw_batch)

    def observed_state_count(self, raw_batch: Any) -> int:
        return self.base.observed_state_count(raw_batch)

    def prepare(self, raw_batch: Any) -> Stage3TrajectoryBatch:
        base = self.base.prepare(raw_batch)
        if len(base.observed_image_paths) != len(base.state_keys) or not all(base.observed_image_paths):
            raise ValueError("DINO supervision requires every observed image including terminal")
        observed = self.targets.load(base.observed_image_paths, device=base.sample_weights.device)
        return replace(
            base,
            observed_dino_target=observed,
            dino_grid_target=observed[base.next_indices],
            current_dino_target=observed[base.current_indices],
        )
__all__ = ["DINOGridBatchAssembler"]
