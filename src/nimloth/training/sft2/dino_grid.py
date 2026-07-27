"""SFT2 DINO-grid target 装配。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from nimloth.backbone.dino_grid import CachedDINOGridTargets
from nimloth.training.sft2.batch import (
    SFT2Batch,
    SFT2BatchAssembler,
    SFT2RolloutBatch,
)


class DINOGridBatchAssembler:
    """为当前 step 读取 next image 的 cached DINO grid。"""

    def __init__(
        self,
        base: SFT2BatchAssembler,
        targets: CachedDINOGridTargets,
    ) -> None:
        if targets.grid_size != 4 or targets.identity.hidden_size != 1024:
            raise ValueError(
                "SFT2 DINO supervision requires a 4x4 grid with hidden size 1024"
            )
        self.base = base
        self.targets = targets

    @property
    def processor(self) -> Any:
        return self.base.processor

    def collate_transition_samples(self, batch: list[Any]) -> Any:
        return self.base.collate_transition_samples(batch)

    def prepare(self, raw_batch: Any) -> SFT2Batch | SFT2RolloutBatch:
        base = self.base.prepare(raw_batch)
        target_count = base.batch_size
        if isinstance(base, SFT2RolloutBatch):
            target_count *= base.prediction_horizon
        if len(base.next_image_paths) != target_count or any(
            not path for path in base.next_image_paths
        ):
            raise ValueError(
                "DINO grid supervision requires one next_image_path per predicted state"
            )
        targets = self.targets.load(
            base.next_image_paths,
            device=base.sample_weights.device,
        )
        if isinstance(base, SFT2RolloutBatch):
            targets = targets.reshape(
                base.batch_size,
                base.prediction_horizon,
                *targets.shape[1:],
            )
        return replace(
            base,
            dino_grid_target=targets,
        )
__all__ = ["DINOGridBatchAssembler"]
