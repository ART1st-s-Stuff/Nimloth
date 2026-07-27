"""SFT2 DINO-grid target 装配与显式的 grid 重建 loss。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.backbone.dino_grid import CachedDINOGridTargets
from nimloth.training.sft2.batch import SFT2Batch, SFT2BatchAssembler


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

    def prepare(self, raw_batch: Any) -> SFT2Batch:
        base = self.base.prepare(raw_batch)
        if len(base.next_image_paths) != base.batch_size or any(
            not path for path in base.next_image_paths
        ):
            raise ValueError(
                "DINO grid supervision requires one next_image_path per current step"
            )
        return replace(
            base,
            dino_grid_target=self.targets.load(
                base.next_image_paths,
                device=base.sample_weights.device,
            ),
        )


def dino_grid_mse(
    predicted_next_state: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """直接监督 WM 预测的 next state 对齐 cached DINO grid。"""

    if predicted_next_state.shape != target.shape:
        raise ValueError(
            "WM predicted state and DINO-grid target must match: "
            f"{tuple(predicted_next_state.shape)} != {tuple(target.shape)}"
        )
    return F.mse_loss(predicted_next_state.float(), target.detach().float())


__all__ = ["DINOGridBatchAssembler", "dino_grid_mse"]
