"""DINO-grid target 装配与可配置附加 loss。

本模块不实现或继承 SFT2 训练算法。current/target、CE、WM、value 和 SIGReg 始终
由唯一的 ``SFT2Algorithm`` 执行；这里仅提供该核心 step 的 target 与 loss 增量。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.backbone.dino_grid import CachedDINOGridTargets
from nimloth.training.sft2.algorithm import SFT2AuxiliaryLossOutput
from nimloth.training.sft2.batch import SFT2Batch, SFT2BatchAssembler


class DINOGridBatchAssembler:
    """在公共 ``SFT2Batch`` 上附加 next-image DINO target。"""

    target_name = "dino_grid"

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

    def collate_cached_transition_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> Any:
        return self.base.collate_cached_transition_batch(batch)

    def prepare(self, raw_batch: Any) -> SFT2Batch:
        base = self.base.prepare(raw_batch)
        if len(base.next_image_paths) != base.batch_size or any(
            not path for path in base.next_image_paths
        ):
            raise ValueError(
                "DINO grid supervision requires one next_image_path per current step"
            )
        targets = dict(base.auxiliary_targets)
        targets[self.target_name] = self.targets.load(
            base.next_image_paths,
            device=base.sample_weights.device,
        )
        return replace(base, auxiliary_targets=targets)


class DINOGridLoss:
    """比较 WM prediction 的 decoded grid 与 cached DINO target。"""

    name = "dino"
    target_name = DINOGridBatchAssembler.target_name

    def __init__(self, *, weight: float) -> None:
        self.weight = float(weight)
        if self.weight < 0.0:
            raise ValueError(f"DINO loss weight must be non-negative, got {self.weight}")

    def __call__(
        self,
        world_model: torch.nn.Module,
        batch: SFT2Batch,
        predicted_next_state: torch.Tensor,
    ) -> SFT2AuxiliaryLossOutput:
        target = batch.auxiliary_targets.get(self.target_name)
        if target is None:
            raise ValueError(
                f"DINO loss requires auxiliary target {self.target_name!r}"
            )
        decode = getattr(world_model, "decode_prediction", None)
        if decode is None:
            raise TypeError("DINO loss requires WorldModel.decode_prediction()")
        decoded = decode(predicted_next_state)
        if decoded.shape != target.shape:
            raise ValueError(
                "decoded WM prediction and DINO grid target must match: "
                f"{tuple(decoded.shape)} != {tuple(target.shape)}"
            )
        raw_loss = F.mse_loss(decoded.float(), target.detach().float())
        return SFT2AuxiliaryLossOutput(
            name=self.name,
            raw_loss=raw_loss,
            weighted_loss=self.weight * raw_loss,
            metrics={
                "dino_grid_mse": float(raw_loss.detach().item()),
                "lambda_dino": self.weight,
            },
        )


__all__ = ["DINOGridBatchAssembler", "DINOGridLoss"]
