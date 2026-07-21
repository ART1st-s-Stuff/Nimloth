"""RL optimizer、梯度裁剪与 EMA 更新运行期。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import torch

from nimloth.backbone import BackboneEMA
from nimloth.rollout import EncodedTransition
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.batch import select_transition_batch


@dataclass
class RLUpdater:
    """执行一次可恢复的 RL optimizer update。"""

    algorithm: RLAlgorithm
    optimizer: torch.optim.Optimizer
    device: torch.device
    vision_ema: BackboneEMA | None

    def update(
        self,
        transitions: Sequence[EncodedTransition],
        *,
        batch_size: int,
        batch_seed: int,
    ) -> dict[str, float]:
        batch = select_transition_batch(
            transitions,
            batch_size=batch_size,
            seed=batch_seed,
            device=self.device,
        )
        if self.algorithm.policy_replay is not None:
            torch.cuda.empty_cache()
            gc.collect()
        output = self.algorithm.training_step(batch)

        self.optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            ],
            1.0,
        )
        self.optimizer.step()
        if self.vision_ema is not None:
            self.vision_ema.update(self.algorithm.agent.backbone.model)
        return output.metrics


__all__ = ["RLUpdater"]
