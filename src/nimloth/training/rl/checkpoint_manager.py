"""把 RL trainer 状态映射到统一 checkpoint artifact。"""

from __future__ import annotations

import argparse
from pathlib import Path

from nimloth.backbone.qwen25vl.tuning import uses_lora
from nimloth.config.rl import RLConfig
from nimloth.training.rl.checkpoint import save_rl_checkpoint
from nimloth.training.rl.components import RLComponents


class RLCheckpointManager:
    """持有静态组件，保存时只接收会随 iteration 改变的状态。"""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        config: RLConfig,
        components: RLComponents,
    ) -> None:
        self._args = args
        self._config = config
        self._components = components

    def save(
        self,
        path: Path,
        *,
        iteration: int,
        global_step: int,
        best_eval_metric: float,
    ) -> None:
        components = self._components
        save_rl_checkpoint(
            path,
            nimloth_model=components.nimloth_model,
            processor=components.processor,
            vision_ema=components.vision_ema,
            optimizer=components.optimizer,
            iteration=iteration,
            global_step=global_step,
            best_eval_metric=best_eval_metric,
            checkpoint_metric=self._config.validation.checkpoint_metric,
            lora=uses_lora(self._args),
            llm_tune=components.llm_tune,
            vision_tune=components.vision_tune,
            base_model_path=components.base_model_path,
        )
