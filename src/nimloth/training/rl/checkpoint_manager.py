"""把 RL trainer 状态映射到统一 checkpoint artifact。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nimloth.agent import Agent
from nimloth.backbone import BackboneEMA
from nimloth.config.rl import RLConfig
from nimloth.training.rl.checkpoint import save_rl_checkpoint


class RLCheckpointManager:
    """持有静态组件，保存时只接收会随 iteration 改变的状态。"""

    def __init__(
        self,
        *,
        config: RLConfig,
        agent: Agent,
        processor: Any,
        vision_ema: BackboneEMA | None,
        optimizer: torch.optim.Optimizer,
        base_model_path: str,
        llm_tune: str,
        vision_tune: str,
    ) -> None:
        self._config = config
        self._agent = agent
        self._processor = processor
        self._vision_ema = vision_ema
        self._optimizer = optimizer
        self._base_model_path = base_model_path
        self._llm_tune = llm_tune
        self._vision_tune = vision_tune

    def save(
        self,
        path: Path,
        *,
        iteration: int,
        global_step: int,
        best_eval_metric: float,
    ) -> None:
        save_rl_checkpoint(
            path,
            agent=self._agent,
            processor=self._processor,
            vision_ema=self._vision_ema,
            optimizer=self._optimizer,
            iteration=iteration,
            global_step=global_step,
            best_eval_metric=best_eval_metric,
            checkpoint_metric=self._config.validation.checkpoint_metric,
            lora=self._llm_tune == "lora" or self._vision_tune == "lora",
            llm_tune=self._llm_tune,
            vision_tune=self._vision_tune,
            base_model_path=self._base_model_path,
        )
