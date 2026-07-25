"""RL 阶段对公共 CSV 与 W&B 工具的轻量封装。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nimloth.config.rl import RLConfig
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.distributed import is_main
from nimloth.util.wandb import init_wandb_run, log_metrics


_LOG_COLUMNS = (
    "time",
    "iteration",
    "global_step",
    "wm_mse",
    "sigreg_loss",
    "value_loss",
    "total_loss",
    "num_rollouts",
    "num_transitions",
    "success_rate",
    "val_success_rate",
    "val_avg_reward",
    "val_avg_steps",
    "actor_loss",
    "entropy",
    "clip_fraction",
    "mean_advantage",
    "token_value_loss",
    "action_distillation_loss",
    "action_distillation_kl",
    "reference_kl_loss",
    "mean_ratio",
    "policy_tokens",
)


class RLReporter:
    """统一 RL iteration 的终端、CSV 与 W&B 记录。"""

    def __init__(
        self,
        *,
        rank: int,
        output_dir: Path,
        run_name: str | None,
        config: RLConfig,
    ) -> None:
        self._writer = CSVRecordWriter(
            output_dir / "train_step_log.csv",
            _LOG_COLUMNS,
        )
        if is_main():
            self._writer.ensure_header()
        self._wandb_run: Any | None = init_wandb_run(
            rank=rank,
            output_dir=output_dir,
            enabled=True,
            default_project="nimloth-rl",
            run_name=run_name,
            config=config.to_dict(),
            metric_definitions=(("global_step", None), ("train/*", "global_step")),
        )

    def log_iteration(
        self,
        *,
        iteration: int,
        global_step: int,
        metrics: dict[str, float],
        elapsed_seconds: float,
    ) -> None:
        if not is_main():
            return
        self._writer.append(
            {
                "time": time.time(),
                "iteration": iteration,
                "global_step": global_step,
                **metrics,
            }
        )
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "global_step": global_step,
                    "metrics": metrics,
                    "elapsed_s": round(elapsed_seconds, 1),
                }
            )
        )
        log_metrics(
            self._wandb_run,
            namespace="train",
            metrics=metrics,
            step=global_step,
            context={"global_step": global_step, "iteration": iteration},
        )

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
