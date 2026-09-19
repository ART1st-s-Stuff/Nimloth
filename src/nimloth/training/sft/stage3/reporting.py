"""SFT2 训练与验证指标的统一汇报入口。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.distributed import is_main
from nimloth.util.wandb import log_metrics


STAGE3_LOG_COLUMNS = (
    "time",
    "epoch",
    "global_step",
    "total_loss",
    "wm_mse",
    "wm_spatial_mse",
    "wm_cls_mse",
    "dino_grid_mse",
    "dino_spatial_mse",
    "dino_cls_mse",
    "predicted_dino_grid_mse",
    "predicted_dino_spatial_mse",
    "predicted_dino_cls_mse",
    "sigreg_loss",
    "sigreg_global_batch_size",
    "value_total",
    "value_mc_mse",
    "lm_ce",
    "outcome_bce",
    "outcome_count",
    "lambda_wm",
    "lambda_dino",
    "lambda_sigreg",
    "qwen_lr",
    "context_length",
    "prediction_horizon",
    "current_batch_size",
    "trajectory_batch_size",
    "val_wm_mse",
)


@dataclass(frozen=True)
class SFT2Reporter:
    """统一 SFT2 的 CSV、W&B 与 epoch 摘要，loop 只提交结构化指标。"""

    log_writer: CSVRecordWriter
    wandb_run: Any
    llm_tune: str
    vision_tune: str

    def log_train_step(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
        wm_weight: float,
        sigreg_weight: float,
        qwen_lr: float,
    ) -> None:
        if not is_main():
            return
        self.log_writer.append(
            {
                "time": time.time(),
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
                "lambda_wm": wm_weight,
                "lambda_sigreg": sigreg_weight,
                "qwen_lr": qwen_lr,
            }
        )
        log_metrics(
            self.wandb_run,
            namespace="train",
            metrics=metrics,
            step=global_step,
            context={"global_step": global_step},
        )

    def log_validation(
        self,
        *,
        epoch: int,
        global_step: int,
        metrics: dict[str, float],
        best_val_wm_mse: float,
        checkpoint_metric: str,
    ) -> None:
        if not is_main():
            return
        log_metrics(
            self.wandb_run,
            namespace="val",
            metrics=metrics,
            step=global_step,
            context={"epoch": epoch},
        )
        self.log_writer.append(
            {
                "time": time.time(),
                "epoch": epoch,
                "global_step": global_step,
                **metrics,
                "val_wm_mse": metrics.get("wm_mse", ""),
            }
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_metrics": metrics,
                    "best_val_wm_mse": best_val_wm_mse,
                    "checkpoint_metric": checkpoint_metric,
                    "llm_tune": self.llm_tune,
                    "vision_tune": self.vision_tune,
                }
            )
        )


__all__ = ["SFT2Reporter", "STAGE3_LOG_COLUMNS"]
