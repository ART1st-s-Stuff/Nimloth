"""RL 阶段对公共 CSV 与 W&B 工具的轻量封装。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import torch

from nimloth.config.rl import RLConfig
from nimloth.training.rl.policy import PPOPolicyLoss
from nimloth.training.rl.value import PPOActionValueLoss
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.distributed import is_main
from nimloth.util.wandb import init_wandb_run, log_metrics


def planner_step_metrics(
    *,
    losses: dict[str, torch.Tensor | None],
    total_loss: torch.Tensor,
    old_action_value: torch.Tensor,
    selected_action_values: torch.Tensor,
    value_objective: PPOActionValueLoss | None,
    policy_objective: PPOPolicyLoss | None,
    policy_advantage: torch.Tensor | None,
    total_transitions: int,
    world_model_weight: float,
    dino_grid_weight: float,
) -> dict[str, float]:
    """从 planner objective 计算 CSV/W&B 指标，不参与训练计算图。"""

    def scalar(value: torch.Tensor | int | float) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    value_loss = cast(torch.Tensor, losses["value"])
    policy_loss = losses["policy"]
    return {
        "wm_mse": scalar(losses["wm"]),
        "dino_grid_mse": scalar(
            losses["dino"] if losses["dino"] is not None else 0
        ),
        "lambda_wm": world_model_weight,
        "lambda_dino": dino_grid_weight,
        "sigreg_loss": 0.0,
        "value_loss": scalar(value_loss),
        "value_mc_mse": scalar(value_loss),
        "value_clipped_mse": scalar(
            value_objective.clipped_mse / total_transitions
            if value_objective is not None
            else 0
        ),
        "value_clip_fraction": scalar(
            value_objective.clip_fraction / total_transitions
            if value_objective is not None
            else 0
        ),
        "value_old_mean": scalar(old_action_value / total_transitions),
        "value_delta_abs_mean": scalar(
            (
                selected_action_values.detach()
                - old_action_value.to(
                    device=selected_action_values.device,
                    dtype=selected_action_values.dtype,
                )
            ).abs().mean()
            / total_transitions
        ),
        "value_rank": 0.0,
        "total_loss": scalar(total_loss),
        "actor_loss": 0.0,
        "planner_policy_loss": scalar(
            policy_loss if policy_loss is not None else 0
        ),
        "planner_policy_entropy": scalar(
            policy_objective.entropy / total_transitions
            if policy_objective is not None
            else 0
        ),
        "planner_policy_clip_fraction": scalar(
            policy_objective.clip_fraction / total_transitions
            if policy_objective is not None
            else 0
        ),
        "planner_policy_mean_ratio": scalar(
            policy_objective.probability_ratio.mean() / total_transitions
            if policy_objective is not None
            else 0
        ),
        "planner_policy_mean_advantage": scalar(
            policy_advantage / total_transitions
            if policy_advantage is not None
            else 0
        ),
        "planner_policy_actions": (
            1.0 / total_transitions if policy_objective is not None else 0.0
        ),
        "token_value_loss": 0.0,
        "reference_kl_loss": 0.0,
        "policy_tokens": 0.0,
    }


_LOG_COLUMNS = (
    "time",
    "iteration",
    "global_step",
    "wm_mse",
    "dino_grid_mse",
    "lambda_wm",
    "lambda_dino",
    "sigreg_loss",
    "value_loss",
    "value_mc_mse",
    "value_clipped_mse",
    "value_clip_fraction",
    "value_old_mean",
    "value_delta_abs_mean",
    "value_ppo_epochs",
    "planner_policy_ppo_epochs",
    "total_loss",
    "num_rollouts",
    "num_transitions",
    "num_actor_transitions",
    "num_wm_windows",
    "success_rate",
    "val_success_rate",
    "val_avg_reward",
    "val_avg_steps",
    "actor_loss",
    "planner_policy_loss",
    "planner_policy_entropy",
    "planner_policy_clip_fraction",
    "planner_policy_mean_ratio",
    "planner_policy_mean_advantage",
    "planner_policy_actions",
    "entropy",
    "clip_fraction",
    "mean_advantage",
    "token_value_loss",
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
