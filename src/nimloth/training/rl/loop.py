"""RL rollout、更新、验证和 checkpoint 生命周期。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from nimloth.config.rl import RLConfig
from nimloth.rollout import RolloutCollector
from nimloth.rollout import RolloutEncoder
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.evaluation import (
    evaluate_rollout_collector,
    summarize_rollouts,
)
from nimloth.training.rl.reporting import RLReporter
from nimloth.util.distributed import is_main


@dataclass
class RLLoopState:
    """RL checkpoint 需要保存的训练位置与 best 指标。"""

    global_step: int
    best_eval_metric: float


@dataclass
class RLTrainingLoop:
    """按 iteration 执行 collect → encode → update → evaluate。"""

    config: RLConfig
    algorithm: RLAlgorithm
    rollout_encoder: RolloutEncoder
    train_collector: RolloutCollector
    eval_collector: RolloutCollector | None
    output_dir: Path
    checkpoint_manager: RLCheckpointManager
    reporter: RLReporter
    start_iteration: int
    state: RLLoopState

    def run(self) -> RLLoopState:
        for iteration in range(
            self.start_iteration,
            self.config.rl.iterations + 1,
        ):
            self._run_iteration(iteration)

        for checkpoint_dir in (
            self.output_dir / "final",
            self.output_dir / "latest",
        ):
            self.checkpoint_manager.save(
                checkpoint_dir,
                iteration=self.config.rl.iterations,
                global_step=self.state.global_step,
                best_eval_metric=self.state.best_eval_metric,
            )
        return self.state

    def _run_iteration(self, iteration: int) -> None:
        started_at = time.time()
        self._print_phase(iteration)
        trajectories = self.train_collector.collect(
            num_episodes=self.config.rl.envs_per_iteration,
            max_steps_per_episode=self.config.rl.max_steps_per_episode,
            output_dir=self.output_dir / f"rollouts/iter_{iteration:04d}",
        )
        if is_main():
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "trajectories_collected": len(trajectories),
                    }
                )
            )
        if not trajectories:
            self._warn_skip(iteration, "no trajectories collected")
            return

        transitions = self.rollout_encoder(
            trajectories,
            gamma=self.config.rl.gamma,
        )
        torch.cuda.empty_cache()
        if len(transitions) < self.config.rl.batch_size:
            self._warn_skip(
                iteration,
                f"only {len(transitions)} transitions, need {self.config.rl.batch_size}",
            )
            return

        update_metrics = self.algorithm.update(
            transitions,
            batch_size=self.config.rl.batch_size,
            batch_seed=self.config.training.seed + iteration,
        )
        self.state.global_step += 1
        rollout_metrics = summarize_rollouts(trajectories)
        metrics = {
            **update_metrics,
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(len(transitions)),
            "success_rate": float(rollout_metrics["success_rate"]),
        }
        self._barrier()
        self._validate(iteration, metrics)
        self._barrier()
        self._log(iteration, metrics, started_at=started_at)
        self._save_periodic(iteration)
        self._barrier()

    def _validate(self, iteration: int, metrics: dict[str, float]) -> None:
        if not (
            self.config.validation.enabled
            and iteration % self.config.validation.interval == 0
        ):
            return
        assert self.eval_collector is not None
        evaluation = evaluate_rollout_collector(
            self.eval_collector,
            num_episodes=self.config.validation.envs,
            max_steps_per_episode=self.config.rl.max_steps_per_episode,
            output_dir=self.output_dir / f"rollouts/val_{iteration:04d}",
            checkpoint_metric=self.config.validation.checkpoint_metric,
        )
        if evaluation is None:
            return
        metrics.update(evaluation.metrics)
        if evaluation.selected_metric > self.state.best_eval_metric:
            self.state.best_eval_metric = evaluation.selected_metric
            self.checkpoint_manager.save(
                self.output_dir / "best",
                iteration=iteration,
                global_step=self.state.global_step,
                best_eval_metric=self.state.best_eval_metric,
            )
        if is_main():
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        **evaluation.metrics,
                        "val_num_episodes": evaluation.num_episodes,
                    }
                )
            )

    def _log(
        self,
        iteration: int,
        metrics: dict[str, float],
        *,
        started_at: float,
    ) -> None:
        if iteration % self.config.training.log_interval != 0 and iteration != 1:
            return
        self.reporter.log_iteration(
            iteration=iteration,
            global_step=self.state.global_step,
            metrics=metrics,
            elapsed_seconds=time.time() - started_at,
        )

    def _save_periodic(self, iteration: int) -> None:
        if iteration % self.config.training.save_interval != 0:
            return
        for checkpoint_dir in (
            self.output_dir / f"iter_{iteration:04d}",
            self.output_dir / "latest",
        ):
            self.checkpoint_manager.save(
                checkpoint_dir,
                iteration=iteration,
                global_step=self.state.global_step,
                best_eval_metric=self.state.best_eval_metric,
            )

    def _print_phase(self, iteration: int) -> None:
        if is_main():
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "phase": "rollout",
                        "num_episodes": self.config.rl.envs_per_iteration,
                    }
                )
            )

    @staticmethod
    def _warn_skip(iteration: int, reason: str) -> None:
        if is_main():
            print(json.dumps({"iteration": iteration, "warning": f"{reason}, skipping"}))

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


__all__ = ["RLLoopState", "RLTrainingLoop"]
