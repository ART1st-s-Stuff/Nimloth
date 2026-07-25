"""RL rollout、更新、验证和 checkpoint 生命周期。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist

from nimloth.config.rl import RLConfig
from nimloth.rollout import (
    RolloutCollector,
    count_trajectory_windows,
    sample_trajectory_windows,
)
from nimloth.training.rl.algorithm import (
    RLAlgorithm,
    build_rl_batch,
)
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.evaluation import (
    evaluate_rollout_collector,
    summarize_rollouts,
)
from nimloth.training.rl.reporting import RLReporter
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.util.distributed import is_main
from nimloth.util.optim import OptimizationRuntime


@dataclass
class RLLoopState:
    """RL checkpoint 需要保存的训练位置与 best 指标。"""

    global_step: int
    best_eval_metric: float


@dataclass
class RLTrainingLoop:
    """按 iteration 执行 collect → sample → encode/update → evaluate。"""

    config: RLConfig
    algorithm: RLAlgorithm
    model_runtime: RLModelRuntime
    optimization_runtime: OptimizationRuntime
    device: torch.device
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

        num_transitions = sum(
            trajectory.num_steps for trajectory in trajectories
        )
        num_windows = count_trajectory_windows(
            trajectories,
            history_size=self.config.predictor.history_size,
        )
        if num_windows < self.config.rl.batch_size:
            self._warn_skip(
                iteration,
                f"only {num_windows} sequence windows, need {self.config.rl.batch_size}",
            )
            return

        windows = sample_trajectory_windows(
            trajectories,
            history_size=self.config.predictor.history_size,
            batch_size=self.config.rl.batch_size,
            seed=self.config.training.seed + iteration,
        )
        batch = build_rl_batch(
            windows,
            gamma=self.config.rl.gamma,
            truncated_bootstrap=(
                0.0
                if self.config.rl.truncated_bootstrap == "zero"
                else None
            ),
            device=self.device,
        )
        begin_consumption = getattr(self.train_collector, "begin_consumption", None)
        abort_consumption = getattr(self.train_collector, "abort_consumption", None)
        commit_consumption = getattr(self.train_collector, "commit_consumption", None)
        consumption_hooks = (
            begin_consumption,
            abort_consumption,
            commit_consumption,
        )
        if any(hook is not None for hook in consumption_hooks) and not all(
            hook is not None for hook in consumption_hooks
        ):
            raise TypeError("fresh rollout collector has an incomplete consumption API")
        consumption_id = (
            begin_consumption(
                output_dir=self.output_dir,
                global_step=self.state.global_step,
            )
            if begin_consumption is not None
            else None
        )
        optimizer_step_started = False
        try:
            self.optimization_runtime.zero_grad()
            step_output = self.algorithm.training_step(self.model_runtime, batch)
            self.optimization_runtime.backward(step_output.loss)
            optimizer_step_started = True
            self.optimization_runtime.step()
        except Exception:
            if consumption_id is not None and not optimizer_step_started:
                assert abort_consumption is not None
                abort_consumption(consumption_id)
            raise
        self.state.global_step += 1
        rollout_metrics = summarize_rollouts(trajectories)
        metrics = {
            **step_output.metrics,
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(num_transitions),
            "num_wm_windows": float(num_windows),
            "success_rate": float(rollout_metrics["success_rate"]),
        }
        self._barrier()
        self._validate(iteration, metrics)
        self._barrier()
        self._log(iteration, metrics, started_at=started_at)
        self._save_periodic(iteration)
        if consumption_id is not None:
            checkpoint_dir = self.output_dir / "latest"
            if iteration % self.config.training.save_interval != 0:
                self.checkpoint_manager.save(
                    checkpoint_dir,
                    iteration=iteration,
                    global_step=self.state.global_step,
                    best_eval_metric=self.state.best_eval_metric,
                )
            self._barrier()
            assert commit_consumption is not None
            commit_consumption(
                consumption_id,
                checkpoint_path=checkpoint_dir,
                global_step=self.state.global_step,
            )
        self._barrier()

    def _validate(self, iteration: int, metrics: dict[str, float]) -> None:
        if not (
            self.config.validation.enabled
            and iteration % self.config.validation.interval == 0
        ):
            return
        if self.eval_collector is None:
            raise RuntimeError(
                "validation is enabled but no evaluation collector was configured"
            )
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
