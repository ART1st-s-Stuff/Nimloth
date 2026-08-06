"""RL rollout、更新、验证和 checkpoint 生命周期。"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, replace
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
    RLBatch,
    build_rl_batch,
)
from nimloth.training.rl.episodes import (
    ExecutedTransition,
    build_episode_training_batches,
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


@dataclass(frozen=True)
class _PlannerTransitionWork:
    """One rank-local planner forward, including collective-safe padding."""

    global_index: int
    is_padding: bool


def _planner_transition_work(
    total_transitions: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[_PlannerTransitionWork, ...]:
    """Shard transitions once globally while keeping equal DDP forward counts.

    DDP requires every rank to execute the same ordered set of synchronized
    module forwards/backwards.  When ``total_transitions`` is not divisible by
    ``world_size``, shorter ranks therefore replay a real transition with a
    zero-weight loss.  Padding never contributes metrics or gradients.
    """

    if total_transitions < 1:
        raise ValueError("planner training requires at least one transition")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")

    forwards_per_rank = (total_transitions + world_size - 1) // world_size
    work: list[_PlannerTransitionWork] = []
    for local_index in range(forwards_per_rank):
        global_index = local_index * world_size + rank
        is_padding = global_index >= total_transitions
        if is_padding:
            # All ranks hold the complete immutable rollout batch, so any valid
            # item can provide the graph-only padding forward.
            global_index = rank % total_transitions
        work.append(
            _PlannerTransitionWork(
                global_index=global_index,
                is_padding=is_padding,
            )
        )
    return tuple(work)


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
    write_final_checkpoint: bool
    start_iteration: int
    state: RLLoopState
    _latest_checkpoint_global_step: int | None = field(
        init=False,
        default=None,
    )

    def run(self) -> RLLoopState:
        for iteration in range(
            self.start_iteration,
            self.config.rl.iterations + 1,
        ):
            self._run_iteration(iteration)

        latest = self.output_dir / "latest"
        self._ensure_latest_checkpoint(self.config.rl.iterations)
        if self.write_final_checkpoint:
            self.checkpoint_manager.link_snapshot(
                latest,
                self.output_dir / "final",
            )
        return self.state

    def _run_iteration(self, iteration: int) -> None:
        """采集一批轨迹，并用这批轨迹完成恰好一次参数更新。"""

        started_at = time.time()
        self._print_phase(iteration)

        # rollout 只负责产生训练数据；尚未得到可训练 batch 时不占用 fresh
        # rollout，也不会推进 global_step。
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
        truncated_bootstrap = (
            0.0 if self.config.rl.truncated_bootstrap == "zero" else None
        )

        # planner 路线保留完整 episode，并对每个真实环境 transition 重算完整
        # Qwen prefix。第一个 PPO epoch联合计算WM与clipped critic，后续epoch只更新
        # critic及其上游Qwen表征；无planner路线继续使用固定长度sequence objective。
        episode_batches = None
        batch = None
        actor_transitions: tuple[ExecutedTransition, ...] = ()
        transition_returns: tuple[torch.Tensor, ...] = ()
        planner_work: tuple[_PlannerTransitionWork, ...] = ()
        local_actor_transitions: tuple[ExecutedTransition, ...] = ()
        local_transition_returns: tuple[torch.Tensor, ...] = ()
        transition_dino_grid_targets: tuple[torch.Tensor | None, ...] = ()
        if self.config.agent.planning.enabled:
            if len(trajectories) != self.config.rl.batch_size:
                raise RuntimeError(
                    "planner episode batch is incomplete: "
                    f"{len(trajectories)} != {self.config.rl.batch_size}"
                )
            episode_batches = build_episode_training_batches(
                trajectories,
                gamma=self.config.rl.gamma,
                truncated_bootstrap=truncated_bootstrap,
            )
            actor_transitions = tuple(
                transition
                for episode in episode_batches
                for transition in episode.transitions
            )
            transition_returns = tuple(
                target
                for episode in episode_batches
                for target in episode.return_targets.unbind(0)
            )
            rank, world_size = self._distributed_rank_world()
            planner_work = _planner_transition_work(
                len(actor_transitions),
                rank=rank,
                world_size=world_size,
            )
            local_actor_transitions = tuple(
                actor_transitions[item.global_index] for item in planner_work
            )
            local_transition_returns = tuple(
                transition_returns[item.global_index] for item in planner_work
            )
            transition_dino_grid_targets = (
                self._dino_grid_targets_for_actor_transitions(
                    local_actor_transitions
                )
            )
            training_unit_metrics = {
                "num_actor_transitions": float(len(actor_transitions))
            }
        else:
            num_windows = count_trajectory_windows(
                trajectories,
                history_size=self.config.predictor.history_size,
            )
            if num_windows < self.config.rl.batch_size:
                self._warn_skip(
                    iteration,
                    f"only {num_windows} sequence windows, "
                    f"need {self.config.rl.batch_size}",
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
                truncated_bootstrap=truncated_bootstrap,
                device=self.device,
            )
            batch = self._with_sequence_dino_grid_target(batch)
            training_unit_metrics = {"num_wm_windows": float(num_windows)}

        # FreshRolloutCollector 用这组三阶段 hook 提供消费事务；普通在线 collector
        # 没有这些方法。三者必须同时存在，避免只标记开始却无法回滚或提交。
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

        # 一个 planner iteration 对同一 frozen old value执行多个PPO epoch。只有在首个
        # optimizer.step()尚未开始时，异常才可以确定参数未变并安全释放fresh rollout；
        # 任一step开始后，消费记录必须保留为in_progress供恢复逻辑处理。
        optimizer_step_started = False
        try:
            step_metrics: dict[str, float] = {}
            if episode_batches is not None:
                total_actor_transitions = len(actor_transitions)
                _, training_world_size = self._distributed_rank_world()
                local_old_action_values = tuple(
                    self.algorithm.planner_old_action_value(
                        self.model_runtime,
                        transition,
                    )
                    for transition in local_actor_transitions
                )
                epoch_metrics: list[dict[str, float]] = []
                for ppo_epoch in range(self.config.value_head.ppo_epochs):
                    self.optimization_runtime.zero_grad()
                    current_metrics: dict[str, float] = {}
                    for (
                        work,
                        transition,
                        return_target,
                        old_action_value,
                        dino_grid_target,
                    ) in zip(
                        planner_work,
                        local_actor_transitions,
                        local_transition_returns,
                        local_old_action_values,
                        transition_dino_grid_targets,
                        strict=True,
                    ):
                        output = self.algorithm.actor_transition_step(
                            self.model_runtime,
                            transition,
                            return_target=return_target,
                            old_action_value=old_action_value,
                            total_transitions=total_actor_transitions,
                            dino_grid_target=dino_grid_target,
                            include_world_model=ppo_epoch == 0,
                        )
                        loss_weight = (
                            0.0 if work.is_padding else float(training_world_size)
                        )
                        self.optimization_runtime.backward(output.loss * loss_weight)
                        self._synchronize_planner_backward()
                        self._accumulate_metrics(
                            current_metrics,
                            output.metrics,
                            include=not work.is_padding,
                        )
                        del output
                    optimizer_step_started = True
                    self.optimization_runtime.step()
                    epoch_metrics.append(
                        self._reduce_planner_step_metrics(current_metrics)
                    )
                step_metrics = self._summarize_planner_ppo_epochs(epoch_metrics)
            else:
                assert batch is not None
                self.optimization_runtime.zero_grad()
                output = self.algorithm.sequence_step(
                    self.model_runtime,
                    batch,
                )
                self.optimization_runtime.backward(output.loss)
                self._accumulate_metrics(step_metrics, output.metrics)
                del output
                optimizer_step_started = True
                self.optimization_runtime.step()
        except Exception as error:
            if consumption_id is not None and not optimizer_step_started:
                assert abort_consumption is not None
                rank, world_size = self._distributed_rank_world()
                print(
                    json.dumps(
                        {
                            "phase": "planner_optimizer_update",
                            "rank": rank,
                            "world_size": world_size,
                            "exception_type": type(error).__name__,
                            "exception": str(error),
                            "optimizer_step_started": False,
                            "consumption_state": (
                                "aborting"
                                if world_size == 1
                                else "left_in_progress_after_rank_local_failure"
                            ),
                        },
                        default=str,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                # A distributed exception is not a safe point for another
                # collective. Calling the rank-zero consumption transaction here
                # previously hid the original rank-local error behind a mismatched
                # one-element broadcast. Leave the claim fail-closed instead.
                if world_size == 1:
                    abort_consumption(consumption_id)
            raise

        # global_step统计完整消费的一批fresh rollout；planner内部的多个critic optimizer
        # epoch共享同一个global_step，因此每个成功iteration仍只递增一次。
        self.state.global_step += 1
        rollout_metrics = summarize_rollouts(trajectories)
        metrics = {
            **step_metrics,
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(num_transitions),
            **training_unit_metrics,
            "success_rate": float(rollout_metrics["success_rate"]),
        }

        # 所有 rank 先完成训练，再进入 validation、日志和 checkpoint 阶段，避免
        # 某个 rank 提前开始下一阶段而破坏 collective 的调用顺序。
        self._barrier()
        self._validate(iteration, metrics)
        self._barrier()
        self._log(iteration, metrics, started_at=started_at)
        self._save_periodic(iteration)

        # fresh rollout 只有在更新后的 latest checkpoint 已经完整落盘后才算消费
        # 成功。这样 committed 记录总能指向可恢复的模型状态，即使本轮不保存周期
        # checkpoint，也会在这里补写 latest。
        if consumption_id is not None:
            checkpoint_dir = self.output_dir / "latest"
            self._ensure_latest_checkpoint(iteration)
            self._barrier()
            assert commit_consumption is not None
            commit_consumption(
                consumption_id,
                checkpoint_path=checkpoint_dir,
                global_step=self.state.global_step,
            )
        self._barrier()

    def _load_dino_grid_target_batch(
        self,
        image_paths: tuple[str, ...],
    ) -> torch.Tensor | None:
        """在训练前一次读取当前更新实际需要的 frozen DINO targets。"""

        if not (
            self.algorithm.train_world_model
            and self.algorithm.dino_grid_weight != 0.0
        ):
            return None
        source = self.model_runtime.dino_grid_targets
        if source is None:
            raise RuntimeError("RL DINO-grid loss has no frozen DINO target source")
        return source.load(
            image_paths,
            device=torch.device("cpu"),
        ).to(dtype=torch.float32)

    def _dino_grid_targets_for_actor_transitions(
        self,
        transitions: tuple[ExecutedTransition, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        """按 transition 顺序装配 next-image target，再逐个搬上 GPU。"""

        targets = self._load_dino_grid_target_batch(
            tuple(transition.next_image_path for transition in transitions)
        )
        if targets is None:
            return (None,) * len(transitions)
        return tuple(target.unsqueeze(0) for target in targets.unbind(0))

    def _with_sequence_dino_grid_target(self, batch: RLBatch) -> RLBatch:
        """把扁平 next-image targets 还原成 sequence objective 的 ``(B,H,...)``。"""

        targets = self._load_dino_grid_target_batch(batch.next_image_paths)
        if targets is None:
            return batch
        target_shape = (*batch.action_indices.shape, *targets.shape[1:])
        return replace(
            batch,
            dino_grid_target=targets.reshape(target_shape),
        )

    @staticmethod
    def _accumulate_metrics(
        totals: dict[str, float],
        current: dict[str, float],
        *,
        include: bool = True,
    ) -> None:
        for name, value in current.items():
            if name.startswith("lambda_"):
                totals[name] = value
            else:
                totals[name] = totals.get(name, 0.0) + (value if include else 0.0)

    def _reduce_planner_step_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        """Return global planner metrics after rank-local transition sharding."""

        if not (dist.is_available() and dist.is_initialized()):
            return metrics
        names = tuple(sorted(metrics))
        gathered_names: list[tuple[str, ...] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered_names, names)
        if any(current != names for current in gathered_names):
            raise RuntimeError(
                "planner metric keys differ across distributed training ranks"
            )
        values = torch.tensor(
            [metrics[name] for name in names],
            dtype=torch.float64,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
        return {
            name: (
                float(values[index].item()) / world_size
                if name.startswith("lambda_")
                else float(values[index].item())
            )
            for index, name in enumerate(names)
        }

    @staticmethod
    def _summarize_planner_ppo_epochs(
        epochs: list[dict[str, float]],
    ) -> dict[str, float]:
        """Keep first-epoch WM metrics and average critic metrics across PPO epochs."""

        if len(epochs) < 2:
            raise ValueError("planner PPO critic requires at least two epoch metrics")
        summary = dict(epochs[0])
        averaged = {
            "value_loss",
            "value_mc_mse",
            "value_clipped_mse",
            "value_clip_fraction",
            "value_old_mean",
            "value_delta_abs_mean",
        }
        for name in averaged:
            if any(name not in metrics for metrics in epochs):
                raise RuntimeError(f"planner PPO epoch metric is missing {name!r}")
            summary[name] = sum(metrics[name] for metrics in epochs) / len(epochs)
        # WM/DINO只在首个epoch计算；total_loss保留该首轮辅助objective，再加上
        # 各critic epoch的平均value loss，避免将WM项错误地除以PPO epoch数。
        if any("total_loss" not in metrics for metrics in epochs):
            raise RuntimeError("planner PPO epoch metric is missing 'total_loss'")
        summary["total_loss"] = (
            epochs[0]["total_loss"]
            - epochs[0]["value_loss"]
            + summary["value_loss"]
        )
        summary["value_ppo_epochs"] = float(len(epochs))
        return summary

    @staticmethod
    def _distributed_rank_world() -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

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
        iteration_checkpoint = self.output_dir / f"iter_{iteration:04d}"
        latest = self.output_dir / "latest"
        if iteration == self.config.rl.iterations:
            self._ensure_latest_checkpoint(iteration)
            self.checkpoint_manager.link_snapshot(latest, iteration_checkpoint)
            return
        self.checkpoint_manager.save(
            iteration_checkpoint,
            iteration=iteration,
            global_step=self.state.global_step,
            best_eval_metric=self.state.best_eval_metric,
        )
        self._save_latest_checkpoint(iteration)

    def _ensure_latest_checkpoint(self, iteration: int) -> None:
        if self._latest_checkpoint_global_step == self.state.global_step:
            return
        self._save_latest_checkpoint(iteration)

    def _save_latest_checkpoint(self, iteration: int) -> None:
        self.checkpoint_manager.save(
            self.output_dir / "latest",
            iteration=iteration,
            global_step=self.state.global_step,
            best_eval_metric=self.state.best_eval_metric,
        )
        self._latest_checkpoint_global_step = self.state.global_step

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
            print(
                json.dumps(
                    {"iteration": iteration, "warning": f"{reason}, skipping"}
                )
            )

    @staticmethod
    def _synchronize_planner_backward() -> None:
        """让多个DDP wrapper共享明确的逐transition通信边界。

        完整Qwen prefix的计算时间随rank而异。缺少此边界时，快rank可能在慢rank
        仍归约上一次backward时进入另一个wrapper的forward，最终让相同NCCL序号
        对应不同collective类型。
        """

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


__all__ = ["RLLoopState", "RLTrainingLoop"]
