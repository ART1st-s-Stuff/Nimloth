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
    SequenceLossNormalization,
    build_rl_batch,
    normalized_monte_carlo_advantages,
    slice_rl_batch,
)
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.episodes import (
    ExecutedTransition,
    build_episode_training_batches,
)
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


def _count_outcome_labels(
    transitions: tuple[ExecutedTransition, ...],
    *,
    enabled: bool,
) -> int | None:
    """Count masked BCE labels without imposing Outcome fields on legacy runs."""

    if not enabled:
        return None
    count = sum(
        getattr(transition, "action_success", None) is not None
        for transition in transitions
    )
    if count == 0:
        raise ValueError(
            "OutcomeHead training requires at least one fresh action_success label"
        )
    return count


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
        # Qwen prefix。每个fresh batch只有一个optimizer epoch，在同一次更新中联合
        # 计算WM、DINO、ValueHead和可选PlannerPolicyHead objective。
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
                outcome_enabled = bool(
                    getattr(
                        getattr(self.config, "outcome_head", None),
                        "enabled",
                        False,
                    )
                )
                total_outcomes = _count_outcome_labels(
                    actor_transitions,
                    enabled=outcome_enabled,
                )
                _, training_world_size = self._distributed_rank_world()
                if self.config.planner_policy.enabled:
                    local_old_statistics = tuple(
                        self.algorithm.planner_old_policy_statistics(
                            self.model_runtime,
                            transition,
                        )
                        for transition in local_actor_transitions
                    )
                    local_old_action_values = tuple(
                        statistics.selected_action_value
                        for statistics in local_old_statistics
                    )
                    local_old_policy_log_probs = tuple(
                        statistics.selected_log_prob
                        for statistics in local_old_statistics
                    )
                    local_policy_advantages = tuple(
                        return_target - statistics.state_value
                        for return_target, statistics in zip(
                            local_transition_returns,
                            local_old_statistics,
                            strict=True,
                        )
                    )
                    ppo_epochs = self.config.planner_policy.ppo_epochs
                else:
                    local_old_action_values = tuple(
                        self.algorithm.planner_old_action_value(
                            self.model_runtime,
                            transition,
                        )
                        for transition in local_actor_transitions
                    )
                    local_old_policy_log_probs = (None,) * len(
                        local_actor_transitions
                    )
                    local_policy_advantages = (None,) * len(
                        local_actor_transitions
                    )
                    ppo_epochs = self.config.value_head.ppo_epochs
                if ppo_epochs != 1:
                    raise ValueError(
                        "each fresh planner rollout batch requires exactly one "
                        "optimizer epoch"
                    )
                planner_micro_batch_size = int(
                    getattr(
                        self.config.training,
                        "planner_micro_batch_size",
                        1,
                    )
                )
                if planner_micro_batch_size < 1:
                    raise ValueError("planner_micro_batch_size must be positive")
                epoch_metrics: list[dict[str, float]] = []
                for ppo_epoch in range(ppo_epochs):
                    self.optimization_runtime.zero_grad()
                    current_metrics: dict[str, float] = {}
                    planner_rows = tuple(
                        zip(
                            planner_work,
                            local_actor_transitions,
                            local_transition_returns,
                            local_old_action_values,
                            local_old_policy_log_probs,
                            local_policy_advantages,
                            transition_dino_grid_targets,
                            strict=True,
                        )
                    )
                    for offset in range(
                        0,
                        len(planner_rows),
                        planner_micro_batch_size,
                    ):
                        micro_batch = planner_rows[
                            offset : offset + planner_micro_batch_size
                        ]
                        if planner_micro_batch_size == 1:
                            (
                                work,
                                transition,
                                return_target,
                                old_action_value,
                                old_policy_log_prob,
                                policy_advantage,
                                dino_grid_target,
                            ) = micro_batch[0]
                            output = self.algorithm.planner_transition_step(
                                self.model_runtime,
                                transition,
                                return_target=return_target,
                                old_action_value=old_action_value,
                                old_policy_log_prob=old_policy_log_prob,
                                policy_advantage=policy_advantage,
                                total_transitions=total_actor_transitions,
                                total_outcomes=total_outcomes,
                                dino_grid_target=dino_grid_target,
                                include_world_model=True,
                            )
                            loss_weight = (
                                0.0
                                if work.is_padding
                                else float(training_world_size)
                            )
                            self.optimization_runtime.backward(
                                output.loss * loss_weight
                            )
                            self._accumulate_metrics(
                                current_metrics,
                                output.metrics,
                                include=not work.is_padding,
                            )
                        else:
                            output = self.algorithm.planner_transition_batch_step(
                                self.model_runtime,
                                tuple(row[1] for row in micro_batch),
                                return_targets=tuple(row[2] for row in micro_batch),
                                old_action_values=tuple(row[3] for row in micro_batch),
                                old_policy_log_probs=tuple(
                                    row[4] for row in micro_batch
                                ),
                                policy_advantages=tuple(
                                    row[5] for row in micro_batch
                                ),
                                total_transitions=total_actor_transitions,
                                total_outcomes=total_outcomes,
                                dino_grid_targets=tuple(
                                    row[6] for row in micro_batch
                                ),
                                loss_weights=tuple(
                                    0.0
                                    if row[0].is_padding
                                    else float(training_world_size)
                                    for row in micro_batch
                                ),
                                include_world_model=True,
                            )
                            self.optimization_runtime.backward(output.loss)
                            self._accumulate_metrics(
                                current_metrics,
                                output.metrics,
                            )
                        self._synchronize_planner_backward()
                        del output
                    current_metrics.update(self._optimizer_gradient_metrics())
                    current_metrics["loss_finite"] = 1.0
                    optimizer_step_started = True
                    self.optimization_runtime.step()
                    current_metrics["optimizer_updates"] = 1.0
                    epoch_metrics.append(
                        self._reduce_planner_step_metrics(current_metrics)
                    )
                step_metrics = self._summarize_planner_update(
                    epoch_metrics,
                    planner_policy_enabled=self.config.planner_policy.enabled,
                )
            else:
                assert batch is not None
                self.optimization_runtime.zero_grad()
                configured_micro_batch_size = (
                    self.config.training.sequence_micro_batch_size
                )
                sequence_micro_batch_size = (
                    len(batch.windows)
                    if configured_micro_batch_size is None
                    else min(configured_micro_batch_size, len(batch.windows))
                )
                if sequence_micro_batch_size < len(batch.windows):
                    old_action_values = []
                    for offset in range(
                        0,
                        len(batch.windows),
                        sequence_micro_batch_size,
                    ):
                        old_value_batch = slice_rl_batch(
                            batch,
                            offset,
                            min(
                                offset + sequence_micro_batch_size,
                                len(batch.windows),
                            ),
                        )
                        old_action_values.append(
                            self.algorithm.sequence_old_action_values(
                                self.model_runtime,
                                old_value_batch,
                            )
                        )
                    full_old_action_values = torch.cat(old_action_values, dim=0)
                    full_step_advantages = normalized_monte_carlo_advantages(
                        return_targets=batch.return_targets.to(
                            device=full_old_action_values.device,
                            dtype=full_old_action_values.dtype,
                        ).flatten(),
                        predicted_values=full_old_action_values.flatten(),
                    ).reshape_as(batch.return_targets)
                    batch = replace(
                        batch,
                        policy_step_advantages=full_step_advantages,
                    )
                    total_outcomes = (
                        int(batch.action_success_mask.sum().item())
                        if batch.action_success_mask is not None
                        else 0
                    )
                    if self.config.outcome_head.enabled and total_outcomes == 0:
                        raise ValueError(
                            "OutcomeHead training requires at least one fresh "
                            "action_success label"
                        )
                    total_policy_tokens = int(batch.old_log_probs.numel())
                    if (
                        self.model_runtime.policy_replay is not None
                        and total_policy_tokens == 0
                    ):
                        raise ValueError(
                            "direct PPO sequence update requires policy tokens"
                        )
                    normalization = SequenceLossNormalization(
                        action_positions=batch.action_indices.numel(),
                        outcome_labels=total_outcomes,
                        policy_tokens=total_policy_tokens,
                    )
                    for offset in range(
                        0,
                        len(batch.windows),
                        sequence_micro_batch_size,
                    ):
                        micro_batch = slice_rl_batch(
                            batch,
                            offset,
                            min(
                                offset + sequence_micro_batch_size,
                                len(batch.windows),
                            ),
                        )
                        output = self.algorithm.sequence_step(
                            self.model_runtime,
                            micro_batch,
                            normalization=normalization,
                        )
                        if not torch.isfinite(output.loss):
                            raise FloatingPointError(
                                "sequence update produced a non-finite total loss"
                            )
                        self.optimization_runtime.backward(output.loss)
                        self._accumulate_metrics(step_metrics, output.metrics)
                        del output
                else:
                    output = self.algorithm.sequence_step(
                        self.model_runtime,
                        batch,
                    )
                    if not torch.isfinite(output.loss):
                        raise FloatingPointError(
                            "sequence update produced a non-finite total loss"
                        )
                    self.optimization_runtime.backward(output.loss)
                    self._accumulate_metrics(step_metrics, output.metrics)
                    del output
                step_metrics.update(self._optimizer_gradient_metrics())
                step_metrics["loss_finite"] = 1.0
                optimizer_step_started = True
                self.optimization_runtime.step()
                step_metrics["optimizer_updates"] = 1.0
        except Exception as error:
            if consumption_id is not None and not optimizer_step_started:
                assert abort_consumption is not None
                rank, world_size = self._distributed_rank_world()
                print(
                    json.dumps(
                        {
                            "phase": (
                                "planner_optimizer_update"
                                if episode_batches is not None
                                else "sequence_optimizer_update"
                            ),
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
            self.algorithm.config.predictor.train_wm
            and self.algorithm.config.predictor.lambda_dino != 0.0
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
        """按 transition 顺序装配current-image target，再逐个搬上 GPU。"""

        targets = self._load_dino_grid_target_batch(
            tuple(transition.current_image_path for transition in transitions)
        )
        if targets is None:
            return (None,) * len(transitions)
        return tuple(target.unsqueeze(0) for target in targets.unbind(0))

    def _with_sequence_dino_grid_target(self, batch: RLBatch) -> RLBatch:
        """把扁平 current-image targets 还原成 sequence objective 的 ``(B,H,...)``。"""

        targets = self._load_dino_grid_target_batch(batch.current_image_paths)
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

    def _optimizer_gradient_metrics(self) -> dict[str, float]:
        """Validate and summarize the gradients used by the imminent update."""

        squared_norm = torch.zeros((), dtype=torch.float64, device=self.device)
        parameter_count = 0
        group_squared_norms: dict[str, torch.Tensor] = {}
        group_parameter_counts: dict[str, int] = {}
        for group in self.optimization_runtime.optimizer.param_groups:
            group_name = str(group.get("name", "unnamed"))
            group_squared_norm = torch.zeros(
                (), dtype=torch.float64, device=self.device
            )
            group_parameter_count = 0
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(
                        "non-finite gradient in optimizer group "
                        f"{group.get('name', '<unnamed>')}"
                    )
                gradient_squared_norm = gradient.detach().double().square().sum()
                squared_norm = squared_norm + gradient_squared_norm
                group_squared_norm = group_squared_norm + gradient_squared_norm
                parameter_count += int(gradient.numel())
                group_parameter_count += int(gradient.numel())
            group_squared_norms[group_name] = group_squared_norm
            group_parameter_counts[group_name] = group_parameter_count
        if parameter_count == 0:
            raise RuntimeError("RL update produced no optimizer gradients")
        metrics = {
            "gradient_finite": 1.0,
            "gradient_parameter_count": float(parameter_count),
            "gradient_l2": float(squared_norm.sqrt().item()),
        }
        for group_name, group_squared_norm in group_squared_norms.items():
            metrics[f"gradient_{group_name}_parameter_count"] = float(
                group_parameter_counts[group_name]
            )
            metrics[f"gradient_{group_name}_l2"] = float(
                group_squared_norm.sqrt().item()
            )
        return metrics

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
        rank_mean_metrics = {
            "loss_finite",
            "gradient_finite",
            "gradient_l2",
            "optimizer_updates",
        }
        return {
            name: (
                float(values[index].item()) / world_size
                if name.startswith("lambda_") or name in rank_mean_metrics
                else float(values[index].item())
            )
            for index, name in enumerate(names)
        }

    @staticmethod
    def _summarize_planner_update(
        epochs: list[dict[str, float]],
        *,
        planner_policy_enabled: bool = False,
    ) -> dict[str, float]:
        """Return the one complete-objective optimizer epoch unchanged."""

        if len(epochs) != 1:
            raise ValueError(
                "planner update requires exactly one optimizer epoch metric"
            )
        summary = dict(epochs[0])
        if planner_policy_enabled:
            summary["planner_policy_ppo_epochs"] = 1.0
            summary["value_ppo_epochs"] = 0.0
        else:
            summary["value_ppo_epochs"] = 1.0
            summary["planner_policy_ppo_epochs"] = 0.0
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
