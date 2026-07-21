"""RL 阶段训练编排：rollout、encode、optimizer、validation、save。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes
from nimloth.backbone.qwen25vl.rollout import encode_rollout_transitions
from nimloth.config.rl import RLConfig
from nimloth.rollout import RolloutCollector
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.components import build_rl_components
from nimloth.training.rl.evaluation import (
    evaluate_rollout_collector,
    summarize_rollouts,
)
from nimloth.training.rl.reporting import RLReporter
from nimloth.training.rl.rollout_runtime import (
    bind_online_collectors,
    validate_collector_configuration,
)
from nimloth.training.rl.step import run_rl_optimizer_step
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def train_rl(
    *,
    args: argparse.Namespace,
    config: RLConfig,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    output_dir: Path,
) -> int:
    """运行 RL 阶段训练循环。"""

    llm_tune, vision_tune = resolve_tune_modes(args)
    actor_enabled = llm_tune != "freeze" or vision_tune != "freeze"
    validate_collector_configuration(
        actor_enabled=actor_enabled,
        train_collector=train_collector,
        eval_collector=eval_collector,
        validation_enabled=config.validation.enabled,
    )

    rank, world, _, device = setup_dist()
    if actor_enabled and world > 1:
        raise RuntimeError(
            "multi-rank PPO actor is disabled until rollout freshness and FSDP "
            "forward/EMA semantics have dedicated integration coverage"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.training.seed)
    reporter = RLReporter(
        rank=rank,
        output_dir=output_dir,
        run_name=args.experiment_name,
        config=config,
    )

    try:
        components = build_rl_components(
            args,
            config,
            output_dir=output_dir,
            device=device,
            rank=rank,
            world_size=world,
        )
        bind_online_collectors(
            train_collector=train_collector,
            eval_collector=eval_collector,
            model=components.model,
            processor=components.processor,
            device=device,
            world_size=world,
        )
        checkpoint_manager = RLCheckpointManager(
            args=args,
            config=config,
            components=components,
        )

        global_step = components.resume.global_step
        best_eval_metric = components.resume.best_eval_metric
        for iteration in range(
            components.resume.start_iteration,
            config.rl.iterations + 1,
        ):
            iteration_started = time.time()
            if is_main():
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "phase": "rollout",
                            "num_episodes": config.rl.envs_per_iteration,
                        }
                    )
                )
            trajectories = train_collector.collect(
                num_episodes=config.rl.envs_per_iteration,
                max_steps_per_episode=config.rl.max_steps_per_episode,
                output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
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
                if is_main():
                    print(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "warning": "no trajectories collected, skipping",
                            }
                        )
                    )
                continue

            transitions = encode_rollout_transitions(
                trajectories,
                components.model,
                components.processor,
                components.token_id_map,
                device,
                gamma=config.rl.gamma,
            )
            torch.cuda.empty_cache()
            if len(transitions) < config.rl.batch_size:
                if is_main():
                    print(
                        json.dumps(
                            {
                                "iteration": iteration,
                                "warning": (
                                    f"only {len(transitions)} transitions, "
                                    f"need {config.rl.batch_size}"
                                ),
                            }
                        )
                    )
                continue

            step_result = run_rl_optimizer_step(
                transitions=transitions,
                batch_size=config.rl.batch_size,
                batch_seed=config.training.seed + iteration,
                model=components.model,
                processor=components.processor,
                token_id_map=components.token_id_map,
                state_proj=components.state_proj,
                wm_predictor=components.wm_predictor,
                value_head=components.value_head,
                optimizer=components.optimizer,
                vision_ema=components.vision_ema,
                config=config,
                actor_enabled=actor_enabled,
                device=device,
            )
            global_step += 1
            rollout_metrics = summarize_rollouts(trajectories)
            iteration_metrics = {
                **step_result.metrics,
                "num_rollouts": float(len(trajectories)),
                "num_transitions": float(len(transitions)),
                "success_rate": float(rollout_metrics["success_rate"]),
            }
            _barrier()

            should_validate = (
                config.validation.enabled
                and iteration % config.validation.interval == 0
            )
            if should_validate:
                assert eval_collector is not None
                evaluation = evaluate_rollout_collector(
                    eval_collector,
                    num_episodes=config.validation.envs,
                    max_steps_per_episode=config.rl.max_steps_per_episode,
                    output_dir=output_dir / f"rollouts/val_{iteration:04d}",
                    checkpoint_metric=config.validation.checkpoint_metric,
                )
                if evaluation is not None:
                    iteration_metrics.update(evaluation.metrics)
                    if evaluation.selected_metric > best_eval_metric:
                        best_eval_metric = evaluation.selected_metric
                        checkpoint_manager.save(
                            output_dir / "best",
                            iteration=iteration,
                            global_step=global_step,
                            best_eval_metric=best_eval_metric,
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
            _barrier()

            should_log = (
                iteration % config.training.log_interval == 0 or iteration == 1
            )
            if should_log:
                reporter.log_iteration(
                    iteration=iteration,
                    global_step=global_step,
                    metrics=iteration_metrics,
                    elapsed_seconds=time.time() - iteration_started,
                )

            if iteration % config.training.save_interval == 0:
                for checkpoint_dir in (
                    output_dir / f"iter_{iteration:04d}",
                    output_dir / "latest",
                ):
                    checkpoint_manager.save(
                        checkpoint_dir,
                        iteration=iteration,
                        global_step=global_step,
                        best_eval_metric=best_eval_metric,
                    )
            _barrier()

        for checkpoint_dir in (output_dir / "final", output_dir / "latest"):
            checkpoint_manager.save(
                checkpoint_dir,
                iteration=config.rl.iterations,
                global_step=global_step,
                best_eval_metric=best_eval_metric,
            )
        return 0
    finally:
        reporter.finish()
        cleanup_dist()
