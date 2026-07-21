"""RL 训练入口：校验运行模式、装配依赖并启动训练循环。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nimloth.backbone import resolve_tune_modes
from nimloth.config.rl import RLConfig
from nimloth.rollout import RolloutCollector
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.objective import RLObjective
from nimloth.training.rl.checkpoint_manager import RLCheckpointManager
from nimloth.training.rl.components import build_rl_components
from nimloth.training.rl.loop import RLLoopState, RLTrainingLoop
from nimloth.training.rl.reporting import RLReporter
from nimloth.training.rl.rollout_runtime import (
    bind_online_collectors,
    validate_collector_configuration,
)
from nimloth.training.rl.update import RLUpdater
from nimloth.util.distributed import cleanup_dist, setup_dist


def train_rl(
    *,
    args: argparse.Namespace,
    config: RLConfig,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    output_dir: Path,
) -> int:
    """装配 RL runtime；核心 batch 算法见 ``RLAlgorithm``。"""

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
            policy=components.adapters.policy,
            latent_token_count=1,
            world_size=world,
        )
        checkpoint_manager = RLCheckpointManager(
            args=args,
            config=config,
            components=components,
        )
        algorithm = RLAlgorithm(
            agent=components.agent,
            objective=RLObjective(
                value_rank_margin=config.value_head.rank_margin,
                value_rank_weight=config.value_head.lambda_rank,
                ppo_clip_ratio=config.actor.clip_ratio,
                entropy_weight=config.actor.entropy_coeff,
            ),
            policy_replay=(
                components.adapters.policy_replay if actor_enabled else None
            ),
        )
        updater = RLUpdater(
            algorithm=algorithm,
            optimizer=components.optimizer,
            device=device,
            vision_ema=components.vision_ema,
        )
        loop = RLTrainingLoop(
            config=config,
            updater=updater,
            rollout_encoder=components.adapters.rollout_encoder,
            train_collector=train_collector,
            eval_collector=eval_collector,
            output_dir=output_dir,
            checkpoint_manager=checkpoint_manager,
            reporter=reporter,
            start_iteration=components.resume.start_iteration,
            state=RLLoopState(
                global_step=components.resume.global_step,
                best_eval_metric=components.resume.best_eval_metric,
            ),
        )
        loop.run()
        return 0
    finally:
        reporter.finish()
        cleanup_dist()
