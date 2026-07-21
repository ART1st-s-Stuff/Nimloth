"""RL 阶段训练循环：rollout → encode → train → repeat。

Qwen model loading is handled inside ``train_rl`` via
``configure_qwen_tuning`` (supports LLM freeze/lora/full +
vision freeze/lora/full).  Resume from a previous RL checkpoint
(``--resume``) reloads the Qwen model, WM heads, and optimizer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from nimloth.config.rl import RLConfig
from nimloth.backbone.qwen25vl.tuning import resolve_tune_modes, uses_lora
from nimloth.training.rl.checkpoint import save_rl_checkpoint
from nimloth.training.rl.components import build_rl_components
from nimloth.rollout import (
    JSONLRolloutCollector,
    RolloutCollector,
    encode_rollout_transitions,
)
from nimloth.training.rl.step import run_rl_optimizer_step
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.wandb import init_wandb_run, log_metrics


def train_rl(
    *,
    args: argparse.Namespace,
    config: RLConfig,
    train_collector: RolloutCollector,
    eval_collector: RolloutCollector | None,
    output_dir: Path,
) -> int:
    """运行 RL 阶段训练循环。"""

    iterations = config.rl.iterations
    envs_per_iter = config.rl.envs_per_iteration
    max_steps_per_ep = config.rl.max_steps_per_episode
    gamma = config.rl.gamma
    batch_size = config.rl.batch_size
    log_interval = config.training.log_interval
    save_interval = config.training.save_interval
    val_enabled = config.validation.enabled
    val_interval = config.validation.interval
    val_envs = config.validation.envs
    seed = config.training.seed

    # --- tuning modes --------------------------------------------------------
    llm_tune, vision_tune = resolve_tune_modes(args)
    actor_enabled = llm_tune != "freeze" or vision_tune != "freeze"
    if actor_enabled and isinstance(train_collector, JSONLRolloutCollector):
        raise ValueError(
            "PPO actor requires fresh trajectories from the current policy; "
            "static JSONL rollout is only supported for WM/value training"
        )
    if val_enabled and eval_collector is None:
        raise ValueError("validation.enabled requires a separate eval collector")

    # --- distributed setup ---------------------------------------------------
    rank, world, local_rank, device = setup_dist()
    if actor_enabled and world > 1:
        raise RuntimeError(
            "multi-rank PPO actor is disabled until rollout freshness and FSDP "
            "forward/EMA semantics have dedicated integration coverage"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    wandb_run = init_wandb_run(
        rank=rank,
        output_dir=output_dir,
        enabled=True,
        default_project="nimloth-rl",
        run_name=args.experiment_name,
        config=config.to_dict(),
        metric_definitions=(("global_step", None), ("train/*", "global_step")),
    )

    components = build_rl_components(
        args,
        config,
        output_dir=output_dir,
        device=device,
        rank=rank,
        world_size=world,
    )
    model = components.model
    processor = components.processor
    token_id_map = components.token_id_map
    state_proj = components.state_proj
    wm_predictor = components.wm_predictor
    value_head = components.value_head
    vision_ema = components.vision_ema
    optimizer = components.optimizer
    base_model_path = components.base_model_path
    start_iteration = components.resume.start_iteration
    global_step = components.resume.global_step
    best_eval_metric = components.resume.best_eval_metric
    best_checkpoint_dir = output_dir / "best"
    latest_checkpoint_dir = output_dir / "latest"

    def _save_checkpoint(path: Path, iteration: int) -> None:
        """用同一份 artifact 契约保存周期、best 和 final checkpoint。"""

        save_rl_checkpoint(
            path,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
            model=model,
            processor=processor,
            vision_ema=vision_ema,
            optimizer=optimizer,
            iteration=iteration,
            global_step=global_step,
            best_eval_metric=best_eval_metric,
            checkpoint_metric=config.validation.checkpoint_metric,
            lora=uses_lora(args),
            llm_tune=llm_tune,
            vision_tune=vision_tune,
            base_model_path=base_model_path,
        )

    # 在线采集器复用 trainer 已加载的模型；多卡仍需使用独立 rollout 进程。
    from nimloth.rollout import VAGENNavigationRolloutCollector

    online_collectors = [
        candidate
        for candidate in (train_collector, eval_collector)
        if isinstance(candidate, VAGENNavigationRolloutCollector)
    ]
    if online_collectors:
        if world > 1:
            raise RuntimeError(
                "分布式/FSDP trainer 不能直接让在线 rollout collector 使用 "
                "FSDP-wrapped Qwen "
                "做动态 env rollout。各 rank 的 episode 长度、图片数、失败时机不同，会导致 "
                "FSDP forward 触碰次数和形状不一致，可能 deadlock 或错误训练。\n"
                "请先使用独立 rollout backend（如 experiments/training/rl/rollout_env.py）"
                "生成 JSONL 文件，再用 --use-jsonl-rollout --jsonl-sources 指定 JSONL 路径训练。"
            )
        for online_collector in online_collectors:
            online_collector.bind_policy(model, processor, device)
        if is_main():
            print(json.dumps({"env_collector": "wired", "device": str(device)}))

    # --- logging --------------------------------------------------------------
    log_writer = CSVRecordWriter(
        output_dir / "train_step_log.csv",
        (
            "time",
            "iteration",
            "global_step",
            "wm_mse",
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
        ),
    )
    if is_main():
        log_writer.ensure_header()

    # --- main loop ------------------------------------------------------------
    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # 1. Collect trajectories -------------------------------------------------
        if is_main():
            print(json.dumps({"iteration": iteration, "phase": "rollout",
                              "num_episodes": envs_per_iter}))
        trajectories = train_collector.collect(
            num_episodes=envs_per_iter,
            max_steps_per_episode=max_steps_per_ep,
            output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
        )
        if is_main():
            print(json.dumps({"iteration": iteration,
                              "trajectories_collected": len(trajectories)}))

        if not trajectories:
            if is_main():
                print(json.dumps({"iteration": iteration,
                                  "warning": "no trajectories collected, skipping"}))
            continue

        # 2. Encode → transitions ------------------------------------------------
        transitions = encode_rollout_transitions(
            trajectories,
            model,
            processor,
            token_id_map,
            device,
            gamma=gamma,
        )
        # Free GPU memory before PPO forward (Qwen+LoRA+gradients needs extra VRAM)
        torch.cuda.empty_cache()
        if len(transitions) < batch_size:
            if is_main():
                print(json.dumps({
                    "iteration": iteration,
                    "warning": f"only {len(transitions)} transitions, need {batch_size}",
                }))
            continue

        # 3. 每个 iteration 只执行一次联合 optimizer step。
        step_result = run_rl_optimizer_step(
            transitions=transitions,
            batch_size=batch_size,
            batch_seed=seed + iteration,
            model=model,
            processor=processor,
            token_id_map=token_id_map,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
            value_head=value_head,
            optimizer=optimizer,
            vision_ema=vision_ema,
            config=config,
            actor_enabled=actor_enabled,
            device=device,
        )
        global_step += 1
        iter_metrics = {
            **step_result.metrics,
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(len(transitions)),
            "success_rate": float(
                sum(1 for t in trajectories if t.success) / max(1, len(trajectories))
            ),
        }

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- validation rollout -------------------------------------------------
        if val_enabled and iteration % val_interval == 0:
            assert eval_collector is not None
            val_trajectories = eval_collector.collect(
                num_episodes=val_envs,
                max_steps_per_episode=max_steps_per_ep,
                output_dir=output_dir / f"rollouts/val_{iteration:04d}",
            )
            if val_trajectories:
                val_success = sum(1 for t in val_trajectories if t.success) / len(val_trajectories)
                val_avg_reward = sum(t.reward for t in val_trajectories) / len(val_trajectories)
                val_avg_steps = sum(t.num_steps for t in val_trajectories) / len(val_trajectories)
                iter_metrics["val_success_rate"] = float(val_success)
                iter_metrics["val_avg_reward"] = float(val_avg_reward)
                iter_metrics["val_avg_steps"] = float(val_avg_steps)
                selected_metric = (
                    val_success
                    if config.validation.checkpoint_metric == "success_rate"
                    else val_avg_reward
                )
                if selected_metric > best_eval_metric:
                    best_eval_metric = float(selected_metric)
                    _save_checkpoint(best_checkpoint_dir, iteration)
                if is_main():
                    print(json.dumps({
                        "iteration": iteration,
                        "val_success_rate": val_success,
                        "val_avg_reward": val_avg_reward,
                        "val_num_episodes": len(val_trajectories),
                    }))

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- logging -----------------------------------------------------------
        if is_main() and (iteration % log_interval == 0 or iteration == 1):
            log_writer.append(
                {
                    "time": time.time(),
                    "iteration": iteration,
                    "global_step": global_step,
                    **iter_metrics,
                }
            )
            elapsed = time.time() - iter_start
            print(json.dumps({
                "iteration": iteration,
                "global_step": global_step,
                "metrics": iter_metrics,
                "elapsed_s": round(elapsed, 1),
            }))
            log_metrics(
                wandb_run,
                namespace="train",
                metrics=iter_metrics,
                step=global_step,
                context={"global_step": global_step, "iteration": iteration},
            )

        # --- checkpoint --------------------------------------------------------
        if iteration % save_interval == 0:
            _save_checkpoint(output_dir / f"iter_{iteration:04d}", iteration)
            _save_checkpoint(latest_checkpoint_dir, iteration)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # --- final checkpoint -----------------------------------------------------
    _save_checkpoint(output_dir / "final", iterations)
    _save_checkpoint(latest_checkpoint_dir, iterations)
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0
