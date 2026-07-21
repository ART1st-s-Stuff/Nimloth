"""RL 阶段训练循环：rollout → encode → train → repeat。

Qwen model loading is handled inside ``train_rl`` via
``configure_qwen_tuning`` (supports LLM freeze/lora/full +
vision freeze/lora/full).  Resume from a previous RL checkpoint
(``--resume``) reloads the Qwen model, WM heads, and optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.config.rl import RLConfig
from nimloth.backbone.qwen25vl.policy import (
    validate_agent_policy_protocol,
)
from nimloth.backbone.qwen25vl.tuning import (
    configure_qwen_tuning,
    resolve_tune_modes,
    uses_lora,
)
from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.rl.checkpoint import (
    load_lora_adapter_state,
    load_rl_wm_checkpoint,
    save_rl_checkpoint,
)
from nimloth.training.rl.loss import compute_predictor_loss, compute_value_loss
from nimloth.rollout import RolloutCollector, encode_rollout_transitions
from nimloth.training.rl.actor import compute_current_policy_log_probs
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.wandb import init_wandb_run


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def _broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    """Synchronize a small non-FSDP module across ranks.

    JSONL/FSDP mode intentionally makes every rank consume identical data so the
    small WM/value modules can remain local replicas.  This only works if their
    initial parameters are identical; CLI construction happens before
    ``setup_dist()``, so we explicitly broadcast rank-0 state after device setup.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def train_rl(
    *,
    args: argparse.Namespace,
    config: RLConfig,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
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
    pred_lr = config.predictor.lr
    vh_lr = config.value_head.lr
    rank_margin = config.value_head.rank_margin
    lambda_rank = config.value_head.lambda_rank
    actor_lr = config.actor.lr
    entropy_coeff = config.actor.entropy_coeff
    clip_ratio = config.actor.clip_ratio
    freeze_state_proj = config.freeze.state_proj
    log_interval = config.training.log_interval
    save_interval = config.training.save_interval
    val_enabled = config.validation.enabled
    val_interval = config.validation.interval
    val_envs = config.validation.envs
    seed = config.training.seed

    # --- tuning modes --------------------------------------------------------
    llm_tune, vision_tune = resolve_tune_modes(args)
    actor_enabled = llm_tune != "freeze" or vision_tune != "freeze"
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)
    if val_enabled and eval_collector is None:
        raise ValueError("validation.enabled requires a separate eval collector")

    # --- distributed setup ---------------------------------------------------
    rank, world, local_rank, device = setup_dist()
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

    # --- Qwen model loading --------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    n_added = add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)
    tokenizer_vocab = len(processor.tokenizer)

    resume_ckpt_dir = output_dir / "best"
    resume_state_path = resume_ckpt_dir / "rl_state.pt"
    resume_adapter = resume_ckpt_dir / "adapter_config.json"
    base_model_path = str(args.model)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    validate_agent_policy_protocol(model.config)
    model_vocab_before = model.get_input_embeddings().weight.shape[0]
    # Log model embedding info before resize
    embed = model.get_input_embeddings()
    pad_idx = getattr(embed, "padding_idx", None)
    print(json.dumps({
        "rank": rank,
        "model_vocab_before": model_vocab_before,
        "tokenizer_vocab": tokenizer_vocab,
        "n_added": n_added,
        "padding_idx": pad_idx,
        "embed_weight_shape": list(embed.weight.shape),
    }), flush=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if n_added > 0:
        model.resize_token_embeddings(tokenizer_vocab)
        model_vocab_after = model.get_input_embeddings().weight.shape[0]
        print(json.dumps({"rank": rank, "resized": True,
                          "model_vocab_after": model_vocab_after}), flush=True)

    # Resume branches
    resume_aux_ckpt: Path | None = None  # for loading WM + optimizer later

    if args.resume and resume_state_path.exists() and resume_adapter.exists():
        if not uses_lora(args):
            raise ValueError("--resume with LoRA adapter requires llm_tune and/or vision_tune lora")
        saved = torch.load(resume_state_path, map_location="cpu", weights_only=False)
        saved_base = saved.get("base_model_path")
        if saved_base:
            base_model_path = str(saved_base)
        if is_main():
            print(json.dumps({"resume_lora_adapter": str(resume_ckpt_dir),
                              "base_model_path": base_model_path}))
        model = configure_qwen_tuning(model, args)
        load_lora_adapter_state(model, resume_ckpt_dir)
        resume_aux_ckpt = resume_ckpt_dir

    elif args.resume and resume_state_path.exists() and (resume_ckpt_dir / "config.json").exists():
        if uses_lora(args):
            raise ValueError("cannot --resume full HF checkpoint with lora tuning")
        if is_main():
            print(json.dumps({"resume_full": str(resume_ckpt_dir)}))
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resume_ckpt_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        model.resize_token_embeddings(len(processor.tokenizer))
        model = configure_qwen_tuning(model, args)
        resume_aux_ckpt = resume_ckpt_dir

    else:
        model = configure_qwen_tuning(model, args)
        if is_main():
            print(json.dumps({"init": "configured_tuning",
                              "base_model_path": base_model_path,
                              "llm_tune": llm_tune,
                              "vision_tune": vision_tune}))

    model.to(device)

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

    # --- freeze WM-encoding pathway if requested -----------------------------
    if freeze_state_proj:
        _freeze(state_proj)

    state_proj.to(device)
    wm_predictor.to(device)
    value_head.to(device)
    if world > 1:
        _broadcast_module_state(state_proj)
        _broadcast_module_state(wm_predictor)
        _broadcast_module_state(value_head)
        if is_main():
            print(json.dumps({"synced_local_wm_modules": True, "world_size": world}))

    # --- Vision EMA -----------------------------------------------------------
    vision_ema: VisionEncoderEMA | None = None
    if vision_ema_enabled:
        vision_ema = VisionEncoderEMA(decay=args.vision_ema_decay)
        vision_ema.reset(model)
        ema_path = resume_ckpt_dir / "vision_ema.pt"
        if args.resume and ema_path.is_file():
            loaded_ema = VisionEncoderEMA.load_checkpoint(ema_path, map_location=device)
            vision_ema.decay = loaded_ema.decay
            vision_ema.shadow = {k: v.to(device) for k, v in loaded_ema.shadow.items()}
        if is_main():
            print(json.dumps({"vision_ema": True,
                              "shadow_params": len(vision_ema.shadow),
                              "decay": vision_ema.decay}))

    # --- FSDP wrap ------------------------------------------------------------
    if world > 1:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
        )

        # FULL_SHARD splits the embedding across ranks. If the padding_idx
        # row doesn't fall on every rank's shard, FSDP forward hits:
        #   assert padding_idx < weight.size(0)
        # Clearing padding_idx is safe: it only zeroes the padding embedding
        # row during forward, which the model doesn't rely on.
        embed = model.get_input_embeddings()
        if hasattr(embed, "padding_idx") and embed.padding_idx is not None:
            embed.padding_idx = None
            if is_main():
                print(json.dumps({"cleared_padding_idx": True}))

        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
            use_orig_params=True,
        )
        if is_main():
            print(json.dumps({"fsdp": "wrapped", "world_size": world}))
    # FSDP handles multi-GPU; small modules stay on device if world==1.

    # --- optimizer ------------------------------------------------------------
    param_groups = [
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "lr": actor_lr,
            "name": "qwen",
        },
        {"params": state_proj.parameters(), "lr": pred_lr, "name": "state_proj"},
        {"params": value_head.parameters(), "lr": vh_lr, "name": "value_head"},
        {"params": wm_predictor.parameters(), "lr": pred_lr, "name": "wm_predictor"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # --- resume training state ------------------------------------------------
    start_iteration = 1
    global_step = 0
    best_value_loss = float("inf")
    if resume_aux_ckpt is not None:
        resume_state = load_rl_wm_checkpoint(
            resume_aux_ckpt, state_proj, wm_predictor, value_head, device
        )
        if resume_state:
            start_iteration = int(resume_state.get("iteration", 0)) + 1
            global_step = int(resume_state.get("global_step", 0))
            best_value_loss = float(resume_state.get("best_value_loss", float("inf")))
            if world > 1:
                saved_world = int(resume_state.get("optimizer_world_size", 0))
                if saved_world != world:
                    raise RuntimeError(
                        f"FSDP optimizer checkpoint world_size={saved_world}, current={world}"
                    )
                optimizer_path = resume_aux_ckpt / f"optimizer_rank_{rank:05d}.pt"
                if not optimizer_path.is_file():
                    raise FileNotFoundError(f"missing rank optimizer checkpoint: {optimizer_path}")
                optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location="cpu", weights_only=False)
                )
            elif resume_state.get("optimizer") is not None:
                optimizer.load_state_dict(resume_state["optimizer"])
            if is_main():
                print(json.dumps({"resume": True, "start_iteration": start_iteration,
                                  "global_step": global_step}))

    # --- logging --------------------------------------------------------------
    log_path = output_dir / "train_step_log.csv"
    if is_main() and not log_path.exists():
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "time", "iteration", "global_step",
                "wm_mse", "value_loss", "total_loss",
                "num_rollouts", "num_transitions", "success_rate",
                "val_success_rate", "val_avg_reward", "val_avg_steps",
                "actor_loss", "entropy", "clip_fraction", "mean_advantage",
            ])

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

        # 3. Train predictor + value head (1 step per iteration) ---------------
        # 使用 per-iteration 确定性 generator 保证所有 rank 选同一 batch，
        # 不依赖全局 RNG 状态同步（FSDP 等可能引入细微信号差异）。
        g = torch.Generator(device="cpu")
        g.manual_seed(seed + iteration)
        indices = torch.randperm(len(transitions), generator=g)[:batch_size]
        batch = [transitions[i] for i in indices]

        hidden_cur = torch.stack(
            [transition.qwen_hidden_current for transition in batch]
        ).to(device)
        hidden_next = torch.stack(
            [transition.qwen_hidden_next for transition in batch]
        ).to(device)
        actions = torch.tensor(
            [transition.action_index for transition in batch],
            dtype=torch.long,
            device=device,
        )
        value_targets = torch.tensor(
            [transition.value_target for transition in batch],
            dtype=torch.float32,
            device=device,
        )

        pred_loss, pred_metrics = compute_predictor_loss(
            qwen_hidden_current=hidden_cur,
            qwen_hidden_next=hidden_next,
            action_indices=actions,
            state_proj=state_proj,
            wm_predictor=wm_predictor,
        )

        sp = _unwrap(state_proj)
        wm_state = sp(hidden_cur).float().detach()
        val_loss, val_metrics = compute_value_loss(
            state_emb=wm_state,
            action_indices=actions,
            action_value_targets=value_targets,
            value_head=value_head,
            rank_margin=rank_margin,
            lambda_rank=lambda_rank,
        )

        # --- PPO actor loss (Qwen update) ---
        actor_metrics: dict[str, float] = {}
        if actor_enabled:
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            from nimloth.training.rl.loss import (
                compute_action_entropy_from_log_probs,
                compute_actor_loss,
            )

            # advantages from value head
            with torch.no_grad():
                all_values = value_head(wm_state).float()
                chosen_values = all_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            advantages = (value_targets.to(device=chosen_values.device, dtype=chosen_values.dtype)
                          - chosen_values.detach())
            # unbiased=False 避免 batch size=1 时 std 产生 NaN
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # 使用 rollout 保存的精确 prompt 和采样参数重放当前 policy。
            new_log_probs, action_log_probs = compute_current_policy_log_probs(
                batch,
                model,
                processor,
                token_id_map,
                device,
            )
            old_log_probs = torch.tensor(
                [transition.old_log_prob for transition in batch],
                device=new_log_probs.device, dtype=new_log_probs.dtype,
            )

            actor_loss, actor_metrics = compute_actor_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages.to(device=new_log_probs.device, dtype=new_log_probs.dtype),
                clip_ratio=clip_ratio,
            )
            entropy = compute_action_entropy_from_log_probs(action_log_probs)
            total_loss = pred_loss + val_loss + actor_loss - entropy_coeff * entropy
            actor_metrics["entropy"] = float(entropy.detach().item())
            actor_metrics["mean_advantage"] = float(advantages.mean().item())
        else:
            total_loss = pred_loss + val_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], 1.0,
        )
        optimizer.step()
        if vision_ema is not None:
            vision_ema.update(model)

        global_step += 1
        iter_metrics: dict[str, float] = {
            "wm_mse": float(pred_metrics.get("wm_mse", 0.0)),
            "value_loss": float(val_metrics.get("value_loss",
                                                val_metrics.get("value_total", 0.0))),
            "total_loss": float(total_loss.detach().item()),
            "num_rollouts": float(len(trajectories)),
            "num_transitions": float(len(transitions)),
            "success_rate": float(
                sum(1 for t in trajectories if t.success) / max(1, len(trajectories))
            ),
        }
        iter_metrics.update({k: v for k, v in actor_metrics.items() if k != "actor_loss"})
        iter_metrics["actor_loss"] = float(actor_metrics.get("actor_loss", 0.0))

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
        current_val = iter_metrics.get("value_loss", float("inf"))

        if is_main() and (iteration % log_interval == 0 or iteration == 1):
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    time.time(), iteration, global_step,
                    iter_metrics.get("wm_mse", ""),
                    iter_metrics.get("value_loss", ""),
                    iter_metrics.get("total_loss", ""),
                    iter_metrics.get("num_rollouts", ""),
                    iter_metrics.get("num_transitions", ""),
                    iter_metrics.get("success_rate", ""),
                    iter_metrics.get("val_success_rate", ""),
                    iter_metrics.get("val_avg_reward", ""),
                    iter_metrics.get("val_avg_steps", ""),
                    iter_metrics.get("actor_loss", ""),
                    iter_metrics.get("entropy", ""),
                    iter_metrics.get("clip_fraction", ""),
                    iter_metrics.get("mean_advantage", ""),
                ])
            elapsed = time.time() - iter_start
            print(json.dumps({
                "iteration": iteration,
                "global_step": global_step,
                "metrics": iter_metrics,
                "elapsed_s": round(elapsed, 1),
            }))
            if wandb_run is not None:
                wandb_run.log(
                    {
                        **{f"train/{key}": value for key, value in iter_metrics.items()},
                        "global_step": global_step,
                        "iteration": iteration,
                    },
                    step=global_step,
                )

        # --- checkpoint --------------------------------------------------------
        if iteration % save_interval == 0:
            save_rl_checkpoint(
                output_dir / f"iter_{iteration:04d}",
                state_proj=state_proj,
                wm_predictor=wm_predictor,
                value_head=value_head,
                model=model,
                processor=processor,
                vision_ema=vision_ema,
                optimizer=optimizer,
                iteration=iteration,
                global_step=global_step,
                best_value_loss=best_value_loss,
                lora=uses_lora(args),
                llm_tune=llm_tune,
                vision_tune=vision_tune,
                base_model_path=base_model_path,
            )
            if current_val < best_value_loss:
                best_value_loss = current_val
                save_rl_checkpoint(
                    resume_ckpt_dir,  # "best/"
                    state_proj=state_proj,
                    wm_predictor=wm_predictor,
                    value_head=value_head,
                    model=model,
                    processor=processor,
                    vision_ema=vision_ema,
                    optimizer=optimizer,
                    iteration=iteration,
                    global_step=global_step,
                    best_value_loss=best_value_loss,
                    lora=uses_lora(args),
                    llm_tune=llm_tune,
                    vision_tune=vision_tune,
                    base_model_path=base_model_path,
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # --- final checkpoint -----------------------------------------------------
    save_rl_checkpoint(
        output_dir / "final",
        state_proj=state_proj,
        wm_predictor=wm_predictor,
        value_head=value_head,
        model=model,
        processor=processor,
        vision_ema=vision_ema,
        optimizer=optimizer,
        iteration=iterations,
        global_step=global_step,
        best_value_loss=best_value_loss,
        lora=uses_lora(args),
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        base_model_path=base_model_path,
    )
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0
