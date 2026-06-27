"""Online RL training loop: rollout → encode → train → repeat.

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

from nimloth.backbone.qwen_tuning import (
    configure_qwen_tuning,
    resolve_tune_modes,
    uses_lora,
)
from nimloth.backbone.vision_ema import VisionEncoderEMA, resolve_vision_ema
from nimloth.latent import add_special_tokens, special_token_ids
from nimloth.training.common.dist import cleanup_dist, is_main, setup_dist
from nimloth.training.rl.checkpoint import (
    load_lora_adapter_state,
    load_rl_wm_checkpoint,
    save_rl_checkpoint,
)
from nimloth.training.rl.loss import compute_predictor_loss, compute_value_loss
from nimloth.training.rl.rollout import RolloutCollector, RolloutTrajectory
from nimloth.wm.dataset import discounted_action_value_targets
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


# ---------------------------------------------------------------------------
# Latent encoding (Qwen → hidden states)
# ---------------------------------------------------------------------------


def encode_trajectory_hiddens(
    trajectory: RolloutTrajectory,
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
) -> list[torch.Tensor]:
    """Run Qwen on each frame of a trajectory, return hidden states.

    Returns:
        List of ``(hidden_dim,)`` tensors, one per frame (len = num_steps + 1).
    """
    from nimloth.latent.extraction import (
        LatentActionTokens,
        extract_latent_state,
        find_last_latent_state_index,
        last_hidden_state,
    )
    from nimloth.training.common.qwen_batch import build_qwen_batch

    states: list[torch.Tensor] = []
    tokens = LatentActionTokens()

    for image_path in trajectory.image_paths:
        item = {
            "prefix_messages": trajectory.messages[:1],  # system prompt only
            "prefix_image_paths": [],
            "current_image_path": image_path,
            "next_image_path": image_path,
            "action_index": 0,
            "action_value_target": 0.0,
            "success": trajectory.success,
            "split": trajectory.split,
        }
        enc = build_qwen_batch([item], processor, max_length=2048)
        model_inputs = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            output = qwen_model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = last_hidden_state(output)
        latent_idx = find_last_latent_state_index(
            enc["input_ids"][0], token_id_map, tokens
        )
        latent = extract_latent_state(hidden[0:1], latent_idx)  # (1, hidden_dim)
        states.append(latent.squeeze(0).detach().cpu())

    return states


# ---------------------------------------------------------------------------
# Transition builder
# ---------------------------------------------------------------------------


def build_rl_transitions(
    trajectories: list[RolloutTrajectory],
    qwen_model: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    gamma: float = 0.99,
) -> list[dict[str, torch.Tensor]]:
    """Encode trajectories → list of transition dicts (CPU tensors)."""

    transitions: list[dict[str, torch.Tensor]] = []
    for traj in trajectories:
        hiddens = encode_trajectory_hiddens(
            traj, qwen_model, processor, token_id_map, device
        )
        if len(hiddens) < 2:
            continue

        record = traj.to_record()
        value_targets = discounted_action_value_targets(record, gamma=gamma)

        for t in range(traj.num_steps):
            transitions.append({
                "qwen_hidden_current": hiddens[t],
                "qwen_hidden_next": hiddens[t + 1],
                "action_index": torch.tensor(traj.action_indices[t], dtype=torch.long),
                "value_target": torch.tensor(
                    value_targets[t] if t < len(value_targets) else 0.0,
                    dtype=torch.float32,
                ),
            })

    return transitions


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def train_rl(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    collector: RolloutCollector,
    output_dir: Path,
) -> int:
    """Run the online RL training loop."""

    # --- unpack config -------------------------------------------------------
    rl_cfg: dict = config.get("rl", {})
    freeze_cfg: dict = config.get("freeze", {})
    pred_cfg: dict = config.get("predictor", {})
    vh_cfg: dict = config.get("value_head", {})
    val_cfg: dict = config.get("validation", {})
    train_cfg: dict = config.get("training", {})

    iterations: int = rl_cfg.get("iterations", 1000)
    envs_per_iter: int = rl_cfg.get("envs_per_iteration", 8)
    max_steps_per_ep: int = rl_cfg.get("max_steps_per_episode", 20)
    gamma: float = rl_cfg.get("gamma", 0.99)
    batch_size: int = rl_cfg.get("batch_size", 32)
    # One optimizer step per iteration → 1 iteration = 1 global_step.

    pred_lr: float = pred_cfg.get("lr", 1e-3)
    vh_lr: float = vh_cfg.get("lr", 1e-3)
    rank_margin: float = vh_cfg.get("rank_margin", 0.1)
    lambda_rank: float = vh_cfg.get("lambda_rank", 1.0)

    # Config-controlled freeze is advisory — actual tuning is via --llm-tune / --vision-tune
    freeze_qwen: bool = freeze_cfg.get("qwen", True)
    freeze_state_proj: bool = freeze_cfg.get("state_proj", True)

    log_interval: int = train_cfg.get("log_interval", 10)
    save_interval: int = train_cfg.get("save_interval", 50)
    val_enabled: bool = val_cfg.get("enabled", True)
    val_interval: int = val_cfg.get("interval", 50)
    val_envs: int = val_cfg.get("envs", 16)
    seed: int = train_cfg.get("seed", 42)

    # --- tuning modes --------------------------------------------------------
    llm_tune, vision_tune = resolve_tune_modes(args)
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)

    # --- distributed setup ---------------------------------------------------
    rank, world, local_rank, device = setup_dist()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    # --- Qwen model loading --------------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.image_processor.min_pixels = 3136
    processor.image_processor.max_pixels = args.max_pixels
    add_special_tokens(processor.tokenizer)
    token_id_map = special_token_ids(processor.tokenizer)

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
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.resize_token_embeddings(len(processor.tokenizer))

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

    # --- Wire up EnvRolloutCollector with loaded model -----------------------
    from nimloth.training.rl.rollout import EnvRolloutCollector
    if isinstance(collector, EnvRolloutCollector):
        collector._model = model
        collector._processor = processor
        collector._device = device
        if is_main():
            print(json.dumps({"env_collector": "wired", "device": str(device)}))

    # --- freeze WM-encoding pathway if requested -----------------------------
    if freeze_qwen and llm_tune == "freeze" and vision_tune == "freeze":
        _freeze(model)
    if freeze_state_proj:
        _freeze(state_proj)

    state_proj.to(device)
    wm_predictor.to(device)
    value_head.to(device)

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

    # --- DDP wrap -------------------------------------------------------------
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                     find_unused_parameters=False)
        if uses_lora(args):
            model._set_static_graph()
        state_proj = DDP(state_proj, device_ids=[local_rank], output_device=local_rank,
                         find_unused_parameters=False)
        wm_predictor = DDP(wm_predictor, device_ids=[local_rank], output_device=local_rank,
                           find_unused_parameters=False)
        value_head = DDP(value_head, device_ids=[local_rank], output_device=local_rank,
                         find_unused_parameters=False)

    # --- optimizer ------------------------------------------------------------
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-8, "name": "qwen"},
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
            if resume_state.get("optimizer") is not None:
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
            ])

    # --- main loop ------------------------------------------------------------
    for iteration in range(start_iteration, iterations + 1):
        iter_start = time.time()

        # 1. Collect trajectories -------------------------------------------------
        if is_main():
            print(json.dumps({"iteration": iteration, "phase": "rollout",
                              "num_episodes": envs_per_iter}))
        trajectories = collector.collect(
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
        transitions = build_rl_transitions(
            trajectories, model, processor, token_id_map, device, gamma=gamma,
        )
        if len(transitions) < batch_size:
            if is_main():
                print(json.dumps({
                    "iteration": iteration,
                    "warning": f"only {len(transitions)} transitions, need {batch_size}",
                }))
            continue

        # 3. Train predictor + value head (1 step per iteration) ---------------
        indices = torch.randperm(len(transitions))[:batch_size]
        batch = [transitions[i] for i in indices]

        hidden_cur = torch.stack([b["qwen_hidden_current"] for b in batch]).to(device)
        hidden_next = torch.stack([b["qwen_hidden_next"] for b in batch]).to(device)
        actions = torch.stack([b["action_index"] for b in batch]).to(device)
        value_targets = torch.stack([b["value_target"] for b in batch]).to(device)

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

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # --- validation rollout -------------------------------------------------
        if val_enabled and iteration % val_interval == 0:
            val_trajectories = collector.collect(
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
                ])
            elapsed = time.time() - iter_start
            print(json.dumps({
                "iteration": iteration,
                "global_step": global_step,
                "metrics": iter_metrics,
                "elapsed_s": round(elapsed, 1),
            }))

        # --- checkpoint --------------------------------------------------------
        if iteration % save_interval == 0:
            if is_main():
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
    if is_main():
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
    cleanup_dist()
    return 0
