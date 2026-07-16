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
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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
from nimloth.training.rl.rollout import (
    RolloutCollector,
    RolloutTrajectory,
    validate_rl_policy_protocol,
)
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

    # System message from trajectory
    system_msg = trajectory.messages[0] if trajectory.messages else {
        "role": "system", "content": "You are a navigation agent."
    }

    for i, image_path in enumerate(trajectory.image_paths):
        # Build messages: system + image observation + brief assistant
        # so Qwen encodes the conversation context including <|latent_state|>.
        messages = [
            system_msg,
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Observe the scene from the current viewpoint."},
            ]},
            # Include <|latent_state|> in assistant so we can extract it
            {"role": "assistant", "content": [
                {"type": "text", "text": f"<|latent_state|>"},
            ]},
        ]
        item = {"messages": messages}
        enc = build_qwen_batch([item], processor, max_length=999999)  # effectively no truncation
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
            # old_log_prob for the taken action at step t
            log_probs = traj.action_log_probs[t] if t < len(traj.action_log_probs) else []
            old_lp = float(log_probs[traj.action_indices[t]]) if len(log_probs) > traj.action_indices[t] else 0.0

            transitions.append({
                "qwen_hidden_current": hiddens[t],
                "qwen_hidden_next": hiddens[t + 1],
                "action_index": torch.tensor(traj.action_indices[t], dtype=torch.long),
                "value_target": torch.tensor(
                    value_targets[t] if t < len(value_targets) else 0.0,
                    dtype=torch.float32,
                ),
                "old_log_prob": old_lp,
                "nav_instruction": traj.nav_instruction,
                "action_history_names": traj.action_names[:t],
                "image_path": traj.image_paths[t],
                "image_history_paths": traj.image_paths[:t + 1],
            })

    return transitions


# ---------------------------------------------------------------------------
# PPO forward pass (Qwen with gradients)
# ---------------------------------------------------------------------------

def compute_new_log_probs_for_batch(
    ppo_items: list[dict],
    model,
    processor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    history_window: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Qwen forward WITH gradients, returning new log-probs and action logits.

    Each ppo_item must have:
        - "image_history_paths": observation paths through the current step
        - "nav_instruction": navigation instruction
        - "action_history_names": list of VAGEN text action names before this step
        - "taken_action_idx": the action that was actually taken

    Returns (new_log_probs, action_logits) where:
        new_log_probs: (B,) log-prob of taken actions under current policy
        action_logits: (B, 8) raw logits for all 8 actions
    """
    from nimloth.latent.extraction import LatentActionTokens
    from nimloth.training.rl.rollout import build_nimloth_policy_messages

    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    tokens = LatentActionTokens()
    action_token_ids = [token_id_map[token] for token in tokens.action_tokens]

    new_log_probs_list: list[torch.Tensor] = []
    policy_logits_list: list[torch.Tensor] = []
    for item in ppo_items:
        messages, images = build_nimloth_policy_messages(
            item["image_history_paths"],
            item["nav_instruction"],
            item["action_history_names"],
            history_window=history_window,
        )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(
            **model_inputs, output_hidden_states=False, return_dict=True
        )
        input_ids = encoded["input_ids"][0]
        positions = (
            input_ids == token_id_map[tokens.action_start]
        ).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            raise RuntimeError("<|action_start|> token not found in PPO prompt")
        position = int(positions[-1].item())
        action_ids = torch.tensor(action_token_ids, device=outputs.logits.device)
        raw_logits = outputs.logits[0, position, action_ids].float()
        policy_logits = raw_logits if temperature == 0 else raw_logits / temperature
        policy_logits_list.append(policy_logits)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        new_log_probs_list.append(log_probs[int(item["taken_action_idx"])])

    return torch.stack(new_log_probs_list), torch.stack(policy_logits_list)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


@contextmanager
def _temporary_eval(module: torch.nn.Module):
    """Run inference without permanently changing per-submodule train modes."""

    training_modes = [(submodule, submodule.training) for submodule in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for submodule, was_training in training_modes:
            submodule.training = was_training


def _maybe_init_wandb(
    *,
    rank: int,
    output_dir: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
):
    """Initialize one resumable W&B run after distributed rank is known."""

    if rank != 0:
        return None
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "disabled" and not os.environ.get("WANDB_API_KEY"):
        print(json.dumps({"wandb": "skipped", "reason": "WANDB_API_KEY not set"}))
        return None

    import wandb

    run_id_path = output_dir / "wandb_run_id.txt"
    requested_run_id = os.environ.get("WANDB_RUN_ID")
    if requested_run_id is None and run_id_path.is_file():
        requested_run_id = run_id_path.read_text(encoding="utf-8").strip() or None
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nimloth-rl"),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME") or args.experiment_name,
        id=requested_run_id,
        resume="allow" if requested_run_id is not None else None,
        mode=mode,
        config=config,
        dir=os.environ.get("WANDB_DIR"),
    )
    run_id_path.write_text(f"{run.id}\n", encoding="utf-8")
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    print(json.dumps({"wandb": "initialized", "run_id": run.id, "resume": requested_run_id is not None}))
    return run


def _require_optimizer_progress(global_step: int) -> None:
    """Refuse to materialize a misleading final checkpoint with zero updates."""

    if int(global_step) <= 0:
        raise RuntimeError(
            "RL completed zero optimizer steps; refusing to write final checkpoint"
        )


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

    pred_lr: float = float(pred_cfg.get("lr", 1e-3))
    vh_lr: float = float(vh_cfg.get("lr", 1e-3))
    rank_margin: float = vh_cfg.get("rank_margin", 0.1)
    lambda_rank: float = vh_cfg.get("lambda_rank", 1.0)

    # Actor (Qwen PPO) config
    actor_cfg: dict = config.get("actor", {})
    actor_enabled: bool = bool(actor_cfg) and not freeze_cfg.get("qwen", True)
    actor_lr: float = float(actor_cfg.get("lr", 1e-6))
    entropy_coeff: float = float(actor_cfg.get("entropy_coeff", 0.0))
    clip_ratio: float = float(actor_cfg.get("clip_ratio", 0.2))

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
    wandb_run = _maybe_init_wandb(
        rank=rank,
        output_dir=output_dir,
        args=args,
        config=config,
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
    validate_rl_policy_protocol(model.config)
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

    # --- freeze WM-encoding pathway if requested -----------------------------
    if freeze_qwen and llm_tune == "freeze" and vision_tune == "freeze":
        _freeze(model)
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

    # --- Wire online env rollout after FSDP wrapping -------------------------
    from nimloth.training.rl.rollout import EnvRolloutCollector
    if isinstance(collector, EnvRolloutCollector):
        if val_enabled:
            raise ValueError(
                "dynamic training rollout uses explicit *_train datasets; "
                "validation.enabled must be false until a separate heldout env collector is configured"
            )
        if world > 1:
            from nimloth.training.rl.distributed_rollout import (
                DistributedEnvRolloutCollector,
            )
            collector = DistributedEnvRolloutCollector.from_collector(collector)
        collector._model = model
        collector._processor = processor
        collector._device = device
        if is_main():
            print(json.dumps({
                "env_collector": "distributed_wired" if world > 1 else "wired",
                "device": str(device),
                "world_size": world,
                "history_window": collector._history_window,
                "eval_sets": collector._eval_sets,
            }))

    if isinstance(collector, EnvRolloutCollector):
        rollout_protocol: dict[str, Any] = {
            "mode": "dynamic_env",
            "split": collector._split,
            "eval_sets": list(collector._eval_sets),
            "history_window": int(collector._history_window),
            "temperature": float(rl_cfg.get("temperature", 1.0)),
            "top_p": float(rl_cfg.get("top_p", 1.0)),
            "seed_offset": int(collector._base_seed_offset),
            "env_timeout": int(collector._env_timeout),
            "control_backend": "gloo" if world > 1 else "local",
        }
    else:
        rollout_protocol = {
            "mode": "jsonl",
            "sources": [str(path) for path in getattr(collector, "_sources", [])],
        }
    ppo_temperature = (
        float(rl_cfg.get("temperature", 1.0))
        if rollout_protocol["mode"] == "dynamic_env"
        else 1.0
    )

    # --- optimizer ------------------------------------------------------------
    param_groups = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": actor_lr, "name": "qwen"},
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
            saved_rollout_protocol = resume_state.get("rollout_protocol")
            if saved_rollout_protocol is None:
                raise RuntimeError(
                    "RL checkpoint has no rollout_protocol metadata; refusing an unverifiable resume"
                )
            if saved_rollout_protocol != rollout_protocol:
                raise RuntimeError(
                    "rollout protocol mismatch on resume: "
                    f"saved={saved_rollout_protocol}, current={rollout_protocol}"
                )
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

    if isinstance(collector, EnvRolloutCollector):
        collector.set_resume_iteration(
            start_iteration=start_iteration,
            envs_per_iteration=envs_per_iter,
            validation_enabled=val_enabled,
            validation_interval=val_interval,
            validation_envs=val_envs,
        )
        if is_main():
            print(json.dumps({
                "env_seed_cursor": collector._ep_counter,
                "base_seed_offset": collector._base_seed_offset,
                "start_iteration": start_iteration,
            }))

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
        if isinstance(collector, EnvRolloutCollector):
            with _temporary_eval(model):
                trajectories = collector.collect(
                    num_episodes=envs_per_iter,
                    max_steps_per_episode=max_steps_per_ep,
                    output_dir=output_dir / f"rollouts/iter_{iteration:04d}",
                )
        else:
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
        with _temporary_eval(model):
            transitions = build_rl_transitions(
                trajectories, model, processor, token_id_map, device, gamma=gamma,
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

        # --- PPO actor loss (Qwen update) ---
        actor_metrics: dict[str, float] = {}
        if actor_enabled:
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            from nimloth.training.rl.loss import compute_actor_loss, compute_action_entropy

            # advantages from value head
            with torch.no_grad():
                all_values = value_head(wm_state).float()
                chosen_values = all_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            advantages = (value_targets.to(device=chosen_values.device, dtype=chosen_values.dtype)
                          - chosen_values.detach())
            # unbiased=False 避免 batch size=1 时 std 产生 NaN
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            # Build PPO items
            ppo_items = []
            for i in range(len(batch)):
                b = batch[i]
                ppo_items.append({
                    "image_history_paths": b["image_history_paths"],
                    "nav_instruction": b["nav_instruction"],
                    "action_history_names": b["action_history_names"],
                    "taken_action_idx": int(b["action_index"].item()),
                })

            # Qwen forward with gradients
            new_log_probs, action_logits = compute_new_log_probs_for_batch(
                ppo_items,
                model,
                processor,
                token_id_map,
                device,
                history_window=int(getattr(collector, "_history_window", 4)),
                temperature=ppo_temperature,
            )
            old_log_probs = torch.tensor(
                [b["old_log_prob"] for b in batch],
                device=new_log_probs.device, dtype=new_log_probs.dtype,
            )

            actor_loss, actor_metrics = compute_actor_loss(
                new_log_probs=new_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages.to(device=new_log_probs.device, dtype=new_log_probs.dtype),
                clip_ratio=clip_ratio,
            )
            entropy = compute_action_entropy(action_logits)
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
                rollout_protocol=rollout_protocol,
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
                    rollout_protocol=rollout_protocol,
                )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # --- final checkpoint -----------------------------------------------------
    try:
        _require_optimizer_progress(global_step)
    except RuntimeError:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1)
        cleanup_dist()
        raise
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
        rollout_protocol=rollout_protocol,
    )
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0
