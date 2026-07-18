"""Configuration and audit helpers for the exact-transcript VERL worker gate."""

from __future__ import annotations

from pathlib import Path


def install_verl_zero_warmup_scheduler_patch() -> None:
    """Make a zero-warmup pinned-VERL scheduler start at configured LR."""
    from torch.optim.lr_scheduler import LambdaLR
    from verl.utils import torch_functional

    current = torch_functional.get_constant_schedule_with_warmup
    if getattr(current, "_nimloth_zero_warmup_fixed", False):
        return
    original = current

    def fixed_schedule(optimizer, num_warmup_steps: int, last_epoch: int = -1):
        if num_warmup_steps == 0:
            return LambdaLR(optimizer, lambda _step: 1.0, last_epoch)
        return original(optimizer, num_warmup_steps, last_epoch)

    fixed_schedule._nimloth_zero_warmup_fixed = True
    torch_functional.get_constant_schedule_with_warmup = fixed_schedule


def configure_nimloth_wm_auxiliary(
    config,
    *,
    latent_token_count: int,
    checkpoint_dir: str | Path | None,
    loss_coef: float = 1.0,
    learning_rate: float = 1e-5,
    allow_random_init: bool = False,
) -> None:
    """Enable actor-side StateProjector + predictor loss and checkpointing."""
    latent_token_count = int(latent_token_count)
    if latent_token_count < 1:
        raise ValueError("WM auxiliary latent_token_count must be >= 1")
    if float(loss_coef) <= 0 or float(learning_rate) <= 0:
        raise ValueError("WM auxiliary loss coefficient and LR must be positive")
    if checkpoint_dir is None and not allow_random_init:
        raise ValueError("WM auxiliary requires an SFT2 checkpoint")
    wm_config = config.actor_rollout_ref.actor.nimloth_wm_aux
    wm_config.enabled = True
    wm_config.checkpoint_dir = (
        str(Path(checkpoint_dir).expanduser().resolve())
        if checkpoint_dir is not None
        else None
    )
    wm_config.allow_random_init = bool(allow_random_init)
    wm_config.latent_token_count = latent_token_count
    wm_config.loss_coef = float(loss_coef)
    wm_config.lr = float(learning_rate)


def configure_nimloth_online_rollout(
    config,
    *,
    latent_token_count: int,
    max_think_tokens: int = 512,
    use_service: bool = True,
) -> None:
    """Enable the strict staged inject manager on an existing VAGEN config."""
    latent_token_count = int(latent_token_count)
    max_think_tokens = int(max_think_tokens)
    if latent_token_count < 1:
        raise ValueError("online rollout latent_token_count must be >= 1")
    if max_think_tokens < 1:
        raise ValueError("online rollout max_think_tokens must be >= 1")
    manager = config.rollout_manager
    manager.use_service = bool(use_service)
    manager.latent_token_count = latent_token_count
    manager.latent_query_mode = "inject"
    manager.max_think_tokens = max_think_tokens
    manager.temperature = float(config.actor_rollout_ref.rollout.temperature)
    if use_service:
        manager.manager_class = None
        manager.service_manager_class = (
            "nimloth.training.rl.vagen_online_rollout."
            "NimlothQwenVLRolloutManagerService"
        )
    else:
        manager.service_manager_class = None
        manager.manager_class = (
            "nimloth.training.rl.vagen_online_rollout."
            "NimlothQwenVLRolloutManager"
        )


def build_exact_replay_worker_config(
    base_config_path: str | Path,
    *,
    model_path: str | Path,
    world_size: int,
    max_token_length: int = 8192,
):
    """Compose a fail-closed full actor/ref/critic config for one PPO update."""

    from omegaconf import OmegaConf, open_dict

    world_size = int(world_size)
    max_token_length = int(max_token_length)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if max_token_length <= 0:
        raise ValueError(
            f"max_token_length must be positive, got {max_token_length}"
        )
    model = str(Path(model_path).expanduser().resolve())
    config = OmegaConf.load(Path(base_config_path))
    OmegaConf.set_struct(config, False)
    with open_dict(config):
        config.algorithm.adv_estimator = "masked_gae"
        config.algorithm.gamma = 1.0
        config.algorithm.lam = 1.0
        config.algorithm.kl_ctrl.kl_coef = 0.0

        config.data.max_prompt_length = 1
        config.data.max_response_length = max_token_length
        config.data.max_trajectory_length = max_token_length
        config.data.train_batch_size = world_size

        config.actor_rollout_ref.model.path = model
        config.actor_rollout_ref.model.external_lib = (
            "nimloth.training.rl.verl_runtime_patch"
        )
        config.actor_rollout_ref.model.enable_gradient_checkpointing = True
        config.actor_rollout_ref.model.use_remove_padding = False
        config.actor_rollout_ref.model.override_config = {
            "use_cache": False,
            "tie_word_embeddings": True,
        }
        config.actor_rollout_ref.actor.strategy = "fsdp"
        config.actor_rollout_ref.actor.ppo_mini_batch_size = world_size
        config.actor_rollout_ref.actor.ppo_micro_batch_size = None
        config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu = max_token_length
        config.actor_rollout_ref.actor.ppo_epochs = 1
        config.actor_rollout_ref.actor.shuffle = False
        config.actor_rollout_ref.actor.entropy_coeff = 0.0
        config.actor_rollout_ref.actor.use_kl_loss = True
        config.actor_rollout_ref.actor.kl_loss_coef = 0.001
        config.actor_rollout_ref.actor.kl_loss_type = "low_var_kl"
        config.actor_rollout_ref.actor.grad_norm_threshold = None
        config.actor_rollout_ref.actor.optim.total_training_steps = 1
        config.actor_rollout_ref.actor.fsdp_config.param_offload = True
        config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
        config.actor_rollout_ref.ref.log_prob_micro_batch_size = None
        config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu = max_token_length
        config.actor_rollout_ref.ref.fsdp_config.param_offload = True
        config.actor_rollout_ref.rollout.name = "hf"
        config.actor_rollout_ref.rollout.n = 1
        config.actor_rollout_ref.rollout.temperature = 1.0
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size = None
        config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
        config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu = (
            max_token_length
        )
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

        config.critic.strategy = "fsdp"
        config.critic.model.path = model
        config.critic.model.tokenizer_path = model
        config.critic.model.external_lib = "nimloth.training.rl.verl_runtime_patch"
        config.critic.model.enable_gradient_checkpointing = True
        config.critic.model.use_remove_padding = False
        config.critic.model.override_config = {"use_cache": False}
        config.critic.model.fsdp_config.param_offload = True
        config.critic.model.fsdp_config.optimizer_offload = True
        # Keep optimizer/master parameters in fp32. FSDP mixed precision still
        # runs forward/backward in bf16; storing trainable params as bf16 makes
        # lr=1e-5 AdamW updates quantize away.
        config.critic.model.fsdp_config.model_dtype = "fp32"
        config.critic.ppo_mini_batch_size = world_size
        config.critic.ppo_micro_batch_size = None
        config.critic.ppo_micro_batch_size_per_gpu = 1
        config.critic.forward_micro_batch_size = None
        config.critic.forward_micro_batch_size_per_gpu = 1
        config.critic.ppo_max_token_len_per_gpu = max_token_length
        config.critic.forward_max_token_len_per_gpu = max_token_length
        config.critic.ppo_epochs = 1
        config.critic.shuffle = False
        config.critic.optim.total_training_steps = 1

        config.rollout_manager.use_loss_mask = True
        config.rollout_manager.use_gae_mask = True
        config.rollout_manager.use_multi_turn_reward = True
        config.trainer.n_gpus_per_node = world_size
        config.trainer.nnodes = 1

    OmegaConf.resolve(config)
    serialized = OmegaConf.to_yaml(config).lower()
    for forbidden in ("lora", "peft", "freeze_vision", "freeze_language"):
        if forbidden in serialized:
            raise ValueError(
                f"formal VERL exact replay config contains forbidden tuning key {forbidden}"
            )
    return config
