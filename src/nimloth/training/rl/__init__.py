"""Online RL training with WM predictor + value head."""

from nimloth.training.rl.checkpoint import (
    load_lora_adapter_state,
    load_rl_checkpoint,
    load_rl_wm_checkpoint,
    save_rl_checkpoint,
    validate_rl_checkpoint_metadata,
)
from nimloth.training.rl.loss import (
    compute_actor_loss,
    compute_action_entropy,
    compute_advantages,
    compute_multistep_predictor_loss,
    compute_predictor_loss,
    compute_value_loss,
)
from nimloth.training.rl.rollout import (
    EnvRolloutCollector,
    JSONLRolloutCollector,
    RLPolicyProtocol,
    RolloutTrajectory,
    VAGENRolloutCollector,
    WMValueFastPathController,
    load_trajectories,
    save_trajectories,
)
from nimloth.training.rl.trainer import (
    build_rl_transitions,
    encode_trajectory_hiddens,
    train_rl,
)

__all__ = [
    "build_rl_transitions",
    "compute_actor_loss",
    "compute_action_entropy",
    "compute_advantages",
    "compute_multistep_predictor_loss",
    "compute_predictor_loss",
    "compute_value_loss",
    "encode_trajectory_hiddens",
    "EnvRolloutCollector",
    "JSONLRolloutCollector",
    "load_lora_adapter_state",
    "load_rl_checkpoint",
    "load_rl_wm_checkpoint",
    "load_trajectories",
    "RLPolicyProtocol",
    "RolloutTrajectory",
    "save_rl_checkpoint",
    "save_trajectories",
    "train_rl",
    "validate_rl_checkpoint_metadata",
    "VAGENRolloutCollector",
    "WMValueFastPathController",
]
