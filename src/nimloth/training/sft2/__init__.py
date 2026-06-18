"""SFT2 training utilities (Qwen latent ↔ WM predictor + value head)."""

from nimloth.training.common.metrics import MetricAccumulator
from nimloth.training.common.schedules import qwen_lr_schedule, set_optimizer_group_lr
from nimloth.training.sft2.loss import (
    StateProjector,
    compute_combined_loss,
    compute_value_loss,
    compute_wm_latent_loss,
    wm_loss_weight_schedule,
)
from nimloth.wm.collate import messages_with_image_paths, transition_collate_for_qwen
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.value_head import ValueHead

__all__ = [
    "LatentWMPredictor",
    "MetricAccumulator",
    "StateProjector",
    "ValueHead",
    "compute_combined_loss",
    "compute_value_loss",
    "compute_wm_latent_loss",
    "messages_with_image_paths",
    "qwen_lr_schedule",
    "set_optimizer_group_lr",
    "transition_collate_for_qwen",
    "wm_loss_weight_schedule",
]
