"""SFT2 training utilities (Qwen latent ↔ frozen LeWM alignment)."""

from nimloth.sft2.collate import messages_with_image_paths, transition_collate_for_qwen
from nimloth.sft2.loss import StateProjector, compute_combined_loss, compute_end_to_end_step_loss, compute_wm_alignment_loss, wm_loss_weight_schedule
from nimloth.sft2.metrics import MetricAccumulator
from nimloth.sft2.schedules import qwen_lr_schedule, set_optimizer_group_lr

__all__ = [
    "MetricAccumulator",
    "StateProjector",
    "compute_combined_loss",
    "compute_end_to_end_step_loss",
    "compute_wm_alignment_loss",
    "messages_with_image_paths",
    "qwen_lr_schedule",
    "set_optimizer_group_lr",
    "transition_collate_for_qwen",
    "wm_loss_weight_schedule",
]
