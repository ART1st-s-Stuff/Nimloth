"""SFT2 YAML 配置 schema 与 argparse 接入。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nimloth.config.io import load_yaml_config

_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("objective", "name"): "objective",
    ("init", "sft1_checkpoint"): "model",
    ("init", "wm_predictor_checkpoint"): "wm_predictor_checkpoint",
    ("supervision", "dino_grid_cache"): "dino_grid_cache",
    ("supervision", "stage2_aligned_dino_cache"): "stage2_aligned_dino_cache",
    ("data", "train_jsonl"): "train_jsonl",
    ("data", "val_jsonl"): "val_jsonl",
    ("data", "include_failed_rollouts"): "include_failed_rollouts",
    ("data", "max_train_records"): "max_train_records",
    ("data", "max_val_records"): "max_val_records",
    ("tuning", "llm_tune"): "llm_tune",
    ("tuning", "vision_tune"): "vision_tune",
    ("tuning", "vision_ema"): "vision_ema",
    ("tuning", "vision_ema_decay"): "vision_ema_decay",
    ("tuning", "lora_r"): "lora_r",
    ("tuning", "lora_alpha"): "lora_alpha",
    ("tuning", "lora_dropout"): "lora_dropout",
    ("train", "epochs"): "epochs",
    ("train", "schedule_total_steps"): "schedule_total_steps",
    ("train", "early_stop_metric"): "early_stop_metric",
    ("train", "early_stop_relative_improvement"): "early_stop_relative_improvement",
    ("train", "early_stop_patience"): "early_stop_patience",
    ("train", "early_stop_baseline"): "early_stop_baseline",
    ("train", "distributed_strategy"): "distributed_strategy",
    ("train", "activation_offload"): "activation_offload",
    ("train", "stop_after_steps"): "stop_after_steps",
    ("train", "diagnostic_steps"): "diagnostic_steps",
    ("train", "diagnostic_dir"): "diagnostic_dir",
    ("train", "diagnose_outcome_gradients"): "diagnose_outcome_gradients",
    ("train", "batch_size"): "batch_size",
    ("train", "grad_accum"): "grad_accum",
    ("train", "lr_qwen_start"): "lr_qwen_start",
    ("train", "lr_qwen_peak"): "lr_qwen_peak",
    ("train", "qwen_lr_warmup_ratio"): "qwen_lr_warmup_ratio",
    ("train", "state_proj_lr"): "state_proj_lr",
    ("train", "wm_predictor_lr"): "wm_predictor_lr",
    ("train", "value_head_lr"): "value_head_lr",
    ("train", "weight_decay"): "weight_decay",
    ("train", "train_wm_predictor"): "train_wm_predictor",
    ("train", "max_length"): "max_length",
    ("train", "max_pixels"): "max_pixels",
    ("train", "emb_dim"): "emb_dim",
    ("train", "history_size"): "history_size",
    ("train", "prediction_horizon"): "prediction_horizon",
    ("grid", "size"): "grid_size",
    ("grid", "predictor_kind"): "grid_predictor_kind",
    ("grid", "global_tokens"): "grid_global_tokens",
    ("grid", "position_encoding"): "grid_position_encoding",
    ("grid", "wm_depth"): "grid_wm_depth",
    ("grid", "wm_heads"): "grid_wm_heads",
    ("grid", "wm_dim_head"): "grid_wm_dim_head",
    ("grid", "wm_mlp_dim"): "grid_wm_mlp_dim",
    ("grid", "wm_dropout"): "grid_wm_dropout",
    ("train", "batch_mode"): "batch_mode",
    ("train", "attn_implementation"): "attn_implementation",
    ("train", "gradient_checkpointing"): "gradient_checkpointing",
    ("train", "preprocess_cache_dir"): "preprocess_cache_dir",
    ("train", "preprocess_cache_processor_source"): "preprocess_cache_processor_source",
    ("train", "preprocess_cache_reuse_image_root"): "preprocess_cache_reuse_image_root",
    ("train", "preprocess_cache_reuse_processor_source"): "preprocess_cache_reuse_processor_source",
    ("train", "preprocess_workers"): "preprocess_workers",
    ("train", "preprocess_cache_image_dtype"): "preprocess_cache_image_dtype",
    ("train", "preprocess_cache_image_shard_size"): "preprocess_cache_image_shard_size",
    ("train", "preprocess_cache_transition_shard_size"): "preprocess_cache_transition_shard_size",
    ("train", "preprocess_cache_shard_lru"): "preprocess_cache_shard_lru",
    ("train", "require_prebuilt_cache"): "require_prebuilt_cache",
    ("train", "force_rebuild_cache"): "force_rebuild_cache",
    ("train", "dataloader_workers"): "dataloader_workers",
    ("train", "dataloader_prefetch_factor"): "dataloader_prefetch_factor",
    ("train", "step_timing"): "step_timing",
    ("train", "step_timing_interval"): "step_timing_interval",
    ("train", "step_timing_sample_interval"): "step_timing_sample_interval",
    ("train", "checkpoint_interval_minutes"): "checkpoint_interval_minutes",
    ("train", "checkpoint_interval_steps"): "checkpoint_interval_steps",
    ("train", "deduplicate_epoch_checkpoints"): "deduplicate_epoch_checkpoints",
    ("train", "checkpoint_latest_only"): "checkpoint_latest_only",
    ("train", "checkpoint_keep_last"): "checkpoint_keep_last",
    ("latent", "token_count"): "latent_token_count",
    ("latent", "query_mode"): "latent_query_mode",
    ("latent", "query_tune"): "query_tune",
    ("latent", "query_lr"): "query_lr",
    ("latent", "protocol_lr"): "protocol_lr",
    ("loss", "lambda_wm_start"): "lambda_wm_start",
    ("loss", "lambda_wm_end"): "lambda_wm_end",
    ("loss", "lambda_ce"): "lambda_ce",
    ("loss", "lambda_dino"): "lambda_dino",
    ("loss", "lambda_value"): "lambda_value",
    ("loss", "lambda_outcome"): "lambda_outcome",
    ("train", "outcome_head_lr"): "outcome_head_lr",
    ("train", "outcome_head"): "outcome_head",
    ("monitor", "outcome_eval_dir"): "outcome_eval_dir",
    ("loss", "value_gamma"): "value_gamma",
    ("loss", "lambda_sigreg"): "lambda_sigreg",
    ("loss", "wm_value_backbone_grad"): "wm_value_backbone_grad",
    ("loss", "sigreg_num_proj"): "sigreg_num_proj",
    ("loss", "sigreg_knots"): "sigreg_knots",
    ("monitor", "wandb"): "wandb_enabled",
    ("monitor", "wandb_run_name"): "wandb_run_name",
    ("monitor", "checkpoint_metric"): "checkpoint_metric",
}


@dataclass(frozen=True)
class SFT2LoopConfig:
    """训练 loop 实际消费的最小类型化配置。"""

    epochs: int
    grad_accum: int
    seed: int
    max_val_batches: int
    lambda_sigreg: float
    checkpoint_metric: str
    step_timing: bool
    step_timing_interval: int
    step_timing_sample_interval: int = 1
    stop_after_steps: int = 0
    diagnose_outcome_gradients: bool = False
    activation_offload: bool = False
    early_stop_metric: str | None = None
    early_stop_relative_improvement: float = .01
    early_stop_patience: int = 2
    early_stop_baseline: float | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> SFT2LoopConfig:
        return cls(
            epochs=int(args.epochs),
            early_stop_metric=getattr(args, "early_stop_metric", None),
            early_stop_relative_improvement=getattr(args, "early_stop_relative_improvement", .01),
            early_stop_patience=getattr(args, "early_stop_patience", 2),
            early_stop_baseline=getattr(args, "early_stop_baseline", None),
            activation_offload=bool(getattr(args, "activation_offload", False)),
            stop_after_steps=int(getattr(args, "stop_after_steps", 0)),
            diagnose_outcome_gradients=bool(getattr(args, "diagnose_outcome_gradients", False)),
            grad_accum=int(args.grad_accum),
            seed=int(args.seed),
            max_val_batches=int(args.max_val_batches),
            lambda_sigreg=float(args.lambda_sigreg),
            checkpoint_metric=str(args.checkpoint_metric),
            step_timing=bool(args.step_timing),
            step_timing_interval=int(args.step_timing_interval),
            step_timing_sample_interval=int(getattr(args, "step_timing_sample_interval", 1)),
        )


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "training" / "sft2" / "latent_wm_value.yaml"


def flatten_sft2_yaml_config(config: dict[str, Any]) -> dict[str, Any]:
    """把校验后的嵌套 SFT2 YAML 转成 argparse 默认值。"""

    flat: dict[str, Any] = {}
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ValueError(f"SFT2 config section {section!r} must be a mapping")
        for key, value in values.items():
            destination = _YAML_TO_ARG.get((section, key))
            if destination is None:
                raise ValueError(f"unknown SFT2 config field: {section}.{key}")
            flat[destination] = value

    if "activation_offload" in flat and not isinstance(flat["activation_offload"], bool):
        raise ValueError("train.activation_offload must be a boolean")
    if "wm_value_backbone_grad" in flat and not isinstance(flat["wm_value_backbone_grad"], bool):
        raise ValueError("loss.wm_value_backbone_grad must be a boolean")
    if "include_failed_rollouts" in flat:
        flat["success_only"] = not bool(flat.pop("include_failed_rollouts"))
    if "wandb_enabled" in flat:
        flat["no_wandb"] = not bool(flat.pop("wandb_enabled"))
    return flat


def apply_sft2_yaml_defaults(
    parser: argparse.ArgumentParser,
    config_path: Path | None,
) -> Path | None:
    path = config_path or default_config_path()
    if not path.is_file():
        return config_path
    parser.set_defaults(**flatten_sft2_yaml_config(load_yaml_config(path)))
    return path
