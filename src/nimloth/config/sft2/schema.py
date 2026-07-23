"""SFT2 YAML 配置 schema 与 argparse 接入。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nimloth.config.io import load_yaml_config


_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("init", "sft1_checkpoint"): "model",
    ("init", "wm_predictor_checkpoint"): "wm_predictor_checkpoint",
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
    ("train", "backbone_rows_per_forward"): "backbone_rows_per_forward",
    ("train", "offload_backbone_chunk_activations"): "offload_backbone_chunk_activations",
    ("train", "batch_mode"): "batch_mode",
    ("train", "max_images_per_batch"): "max_images_per_batch",
    ("train", "max_steps_per_trajectory"): "max_steps_per_trajectory",
    ("train", "attn_implementation"): "attn_implementation",
    ("train", "gradient_checkpointing"): "gradient_checkpointing",
    ("train", "preprocess_cache_dir"): "preprocess_cache_dir",
    ("train", "preprocess_workers"): "preprocess_workers",
    ("train", "preprocess_cache_format"): "preprocess_cache_format",
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
    ("train", "checkpoint_interval_minutes"): "checkpoint_interval_minutes",
    ("train", "checkpoint_interval_steps"): "checkpoint_interval_steps",
    ("train", "checkpoint_keep_last"): "checkpoint_keep_last",
    ("latent", "token_count"): "latent_token_count",
    ("latent", "query_mode"): "latent_query_mode",
    ("latent", "query_tune"): "query_tune",
    ("latent", "query_lr"): "query_lr",
    ("latent", "mask_query_labels"): "mask_latent_query_labels",
    ("loss", "lambda_wm_start"): "lambda_wm_start",
    ("loss", "lambda_wm_end"): "lambda_wm_end",
    ("loss", "lambda_ce"): "lambda_ce",
    ("loss", "lambda_value"): "lambda_value",
    ("loss", "value_rank_margin"): "value_rank_margin",
    ("loss", "value_rank_lambda"): "value_rank_lambda",
    ("loss", "value_gamma"): "value_gamma",
    ("loss", "lambda_sigreg"): "lambda_sigreg",
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

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "SFT2LoopConfig":
        return cls(
            epochs=int(args.epochs),
            grad_accum=int(args.grad_accum),
            seed=int(args.seed),
            max_val_batches=int(args.max_val_batches),
            lambda_sigreg=float(args.lambda_sigreg),
            checkpoint_metric=str(args.checkpoint_metric),
            step_timing=bool(args.step_timing),
            step_timing_interval=int(args.step_timing_interval),
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
