"""YAML defaults for the CLI-driven SFT1 entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimloth.config import load_yaml_config

_SFT1_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("train", "success_eval_concurrency"): "success_eval_concurrency",
    ("train", "success_eval_env_url"): "success_eval_env_url",
    ("query_alignment", "grid_size"): "grid_size",
    ("query_alignment", "projector_hidden_dim"): "projector_hidden_dim",
    ("query_alignment", "weight_lm"): "weight_lm",
    ("query_alignment", "weight_dino"): "weight_dino",
    ("query_alignment", "dino_cache_root"): "dino_cache_root",
    ("data", "train_jsonl"): "train_jsonl",
    ("data", "val_jsonl"): "val_jsonl",
    ("latent", "token_count"): "latent_token_count",
    ("latent", "query_mode"): "latent_query_mode",
    ("latent", "mask_query_labels"): "mask_latent_query_labels",
    ("tuning", "lora"): "lora",
    ("tuning", "lora_r"): "lora_r",
    ("tuning", "lora_alpha"): "lora_alpha",
    ("tuning", "lora_dropout"): "lora_dropout",
    ("train", "distributed_strategy"): "distributed_strategy",
    ("train", "epochs"): "epochs",
    ("train", "until_converged"): "until_converged",
    ("train", "convergence_metric"): "convergence_metric",
    ("train", "convergence_format_min_rate"): "convergence_format_min_rate",
    ("train", "convergence_min_epochs"): "convergence_min_epochs",
    ("train", "convergence_patience_epochs"): "convergence_patience_epochs",
    ("train", "convergence_min_relative_improvement"): "convergence_min_relative_improvement",
    ("train", "resume_save_steps"): "resume_save_steps",
    ("train", "prune_intermediate_checkpoints"): "prune_intermediate_checkpoints",
    ("train", "min_pixels"): "min_pixels",
    ("train", "weight_decay"): "weight_decay",
    ("train", "warmup_ratio"): "warmup_ratio",
    ("train", "attn_implementation"): "attn_implementation",
    ("train", "gradient_checkpointing"): "gradient_checkpointing",
    ("train", "seed"): "seed",
    ("train", "cache_pixel_dtype"): "cache_pixel_dtype",
    ("train", "preprocess_workers"): "preprocess_workers",
    ("train", "num_workers"): "num_workers",
    ("train", "format_eval_temperature"): "format_eval_temperature",
    ("train", "format_eval_top_p"): "format_eval_top_p",
    ("train", "format_eval_max_new_tokens"): "format_eval_max_new_tokens",
    ("train", "format_eval_generation_seed"): "format_eval_generation_seed",
    ("train", "format_eval_samples"): "format_eval_samples",
    ("train", "format_eval_batch_size"): "format_eval_batch_size",
    ("train", "no_wandb"): "no_wandb",

    ("train", "batch_size"): "batch_size",
    ("train", "grad_accum"): "grad_accum",
    ("train", "action_token_loss_weight"): "action_token_loss_weight",
    ("train", "boundary_token_loss_weight"): "boundary_token_loss_weight",
    ("train", "lr"): "lr",
    ("train", "embedding_lr"): "embedding_lr",
    ("train", "embedding_master_dtype"): "embedding_master_dtype",
    ("train", "max_length"): "max_length",
    ("train", "max_pixels"): "max_pixels",
}


def sft1_yaml_defaults(path: Path, *, stage: str = "format") -> dict[str, Any]:
    cfg = load_yaml_config(path)
    if stage == "format" and any(key in cfg for key in ("latent", "query_alignment")):
        raise ValueError(
            "stage1 format supervision does not accept latent/query configuration"
        )
    defaults: dict[str, Any] = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            dest = _SFT1_YAML_TO_ARG.get((section, key))
            if dest is not None and value is not None:
                defaults[dest] = value
    return defaults
