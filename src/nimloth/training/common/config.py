"""Load YAML training configs into argparse namespaces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# (yaml_section, yaml_key) -> argparse dest (snake_case)
_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("init", "wm_predictor_checkpoint"): "wm_predictor_checkpoint",
    ("data", "train_jsonl"): "train_jsonl",
    ("data", "val_jsonl"): "val_jsonl",
    ("data", "include_failed_rollouts"): "include_failed_rollouts",
    ("tuning", "llm_tune"): "llm_tune",
    ("tuning", "vision_tune"): "vision_tune",
    ("tuning", "vision_ema"): "vision_ema",
    ("tuning", "vision_ema_decay"): "vision_ema_decay",
    ("tuning", "lora_r"): "lora_r",
    ("tuning", "lora_alpha"): "lora_alpha",
    ("train", "epochs"): "epochs",
    ("train", "batch_size"): "batch_size",
    ("train", "grad_accum"): "grad_accum",
    ("train", "lr_qwen_start"): "lr_qwen_start",
    ("train", "lr_qwen_peak"): "lr_qwen_peak",
    ("train", "state_proj_lr"): "state_proj_lr",
    ("train", "wm_predictor_lr"): "wm_predictor_lr",
    ("train", "value_head_lr"): "value_head_lr",
    ("train", "train_wm_predictor"): "train_wm_predictor",
    ("train", "max_length"): "max_length",
    ("train", "max_pixels"): "max_pixels",
    ("train", "emb_dim"): "emb_dim",
    ("train", "full_trajectory_batching"): "full_trajectory_batching",
    ("train", "max_steps_per_trajectory"): "max_steps_per_trajectory",
    ("train", "attn_implementation"): "attn_implementation",
    ("train", "gradient_checkpointing"): "gradient_checkpointing",
    ("train", "preprocess_cache_dir"): "preprocess_cache_dir",
    ("train", "preprocess_workers"): "preprocess_workers",
    ("train", "dataloader_workers"): "dataloader_workers",
    ("train", "step_timing"): "step_timing",
    ("train", "step_timing_interval"): "step_timing_interval",
    ("train", "checkpoint_interval_minutes"): "checkpoint_interval_minutes",
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
    ("monitor", "early_stop_metric"): "early_stop_metric",
}


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for --config") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def flatten_yaml_config(cfg: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            arg_name = _YAML_TO_ARG.get((section, key))
            if arg_name is None:
                continue
            flat[arg_name] = value
    if "include_failed_rollouts" in flat:
        flat["success_only"] = not bool(flat.pop("include_failed_rollouts"))
    if "wandb_enabled" in flat:
        flat["no_wandb"] = not bool(flat.pop("wandb_enabled"))
    return flat


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[4] / "configs" / "training" / "sft2" / "latent_wm_value.yaml"


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: Path | None) -> Path | None:
    path = config_path or default_config_path()
    if not path.is_file():
        return config_path
    flat = flatten_yaml_config(load_yaml_config(path))
    parser.set_defaults(**flat)
    return path


def merge_cli_over_yaml(args: argparse.Namespace, config_path: Path | None) -> None:
    """No-op placeholder: argparse already applies CLI over defaults."""

    _ = (args, config_path)
