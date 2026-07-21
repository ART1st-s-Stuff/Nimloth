"""YAML defaults for the CLI-driven SFT1 entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nimloth.config import load_yaml_config

_SFT1_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("data", "train_jsonl"): "train_jsonl",
    ("data", "val_jsonl"): "val_jsonl",
    ("latent", "token_count"): "latent_token_count",
    ("latent", "query_mode"): "latent_query_mode",
    ("latent", "mask_query_labels"): "mask_latent_query_labels",
    ("tuning", "lora"): "lora",
    ("tuning", "lora_r"): "lora_r",
    ("tuning", "lora_alpha"): "lora_alpha",
    ("train", "epochs"): "epochs",
    ("train", "batch_size"): "batch_size",
    ("train", "grad_accum"): "grad_accum",
    ("train", "lr"): "lr",
    ("train", "embedding_lr"): "embedding_lr",
    ("train", "max_length"): "max_length",
    ("train", "max_pixels"): "max_pixels",
}


def sft1_yaml_defaults(path: Path) -> dict[str, Any]:
    cfg = load_yaml_config(path)
    defaults: dict[str, Any] = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            dest = _SFT1_YAML_TO_ARG.get((section, key))
            if dest is not None and value is not None:
                defaults[dest] = value
    return defaults
