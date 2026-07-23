from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.config.sft2 import flatten_sft2_yaml_config


ROOT = Path(__file__).resolve().parents[3]
K8_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8.yaml"
K1_CONTROL_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k1_control.yaml"
REQUIRED = [
    "--model",
    "/tmp/model",
    "--train-jsonl",
    "/tmp/train.jsonl",
    "--val-jsonl",
    "/tmp/val.jsonl",
    "--output-dir",
    "/tmp/out",
]


def test_yaml_defaults_apply_after_argument_registration() -> None:
    args = parse_sft2_args(["--config", str(K8_CONFIG), *REQUIRED])

    assert args.config == K8_CONFIG
    assert args.latent_token_count == 8
    assert args.latent_query_mode == "inject"
    assert args.mask_latent_query_labels is True
    assert args.query_tune == "adapter"
    assert args.query_lr == pytest.approx(5e-5)
    assert args.checkpoint_metric == "val_wm_mse"
    assert args.batch_mode == "trajectory_online_cache"
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.max_length == 12000
    assert args.max_pixels == 100352
    assert args.history_size == 4
    assert not hasattr(args, "backbone_rows_per_forward")
    assert not hasattr(args, "offload_backbone_chunk_activations")
    assert args.preprocess_cache_format == "compact"
    assert args.preprocess_cache_image_dtype == "bfloat16"
    assert args.preprocess_workers == 16


def test_k1_control_only_changes_latent_capacity_not_runtime_budget() -> None:
    args = parse_sft2_args(["--config", str(K1_CONTROL_CONFIG), *REQUIRED])

    assert args.latent_token_count == 1
    assert args.latent_query_mode == "inject"
    assert args.query_tune == "adapter"
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.max_pixels == 100352
    assert args.history_size == 4
    assert args.batch_mode == "trajectory_online_cache"
    assert not hasattr(args, "backbone_rows_per_forward")
    assert not hasattr(args, "offload_backbone_chunk_activations")
    assert args.checkpoint_metric == "val_wm_mse"


def test_cli_values_override_yaml_defaults() -> None:
    args = parse_sft2_args(
        [
            "--config",
            str(K8_CONFIG),
            *REQUIRED,
            "--latent-token-count",
            "3",
            "--latent-query-mode",
            "generate",
            "--epochs",
            "2",
            "--preprocess-workers",
            "1",
        ]
    )

    assert args.latent_token_count == 3
    assert args.latent_query_mode == "generate"
    assert args.mask_latent_query_labels is False
    assert args.epochs == 2
    assert args.preprocess_workers == 1


def test_cli_rejects_conflicting_mode_and_legacy_mask() -> None:
    with pytest.raises(ValueError, match="conflicting latent query settings"):
        parse_sft2_args(
            [
                "--config",
                str(K8_CONFIG),
                *REQUIRED,
                "--latent-query-mode",
                "inject",
                "--no-mask-latent-query-labels",
            ]
        )


def test_sft2_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown SFT2 config field: train.typo"):
        flatten_sft2_yaml_config({"train": {"typo": True}})


def test_sft2_config_rejects_removed_oom_emergency_fields() -> None:
    with pytest.raises(
        ValueError,
        match="unknown SFT2 config field: train.offload_backbone_chunk_activations",
    ):
        flatten_sft2_yaml_config(
            {"train": {"offload_backbone_chunk_activations": True}}
        )
