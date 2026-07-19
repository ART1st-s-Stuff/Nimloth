from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.training.sft2.cli import parse_sft2_args


ROOT = Path(__file__).resolve().parents[3]
K8_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8.yaml"
K8_DINOV2_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8_dinov2.yaml"
K8_DINOV2_CACHED_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8_dinov2_cached.yaml"
K8_DINOV3_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8_dinov3.yaml"
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
    assert args.early_stop_metric == "val_wm_mse"
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.max_length == 12000
    assert args.max_pixels == 100352
    assert args.max_images_per_batch == 12
    assert args.preprocess_cache_format == "compact"
    assert args.preprocess_cache_image_dtype == "bfloat16"
    assert args.preprocess_workers == 16


def test_dinov2_config_aligns_current_rgb_global_feature() -> None:
    args = parse_sft2_args(["--config", str(K8_DINOV2_CONFIG), *REQUIRED])

    assert args.latent_token_count == 8
    assert args.dino_align is True
    assert args.dino_model == "facebook/dinov2-large"
    assert args.dino_feature == "cls"
    assert args.lambda_dino == pytest.approx(1.0)


def test_cached_dinov2_config_fails_closed_without_sidecar_path() -> None:
    args = parse_sft2_args(["--config", str(K8_DINOV2_CACHED_CONFIG), *REQUIRED])

    assert args.dino_align is True
    assert args.dino_cache_dir is None
    assert args.require_dino_cache is True


def test_dino_cache_cli_is_explicit() -> None:
    args = parse_sft2_args(
        [
            "--config",
            str(K8_DINOV2_CONFIG),
            *REQUIRED,
            "--dino-cache-dir",
            "/tmp/dino-cache",
            "--require-dino-cache",
        ]
    )

    assert args.dino_cache_dir == Path("/tmp/dino-cache")
    assert args.require_dino_cache is True


def test_dinov3_config_remains_available() -> None:
    args = parse_sft2_args(["--config", str(K8_DINOV3_CONFIG), *REQUIRED])

    assert args.dino_align is True
    assert args.dino_model == "facebook/dinov3-vitl16-pretrain-lvd1689m"


def test_k1_control_only_changes_latent_capacity_not_runtime_budget() -> None:
    args = parse_sft2_args(["--config", str(K1_CONTROL_CONFIG), *REQUIRED])

    assert args.latent_token_count == 1
    assert args.latent_query_mode == "inject"
    assert args.query_tune == "adapter"
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.max_pixels == 100352
    assert args.max_images_per_batch == 12
    assert args.early_stop_metric == "val_wm_mse"


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
