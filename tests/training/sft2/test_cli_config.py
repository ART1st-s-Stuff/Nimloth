from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.training.sft2.cli import parse_sft2_args


ROOT = Path(__file__).resolve().parents[3]
K8_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8.yaml"
K8_STATE8192_CONFIG = (
    ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8_state8192.yaml"
)
K8_STATE8192_FACTORIZED_CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "sft2"
    / "latent_wm_value_k8_state8192_factorized.yaml"
)
K1_CONTROL_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k1_control.yaml"
T4_TOKENIZED_CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "sft2"
    / "latent_wm_value_k8_state8192_t4_tok8_residual_ep5_img8.yaml"
)
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
    assert args.wm_history_size == 1
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


def test_state8192_config_removes_projector_hidden_bottleneck() -> None:
    args = parse_sft2_args(["--config", str(K8_STATE8192_CONFIG), *REQUIRED])

    assert args.latent_token_count == 8
    assert args.emb_dim == 8192
    assert args.projector_hidden_dim == 8192
    assert args.value_hidden_dim == 1024
    assert args.epochs == 2
    assert args.max_pixels == 100352


def test_state8192_factorized_config_keeps_wide_state_and_narrow_dynamics() -> None:
    args = parse_sft2_args(
        ["--config", str(K8_STATE8192_FACTORIZED_CONFIG), *REQUIRED]
    )

    assert args.emb_dim == 8192
    assert args.projector_hidden_dim == 8192
    assert args.wm_dynamics_dim == 2048
    assert args.value_hidden_dim == 1024
    assert args.epochs == 2


def test_t4_tokenized_residual_config_has_exact_history_contract() -> None:
    args = parse_sft2_args(["--config", str(T4_TOKENIZED_CONFIG), *REQUIRED])
    assert args.emb_dim == 8192
    assert args.wm_history_size == 4
    assert args.wm_state_token_count == 8
    assert args.wm_residual_prediction is True
    assert args.wm_predictor_hidden_dim == 1024
    assert args.wm_predictor_heads * args.wm_predictor_dim_head == 1024
    assert args.full_trajectory_batching is True


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
