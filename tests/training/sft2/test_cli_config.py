from __future__ import annotations

from pathlib import Path

import pytest

from nimloth.config.io import load_yaml_config
from nimloth.config.sft2 import flatten_sft2_yaml_config
from nimloth.training.sft.stage3.cli import parse_sft2_args

ROOT = Path(__file__).resolve().parents[3]
K8_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8.yaml"
K1_CONTROL_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k1_control.yaml"
DINO_GRID_CONFIG = (
    ROOT / "configs" / "training" / "sft2" / "dino_grid_k16_h4.yaml"
)
DINO_GRID_H1_T4_CONFIG = (
    ROOT / "configs" / "training" / "sft2" / "dino_grid_k16_h1_t4.yaml"
)
CLS_FIXED2D_CONFIG = (
    ROOT
    / "configs"
    / "training"
    / "sft2"
    / "action_outcome_k64_cls_fixed2d_h1_t4_eval.yaml"
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


def test_backbone_gradient_boundary_is_explicit_opt_in():
    args = parse_sft2_args([*REQUIRED, "--history-size", "1"])
    assert args.wm_value_backbone_grad is True
    args = parse_sft2_args([*REQUIRED, "--history-size", "1", "--no-wm-value-backbone-grad"])
    assert args.wm_value_backbone_grad is False
    assert args.preprocess_cache_reuse_image_root is None
    assert args.require_prebuilt_cache is True


def test_stage3_image_cache_reuse_cli_is_explicit_and_nonoverlapping(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    processor_source = tmp_path / "epoch16"
    processor_source.mkdir()
    for split in ("train", "val"):
        split_dir = source / split
        split_dir.mkdir(parents=True)
        (split_dir / "manifest.json").write_text("{}", encoding="utf-8")
    common = [
        *REQUIRED,
        "--preprocess-cache-dir",
        str(destination),
        "--preprocess-cache-reuse-image-root",
        str(source),
        "--preprocess-cache-reuse-processor-source",
        str(processor_source),
    ]
    args = parse_sft2_args(common)
    assert args.preprocess_cache_reuse_image_root == source
    assert args.preprocess_cache_reuse_processor_source == processor_source
    assert args.preprocess_cache_processor_source is None

    with pytest.raises(SystemExit):
        parse_sft2_args([*common, "--require-prebuilt-cache"])
    with pytest.raises(SystemExit):
        parse_sft2_args(
            [
                *REQUIRED,
                "--preprocess-cache-dir",
                str(source / "nested"),
                "--preprocess-cache-reuse-image-root",
                str(source),
                "--preprocess-cache-reuse-processor-source",
                str(processor_source),
            ]
        )


def test_legacy_stage3_image_cache_reuse_requires_original_processor(tmp_path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        split_dir = source / split
        split_dir.mkdir(parents=True)
        (split_dir / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_sft2_args(
            [
                *REQUIRED,
                "--preprocess-cache-dir",
                str(tmp_path / "destination"),
                "--preprocess-cache-reuse-image-root",
                str(source),
            ]
        )
    assert flatten_sft2_yaml_config({"loss": {"wm_value_backbone_grad": False}}) == {
        "wm_value_backbone_grad": False}
    with pytest.raises(ValueError, match="must be a boolean"):
        flatten_sft2_yaml_config({"loss": {"wm_value_backbone_grad": "false"}})


def test_yaml_defaults_apply_after_argument_registration() -> None:
    args = parse_sft2_args(["--config", str(K8_CONFIG), *REQUIRED, "--history-size", "1"])

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
    assert args.history_size == 1
    assert not hasattr(args, "backbone_rows_per_forward")
    assert not hasattr(args, "offload_backbone_chunk_activations")
    assert not hasattr(args, "preprocess_cache_format")
    assert args.preprocess_cache_image_dtype == "bfloat16"
    assert args.preprocess_cache_processor_source is None
    assert args.preprocess_workers == 16


def test_k1_control_uses_b1_ga8_for_global_sigreg_batch() -> None:
    args = parse_sft2_args(["--config", str(K1_CONTROL_CONFIG), *REQUIRED, "--history-size", "1"])

    assert args.latent_token_count == 1
    assert args.latent_query_mode == "inject"
    assert args.query_tune == "adapter"
    assert args.epochs == 10
    assert args.batch_size == 1
    assert args.grad_accum == 8
    assert args.max_pixels == 100352
    assert args.history_size == 1
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
            "--history-size",
            "1",
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


def test_dino_grid_config_has_no_retired_wm_ema_or_decoder_options() -> None:
    args = parse_sft2_args(["--config", str(DINO_GRID_CONFIG), *REQUIRED, "--history-size", "1"])

    assert args.objective == "dino_grid"
    assert args.latent_token_count == 16
    assert not hasattr(args, "grid_ema_decay")
    assert not hasattr(args, "grid_encoder_hidden_dim")
    assert not hasattr(args, "grid_decoder_hidden_dim")
    assert not hasattr(args, "dino_decoder_lr")
    assert not hasattr(args, "grid_warmstart")


def test_dino_grid_h1_t4_config_uses_real_value_and_recorded_rollout_contract() -> None:
    args = parse_sft2_args(["--config", str(DINO_GRID_H1_T4_CONFIG), *REQUIRED])

    assert args.objective == "dino_grid"
    assert args.history_size == 1
    assert args.prediction_horizon == 4
    assert args.value_gamma == 1.0
    assert args.lambda_value == 1.0
    assert args.lambda_dino == 0.5
    assert not hasattr(args, "value_rank_lambda")
    flattened = flatten_sft2_yaml_config(load_yaml_config(DINO_GRID_H1_T4_CONFIG))
    assert str(flattened["train_jsonl"]).endswith(
        "train_terminal_cot_migrated.jsonl"
    )
    assert str(flattened["val_jsonl"]).endswith(
        "val_terminal_cot_migrated.jsonl"
    )
    assert "/52_terminalcot_" in str(flattened["preprocess_cache_dir"])


def test_cls_fixed2d_evaluation_config_resolves_explicit_k65_contract() -> None:
    args = parse_sft2_args(["--config", str(CLS_FIXED2D_CONFIG), *REQUIRED])
    assert args.objective == "dino_grid"
    assert args.epochs == 5
    assert args.schedule_total_steps == 46
    assert args.lr_qwen_start == pytest.approx(2e-7)
    assert args.lr_qwen_peak == pytest.approx(2e-7)
    assert args.state_proj_lr == pytest.approx(8e-6)
    assert args.wm_predictor_lr == pytest.approx(3e-4)
    assert args.value_head_lr == pytest.approx(1e-4)
    assert args.outcome_head_lr == pytest.approx(1e-4)
    assert args.query_lr == pytest.approx(1e-5)
    assert args.protocol_lr == pytest.approx(2e-6)
    assert args.max_length == 16384
    assert args.grid_size == 8
    assert args.latent_token_count == 65
    assert args.grid_global_tokens == 1
    assert args.grid_position_encoding == "fixed_2d_sincos_v1"
    assert args.grid_predictor_kind == "residual"
    assert args.lambda_dino == pytest.approx(2.0)
    assert args.lambda_outcome == pytest.approx(1.0)
    assert args.lambda_sigreg == pytest.approx(0.0)
    assert args.wm_value_backbone_grad is False


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


@pytest.mark.parametrize("field", ["value_rank_margin", "value_rank_lambda"])
def test_sft2_config_rejects_retired_value_ranking_fields(field: str) -> None:
    with pytest.raises(
        ValueError,
        match=f"unknown SFT2 config field: loss.{field}",
    ):
        flatten_sft2_yaml_config({"loss": {field: 1.0}})


def test_sft2_config_rejects_retired_grid_ema_and_decoder_fields() -> None:
    with pytest.raises(
        ValueError,
        match="unknown SFT2 config field: grid.ema_decay",
    ):
        flatten_sft2_yaml_config({"grid": {"ema_decay": 0.99}})


@pytest.mark.parametrize("config", [K8_CONFIG, K1_CONTROL_CONFIG, DINO_GRID_CONFIG])
def test_historical_h4_configs_require_explicit_native_override(config):
    with pytest.raises(SystemExit) as error:
        parse_sft2_args(["--config", str(config), *REQUIRED])
    assert error.value.code == 2
