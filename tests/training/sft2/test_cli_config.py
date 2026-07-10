from __future__ import annotations

from pathlib import Path

from nimloth.training.sft2.cli import parse_sft2_args


ROOT = Path(__file__).resolve().parents[3]
K8_CONFIG = ROOT / "configs" / "training" / "sft2" / "latent_wm_value_k8.yaml"
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
    assert args.mask_latent_query_labels is True
    assert args.epochs == 10
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.max_length == 12000
    assert args.max_pixels == 100352
    assert args.max_images_per_batch == 12
    assert args.preprocess_cache_format == "compact"
    assert args.preprocess_cache_image_dtype == "bfloat16"
    assert args.preprocess_workers == 16


def test_cli_values_override_yaml_defaults() -> None:
    args = parse_sft2_args(
        [
            "--config",
            str(K8_CONFIG),
            *REQUIRED,
            "--latent-token-count",
            "3",
            "--epochs",
            "2",
            "--preprocess-workers",
            "1",
        ]
    )

    assert args.latent_token_count == 3
    assert args.epochs == 2
    assert args.preprocess_workers == 1
