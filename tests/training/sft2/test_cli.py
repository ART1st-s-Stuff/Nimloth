from __future__ import annotations

import pytest

from nimloth.training.sft2.cli import parse_sft2_args


def test_parse_sft2_args_applies_yaml_defaults() -> None:
    args = parse_sft2_args(
        [
            "--model",
            "/tmp/model",
            "--train-jsonl",
            "/tmp/train.jsonl",
            "--val-jsonl",
            "/tmp/val.jsonl",
            "--output-dir",
            "/tmp/out",
        ]
    )
    assert args.llm_tune == "freeze"
    assert args.vision_tune == "full"
    assert args.batch_size == 2
    assert args.grad_accum == 4
    assert args.lambda_value == 1.0
    assert args.checkpoint_metric == "val_wm_mse"
    assert args.batch_mode == "trajectory_image_budget"


def test_parse_sft2_args_batch_mode() -> None:
    args = parse_sft2_args(
        [
            "--model",
            "/tmp/model",
            "--train-jsonl",
            "/tmp/train.jsonl",
            "--val-jsonl",
            "/tmp/val.jsonl",
            "--output-dir",
            "/tmp/out",
            "--batch-mode",
            "trajectory",
        ]
    )
    assert args.batch_mode == "trajectory"


def test_production_cli_rejects_packed_forward() -> None:
    with pytest.raises(SystemExit):
        parse_sft2_args(
            [
                "--model",
                "/tmp/model",
                "--train-jsonl",
                "/tmp/train.jsonl",
                "--val-jsonl",
                "/tmp/val.jsonl",
                "--output-dir",
                "/tmp/out",
                "--packed-forward",
            ]
        )
