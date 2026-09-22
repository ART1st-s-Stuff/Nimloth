from __future__ import annotations

import pytest

from nimloth.training.sft.stage3.cli import parse_sft2_args


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
    assert not hasattr(args, "value_rank_margin")
    assert not hasattr(args, "value_rank_lambda")
    assert args.checkpoint_metric == "val_wm_mse"
    assert args.batch_mode == "trajectory_online_cache"
    assert args.history_size == 1


def test_parse_sft2_args_rejects_removed_batch_modes() -> None:
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
                "--batch-mode",
                "trajectory",
            ]
        )


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


def test_step_timing_sampling_cli_and_loop_config() -> None:
    from nimloth.config.sft2.schema import SFT2LoopConfig

    required = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train.jsonl",
                "--val-jsonl", "/tmp/val.jsonl", "--output-dir", "/tmp/out"]
    args = parse_sft2_args(required + ["--step-timing-sample-interval", "10"])
    assert SFT2LoopConfig.from_namespace(args).step_timing_sample_interval == 10
    assert parse_sft2_args(required).step_timing_sample_interval == 1
    with pytest.raises(SystemExit):
        parse_sft2_args(required + ["--step-timing-sample-interval", "0"])


def test_frozen_wm_export_requires_explicit_complete_split() -> None:
    required = [
        "--model", "/tmp/model",
        "--train-jsonl", "/tmp/train.jsonl",
        "--val-jsonl", "/tmp/val.jsonl",
        "--output-dir", "/tmp/out",
        "--objective", "dino_grid",
        "--eval-only",
        "--frozen-wm-cache-dir", "/tmp/frozen",
    ]
    with pytest.raises(SystemExit):
        parse_sft2_args(required)
    train = parse_sft2_args([*required, "--frozen-wm-cache-split", "train"])
    assert train.frozen_wm_cache_split == "train"
    eval_args = parse_sft2_args([*required, "--frozen-wm-cache-split", "eval"])
    assert eval_args.frozen_wm_cache_split == "eval"
    with pytest.raises(SystemExit):
        parse_sft2_args([
            *required,
            "--frozen-wm-cache-split", "train",
            "--max-train-records", "1",
        ])
