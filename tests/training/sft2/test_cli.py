from __future__ import annotations

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
    assert args.early_stop_metric == "val_success_rate"
    assert args.trajectory_aware_batching is False
    assert args.allow_approx_trajectory_once is False


def test_parse_sft2_args_trajectory_flags() -> None:
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
            "--trajectory-aware-batching",
            "--packed-forward",
            "--allow-approx-trajectory-once",
        ]
    )
    assert args.trajectory_aware_batching is True
    assert args.packed_forward is True
    assert args.allow_approx_trajectory_once is True
