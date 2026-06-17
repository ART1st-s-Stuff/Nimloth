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
    assert args.lambda_value == 1.0
    assert args.early_stop_metric == "val_success_rate"
