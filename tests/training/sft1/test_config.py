from pathlib import Path

import pytest

from nimloth.training.sft.stage1.config import sft1_yaml_defaults

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "name", ["qwen25vl_lora_k8.yaml", "qwen25vl_lora_k1_inject.yaml"]
)
def test_old_query_sft1_configs_rejected(name):
    with pytest.raises(ValueError, match="does not accept"):
        sft1_yaml_defaults(ROOT / "configs/training/sft1" / name)


def test_default_sft1_config_has_only_format_parameters():
    defaults = sft1_yaml_defaults(ROOT / "configs/training/sft1/qwen25vl_lora.yaml")
    assert not any("latent" in key for key in defaults)
    assert defaults["epochs"] == 20


def test_fsdp_yaml_is_explicit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  distributed_strategy: fsdp\n")
    assert sft1_yaml_defaults(path)["distributed_strategy"] == "fsdp"


def test_fsdp_cli_supports_format_and_query():
    from nimloth.training.sft.stage1.cli import parse_args

    common = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train.jsonl",
              "--val-jsonl", "/tmp/val.jsonl", "--output-dir", "/tmp/output"]
    assert parse_args(common)[0].distributed_strategy == "ddp"
    assert parse_args(common + ["--distributed-strategy", "fsdp"])[0].distributed_strategy == "fsdp"
    query_args, _ = parse_args(common + ["--distributed-strategy", "fsdp", "--dino-cache-root", "/tmp/dino"], stage="query")
    assert query_args.distributed_strategy == "fsdp"


def test_initial_checkpoint_is_explicit_opt_in():
    from nimloth.training.sft.stage1.cli import parse_args

    common = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train.jsonl",
              "--val-jsonl", "/tmp/val.jsonl", "--output-dir", "/tmp/output"]
    assert not parse_args(common)[0].save_initial_checkpoint
    assert parse_args(common + ["--save-initial-checkpoint"])[0].save_initial_checkpoint
    assert not parse_args(common)[0].keep_step_checkpoints
    assert parse_args(common + ["--keep-step-checkpoints"])[0].keep_step_checkpoints
