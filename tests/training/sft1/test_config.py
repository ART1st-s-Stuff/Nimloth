from pathlib import Path

import pytest

from nimloth.training.sft.stage1.config import sft1_yaml_defaults

ROOT = Path(__file__).resolve().parents[3]


def test_format_config_defaults():
    defaults = sft1_yaml_defaults(ROOT / "configs/training/sft1/format.yaml")
    assert not any("latent" in key for key in defaults)
    assert defaults["until_converged"] is True
    assert defaults["action_token_loss_weight"] == 8
    assert defaults["convergence_patience_epochs"] == 2
    assert defaults["prune_intermediate_checkpoints"] is True


def test_query_configuration_rejected_for_format(tmp_path):
    path = tmp_path / "query.yaml"
    path.write_text("latent:\n  token_count: 16\n")
    with pytest.raises(ValueError, match="does not accept"):
        sft1_yaml_defaults(path)


def test_fsdp_yaml_is_explicit(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  distributed_strategy: fsdp\n")
    assert sft1_yaml_defaults(path)["distributed_strategy"] == "fsdp"


def test_fsdp_cli_supports_both_early_stages():
    from nimloth.training.sft.stage1.cli import parse_args

    query_common = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train.jsonl",
                    "--val-jsonl", "/tmp/val.jsonl", "--output-dir", "/tmp/output"]
    common = [*query_common, "--format-eval-jsonl", "/tmp/format-eval.jsonl"]
    assert parse_args(common)[0].distributed_strategy == "ddp"
    assert parse_args(common + ["--distributed-strategy", "fsdp"])[0].distributed_strategy == "fsdp"
    args, _ = parse_args(query_common + ["--distributed-strategy", "fsdp", "--dino-cache-root", "/tmp/dino"], stage="query")
    assert args.distributed_strategy == "fsdp"
    assert args.action_token_loss_weight == 1


def test_format_cli_can_override_convergence_and_checkpointing():
    from nimloth.training.sft.stage1.cli import parse_args
    common = ['--config', str(ROOT / 'configs/training/sft1/format.yaml'),
              '--model', '/tmp/model', '--train-jsonl', '/tmp/train',
              '--val-jsonl', '/tmp/val', '--format-eval-jsonl', '/tmp/format-eval',
              '--output-dir', '/tmp/run']
    args, _ = parse_args(common + ['--epochs', '2', '--no-until-converged',
                                  '--no-gradient-checkpointing'])
    assert args.epochs == 2
    assert args.until_converged is False
    assert args.gradient_checkpointing is False
    assert args.convergence_min_epochs is None
    assert args.convergence_patience_epochs is None
    assert args.convergence_min_relative_improvement is None
    with pytest.raises(ValueError, match='convergence policy requires'):
        parse_args(common + ['--epochs', '2', '--no-until-converged',
                             '--convergence-min-epochs', '2'])
    with pytest.raises(ValueError, match='exactly 32'):
        parse_args(common + ['--format-eval-samples', '31'])
