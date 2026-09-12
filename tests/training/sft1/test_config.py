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
    assert defaults["format_eval_batch_size"] == 4


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
    assert args.format_eval_batch_size == 4
    assert args.convergence_min_epochs is None
    assert args.convergence_patience_epochs is None
    assert args.convergence_min_relative_improvement is None
    with pytest.raises(ValueError, match='convergence policy requires'):
        parse_args(common + ['--epochs', '2', '--no-until-converged',
                             '--convergence-min-epochs', '2'])
    with pytest.raises(ValueError, match='exactly 32'):
        parse_args(common + ['--format-eval-samples', '31'])
    with pytest.raises(ValueError, match='batch-size must be >= 1'):
        parse_args(common + ['--format-eval-batch-size', '0'])
    overridden, _ = parse_args(
        common
        + [
            '--epochs',
            '2',
            '--no-until-converged',
            '--format-eval-batch-size',
            '2',
        ]
    )
    assert overridden.format_eval_batch_size == 2


def test_format_sampling_defaults_and_overrides():
    from nimloth.training.sft.stage1.cli import parse_args
    common = ["--model", "/tmp/model", "--train-jsonl", "/tmp/train",
              "--val-jsonl", "/tmp/val", "--format-eval-jsonl", "/tmp/format",
              "--output-dir", "/tmp/out"]
    args, _ = parse_args(common)
    assert (args.format_eval_temperature, args.format_eval_top_p,
            args.format_eval_max_new_tokens, args.format_eval_generation_seed) == (0.7, 0.95, 512, 0)
    defaults = sft1_yaml_defaults(ROOT / "configs/training/sft1/format.yaml")
    assert defaults["format_eval_temperature"] == 0.7
    args, _ = parse_args(common + ["--format-eval-temperature", "0.8", "--format-eval-generation-seed", "12"])
    assert args.format_eval_temperature == 0.8
    assert args.format_eval_generation_seed == 12
    for flag, value in [("temperature", "nan"), ("temperature", "-1"), ("top-p", "0"),
                        ("max-new-tokens", "0"), ("generation-seed", "-1")]:
        with pytest.raises(ValueError, match="format-eval"):
            parse_args(common + [f"--format-eval-{flag}", value])


def test_weighted_convergence_cli_and_yaml(tmp_path):
    from nimloth.training.sft.stage1.cli import parse_args
    common = ['--config', str(ROOT / 'configs/training/sft1/format.yaml'),
              '--model', '/tmp/model', '--train-jsonl', '/tmp/train',
              '--val-jsonl', '/tmp/val', '--format-eval-jsonl', '/tmp/format-eval',
              '--output-dir', '/tmp/run']
    args, _ = parse_args(common + ['--convergence-metric', 'validation_weighted_lm_loss',
                                 '--convergence-format-min-rate', '0.96875'])
    assert args.convergence_metric == 'validation_weighted_lm_loss'
    assert args.convergence_format_min_rate == 31 / 32
    for value in ['nan', '-1', '1.01']:
        with pytest.raises(ValueError, match='minimum rate'):
            parse_args(common + ['--convergence-format-min-rate', value])
    path = tmp_path / 'metric.yaml'
    path.write_text('train:\n  convergence_metric: validation_weighted_lm_loss\n  convergence_format_min_rate: 0.96875\n')
    assert sft1_yaml_defaults(path)['convergence_metric'] == 'validation_weighted_lm_loss'
