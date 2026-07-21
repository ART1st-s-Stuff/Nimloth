"""RL 类型化配置的字段与覆盖契约。"""

from __future__ import annotations

from argparse import Namespace

import pytest

from nimloth.config.rl import merge_rl_config_overrides, parse_rl_config
from nimloth.training.rl.cli import main


def _raw_config() -> dict:
    return {
        "freeze": {"state_proj": True},
        "predictor": {"emb_dim": 128, "history_size": 1},
        "rollout": {
            "train_datasets": ["base_train"],
            "eval_datasets": ["base"],
        },
        "validation": {"checkpoint_metric": "success_rate"},
        "rl": {"iterations": 10},
    }


def test_rl_config_rejects_unimplemented_fields() -> None:
    raw = _raw_config()
    raw["qwen"] = {"model": "ignored-before"}
    with pytest.raises(ValueError, match="unknown RL config section: qwen"):
        parse_rl_config(raw)


def test_rl_config_rejects_stale_nested_fields() -> None:
    raw = _raw_config()
    raw["predictor"]["rollout_steps"] = 1
    with pytest.raises(
        ValueError,
        match="unknown RL config field: predictor.rollout_steps",
    ):
        parse_rl_config(raw)


def test_rl_config_builds_immutable_sections_and_cli_overrides() -> None:
    config = parse_rl_config(_raw_config())
    overridden = merge_rl_config_overrides(
        Namespace(seed=7, rl_iterations=20, rl_envs_per_iteration=3),
        config,
    )

    assert config.rl.iterations == 10
    assert overridden.rl.iterations == 20
    assert overridden.rl.envs_per_iteration == 3
    assert overridden.training.seed == 7
    assert overridden.rollout.train_datasets == ("base_train",)
    assert config.predictor.lambda_sigreg == 0.1
    assert config.predictor.sigreg_num_proj == 1024
    assert config.predictor.sigreg_knots == 17


def test_rl_config_rejects_unknown_checkpoint_metric() -> None:
    raw = _raw_config()
    raw["validation"]["checkpoint_metric"] = "train_value_loss"
    with pytest.raises(ValueError, match="validation.checkpoint_metric"):
        parse_rl_config(raw)


@pytest.mark.parametrize(
    ("section", "field"),
    (("freeze", "state_proj"), ("validation", "enabled")),
)
def test_rl_config_rejects_string_booleans(section: str, field: str) -> None:
    raw = _raw_config()
    raw[section][field] = "false"
    with pytest.raises(ValueError, match=f"{section}.{field} must be a boolean"):
        parse_rl_config(raw)


def test_rl_config_validates_discount_and_validation_size() -> None:
    raw = _raw_config()
    raw["rl"]["gamma"] = 1.1
    with pytest.raises(ValueError, match="rl.gamma"):
        parse_rl_config(raw)

    raw = _raw_config()
    raw["validation"]["envs"] = 0
    with pytest.raises(ValueError, match="validation.envs"):
        parse_rl_config(raw)


def test_rl_cli_requires_explicit_rollout_mode(tmp_path) -> None:
    with pytest.raises(ValueError, match="choose --env-url or --use-jsonl-rollout"):
        main(
            [
                "--config",
                str(tmp_path / "unused.yaml"),
                "--model",
                "/tmp/model",
                "--output-dir",
                "/tmp/output",
            ]
        )
