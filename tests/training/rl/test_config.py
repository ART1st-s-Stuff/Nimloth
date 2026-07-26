"""RL 类型化配置的字段与覆盖契约。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from nimloth.config.rl import load_rl_config, merge_rl_config_overrides, parse_rl_config
from nimloth.training.rl.cli import main


def _raw_config() -> dict:
    return {
        "freeze": {"state_proj": True},
        "gradient": {"representation_to_backbone": True},
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
    assert config.actor.enabled is False
    assert config.actor.action_objective == "ppo"
    assert config.actor.credit_assignment == "action"
    assert config.actor.max_response_tokens == 64
    assert config.gradient.representation_to_backbone is True
    assert config.agent.planning.enabled is False
    assert config.distributed.nodes == 1
    assert config.distributed.world_size == 1
    assert config.distributed.gpus_per_rank == 1
    assert config.distributed.total_gpus == 1
    assert config.distributed.rollout_tensor_parallel_size == 1


def test_rl_config_parses_heterogeneous_distributed_topology() -> None:
    raw = _raw_config()
    raw["distributed"] = {
        "nodes": 3,
        "world_size": 4,
        "gpus_per_rank": 2,
        "rollout_tensor_parallel_size": 8,
    }

    config = parse_rl_config(raw)

    assert config.distributed.nodes == 3
    assert config.distributed.world_size == 4
    assert config.distributed.gpus_per_rank == 2
    assert config.distributed.total_gpus == 8
    assert config.distributed.rollout_tensor_parallel_size == 8


def test_formal_h2_config_preserves_validated_online_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_rl_config(
        root / "configs/training/rl/planner_greedy_h2_full.yaml"
    )

    assert config.rl.iterations == 60
    assert config.rl.envs_per_iteration == 8
    assert config.rl.max_steps_per_episode == 20
    assert config.rl.batch_size == 8
    assert config.rollout.train_datasets == (
        "base_train",
        "common_sense_train",
    )
    assert config.agent.planning.horizon == 2
    assert config.agent.planning.search_mode == "greedy"
    assert config.actor.action_objective == "distillation"
    assert config.actor.credit_assignment == "action"
    assert config.actor.max_response_tokens == 512
    assert config.training.save_interval == 10
    assert config.validation.enabled is False
    assert config.distributed.total_gpus == 4


def test_continuation_gate_uses_two_fresh_greedy_updates() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_rl_config(
        root / "configs/training/rl/planner_greedy_h2_continuation_gate.yaml"
    )

    assert config.rl.iterations == 2
    assert config.rl.envs_per_iteration == 4
    assert config.rl.batch_size == 4
    assert config.rollout.train_datasets == (
        "base_train",
        "common_sense_train",
    )
    assert config.agent.planning.search_mode == "greedy"
    assert config.distributed.total_gpus == 4


def test_rl_config_rejects_impossible_distributed_topology() -> None:
    raw = _raw_config()
    raw["distributed"] = {"nodes": 3, "world_size": 2}
    with pytest.raises(ValueError, match="nodes cannot exceed"):
        parse_rl_config(raw)

    raw = _raw_config()
    raw["distributed"] = {
        "world_size": 4,
        "rollout_tensor_parallel_size": 8,
    }
    with pytest.raises(ValueError, match="tensor_parallel_size cannot exceed"):
        parse_rl_config(raw)

    raw = _raw_config()
    raw["distributed"] = {"world_size": 2, "gpus_per_rank": 3}
    with pytest.raises(ValueError, match="gpus_per_rank currently supports"):
        parse_rl_config(raw)


def test_rl_config_parses_agent_planning() -> None:
    raw = _raw_config()
    raw["rl"].update({"envs_per_iteration": 2, "batch_size": 2})
    raw["actor"] = {
        "enabled": True,
        "action_objective": "distillation",
        "credit_assignment": "action",
        "planner_distillation_weight": 0.3,
    }
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 3,
            "search_mode": "beam",
            "beam_width": 6,
            "device": "cpu",
        }
    }

    config = parse_rl_config(raw)

    assert config.agent.planning.enabled is True
    assert config.agent.planning.horizon == 3
    assert config.agent.planning.beam_width == 6


def test_planner_episode_training_requires_every_collected_episode() -> None:
    raw = _raw_config()
    raw["rl"].update({"envs_per_iteration": 2, "batch_size": 1})
    raw["actor"] = {
        "enabled": True,
        "action_objective": "distillation",
        "credit_assignment": "action",
        "planner_distillation_weight": 0.3,
    }
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 2,
            "search_mode": "greedy",
            "device": "cpu",
        }
    }

    with pytest.raises(ValueError, match="batch_size to equal rl.envs_per_iteration"):
        parse_rl_config(raw)


def test_planner_episode_training_requires_action_replay() -> None:
    raw = _raw_config()
    raw["agent"] = {"planning": {"enabled": True}}

    with pytest.raises(ValueError, match="actor.enabled=true"):
        parse_rl_config(raw)


def test_rl_config_parses_turn_credit_assignment() -> None:
    raw = _raw_config()
    raw["actor"] = {
        "enabled": True,
        "credit_assignment": "turn",
        "max_response_tokens": 32,
    }

    config = parse_rl_config(raw)

    assert config.actor.credit_assignment == "turn"
    assert config.actor.max_response_tokens == 32


def test_rl_config_requires_explicit_token_credit_semantics() -> None:
    raw = _raw_config()
    raw["actor"] = {"enabled": True, "credit_assignment": "token"}
    with pytest.raises(ValueError, match="explicit token_credit fields"):
        parse_rl_config(raw)

    raw["token_credit"] = {
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "value_lr": 1e-4,
        "value_loss_weight": 0.5,
        "hidden_dim": 256,
    }
    with pytest.raises(ValueError, match="truncated_bootstrap"):
        parse_rl_config(raw)

    raw["rl"]["truncated_bootstrap"] = "zero"
    config = parse_rl_config(raw)

    assert config.actor.credit_assignment == "token"
    assert config.token_credit.gamma == 0.95
    assert config.token_credit.gae_lambda == 0.9
    assert config.token_credit.value_lr == 1e-4
    assert config.token_credit.value_loss_weight == 0.5
    assert config.token_credit.hidden_dim == 256
    assert config.rl.truncated_bootstrap == "zero"


def test_planner_distillation_requires_explicit_search_and_loss_weight() -> None:
    raw = _raw_config()
    raw["rl"]["batch_size"] = 8
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 2,
            "search_mode": "exhaustive",
        }
    }
    raw["actor"] = {
        "enabled": True,
        "action_objective": "distillation",
        "credit_assignment": "action",
    }
    raw["rl"]["truncated_bootstrap"] = "zero"

    with pytest.raises(ValueError, match="planning.device"):
        parse_rl_config(raw)

    raw["agent"]["planning"]["device"] = "cpu"
    with pytest.raises(ValueError, match="planner_distillation_weight"):
        parse_rl_config(raw)

    raw["actor"]["planner_distillation_weight"] = 0.3
    with pytest.raises(ValueError, match="predictor.train_wm"):
        parse_rl_config(raw)

    raw["predictor"]["train_wm"] = True
    raw["predictor"]["lambda_sigreg"] = 0.0
    config = parse_rl_config(raw)
    assert config.agent.planning.horizon == 2
    assert config.agent.planning.beam_width is None
    assert config.actor.planner_distillation_weight == 0.3
    assert config.predictor.train_wm is True


def test_planner_beam_width_matches_search_mode() -> None:
    raw = _raw_config()
    raw["rl"]["batch_size"] = 8
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 2,
            "search_mode": "beam",
            "device": "cpu",
        }
    }
    raw["actor"] = {
        "enabled": True,
        "action_objective": "distillation",
        "credit_assignment": "action",
        "planner_distillation_weight": 0.3,
    }
    raw["rl"]["truncated_bootstrap"] = "zero"
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})

    with pytest.raises(ValueError, match="beam_width"):
        parse_rl_config(raw)

    raw["agent"]["planning"]["beam_width"] = 4
    assert parse_rl_config(raw).agent.planning.beam_width == 4

    raw["agent"]["planning"]["search_mode"] = "exhaustive"
    with pytest.raises(ValueError, match="only valid for beam"):
        parse_rl_config(raw)


def test_reference_kl_requires_explicit_low_var_loss_type() -> None:
    raw = _raw_config()
    raw["actor"] = {
        "enabled": True,
        "credit_assignment": "token",
        "reference_kl_loss_weight": 0.001,
    }
    raw["token_credit"] = {
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "value_lr": 1e-5,
        "value_loss_weight": 1.0,
        "hidden_dim": 256,
    }
    raw["rl"]["truncated_bootstrap"] = "zero"
    with pytest.raises(ValueError, match="reference_kl_loss_type=low_var_kl"):
        parse_rl_config(raw)

    raw["actor"]["reference_kl_loss_type"] = "low_var_kl"
    config = parse_rl_config(raw)
    assert config.actor.reference_kl_loss_weight == 0.001
    assert config.actor.reference_kl_loss_type == "low_var_kl"


def test_reward_kl_is_rejected_as_unimplemented() -> None:
    raw = _raw_config()
    raw["actor"] = {"reward_kl_weight": 0.001}
    with pytest.raises(ValueError, match="unknown RL config field"):
        parse_rl_config(raw)


def test_rl_config_rejects_unknown_credit_assignment() -> None:
    raw = _raw_config()
    raw["actor"] = {"credit_assignment": "bi_level_gae"}
    with pytest.raises(ValueError, match="credit_assignment"):
        parse_rl_config(raw)


def test_rl_config_rejects_unknown_checkpoint_metric() -> None:
    raw = _raw_config()
    raw["validation"]["checkpoint_metric"] = "train_value_loss"
    with pytest.raises(ValueError, match="validation.checkpoint_metric"):
        parse_rl_config(raw)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("freeze", "state_proj"),
        ("gradient", "representation_to_backbone"),
        ("validation", "enabled"),
    ),
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
