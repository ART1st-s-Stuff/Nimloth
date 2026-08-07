"""RL 类型化配置的字段与覆盖契约。"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import pytest

from nimloth.config.rl import load_rl_config, merge_rl_config_overrides, parse_rl_config
from nimloth.training.rl.cli import main, parse_rl_args


def _raw_config() -> dict:
    return {
        "freeze": {"state_proj": True},
        "gradient": {
            "state_source": "recompute",
            "representation_to_backbone": True,
        },
        "predictor": {"emb_dim": 128, "history_size": 1},
        "rollout": {
            "train_datasets": ["base_train"],
            "eval_datasets": ["base"],
        },
        "validation": {"checkpoint_metric": "success_rate"},
        "rl": {"iterations": 10},
    }


def _configure_planner_value_ppo(raw: dict, *, epochs: int = 4) -> None:
    raw["value_head"] = {
        "lambda_rank": 0.0,
        "ppo_clip_range": 0.2,
        "ppo_epochs": epochs,
    }


def _configure_planner_policy_ppo(raw: dict, *, epochs: int = 4) -> None:
    raw["value_head"] = {"lambda_rank": 0.0}
    raw["planner_policy"] = {
        "enabled": True,
        "lr": 1e-4,
        "clip_ratio": 0.2,
        "entropy_coeff": 0.01,
        "temperature": 1.0,
        "ppo_epochs": epochs,
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
    assert config.rollout.max_episode_attempts == 1
    assert config.predictor.lambda_sigreg == 0.1
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.0
    assert config.predictor.sigreg_num_proj == 1024
    assert config.predictor.sigreg_knots == 17
    assert config.actor.enabled is False
    assert config.actor.credit_assignment == "action"
    assert config.actor.max_response_tokens == 64
    assert config.actor.max_state_tokens is None
    assert config.gradient.representation_to_backbone is True
    assert config.gradient.state_source == "recompute"
    assert config.agent.planning.enabled is False
    assert config.distributed.nodes == 1
    assert config.distributed.world_size == 1
    assert config.distributed.gpus_per_rank == 1
    assert config.distributed.total_gpus == 1
    assert config.distributed.rollout_tensor_parallel_size == 1


def test_rollout_config_requires_positive_episode_attempts() -> None:
    raw = _raw_config()
    raw["rollout"]["max_episode_attempts"] = 0

    with pytest.raises(ValueError, match="max_episode_attempts must be positive"):
        parse_rl_config(raw)


def test_external_validation_is_explicit_and_excludes_builtin_validation() -> None:
    raw = _raw_config()
    raw["validation"] = {
        "enabled": False,
        "external": True,
        "interval": 10,
        "envs": 120,
        "checkpoint_metric": "success_rate",
    }

    config = parse_rl_config(raw)

    assert config.validation.enabled is False
    assert config.validation.external is True
    assert config.validation.interval == 10
    assert config.validation.envs == 120

    raw["validation"]["enabled"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_rl_config(raw)


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
    assert config.actor.enabled is False
    assert config.actor.credit_assignment == "action"
    assert config.actor.max_response_tokens == 512
    assert config.gradient.state_source == "recompute"
    assert config.gradient.representation_to_backbone is True
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.5
    assert config.training.save_interval == 10
    assert config.validation.enabled is False
    assert config.distributed.total_gpus == 4


def test_h1_smoke_trains_qwen_wm_and_value_without_direct_ppo() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_smoke.yaml"
    )

    assert config.agent.planning.horizon == 1
    assert config.agent.planning.search_mode == "greedy"
    assert config.freeze.state_proj is True
    assert config.gradient.state_source == "recompute"
    assert config.gradient.representation_to_backbone is True
    assert config.actor.enabled is False
    assert config.predictor.history_size == 1
    assert config.predictor.train_wm is True
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.5
    assert config.value_head.lambda_rank == 0.0
    assert config.value_head.ppo_clip_range == 0.2
    assert config.value_head.ppo_epochs == 4
    assert config.rl.iterations == 1
    assert config.rl.envs_per_iteration == config.rl.batch_size == 4
    assert config.rl.max_steps_per_episode == 20
    assert config.distributed.total_gpus == 4


def test_formal_h1_config_preserves_corrected_online_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full.yaml"
    )

    assert config.agent.planning.horizon == 1
    assert config.agent.planning.search_mode == "greedy"
    assert config.freeze.state_proj is True
    assert config.gradient.state_source == "recompute"
    assert config.gradient.representation_to_backbone is True
    assert config.actor.enabled is False
    assert config.predictor.history_size == 1
    assert config.predictor.train_wm is True
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.5
    assert config.value_head.lambda_rank == 0.0
    assert config.value_head.ppo_clip_range == 0.2
    assert config.value_head.ppo_epochs == 4
    assert config.rl.iterations == 60
    assert config.rl.envs_per_iteration == config.rl.batch_size == 8
    assert config.rl.max_steps_per_episode == 20
    assert config.rollout.train_datasets == (
        "base_train",
        "common_sense_train",
    )
    assert config.validation.enabled is False
    assert config.validation.external is False
    assert config.training.save_interval == 10
    assert config.distributed.nodes == 1
    assert config.distributed.world_size == 2
    assert config.distributed.gpus_per_rank == 2
    assert config.distributed.total_gpus == 4


def test_formal_h1_32gpu_config_preserves_objective_and_true_sharded_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_32gpu.yaml"
    )

    assert config.agent.planning.horizon == 1
    assert config.agent.planning.search_mode == "greedy"
    assert config.freeze.state_proj is True
    assert config.gradient.state_source == "recompute"
    assert config.gradient.representation_to_backbone is True
    assert config.actor.enabled is False
    assert config.actor.reference_kl_loss_weight == 0.0
    assert config.predictor.history_size == 1
    assert config.predictor.train_wm is True
    assert config.predictor.lambda_wm == 1.0
    assert config.predictor.lambda_dino == 0.5
    assert config.rl.iterations == 60
    assert config.rl.envs_per_iteration == config.rl.batch_size == 128
    assert config.rollout.train_datasets == (
        "base_train",
        "common_sense_train",
    )
    assert config.rollout.eval_datasets == ("base", "common_sense")
    assert config.validation.enabled is False
    assert config.validation.external is True
    assert config.validation.interval == 10
    assert config.validation.envs == 120
    assert config.distributed.nodes == 4
    assert config.distributed.world_size == 16
    assert config.distributed.gpus_per_rank == 2
    assert config.distributed.rollout_tensor_parallel_size == 4
    assert config.distributed.total_gpus == 32


def test_formal_h1_32gpu_88844_config_changes_only_physical_node_count() -> None:
    root = Path(__file__).resolve().parents[3]
    uniform = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_32gpu.yaml"
    )
    heterogeneous = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_32gpu_88844.yaml"
    )

    assert heterogeneous.distributed.nodes == 5
    assert heterogeneous.distributed.world_size == 16
    assert heterogeneous.distributed.gpus_per_rank == 2
    assert heterogeneous.distributed.rollout_tensor_parallel_size == 4
    assert heterogeneous.distributed.total_gpus == 32
    assert heterogeneous.agent == uniform.agent
    assert heterogeneous.freeze == uniform.freeze
    assert heterogeneous.gradient == uniform.gradient
    assert heterogeneous.actor == uniform.actor
    assert heterogeneous.predictor == uniform.predictor
    assert heterogeneous.value_head == uniform.value_head
    assert heterogeneous.rl == uniform.rl
    assert heterogeneous.rollout == uniform.rollout
    assert heterogeneous.validation == uniform.validation
    assert heterogeneous.training == uniform.training


def test_formal_h1_8gpu_44_config_changes_only_distributed_parallelism() -> None:
    root = Path(__file__).resolve().parents[3]
    full32 = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_32gpu.yaml"
    )
    config8 = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_8gpu_44.yaml"
    )

    assert config8.distributed.nodes == 2
    assert config8.distributed.world_size == 4
    assert config8.distributed.gpus_per_rank == 2
    assert config8.distributed.rollout_tensor_parallel_size == 4
    assert config8.distributed.total_gpus == 8
    assert config8.agent == full32.agent
    assert config8.freeze == full32.freeze
    assert config8.gradient == full32.gradient
    assert config8.actor == full32.actor
    assert config8.predictor == full32.predictor
    assert config8.value_head == full32.value_head
    assert config8.rl == full32.rl
    assert config8.rollout == full32.rollout
    assert config8.validation == full32.validation
    assert config8.training == full32.training


def test_formal_h1_8gpu_422_config_changes_only_node_count_from_44() -> None:
    root = Path(__file__).resolve().parents[3]
    uniform = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_8gpu_44.yaml"
    )
    heterogeneous = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_8gpu_422.yaml"
    )

    assert heterogeneous.distributed.nodes == 3
    assert heterogeneous.distributed.world_size == 4
    assert heterogeneous.distributed.gpus_per_rank == 2
    assert heterogeneous.distributed.rollout_tensor_parallel_size == 4
    assert heterogeneous.distributed.total_gpus == 8
    assert heterogeneous.agent == uniform.agent
    assert heterogeneous.freeze == uniform.freeze
    assert heterogeneous.gradient == uniform.gradient
    assert heterogeneous.actor == uniform.actor
    assert heterogeneous.predictor == uniform.predictor
    assert heterogeneous.value_head == uniform.value_head
    assert heterogeneous.rl == uniform.rl
    assert heterogeneous.rollout == uniform.rollout
    assert heterogeneous.validation == uniform.validation
    assert heterogeneous.training == uniform.training


def test_formal_h1_16rollout_22gpu_8662_preserves_objective_and_eval() -> None:
    root = Path(__file__).resolve().parents[3]
    full32 = load_rl_config(
        root / "configs/training/rl/planner_greedy_h1_full_32gpu.yaml"
    )
    config = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_22gpu_8662.yaml"
    )

    assert config.agent == full32.agent
    assert config.freeze == full32.freeze
    assert config.gradient == full32.gradient
    assert config.actor == full32.actor
    assert config.predictor == full32.predictor
    assert config.value_head == full32.value_head
    assert config.rl.iterations == 60
    assert config.rl.envs_per_iteration == config.rl.batch_size == 16
    assert config.rl.max_steps_per_episode == full32.rl.max_steps_per_episode
    assert config.rl.gamma == full32.rl.gamma
    assert config.rl.truncated_bootstrap == full32.rl.truncated_bootstrap
    assert config.rollout == full32.rollout
    assert config.validation == full32.validation
    assert config.training == full32.training
    assert config.distributed.nodes == 4
    assert config.distributed.world_size == 11
    assert config.distributed.gpus_per_rank == 2
    assert config.distributed.rollout_tensor_parallel_size == 4
    assert config.distributed.total_gpus == 22


def test_formal_h1_16rollout_20gpu_8642_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config22 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_22gpu_8662.yaml"
    )
    config20 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_20gpu_8642.yaml"
    )

    assert config20.agent == config22.agent
    assert config20.freeze == config22.freeze
    assert config20.gradient == config22.gradient
    assert config20.actor == config22.actor
    assert config20.predictor == config22.predictor
    assert config20.value_head == config22.value_head
    assert config20.rl == config22.rl
    assert config20.rollout == config22.rollout
    assert config20.validation == config22.validation
    assert config20.training == config22.training
    assert config20.distributed.nodes == 4
    assert config20.distributed.world_size == 10
    assert config20.distributed.gpus_per_rank == 2
    assert config20.distributed.rollout_tensor_parallel_size == 4
    assert config20.distributed.total_gpus == 20


def test_formal_h1_16rollout_12gpu_642_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config20 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_20gpu_8642.yaml"
    )
    config12 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_642.yaml"
    )

    assert config12.agent == config20.agent
    assert config12.freeze == config20.freeze
    assert config12.gradient == config20.gradient
    assert config12.actor == config20.actor
    assert config12.predictor == config20.predictor
    assert config12.value_head == config20.value_head
    assert config12.rl == config20.rl
    assert config12.rollout == config20.rollout
    assert config12.validation == config20.validation
    assert config12.training == config20.training
    assert config12.distributed.nodes == 3
    assert config12.distributed.world_size == 6
    assert config12.distributed.gpus_per_rank == 2
    assert config12.distributed.rollout_tensor_parallel_size == 4
    assert config12.distributed.total_gpus == 12
    assert config12.actor.max_state_tokens == 16384


def test_formal_h1_16rollout_8gpu_44_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config12 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_642.yaml"
    )
    config8 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_44.yaml"
    )

    assert config8.agent == config12.agent
    assert config8.freeze == config12.freeze
    assert config8.gradient == config12.gradient
    assert config8.actor == config12.actor
    assert config8.predictor == config12.predictor
    assert config8.value_head == config12.value_head
    assert config8.rl == config12.rl
    assert config8.rollout.max_episode_attempts == 3
    assert replace(
        config8.rollout,
        max_episode_attempts=config12.rollout.max_episode_attempts,
    ) == config12.rollout
    assert config8.validation == config12.validation
    assert config8.training == config12.training
    assert config8.distributed.nodes == 2
    assert config8.distributed.world_size == 4
    assert config8.distributed.gpus_per_rank == 2
    assert config8.distributed.rollout_tensor_parallel_size == 4
    assert config8.distributed.total_gpus == 8


def test_h1_iter16_smoke_changes_only_training_horizon() -> None:
    root = Path(__file__).resolve().parents[3]
    formal = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml"
    )
    smoke = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_smoke_iter16_16rollout_8gpu_1x8.yaml"
    )

    assert smoke.agent == formal.agent
    assert smoke.freeze == formal.freeze
    assert smoke.gradient == formal.gradient
    assert smoke.actor == formal.actor
    assert smoke.predictor == formal.predictor
    assert smoke.value_head == formal.value_head
    assert smoke.rl.iterations == 16
    assert replace(smoke.rl, iterations=formal.rl.iterations) == formal.rl
    assert smoke.rollout == formal.rollout
    assert smoke.validation == formal.validation
    assert smoke.training == formal.training
    assert smoke.distributed == formal.distributed


def test_formal_h1_16rollout_8gpu_422_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config1x8 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_1x8.yaml"
    )
    config422 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_8gpu_422.yaml"
    )

    assert config422.agent == config1x8.agent
    assert config422.freeze == config1x8.freeze
    assert config422.gradient == config1x8.gradient
    assert config422.actor == config1x8.actor
    assert config422.predictor == config1x8.predictor
    assert config422.value_head == config1x8.value_head
    assert config422.rl == config1x8.rl
    assert config422.rollout == config1x8.rollout
    assert config422.validation == config1x8.validation
    assert config422.training == config1x8.training
    assert config422.distributed.nodes == 3
    assert config422.distributed.world_size == 4
    assert config422.distributed.gpus_per_rank == 2
    assert config422.distributed.rollout_tensor_parallel_size == 4
    assert config422.distributed.total_gpus == 8


def test_formal_h1_16rollout_12gpu_6222_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config642 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_642.yaml"
    )
    config6222 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_6222.yaml"
    )

    assert config6222.agent == config642.agent
    assert config6222.freeze == config642.freeze
    assert config6222.gradient == config642.gradient
    assert config6222.actor == config642.actor
    assert config6222.predictor == config642.predictor
    assert config6222.value_head == config642.value_head
    assert config6222.rl == config642.rl
    assert config6222.rollout == config642.rollout
    assert config6222.validation == config642.validation
    assert config6222.training == config642.training
    assert config6222.distributed.nodes == 4
    assert config6222.distributed.world_size == 6
    assert config6222.distributed.gpus_per_rank == 2
    assert config6222.distributed.rollout_tensor_parallel_size == 4
    assert config6222.distributed.total_gpus == 12


def test_formal_h1_16rollout_24gpu_66642_changes_only_distributed_layout() -> None:
    root = Path(__file__).resolve().parents[3]
    config12 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_12gpu_642.yaml"
    )
    config24 = load_rl_config(
        root
        / "configs/training/rl/planner_greedy_h1_full_16rollout_24gpu_66642.yaml"
    )

    assert config24.agent == config12.agent
    assert config24.freeze == config12.freeze
    assert config24.gradient == config12.gradient
    assert config24.actor == config12.actor
    assert config24.predictor == config12.predictor
    assert config24.value_head == config12.value_head
    assert config24.rl == config12.rl
    assert config24.rollout == config12.rollout
    assert config24.validation == config12.validation
    assert config24.training == config12.training
    assert config24.distributed.nodes == 5
    assert config24.distributed.world_size == 12
    assert config24.distributed.gpus_per_rank == 2
    assert config24.distributed.rollout_tensor_parallel_size == 4
    assert config24.distributed.total_gpus == 24


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
    assert config.distributed.nodes == 1
    assert config.distributed.world_size == 2
    assert config.distributed.gpus_per_rank == 2
    assert config.distributed.total_gpus == 4


def test_rl_cli_preserves_checkpoint_processor_by_default() -> None:
    args = parse_rl_args(
        [
            "--config", "config.yaml",
            "--model", "model",
            "--output-dir", "output",
        ]
    )

    assert args.max_pixels is None


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
        "enabled": False,
        "credit_assignment": "action",
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
    _configure_planner_value_ppo(raw)

    config = parse_rl_config(raw)

    assert config.agent.planning.enabled is True
    assert config.agent.planning.horizon == 3
    assert config.agent.planning.beam_width == 6


def test_planner_episode_training_requires_every_collected_episode() -> None:
    raw = _raw_config()
    raw["rl"].update({"envs_per_iteration": 2, "batch_size": 1})
    raw["actor"] = {
        "enabled": False,
        "credit_assignment": "action",
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
    _configure_planner_value_ppo(raw)

    with pytest.raises(ValueError, match="batch_size to equal rl.envs_per_iteration"):
        parse_rl_config(raw)


def test_planner_rejects_direct_qwen_ppo() -> None:
    raw = _raw_config()
    raw["actor"] = {"enabled": True}
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 2,
            "search_mode": "greedy",
            "device": "cpu",
        }
    }
    _configure_planner_value_ppo(raw)

    with pytest.raises(ValueError, match="actor.enabled must be false"):
        parse_rl_config(raw)


def test_rl_config_parses_turn_credit_assignment() -> None:
    raw = _raw_config()
    raw["actor"] = {
        "enabled": True,
        "credit_assignment": "turn",
        "max_response_tokens": 32,
        "max_state_tokens": 16384,
    }

    config = parse_rl_config(raw)

    assert config.actor.credit_assignment == "turn"
    assert config.actor.max_response_tokens == 32
    assert config.actor.max_state_tokens == 16384


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


def test_planner_requires_explicit_search_and_trainable_world_model() -> None:
    raw = _raw_config()
    raw["rl"]["batch_size"] = 8
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 2,
            "search_mode": "exhaustive",
        }
    }
    _configure_planner_value_ppo(raw)
    raw["actor"] = {"enabled": False, "credit_assignment": "action"}
    raw["rl"]["truncated_bootstrap"] = "zero"

    with pytest.raises(ValueError, match="planning.device"):
        parse_rl_config(raw)

    raw["agent"]["planning"]["device"] = "cpu"
    with pytest.raises(ValueError, match="predictor.train_wm"):
        parse_rl_config(raw)

    raw["predictor"]["train_wm"] = False
    with pytest.raises(ValueError, match="train_wm=true"):
        parse_rl_config(raw)

    raw["predictor"]["train_wm"] = True
    raw["predictor"]["lambda_sigreg"] = 0.0
    config = parse_rl_config(raw)
    assert config.agent.planning.horizon == 2
    assert config.agent.planning.beam_width is None
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
    _configure_planner_value_ppo(raw)
    raw["actor"] = {
        "enabled": False,
        "credit_assignment": "action",
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


def test_planner_value_ppo_requires_explicit_clip_and_multiple_epochs() -> None:
    raw = _raw_config()
    raw["rl"]["batch_size"] = raw["rl"].get("envs_per_iteration", 8)
    raw["actor"] = {"enabled": False, "credit_assignment": "action"}
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 1,
            "search_mode": "greedy",
            "device": "cpu",
        }
    }

    with pytest.raises(ValueError, match="explicit value_head.ppo_clip_range"):
        parse_rl_config(raw)

    _configure_planner_value_ppo(raw, epochs=1)
    with pytest.raises(ValueError, match="ppo_epochs>=2"):
        parse_rl_config(raw)


def test_nonplanner_rejects_unused_value_ppo_fields() -> None:
    raw = _raw_config()
    _configure_planner_value_ppo(raw)

    with pytest.raises(ValueError, match="only valid for planner"):
        parse_rl_config(raw)


def test_planner_policy_ppo_requires_h1_policy_search_and_explicit_fields() -> None:
    raw = _raw_config()
    raw["rl"].update({"envs_per_iteration": 2, "batch_size": 2})
    raw["actor"] = {"enabled": False, "credit_assignment": "action"}
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 1,
            "search_mode": "policy",
            "device": "cpu",
        }
    }
    _configure_planner_policy_ppo(raw)

    config = parse_rl_config(raw)

    assert config.planner_policy.enabled is True
    assert config.planner_policy.ppo_epochs == 4
    assert config.value_head.ppo_clip_range is None

    raw["agent"]["planning"]["horizon"] = 2
    with pytest.raises(ValueError, match="requires horizon=1"):
        parse_rl_config(raw)

    raw["agent"]["planning"]["horizon"] = 1
    raw["planner_policy"].pop("temperature")
    with pytest.raises(ValueError, match="explicit planner_policy fields"):
        parse_rl_config(raw)


def test_planner_policy_ppo_rejects_critic_clipping_fields() -> None:
    raw = _raw_config()
    raw["rl"].update({"envs_per_iteration": 2, "batch_size": 2})
    raw["actor"] = {"enabled": False, "credit_assignment": "action"}
    raw["predictor"].update({"train_wm": True, "lambda_sigreg": 0.0})
    raw["agent"] = {
        "planning": {
            "enabled": True,
            "horizon": 1,
            "search_mode": "policy",
            "device": "cpu",
        }
    }
    _configure_planner_policy_ppo(raw)
    raw["value_head"]["ppo_clip_range"] = 0.2

    with pytest.raises(ValueError, match="ordinary critic regression"):
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


def test_rl_config_requires_explicit_state_source() -> None:
    raw = _raw_config()
    del raw["gradient"]["state_source"]
    with pytest.raises(ValueError, match="gradient.state_source must be explicit"):
        parse_rl_config(raw)


def test_rl_config_rejects_rollout_states_with_backbone_gradient() -> None:
    raw = _raw_config()
    raw["gradient"]["state_source"] = "rollout"
    with pytest.raises(ValueError, match="requires gradient.state_source=recompute"):
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
