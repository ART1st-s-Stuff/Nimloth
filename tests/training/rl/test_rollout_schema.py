"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from experiments.training.rl.rollout_env import validate_split, validate_trajectories
from nimloth.training.rl.rollout import (
    EnvRolloutCollector,
    RolloutTrajectory,
    _build_nimloth_policy_messages,
    build_injected_query_prefix,
    qwen_hidden_size_from_config,
    validate_rl_policy_protocol,
)


def _trajectory() -> RolloutTrajectory:
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=["before.png", "after.png"],
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-2.0] * 8],
        action_log_prob_semantics="sampling_distribution_v1",
        nav_instruction="Move near the couch.",
        split="train",
    )


def test_rl_policy_protocol_accepts_any_positive_k_inject() -> None:
    k1 = validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    ))
    k8 = validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    ))
    assert k1.latent_token_count == 1
    assert k8.latent_token_count == 8
    assert k8.latent_query_mode == "inject"


def test_rl_policy_protocol_rejects_generate_and_invalid_k() -> None:
    with pytest.raises(ValueError, match="inject"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=8,
            nimloth_latent_query_mode="generate",
        ))
    with pytest.raises(ValueError, match="positive"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=0,
            nimloth_latent_query_mode="inject",
        ))


def test_qwen_hidden_size_supports_flat_and_nested_configs() -> None:
    assert qwen_hidden_size_from_config(SimpleNamespace(hidden_size=2048)) == 2048
    assert qwen_hidden_size_from_config(
        SimpleNamespace(text_config=SimpleNamespace(hidden_size=3584))
    ) == 3584


def test_injected_query_prefix_contains_the_full_ordered_k8_block() -> None:
    prefix = build_injected_query_prefix(8, include_action_start=True)
    assert prefix == (
        "<|latent_state|><|latent_state_1|><|latent_state_2|>"
        "<|latent_state_3|><|latent_state_4|><|latent_state_5|>"
        "<|latent_state_6|><|latent_state_7|><|action_start|>"
    )


def test_policy_prompt_preserves_real_observation_history() -> None:
    messages = _build_nimloth_policy_messages(
        "current.png",
        "Find the couch.",
        ["moveahead", "rotateleft"],
        latent_token_count=8,
        include_action_start=True,
        observation_history=["obs0.png", "obs1.png", "obs2.png"],
    )
    user_images = [
        message["content"][0]["image"]
        for message in messages
        if message["role"] == "user"
    ]
    assert user_images == ["obs0.png", "obs1.png", "obs2.png"]


def test_training_split_requires_training_dataset() -> None:
    validate_split("base_train", "train")
    with pytest.raises(ValueError, match="refusing to label eval dataset"):
        validate_split("base", "train")
    with pytest.raises(ValueError, match="must use --split train"):
        validate_split("base_train", "eval")


def test_env_collector_enforces_training_dataset() -> None:
    EnvRolloutCollector(None, None, "http://env", None,
                        eval_sets=("base_train",), split="train")
    with pytest.raises(ValueError, match=r"requires \*_train datasets"):
        EnvRolloutCollector(None, None, "http://env", None,
                            eval_sets=("base",), split="train")


def test_complete_trajectory_schema_passes() -> None:
    validate_trajectories([_trajectory()])


def test_missing_final_observation_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.image_paths.pop()
    with pytest.raises(RuntimeError, match="images=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_action_log_probs_is_rejected_for_qwen_policy() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs.clear()
    with pytest.raises(RuntimeError, match="action_log_probs=0 but actions=1"):
        validate_trajectories([trajectory])


def test_wm_value_schema_has_no_fake_qwen_log_probs() -> None:
    trajectory = _trajectory()
    trajectory.policy_sources = ["wm_value"]
    trajectory.state_sources = ["qwen_gt"]
    trajectory.fast_path_steps = [0]
    trajectory.action_log_probs = [None]
    trajectory.action_log_prob_semantics = None
    trajectory.rollout_policy = "wm_value"
    trajectory.fast_path_horizon = 2
    trajectory.action_temperature = 0.7
    trajectory.action_top_p = 0.95
    validate_trajectories([trajectory])

    restored = RolloutTrajectory.from_record(trajectory.to_record())
    assert restored.policy_sources == ["wm_value"]
    assert restored.state_sources == ["qwen_gt"]
    assert restored.fast_path_steps == [0]
    assert restored.action_log_probs == [None]
    assert restored.fast_path_horizon == 2
    assert restored.action_temperature == 0.7
    assert restored.action_top_p == 0.95


def test_wm_value_schema_rejects_qwen_behavior_log_probs() -> None:
    trajectory = _trajectory()
    trajectory.policy_sources = ["wm_value"]
    trajectory.state_sources = ["qwen_gt"]
    trajectory.fast_path_steps = [0]
    trajectory.action_log_prob_semantics = None
    trajectory.rollout_policy = "wm_value"
    trajectory.fast_path_horizon = 2
    with pytest.raises(RuntimeError, match="must not carry Qwen behavior log-probs"):
        validate_trajectories([trajectory])


def test_wm_value_schema_rejects_broken_segment_provenance() -> None:
    trajectory = _trajectory()
    trajectory.policy_sources = ["wm_value"]
    trajectory.state_sources = ["wm_predicted"]
    trajectory.fast_path_steps = [1]
    trajectory.action_log_probs = [None]
    trajectory.action_log_prob_semantics = None
    trajectory.rollout_policy = "wm_value"
    trajectory.fast_path_horizon = 2
    with pytest.raises(RuntimeError, match="invalid fast-path metadata"):
        validate_trajectories([trajectory])
