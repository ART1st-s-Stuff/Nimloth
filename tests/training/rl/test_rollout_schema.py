"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

from types import SimpleNamespace
import math

import pytest

from experiments.training.rl.rollout_env import validate_split, validate_trajectories
from nimloth.training.rl.rollout import (
    EnvRolloutCollector,
    RolloutTrajectory,
    validate_rl_policy_protocol,
)
from nimloth.training.rl.vagen_protocol import nimloth_assistant_response


INITIAL = """[Initial Observation]:
<image>
Human Instruction: Move near the couch.
Decide your next action(s)."""
NEXT = """After your action, the extracted valid action is move_forward.
The environment feedback is: Last action is executed successfully.
After that, the observation is:
<image>
Decide your next action(s)."""


def _trajectory() -> RolloutTrajectory:
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=["before.png", "after.png"],
        observation_texts=[INITIAL, NEXT],
        task_instruction="Move near the couch.",
        system_prompt="Navigation system prompt.",
        assistant_responses=[nimloth_assistant_response(
            "<think>Move toward the couch.</think>", 0, latent_token_count=8
        )],
        action_indices=[0],
        action_names=["move_forward"],
        action_log_probs=[[-math.log(8.0)] * 8],
        thought_token_ids=[[10, 11, 12]],
        thought_token_log_probs=[[-0.2, -0.3, -0.1]],
        step_rewards=[0.01],
        final_reward=0.0,
        success=False,
        reward=0.01,
        split="train",
        latent_token_count=8,
    )


def test_rl_policy_protocol_requires_inject_queries() -> None:
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    )) == 1
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    )) == 8
    with pytest.raises(ValueError, match="requires an inject checkpoint"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=1,
            nimloth_latent_query_mode="generate",
        ))
    with pytest.raises(ValueError, match="at least one latent query"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=0,
            nimloth_latent_query_mode="inject",
        ))


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
    EnvRolloutCollector(None, None, "http://env", None,
                        eval_sets=("base",), split="validation")
    with pytest.raises(ValueError, match=r"forbids \*_train datasets"):
        EnvRolloutCollector(None, None, "http://env", None,
                            eval_sets=("base_train",), split="validation")


def test_environment_config_matches_pinned_source_eval_reference() -> None:
    collector = EnvRolloutCollector(
        None, None, "http://env", None,
        eval_sets=("base_train",), split="train",
    )
    config = collector._environment_config("base_train")["env_config"]
    assert config == {
        "render_mode": "vision",
        "prompt_format": "source_eval_mode",
        "use_state_reward": False,
        "eval_set": "base_train",
        "max_actions_per_step": 1,
        "action_sep": "|",
        "example_count": 0,
        "format_reward": 0.0,
        "per_turn_format_reward": 0.01,
        "success_reward": 1.0,
        "success_threshold": 1.0,
        "step_length": 0.3,
        "grounding_reward_weight": 0.5,
        "worldmodeling_reward_weight": 0.5,
        "gpu_device": 0,
    }


def test_complete_trajectory_schema_passes() -> None:
    trajectory = _trajectory()
    validate_trajectories([trajectory])
    record = trajectory.to_record()
    assert record["observation_texts"] == [INITIAL, NEXT]
    assert record["task_instruction"] == "Move near the couch."
    assert record["step_rewards"] == [0.01]
    assert record["assistant_responses"] == trajectory.assistant_responses
    assert record["thought_token_ids"] == [[10, 11, 12]]
    assert record["thought_token_log_probs"] == [[-0.2, -0.3, -0.1]]
    assert [message["role"] for message in record["messages"]] == [
        "system", "user", "assistant"
    ]
    assert "nav_instruction" not in record


def test_legacy_taskless_record_is_rejected() -> None:
    record = _trajectory().to_record()
    for key in (
        "task_instruction", "observation_texts", "assistant_responses",
        "thought_token_ids", "thought_token_log_probs", "step_rewards"
    ):
        record.pop(key)
    with pytest.raises(ValueError, match="legacy/taskless"):
        RolloutTrajectory.from_record(record)


def test_missing_final_observation_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.image_paths.pop()
    with pytest.raises(RuntimeError, match="images=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_observation_text_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.observation_texts.pop()
    with pytest.raises(RuntimeError, match="observation_texts=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_step_reward_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.step_rewards.clear()
    with pytest.raises(RuntimeError, match="step_rewards=0 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_action_log_probs_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs.clear()
    with pytest.raises(RuntimeError, match="action_log_probs=0 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_thought_token_trace_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.thought_token_ids.clear()
    with pytest.raises(RuntimeError, match="thought_token_ids=0 but actions=1"):
        validate_trajectories([trajectory])
