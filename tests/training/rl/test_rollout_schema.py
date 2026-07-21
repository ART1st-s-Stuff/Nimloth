"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from experiments.training.rl.rollout_env import validate_split, validate_trajectories
from nimloth.agent import AgentTranscript, NimlothAgentPrompt
from nimloth.training.rl.rollout import (
    EnvRolloutCollector,
    RolloutTrajectory,
    validate_rl_policy_protocol,
)


def _trajectory() -> RolloutTrajectory:
    prompt = NimlothAgentPrompt()
    system_prompt = "Follow the navigation instruction."
    observation_texts = (
        "Human Instruction: Move near the couch.\n<image>",
        "Feedback: Action completed.\n<image>",
    )
    image_paths = ("before.png", "after.png")
    policy_messages = prompt.build_policy_messages(
        AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[:1],
            observation_images=image_paths[:1],
            action_indices=(),
        ),
        bind_images=False,
    )
    full_transcript = AgentTranscript(
        system_prompt=system_prompt,
        observation_texts=observation_texts,
        observation_images=image_paths,
        action_indices=(0,),
    )
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=list(image_paths),
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-math.log(8.0)] * 8],
        nav_instruction="Move near the couch.",
        split="train",
        messages=prompt.build_supervised_messages(
            full_transcript,
            bind_images=False,
        ),
        system_prompt=system_prompt,
        observation_texts=list(observation_texts),
        policy_messages=[policy_messages],
    )


def test_rl_policy_protocol_requires_k1_inject() -> None:
    validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    ))
    with pytest.raises(ValueError, match="k=1 inject"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=8,
            nimloth_latent_query_mode="inject",
        ))
    with pytest.raises(ValueError, match="k=1 inject"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=1,
            nimloth_latent_query_mode="generate",
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


def test_complete_trajectory_schema_passes() -> None:
    validate_trajectories([_trajectory()])


def test_missing_final_observation_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.image_paths.pop()
    with pytest.raises(RuntimeError, match="images=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_action_log_probs_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs.clear()
    with pytest.raises(RuntimeError, match="action_log_probs=0 but actions=1"):
        validate_trajectories([trajectory])


def test_non_normalized_action_log_probs_are_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs[0] = [-2.0] * 8
    with pytest.raises(RuntimeError, match="invalid action probabilities"):
        validate_trajectories([trajectory])


def test_missing_policy_prompt_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.policy_messages.clear()
    with pytest.raises(RuntimeError, match="policy_messages=0 but actions=1"):
        validate_trajectories([trajectory])


def test_stale_prompt_version_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.prompt_version = "old-prompt"
    with pytest.raises(RuntimeError, match="unsupported prompt version"):
        validate_trajectories([trajectory])


def test_policy_prompt_must_match_structured_transcript() -> None:
    trajectory = _trajectory()
    trajectory.policy_messages[0][-1]["content"] = "different prompt"
    with pytest.raises(RuntimeError, match="does not match the shared Agent template"):
        validate_trajectories([trajectory])
