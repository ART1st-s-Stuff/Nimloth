"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from experiments.training.rl.rollout_env import validate_split, validate_trajectories
from nimloth.agent import AgentTranscript, NimlothPromptTemplate
from nimloth.backbone.qwen25vl.policy import validate_agent_policy_protocol
from nimloth.environment.navigation.collector import VAGENNavigationRolloutCollector
from nimloth.rollout import RolloutTrajectory


def _trajectory() -> RolloutTrajectory:
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    system_prompt = "Follow the navigation instruction."
    observation_texts = (
        "Human Instruction: Move near the couch.\n<image>",
        "Feedback: Action completed.\n<image>",
    )
    image_paths = ("before.png", "after.png")
    response = prompt.assistant_response(0, thought="Move toward the couch.")
    policy_messages = prompt.build_response_policy_prompt(
        AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[:1],
            observation_images=image_paths[:1],
            action_indices=(),
        ),
    ).unbound_messages()
    full_transcript = AgentTranscript(
        system_prompt=system_prompt,
        observation_texts=observation_texts,
        observation_images=image_paths,
        action_indices=(0,),
        assistant_responses=(response,),
    )
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=list(image_paths),
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-math.log(8.0)] * 8],
        instruction="Move near the couch.",
        split="train",
        messages=(
            prompt.build_supervised_prompt(full_transcript).unbound_messages()
        ),
        system_prompt=system_prompt,
        observation_texts=list(observation_texts),
        assistant_responses=[response],
        terminal_assistant_prefix=prompt.assistant_prefix(
            thought="Terminal observation."
        ),
        policy_credit_assignment="turn",
        policy_messages=[policy_messages],
        policy_token_ids=[[100, 102, 103]],
        policy_token_log_probs=[[-0.2, -math.log(8.0), None]],
        policy_loss_masks=[[True, True, False]],
        policy_token_roles=[["reasoning", "action", "injected"]],
        policy_action_token_ids=[[102, 202, 203, 204, 205, 206, 207, 208]],
        policy_reasoning_texts=["Move toward the couch."],
        policy_finish_reasons=["stop"],
        policy_reasoning_truncated=[False],
        prompt_template_spec=prompt.spec,
    )


def _turn_trajectory() -> RolloutTrajectory:
    return _trajectory()


def test_rl_policy_protocol_requires_positive_k_inject() -> None:
    assert validate_agent_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    )) == 1
    assert validate_agent_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=16,
        nimloth_latent_query_mode="inject",
    )) == 16
    with pytest.raises(ValueError, match="positive-k inject"):
        validate_agent_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=1,
            nimloth_latent_query_mode="generate",
        ))
    with pytest.raises(ValueError, match="positive-k inject"):
        validate_agent_policy_protocol(SimpleNamespace(
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
    VAGENNavigationRolloutCollector(
        None,
        "http://env",
        eval_sets=("base_train",),
        split="train",
    )
    with pytest.raises(ValueError, match=r"requires \*_train datasets"):
        VAGENNavigationRolloutCollector(
            None,
            "http://env",
            eval_sets=("base",),
            split="train",
        )


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
    trajectory.prompt_template_spec = None
    trajectory.prompt_version = "old-prompt"
    with pytest.raises(RuntimeError, match="unsupported prompt version"):
        validate_trajectories([trajectory])


def test_policy_prompt_must_match_structured_transcript() -> None:
    trajectory = _trajectory()
    trajectory.policy_messages[0][-1]["content"] = "different prompt"
    with pytest.raises(RuntimeError, match="does not match the shared Agent template"):
        validate_trajectories([trajectory])


def test_legacy_latent_count_cannot_drift_from_template_spec() -> None:
    trajectory = _trajectory()
    trajectory.latent_token_count = 8
    with pytest.raises(RuntimeError, match="does not match the prompt template"):
        validate_trajectories([trajectory])


def test_turn_credit_roundtrip_separates_behavior_and_state_prompts() -> None:
    trajectory = _turn_trajectory()

    validate_trajectories([trajectory])
    restored = RolloutTrajectory.from_record(trajectory.to_record())

    assert restored.build_policy_prompt(0).messages[-1]["content"] == "<think>"
    state_prefix = restored.build_state_prompt(0).messages[-1]["content"]
    assert state_prefix.endswith("<|latent_state|><|action_start|>")
    assert "Move toward the couch." in state_prefix
    assert "<|action_(0)|>" not in state_prefix
    terminal_prompt = restored.build_state_prompt(1)
    assert terminal_prompt.messages[-1]["content"] == (
        "<think>Terminal observation.</think><|latent_state|><|action_start|>"
    )
    assert any(
        "Move toward the couch." in str(message["content"])
        for message in terminal_prompt.messages
    )
    assert restored.policy_token_trace(0) == trajectory.policy_token_trace(0)


def test_turn_trace_action_token_must_match_action_index() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_token_ids[0][1] = 202

    with pytest.raises(RuntimeError, match="does not match action_index"):
        validate_trajectories([trajectory])


def test_turn_trace_action_log_prob_must_match_behavior_distribution() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_token_log_probs[0][1] = -0.3

    with pytest.raises(RuntimeError, match="does not match action_log_probs"):
        validate_trajectories([trajectory])


def test_turn_response_must_match_reasoning_and_action_trace() -> None:
    trajectory = _turn_trajectory()
    trajectory.assistant_responses[0] = trajectory.assistant_responses[0].replace(
        "action_(0)",
        "action_(1)",
    )

    with pytest.raises(RuntimeError, match="assistant response does not match"):
        validate_trajectories([trajectory])


def test_reasoning_truncation_metadata_must_be_consistent() -> None:
    trajectory = _turn_trajectory()
    trajectory.policy_reasoning_truncated[0] = True

    with pytest.raises(RuntimeError, match="truncation must match finish_reason"):
        validate_trajectories([trajectory])
