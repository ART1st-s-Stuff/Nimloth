"""Regression tests for the task-text and reward protocol used by dynamic RL."""

from __future__ import annotations

from PIL import Image
import pytest

from nimloth.training.rl.vagen_protocol import (
    ACTION_NAMES,
    extract_human_instruction,
    nimloth_assistant_response,
    normalize_vagen_policy_image,
    observation_text_and_image,
    source_eval_text_to_nimloth,
    task_succeeded,
    trajectory_messages,
    vagen_env_response,
)


SYSTEM_PROMPT = """You are a home robot.
Actions include move_forward and turn_right.
You can optionally think first, then give your action. Respond in this format:
<think>...</think><action>some_action</action>"""
INITIAL_OBSERVATION = """[Initial Observation]:
<image>
Human Instruction: Navigate to a couch and move near to the couch.
Decide your next action(s)."""
STEP_OBSERVATION = """After your action, the extracted valid action is move_forward.
The environment feedback is: Last action is executed successfully.
After that, the observation is:
<image>
Decide your next action(s)."""


def test_source_eval_text_conversion_is_the_same_protocol_used_by_sft() -> None:
    converted_system = source_eval_text_to_nimloth(
        SYSTEM_PROMPT, latent_token_count=8
    )
    converted_initial = source_eval_text_to_nimloth(
        INITIAL_OBSERVATION, latent_token_count=8
    )
    assert "<action>" not in converted_system
    assert "Actions include move_forward and turn_right." in converted_system
    assert "<|latent_state_7|><|action_start|><|action_(idx)|><|action_end|>" in converted_system
    assert converted_initial == INITIAL_OBSERVATION
    assert extract_human_instruction(converted_initial) == (
        "Navigate to a couch and move near to the couch."
    )


def test_sft_converter_delegates_to_the_shared_runtime_rewrite() -> None:
    import re

    from experiments.training.sft1.convert_rollouts import (
        LATENT_STATE_BLOCK,
        rewrite_prompt_instruction,
    )

    latent_count = len(re.findall(
        r"<\|latent_state(?:_\d+)?\|>", LATENT_STATE_BLOCK
    ))
    assert rewrite_prompt_instruction(SYSTEM_PROMPT) == source_eval_text_to_nimloth(
        SYSTEM_PROMPT, latent_token_count=latent_count
    )


def test_vagen_policy_image_normalization_matches_source_rollout_manager() -> None:
    # VAGEN's pinned `verl.utils.dataset.rl_dataset.process_image` upscales
    # raw AI2-THOR frames from255×255 to512×512 before policy use.
    raw = Image.new("RGBA", (255, 255), (1, 2, 3, 255))
    normalized = normalize_vagen_policy_image(raw)
    assert normalized.size == (512, 512)
    assert normalized.mode == "RGB"
    assert normalized.getpixel((0, 0)) == (1, 2, 3)

    source_sized = Image.new("RGB", (512, 512), "black")
    assert normalize_vagen_policy_image(source_sized) is source_sized


def test_observation_extraction_requires_real_obs_str_and_one_image() -> None:
    image = Image.new("RGB", (255, 255), "black")
    text, extracted = observation_text_and_image({
        "obs_str": INITIAL_OBSERVATION,
        "multi_modal_data": {"<image>": [image]},
    }, latent_token_count=8)
    assert text == INITIAL_OBSERVATION
    assert extracted.size == (512, 512)
    assert extracted.mode == "RGB"

    with pytest.raises(ValueError, match="obs_str"):
        observation_text_and_image(
            {"multi_modal_data": {"<image>": [image]}}, latent_token_count=8
        )
    with pytest.raises(ValueError, match="exactly one"):
        observation_text_and_image({
            "obs_str": INITIAL_OBSERVATION,
            "multi_modal_data": {"<image>": []},
        }, latent_token_count=8)


def test_policy_history_replays_environment_text_and_generated_assistant_verbatim() -> None:
    system = source_eval_text_to_nimloth(
        SYSTEM_PROMPT, latent_token_count=8
    )
    observations = [
        source_eval_text_to_nimloth(text, latent_token_count=8)
        for text in (INITIAL_OBSERVATION, STEP_OBSERVATION)
    ]
    assistant = nimloth_assistant_response(
        "<think>I should move closer.</think>", 0, latent_token_count=8
    )
    messages = trajectory_messages(
        system,
        observations,
        [assistant],
        history_window=112,
    )
    assert messages == [
        {"role": "system", "content": system},
        {"role": "user", "content": observations[0]},
        {"role": "assistant", "content": assistant},
        {"role": "user", "content": observations[1]},
    ]
    rendered = "\n".join(message["content"] for message in messages)
    assert "Human Instruction:" in rendered
    assert "environment feedback" in rendered
    assert "Observe the scene" not in rendered
    assert "What should I do next?" not in rendered
    assert "<action>some_action</action>" not in rendered


def test_full_source_history_is_not_arbitrarily_truncated() -> None:
    observations = [f"<image>\nobservation-{index}" for index in range(6)]
    responses = [
        nimloth_assistant_response(
            f"<think>thought-{index}</think>", index % 8, latent_token_count=8
        )
        for index in range(5)
    ]
    messages = trajectory_messages(
        "system", observations, responses, history_window=112
    )
    assert len(messages) == 1 + 2 * 5 + 1
    assert messages[1]["content"] == observations[0]
    assert messages[-1]["content"] == observations[-1]


def test_environment_action_uses_generated_thought_and_canonical_vagen_name() -> None:
    thought = "<think>Need to turn toward the target.</think>"
    assert ACTION_NAMES[4] == "turn_right"
    response = vagen_env_response(thought, 4)
    assert response == f"{thought}<action>turn_right</action>"

    from vagen.env.navigation.prompt import SOURCE_EVAL_MODE
    from vagen.env.utils.parse_utils import PARSE_FUNC_MAP

    parsed = PARSE_FUNC_MAP[SOURCE_EVAL_MODE](
        response, action_sep="|", max_actions=1
    )
    assert parsed["format_correct"] is True
    assert parsed["actions"] == ["turn_right"]


def test_success_comes_from_explicit_vagen_info_not_reward_magnitude() -> None:
    assert task_succeeded({"task_success": True}) is True
    assert task_succeeded({"task_success": False}) is False
    assert task_succeeded({
        "metrics": {"traj_metrics": {"success": True}}
    }) is True
    assert task_succeeded({"reward": 10.0}) is False
