from __future__ import annotations

import pytest

from nimloth.agent import (
    AgentTranscript,
    NimlothPromptTemplate,
    PromptTemplateSpec,
    bind_image_placeholders,
    create_prompt_template,
)


def _transcript(*, observations: int, actions: tuple[int, ...]) -> AgentTranscript:
    return AgentTranscript(
        system_prompt="system",
        observation_texts=tuple(f"observation {index}: <image>" for index in range(observations)),
        observation_images=tuple(f"image-{index}" for index in range(observations)),
        action_indices=actions,
    )


def _template() -> NimlothPromptTemplate:
    return NimlothPromptTemplate(latent_token_count=1, action_count=8)


def test_response_policy_and_supervised_turn_share_the_reasoning_prefix() -> None:
    prompt = _template()
    policy_messages = prompt.build_response_policy_prompt(
        AgentTranscript(
            system_prompt="system",
            observation_texts=("observation 0: <image>", "observation 1: <image>"),
            observation_images=("image-0", "image-1"),
            action_indices=(0,),
            assistant_responses=(
                prompt.assistant_response(0, thought="Move forward."),
            ),
        )
    ).unbound_messages()
    response = prompt.assistant_response(3, thought="Inspect the room.")
    supervised_messages = prompt.build_supervised_prompt(
        AgentTranscript(
            system_prompt="system",
            observation_texts=("observation 0: <image>", "observation 1: <image>"),
            observation_images=("image-0", "image-1"),
            action_indices=(0, 3),
            assistant_responses=(
                prompt.assistant_response(0, thought="Move forward."),
                response,
            ),
        )
    ).unbound_messages()

    assert policy_messages[:-1] == supervised_messages[:-1]
    assert supervised_messages[-1]["content"].startswith(
        policy_messages[-1]["content"]
    )
    assert supervised_messages[-1]["content"].endswith(
        "<|action_(3)|><|action_end|>"
    )


def test_response_policy_prompt_binds_each_real_observation_in_order() -> None:
    prompt = _template()
    transcript = AgentTranscript(
        system_prompt="system",
        observation_texts=tuple(f"observation {index}: <image>" for index in range(3)),
        observation_images=tuple(f"image-{index}" for index in range(3)),
        action_indices=(0, 4),
        assistant_responses=(
            prompt.assistant_response(0, thought="First."),
            prompt.assistant_response(4, thought="Second."),
        ),
    )
    messages = prompt.build_response_policy_prompt(
        transcript
    ).bound_messages()
    bound_images = [
        part["image"]
        for message in messages
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image"
    ]
    assert bound_images == ["image-0", "image-1", "image-2"]


def test_bind_image_placeholders_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError, match="more image placeholders"):
        bind_image_placeholders(
            [{"role": "user", "content": "<image> then <image>"}],
            ["only-one"],
        )
    with pytest.raises(ValueError, match="1 images were provided"):
        bind_image_placeholders(
            [{"role": "user", "content": "no image"}],
            ["unused"],
        )


def test_policy_prefix_requires_one_unacted_observation() -> None:
    prompt = _template()
    with pytest.raises(ValueError, match="one unacted observation"):
        prompt.build_response_policy_prompt(
            _transcript(observations=1, actions=(0,))
        )


def test_action_only_and_fixed_state_prompts_are_rejected() -> None:
    transcript = _transcript(observations=1, actions=())
    with pytest.raises(RuntimeError, match="fixed CoT"):
        _template().build_policy_prompt(transcript)
    with pytest.raises(RuntimeError, match="persisted real CoT"):
        _template().build_state_prompt(transcript)


def test_supervised_response_can_preserve_dataset_thought_text() -> None:
    response = _template().assistant_response(
        4,
        thought="The target is to my right.",
    )
    assert response == (
        "<think>The target is to my right.</think>"
        "<|latent_state|><|action_start|><|action_(4)|><|action_end|>"
    )


def test_prompt_template_spec_rebuilds_the_same_template() -> None:
    original = NimlothPromptTemplate(
        latent_token_count=3,
        action_count=8,
    )

    restored = create_prompt_template(original.spec, action_count=8)
    assert restored.spec == original.spec
    assert restored.assistant_prefix(thought="Inspect the next observation.") == (
        original.assistant_prefix(thought="Inspect the next observation.")
    )


def test_prompt_registry_rejects_unknown_template_and_version() -> None:
    with pytest.raises(ValueError, match="unknown prompt template"):
        create_prompt_template(
            PromptTemplateSpec(identifier="missing", version="v1"),
            action_count=8,
        )
    with pytest.raises(ValueError, match="unsupported prompt version"):
        create_prompt_template(
            PromptTemplateSpec(
                identifier="nimloth-latent-action",
                version="old",
            ),
            action_count=8,
        )


def test_transcript_rejects_multiple_unacted_observations() -> None:
    with pytest.raises(ValueError, match="at most one unacted observation"):
        _transcript(observations=2, actions=())
