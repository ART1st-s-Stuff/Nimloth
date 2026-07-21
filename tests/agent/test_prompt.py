from __future__ import annotations

import pytest

from nimloth.agent import (
    AgentTranscript,
    NimlothAgentPrompt,
    bind_image_placeholders,
)


def _transcript(*, observations: int, actions: tuple[int, ...]) -> AgentTranscript:
    return AgentTranscript(
        system_prompt="system",
        observation_texts=tuple(f"observation {index}: <image>" for index in range(observations)),
        observation_images=tuple(f"image-{index}" for index in range(observations)),
        action_indices=actions,
    )


def test_policy_and_supervised_turn_share_the_exact_action_prefix() -> None:
    prompt = NimlothAgentPrompt()
    policy_messages = prompt.build_policy_messages(
        _transcript(observations=2, actions=(0,)),
        bind_images=False,
    )
    supervised_messages = prompt.build_supervised_messages(
        _transcript(observations=2, actions=(0, 3)),
        bind_images=False,
    )

    assert policy_messages[:-1] == supervised_messages[:-1]
    assert supervised_messages[-1]["content"].startswith(
        policy_messages[-1]["content"]
    )
    assert supervised_messages[-1]["content"].endswith(
        "<|action_(3)|><|action_end|>"
    )


def test_policy_prompt_binds_each_real_observation_in_order() -> None:
    prompt = NimlothAgentPrompt()
    messages = prompt.build_policy_messages(
        _transcript(observations=3, actions=(0, 4)),
        bind_images=True,
    )
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
    prompt = NimlothAgentPrompt()
    with pytest.raises(ValueError, match="one unacted observation"):
        prompt.build_policy_messages(
            _transcript(observations=1, actions=(0,)),
            bind_images=False,
        )


def test_supervised_response_can_preserve_dataset_thought_text() -> None:
    response = NimlothAgentPrompt().assistant_response(
        4,
        thought="The target is to my right.",
    )
    assert response == (
        "<think>The target is to my right.</think>"
        "<|latent_state|><|action_start|><|action_(4)|><|action_end|>"
    )
