from __future__ import annotations

import pytest

from nimloth.environment.navigation import NUM_NAVIGATION_ACTIONS
from nimloth.rollout.record_format import (
    TRAJECTORY_RECORD_FORMAT,
    TRAJECTORY_REWARD_PROVENANCE,
)
from nimloth.rollout.transitions import (
    TransitionSample,
    discounted_action_value_targets,
    expand_record_transitions,
)


def _make_record(num_steps: int = 2) -> dict:
    image_paths = []
    action_indices = []
    observation_texts = []
    assistant_responses = []
    for step in range(num_steps):
        image_paths.append(f"/tmp/img_{step}.png")
        observation_texts.append(f"observe <image> step {step}")
        assistant_responses.append(
            f"<think>t{step}</think><|latent_state|>"
            f"<|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
        )
        action_indices.append(step % NUM_NAVIGATION_ACTIONS)
    image_paths.append(f"/tmp/img_{num_steps}.png")
    observation_texts.append(f"observe <image> step {num_steps}")
    return {
        "record_format": TRAJECTORY_RECORD_FORMAT,
        "id": "train/shard_000/000001",
        "split": "train",
        "success": True,
        "system_prompt": "sys",
        "observation_texts": observation_texts,
        "assistant_responses": assistant_responses,
        "image_paths": image_paths,
        "action_indices": action_indices,
        "action_space_id": "navigation",
        "action_space_version": 1,
        "reward": 1.0,
        "reward_provenance": TRAJECTORY_REWARD_PROVENANCE,
        "terminal_assistant_prefix": (
            "<think>terminal thought</think><|latent_state|><|action_start|>"
        ),
    }


def test_expand_record_transitions_alignment() -> None:
    record = _make_record(num_steps=3)
    transitions = expand_record_transitions(record)
    assert len(transitions) == 3
    t0 = transitions[0]
    assert isinstance(t0, TransitionSample)
    assert t0.step_index == 0
    assert t0.current_image_path == "/tmp/img_0.png"
    assert t0.next_image_path == "/tmp/img_1.png"
    assert t0.action_index == 0
    assert len(t0.prefix_image_paths) == 1
    assert t0.prefix_messages[-1]["role"] == "assistant"
    assert t0.next_prefix_messages is not None
    assert len(t0.next_prefix_image_paths) == 2
    assert t0.action_value_target == pytest.approx(1.0)

    t2 = transitions[2]
    assert t2.current_image_path == "/tmp/img_2.png"
    assert t2.next_image_path == "/tmp/img_3.png"
    assert len(t2.prefix_image_paths) == 3
    assert len(t2.prefix_messages) == 7  # system + 3*(user+assistant)
    assert t2.next_prefix_messages is not None
    assert t2.next_prefix_messages[-2]["role"] == "user"
    assert t2.next_prefix_messages[-1]["role"] == "assistant"
    assert t2.next_prefix_messages[-1]["content"] == (
        "<think>terminal thought</think><|latent_state|><|action_start|>"
    )
    assert t2.next_prefix_image_paths == [
        "/tmp/img_0.png",
        "/tmp/img_1.png",
        "/tmp/img_2.png",
        "/tmp/img_3.png",
    ]


def test_expand_rejects_missing_next_image() -> None:
    record = _make_record(num_steps=1)
    record["image_paths"] = ["/tmp/img_0.png"]
    with pytest.raises(ValueError, match="expected one final image"):
        expand_record_transitions(record)


def test_expand_rejects_invalid_action_index() -> None:
    record = _make_record(num_steps=1)
    record["action_indices"] = [99]
    try:
        expand_record_transitions(record)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_expand_record_transitions_configurable_value_gamma() -> None:
    record = _make_record(num_steps=3)
    transitions = expand_record_transitions(record, value_gamma=0.9)
    assert transitions[0].action_value_target == pytest.approx(0.9 ** 2)
    assert transitions[2].action_value_target == pytest.approx(1.0)


def test_discounted_action_value_targets() -> None:
    record = {
        "action_indices": [0, 1, 2],
        "reward": 1.0,
        "reward_provenance": TRAJECTORY_REWARD_PROVENANCE,
    }
    values = discounted_action_value_targets(record, gamma=0.9)
    assert len(values) == 3
    assert values[0] == pytest.approx(0.9 ** 2)
    assert values[2] == pytest.approx(1.0)


def test_structured_agent_record_uses_shared_prompt_for_sft2_prefixes() -> None:
    record = {
        "record_format": TRAJECTORY_RECORD_FORMAT,
        "id": "structured",
        "split": "train",
        "success": True,
        "reward": 1.0,
        "reward_provenance": TRAJECTORY_REWARD_PROVENANCE,
        "system_prompt": "system",
        "observation_texts": ["first <image>", "second <image>", "final <image>"],
        "image_paths": ["first.png", "second.png", "final.png"],
        "action_indices": [0, 3],
        "action_space_id": "navigation",
        "action_space_version": 1,
        "assistant_responses": [
            (
                "<think>first thought</think><|latent_state|>"
                "<|action_start|><|action_(0)|><|action_end|>"
            ),
            (
                "<think>second thought</think><|latent_state|>"
                "<|action_start|><|action_(3)|><|action_end|>"
            ),
        ],
        "terminal_assistant_prefix": (
            "<think>terminal thought</think><|latent_state|><|action_start|>"
        ),
    }

    transitions = expand_record_transitions(record)
    assert len(transitions) == 2
    assert transitions[1].prefix_messages[-1]["content"].endswith(
        "<|action_(3)|><|action_end|>"
    )
    assert transitions[0].next_prefix_messages is not None
    assert transitions[0].next_prefix_messages[-2]["content"] == "second <image>"
    assert transitions[0].next_prefix_messages[-1]["content"].endswith(
        "<|action_(3)|><|action_end|>"
    )
    assert transitions[0].next_prefix_image_paths == ["first.png", "second.png"]

    assert transitions[1].next_prefix_messages is not None
    assert transitions[1].next_prefix_messages[-1]["content"] == (
        "<think>terminal thought</think><|latent_state|><|action_start|>"
    )


def test_expand_requires_persisted_terminal_cot() -> None:
    record = _make_record(num_steps=1)
    record["terminal_assistant_prefix"] = ""
    with pytest.raises(ValueError, match="generate terminal CoT"):
        expand_record_transitions(record)
