from __future__ import annotations

import pytest

from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS, TransitionSample, discounted_action_value_targets, expand_record_transitions


def _make_record(num_steps: int = 2) -> dict:
    messages = [{"role": "system", "content": "sys"}]
    image_paths = []
    action_indices = []
    for step in range(num_steps):
        image_paths.append(f"/tmp/img_{step}.png")
        messages.append({"role": "user", "content": f"observe <image> step {step}"})
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"<think>t{step}</think><|latent_state|>"
                    f"<|action_start|><|action_({step % NUM_NAVIGATION_ACTIONS})|><|action_end|>"
                ),
            }
        )
        action_indices.append(step % NUM_NAVIGATION_ACTIONS)
    image_paths.append(f"/tmp/img_{num_steps}.png")
    return {
        "id": "train/shard_000/000001",
        "split": "train",
        "success": True,
        "messages": messages,
        "image_paths": image_paths,
        "action_indices": action_indices,
        "reward": 1.0,
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
    assert t2.next_prefix_messages is None


def test_expand_skips_when_no_next_image() -> None:
    record = _make_record(num_steps=1)
    record["image_paths"] = ["/tmp/img_0.png"]
    assert expand_record_transitions(record) == []


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
    record = {"action_indices": [0, 1, 2], "reward": 1.0}
    values = discounted_action_value_targets(record, gamma=0.9)
    assert len(values) == 3
    assert values[0] == pytest.approx(0.9 ** 2)
    assert values[2] == pytest.approx(1.0)
