"""Reward placement regression tests against VAGEN multi-turn semantics."""

from __future__ import annotations

import pytest

from nimloth.wm.dataset import discounted_action_value_targets


def test_step_rewards_are_discounted_from_the_turn_where_vagen_emitted_them() -> None:
    record = {
        "action_indices": [0, 4, 0],
        "step_rewards": [0.01, 0.01, 1.01],
        "final_reward": 0.0,
        "reward": 1.03,
    }
    assert discounted_action_value_targets(record, gamma=1.0) == pytest.approx([
        1.03, 1.02, 1.01
    ])
    assert discounted_action_value_targets(record, gamma=0.5) == pytest.approx([
        0.2675, 0.515, 1.01
    ])


def test_final_environment_reward_is_added_only_to_last_response() -> None:
    record = {
        "action_indices": [0, 4],
        "step_rewards": [0.01, 0.01],
        "final_reward": 0.5,
        "reward": 0.52,
    }
    assert discounted_action_value_targets(record, gamma=1.0) == pytest.approx([
        0.52, 0.51
    ])


def test_misaligned_step_rewards_fail_instead_of_becoming_terminal_credit() -> None:
    with pytest.raises(ValueError, match="step_rewards has length 1"):
        discounted_action_value_targets({
            "action_indices": [0, 1],
            "step_rewards": [1.0],
        })
