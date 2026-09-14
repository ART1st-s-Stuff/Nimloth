"""The dropped final reward must still contribute to retained action returns."""

import copy

import pytest

from nimloth.rollout.finite_horizon import validated_action_value_targets
from nimloth.rollout.record_format import require_trajectory_record
from nimloth.rollout.transitions import discounted_action_value_targets, expand_record_transitions


def _record():
    return {
        "record_format": "nimloth_trajectory_v1", "id": "test", "split": "train",
        "success": True, "reward": 4.0, "reward_provenance": "step_rewards",
        "rewards": [0.0, 0.0], "terminated": False, "truncated": True,
        "action_indices": [0, 1], "action_space_id": "navigation", "action_space_version": 1,
        "image_paths": ["a.png", "b.png", "c.png"], "system_prompt": "sys",
        "observation_texts": ["<image> a", "<image> b", "<image> c"],
        "assistant_responses": ["a", "b"], "terminal_assistant_prefix": "real thought",
        "action_successes": [True, False], "action_value_targets": [1.0, 2.0],
        "finite_horizon_provenance": {
            "format": "finite_horizon_tail_drop_v1", "original_action_count": 3,
            "max_action_horizon": 3, "gamma": 0.5, "bootstrap": 0.0,
            "bootstrap_reason": "task_horizon", "source_record_sha256": "a" * 64,
            "original_dones": [False, False, False],
            "original_rewards": [0.0, 0.0, 4.0], "removed_reward": 4.0,
            "removed_action_index": 2,
        },
    }


def test_full_reward_return_then_slice_and_transition_alignment():
    record = _record()
    assert discounted_action_value_targets(record, gamma=0.5) == [1.0, 2.0]
    samples = expand_record_transitions(record, value_gamma=0.5)
    assert [(s.action_value_target, s.action_success) for s in samples] == [(1.0, True), (2.0, False)]
    assert samples[-1].next_image_path == "c.png"


@pytest.mark.parametrize("field,value", [
    ("gamma", 1.0), ("bootstrap", 2.0), ("original_action_count", 4),
    ("max_action_horizon", 20), ("removed_reward", 0.0),
    ("source_record_sha256", "bad"), ("original_rewards", [0.0, 0.0]),
    ("bootstrap_reason", "unknown"), ("original_dones", [False, True, False]),
    ("original_dones", [False, False, True]), ("removed_action_index", -1),
])
def test_reject_inconsistent_provenance(field, value):
    record = _record()
    record["finite_horizon_provenance"][field] = value
    with pytest.raises(ValueError):
        validated_action_value_targets(record, gamma=0.5)


@pytest.mark.parametrize("targets", [[0.0, 0.0], [float("nan"), 2.0], [1.0], [True, 2.0]])
def test_reject_unverified_targets(targets):
    record = _record()
    record["action_value_targets"] = targets
    with pytest.raises(ValueError):
        require_trajectory_record(record)


def test_missing_provenance_and_mismatched_consumption_gamma():
    record = _record()
    with pytest.raises(ValueError, match="gamma"):
        discounted_action_value_targets(record, gamma=1.0)
    record.pop("finite_horizon_provenance")
    with pytest.raises(ValueError, match="provenance"):
        discounted_action_value_targets(record, gamma=0.5)


@pytest.mark.parametrize("labels", [[1, 0], [True], [True, None]])
def test_reject_non_boolean_or_misaligned_outcomes(labels):
    record = _record()
    record["action_successes"] = labels
    with pytest.raises(ValueError, match="action_successes"):
        require_trajectory_record(record)


def test_legacy_missing_outcome_is_not_failure():
    record = copy.deepcopy(_record())
    for key in ("action_value_targets", "finite_horizon_provenance", "action_successes"):
        record.pop(key)
    record.update(terminated=True, truncated=False)
    assert [s.action_success for s in expand_record_transitions(record)] == [None, None]
    assert discounted_action_value_targets(record) == [0.0, 0.0]
