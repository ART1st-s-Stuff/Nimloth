import pytest

from nimloth.eval.wm_predictor_teacher_forced import build_transition_pairs


def test_build_transition_pairs_respects_records_gaps_and_actions() -> None:
    rows = [
        {"record_id": "a", "step_index": 1, "action_index": 4},
        {"record_id": "b", "step_index": 0, "action_index": 5},
        {"record_id": "a", "step_index": 0, "action_index": 0},
        {"record_id": "b", "step_index": 2, "action_index": 7},
        {"record_id": "a", "step_index": 2, "action_index": 1},
    ]
    assert build_transition_pairs(rows) == [(2, 0, 0), (0, 4, 4)]


def test_build_transition_pairs_rejects_duplicate_steps() -> None:
    rows = [
        {"record_id": "a", "step_index": 0, "action_index": 0},
        {"record_id": "a", "step_index": 0, "action_index": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        build_transition_pairs(rows)
