import pytest
import torch

from nimloth.eval.query_cfm_trajectory import prepare_trajectory_rows


def _record(record_id: str, actions: list[int], offset: float) -> dict[int, dict]:
    record = {}
    for step in range(6):
        record[step] = {
            "record_id": record_id,
            "step_index": step,
            "action_index": actions[min(step, 4)],
            "state_emb": torch.full((8, 2048), offset + step),
            "current_image_path": f"/{record_id}/{step}.png",
        }
    return record


def test_prepare_trajectory_rows_validates_actions_and_pairs_wrong_by_horizon() -> None:
    actions_a = [0, 4, 0, 5, 0]
    actions_b = [0, 0, 4, 0, 5]
    caches = {
        "old": {"a": _record("a", actions_a, 10.0)},
        "current": {"b": _record("b", actions_b, 20.0)},
    }
    selections = [
        {"run_index": 0, "source": "old", "record_id": "a", "expected_actions": actions_a},
        {"run_index": 1, "source": "current", "record_id": "b", "expected_actions": actions_b},
    ]
    rows, correct, wrong = prepare_trajectory_rows(selections, caches)
    assert len(rows) == 10
    assert correct.shape == wrong.shape == (10, 8 * 2048)
    assert rows[1]["action_name"] == "turn_right"
    assert rows[3]["action_name"] == "turn_left"
    torch.testing.assert_close(correct[0], torch.full_like(correct[0], 11.0))
    torch.testing.assert_close(wrong[0], torch.full_like(wrong[0], 21.0))
    torch.testing.assert_close(wrong[5], torch.full_like(wrong[5], 11.0))


def test_prepare_trajectory_rows_rejects_action_mismatch() -> None:
    actions = [0, 4, 0, 5, 0]
    caches = {"old": {"a": _record("a", actions, 0.0)}}
    with pytest.raises(ValueError, match="action mismatch"):
        prepare_trajectory_rows(
            [{"run_index": 0, "source": "old", "record_id": "a", "expected_actions": [0] * 5}],
            caches,
        )
