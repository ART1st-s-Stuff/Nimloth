import pytest
import torch

from nimloth.eval.query_vs_qwen_trajectory import (
    _metric_rows,
    prepare_comparison_rows,
)


def _record(record_id: str, actions: list[int], shape: tuple[int, int]) -> dict[int, dict]:
    return {
        step: {
            "record_id": record_id,
            "step_index": step,
            "action_index": actions[min(step, 4)],
            "state_emb": torch.full(shape, float(step)),
            "current_image_path": f"/{record_id}/{step}.png",
        }
        for step in range(6)
    }


def test_prepare_comparison_rows_aligns_query_and_qwen_conditions() -> None:
    actions = [0, 4, 0, 5, 0]
    selections = [
        {
            "run_index": 7,
            "candidate_index": 59,
            "scene_note": "blue window",
            "record_id": "record",
            "expected_actions": actions,
        }
    ]
    rows, query, qwen = prepare_comparison_rows(
        selections,
        {"record": _record("record", actions, (8, 2048))},
        {"record": _record("record", actions, (16, 512))},
    )
    assert len(rows) == 5
    assert query.shape == (5, 8 * 2048)
    assert qwen.shape == (5, 16 * 512)
    assert rows[1]["action_name"] == "turn_right"
    assert rows[3]["action_name"] == "turn_left"
    assert rows[0]["scene_note"] == "blue window"


def test_prepare_comparison_rows_rejects_image_misalignment() -> None:
    actions = [0] * 5
    query = _record("record", actions, (8, 2048))
    qwen = _record("record", actions, (16, 512))
    qwen[3]["current_image_path"] = "/wrong.png"
    with pytest.raises(ValueError, match="image mismatch"):
        prepare_comparison_rows(
            [{"run_index": 0, "record_id": "record", "expected_actions": actions}],
            {"record": query},
            {"record": qwen},
        )


def test_metric_rows_reports_branch_comparison_and_actions() -> None:
    rows = [{"action_name": "move_forward"}, {"action_name": "turn_left"}]
    gt = torch.zeros(2, 3, 2, 2)
    qwen = torch.stack([torch.full((3, 2, 2), 0.4), torch.full((3, 2, 2), 0.3)])
    query = torch.stack([torch.full((3, 2, 2), 0.2), torch.full((3, 2, 2), 0.6)])
    metrics, by_action = _metric_rows(rows, gt, qwen, query)
    assert metrics["trajectory/query_better_frame_fraction"] == 0.5
    assert by_action["move_forward"]["query_over_qwen"] == pytest.approx(0.5)
    assert by_action["turn_left"]["query_over_qwen"] == pytest.approx(2.0)
