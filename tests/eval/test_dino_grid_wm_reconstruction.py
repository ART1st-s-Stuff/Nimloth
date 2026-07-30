import json
from pathlib import Path

import pytest
import torch

from nimloth.eval.dino_grid_wm_reconstruction import (
    build_parser,
    calculate_metrics,
    prepare_protocol_rows,
)
from nimloth.rollout.transitions import TransitionSample


def _sample(record_id: str, step: int, action: int) -> TransitionSample:
    return TransitionSample(
        record_id=record_id,
        step_index=step,
        prefix_messages=[],
        prefix_image_paths=[],
        action_index=action,
        current_image_path=f"/images/{record_id}_{step}.png",
        next_image_path=f"/images/{record_id}_{step + 1}.png",
    )


def _cache_row(
    record_id: str,
    step: int,
    action: int,
    shape: tuple[int, ...],
) -> dict[str, object]:
    return {
        "id": f"{record_id}:{step}",
        "record_id": record_id,
        "step_index": step,
        "action_index": action,
        "current_image_path": f"/images/{record_id}_{step}.png",
        "state_emb": torch.full(shape, float(step)),
    }


def test_prepare_protocol_rows_aligns_t_plus_one_through_four() -> None:
    record_id = "trajectory-a"
    actions = [0, 4, 2, 6, 1]
    current = {
        record_id: {
            step: _sample(record_id, step, actions[min(step, len(actions) - 1)])
            for step in range(5)
        }
    }
    old_grid = {
        record_id: {
            step: _cache_row(
                record_id,
                step,
                actions[min(step, len(actions) - 1)],
                (16, 1024),
            )
            for step in range(5)
        }
    }
    qwen = {
        record_id: {
            step: _cache_row(
                record_id,
                step,
                actions[min(step, len(actions) - 1)],
                (16, 512),
            )
            for step in range(5)
        }
    }

    rows, trajectories, states = prepare_protocol_rows(
        selections=[
            {
                "run_index": 3,
                "record_id": record_id,
                "expected_actions": actions,
                "scene_note": "test",
            }
        ],
        current_samples=current,
        old_grid_records=old_grid,
        qwen_records=qwen,
        horizon=4,
    )

    assert [row["horizon"] for row in rows] == [1, 2, 3, 4]
    assert [row["action_prefix"] for row in rows] == [
        actions[:1],
        actions[:2],
        actions[:3],
        actions[:4],
    ]
    assert [sample.step_index for sample in trajectories[0]] == [0, 1, 2, 3, 4]
    assert states["old_grid"].shape == (4, 16, 1024)
    assert states["qwen"].shape == (4, 16, 512)


def test_prepare_protocol_rows_rejects_action_mismatch() -> None:
    record_id = "trajectory-a"
    actions = [0, 1, 2, 3]
    current = {
        record_id: {
            step: _sample(record_id, step, actions[min(step, 3)])
            for step in range(5)
        }
    }
    old_grid = {
        record_id: {
            step: _cache_row(record_id, step, actions[min(step, 3)], (16, 1024))
            for step in range(5)
        }
    }
    qwen = {
        record_id: {
            step: _cache_row(record_id, step, actions[min(step, 3)], (16, 512))
            for step in range(5)
        }
    }
    qwen[record_id][2]["action_index"] = 7

    with pytest.raises(ValueError, match="action mismatch"):
        prepare_protocol_rows(
            selections=[
                {
                    "run_index": 0,
                    "record_id": record_id,
                    "expected_actions": actions,
                }
            ],
            current_samples=current,
            old_grid_records=old_grid,
            qwen_records=qwen,
            horizon=4,
        )


def test_metrics_preserve_horizon_units_and_matched_outputs() -> None:
    rows = [
        {"horizon": 1},
        {"horizon": 2},
        {"horizon": 1},
        {"horizon": 2},
    ]
    actual = torch.zeros(4, 2, 3)
    predicted = actual.clone()
    predicted[1::2] = 1.0
    old = torch.full_like(actual, 0.5)
    gt = torch.zeros(4, 3, 2, 2)
    images = {
        "qwen": torch.zeros_like(gt),
        "old_dino_grid": torch.zeros_like(gt),
        "id56_actual": torch.zeros_like(gt),
        "id56_predicted": torch.zeros_like(gt),
    }
    images["id56_predicted"][1::2] = 1.0

    metrics, horizons = calculate_metrics(
        rows=rows,
        states={
            "old_grid": old,
            "qwen": torch.zeros(4, 2, 2),
            "id56_actual": actual,
            "id56_predicted": predicted,
        },
        images=images,
        gt=gt,
    )

    assert metrics["state/predicted_to_actual_mse"] == pytest.approx(0.5)
    assert horizons["1"]["state_predicted_to_actual_mse"] == pytest.approx(0.0)
    assert horizons["2"]["state_predicted_to_actual_mse"] == pytest.approx(1.0)
    assert horizons["1"]["image_id56_predicted_to_gt_l1"] == pytest.approx(0.0)
    assert horizons["2"]["image_id56_predicted_to_gt_l1"] == pytest.approx(1.0)


def test_diverse40_selection_is_forty_five_action_runs() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs/eval/reconstruction/id56_wm_predicted_diverse40.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selections = payload["selections"]
    assert len(selections) == 40
    assert [selection["run_index"] for selection in selections] == list(range(40))
    assert all(len(selection["expected_actions"]) == 5 for selection in selections)


def test_cli_requires_exact_git_commit() -> None:
    parser = build_parser()
    required = {
        action.dest
        for action in parser._actions
        if getattr(action, "required", False)
    }
    assert "git_commit" in required
