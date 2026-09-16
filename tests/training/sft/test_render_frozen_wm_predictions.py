from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from experiments.training.sft.stage3.render_frozen_wm_predictions import (
    join_cache_images,
    load_image_index,
    predict_selected,
    render_pages,
    select_rows,
)


def _images(tmp_path: Path, identity: str, count: int) -> tuple[Path, ...]:
    paths = []
    for index in range(count):
        path = tmp_path / f"{identity}-{index}.png"
        Image.new("RGB", (12, 10), (index * 20, 10, 30)).save(path)
        paths.append(path)
    return tuple(paths)


def test_image_join_requires_unique_identity_exact_state_count_and_existing_files(
    tmp_path: Path,
) -> None:
    paths = _images(tmp_path, "trajectory", 5)
    jsonl = tmp_path / "eval.jsonl"
    jsonl.write_text(
        json.dumps({
            "id": "trajectory",
            "image_paths": [str(path) for path in paths],
            "action_indices": [0, 1, 2, 3],
        })
        + "\n",
        encoding="utf-8",
    )
    images, count = load_image_index(jsonl)
    cache = SimpleNamespace(
        records=({"trajectory_id": "trajectory", "state_count": 5, "action_count": 4},)
    )
    joined = join_cache_images(cache, images, prediction_horizon=4)
    assert count == 1
    assert joined[0]["image_paths"][4] == paths[4]

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(jsonl.read_text() * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate eval trajectory"):
        load_image_index(duplicate)

    cache.records = ({"trajectory_id": "trajectory", "state_count": 4, "action_count": 4},)
    with pytest.raises(ValueError, match="image/state count mismatch"):
        join_cache_images(cache, images, prediction_horizon=4)

    extra = dict(images)
    extra["other"] = {"image_paths": paths, "action_count": 4}
    with pytest.raises(ValueError, match="eligible eval JSONL/cache"):
        join_cache_images(cache, extra, prediction_horizon=4)

    extra["other"]["action_count"] = 3
    cache.records = ({"trajectory_id": "trajectory", "state_count": 5, "action_count": 4},)
    assert len(join_cache_images(cache, extra, prediction_horizon=4)) == 1

    paths[-1].unlink()
    cache.records = ({"trajectory_id": "trajectory", "state_count": 5, "action_count": 4},)
    with pytest.raises(FileNotFoundError, match="missing observation images"):
        join_cache_images(cache, images, prediction_horizon=4)


def test_image_join_rejects_cached_action_count_mismatch(tmp_path: Path) -> None:
    paths = _images(tmp_path, "trajectory", 5)
    images = {
        "trajectory": {"image_paths": paths, "action_count": 4},
    }
    cache = SimpleNamespace(
        records=({"trajectory_id": "trajectory", "state_count": 5, "action_count": 5},)
    )
    with pytest.raises(ValueError, match="action count mismatch"):
        join_cache_images(cache, images, prediction_horizon=4)


def test_selection_is_deterministic_trajectory_diverse_and_step_valid(tmp_path: Path) -> None:
    joined = [
        {
            "cache_index": index,
            "trajectory": f"trajectory-{index:02d}",
            "state_count": 7 + index % 3,
            "image_paths": _images(tmp_path, f"t{index}", 7 + index % 3),
        }
        for index in range(12)
    ]
    first = select_rows(joined, horizon=3, limit=8)
    second = select_rows(list(reversed(joined)), horizon=3, limit=8)
    assert [row["trajectory"] for row in first] == [row["trajectory"] for row in second]
    assert len({row["trajectory"] for row in first}) == 8
    assert all(row["future_index"] == row["window_start"] + 3 for row in first)
    assert all(row["future_index"] < row["state_count"] for row in first)


def test_predictions_consume_full_action_prefix_and_take_requested_horizon() -> None:
    class Cache:
        records = ({"trajectory_id": "trajectory", "state_count": 5},)

        @staticmethod
        def load(index: int) -> dict:
            assert index == 0
            return {
                "states": torch.zeros(5, 4, 3),
                "dino": torch.arange(5, dtype=torch.float32)[:, None, None].expand(5, 4, 3),
                "actions": torch.tensor([1, 2, 3, 4]),
            }

    class Predictor:
        def __init__(self) -> None:
            self.actions = []

        def rollout_from_history(self, state, previous_actions, future_actions):
            assert state.shape == (1, 1, 4, 3)
            assert previous_actions.shape == (1, 0)
            self.actions.append(future_actions.clone())
            horizon = future_actions.shape[1]
            return torch.arange(1, horizon + 1, device=state.device)[:, None, None].expand(
                horizon, 4, 3
            )[None]

    selected = [{
        "cache_index": 0,
        "trajectory": "trajectory",
        "state_count": 5,
        "image_paths": (),
        "window_start": 1,
        "horizon_step": 3,
        "future_index": 4,
    }]
    left, right = Predictor(), Predictor()
    rows = predict_selected(
        Cache(), selected,
        stage2_predictor=left,
        dino_predictor=right,
        device=torch.device("cpu"),
    )
    assert torch.equal(left.actions[0], torch.tensor([[2, 3, 4]]))
    assert torch.equal(right.actions[0], torch.tensor([[2, 3, 4]]))
    assert torch.all(rows[0]["stage2_state_wm"] == 3)
    assert torch.all(rows[0]["dino_wm"] == 3)
    assert torch.all(rows[0]["target"] == 4)


def test_render_contract_uses_two_raw_frames_nearest_grid_and_shared_error_scale(
    tmp_path: Path,
) -> None:
    paths = _images(tmp_path, "render", 4)
    rows = []
    for horizon in (1, 2):
        target = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) / 10
        rows.append(
            {
                "trajectory": "render",
                "window_start": 0,
                "future_index": horizon,
                "horizon_step": horizon,
                "image_paths": paths,
                "actions": list(range(horizon)),
                "target": target,
                "current_dino_copy": target + 0.1,
                "stage2_state_wm": target + 0.2,
                "dino_wm": target + 0.3,
            }
        )
    files = render_pages(
        rows,
        mean=np.zeros(3),
        basis=np.eye(3),
        low=np.zeros(3),
        high=np.ones(3),
        error_high=1.0,
        output=tmp_path,
    )
    assert files == ["horizon_1.png", "horizon_2.png"]
    for name in files:
        with Image.open(tmp_path / name) as image:
            assert image.size == (8 * 128, 42 + 160)


@pytest.mark.parametrize("kind", [None, "direct", "residual"])
def test_checkpoint_loader_dispatches_predictor_kind(tmp_path: Path, kind: str | None) -> None:
    from dataclasses import asdict
    from experiments.training.sft.stage3.frozen_wm_diagnostic import PREDICTOR_TYPES
    from experiments.training.sft.stage3.render_frozen_wm_predictions import _load_predictor
    from nimloth.wm.grid import GridPredictorConfig

    config = GridPredictorConfig(
        grid_tokens=4, emb_dim=8, action_dim=8, history_size=1,
        depth=1, heads=2, dim_head=4, mlp_dim=16, dropout=0.0,
    )
    model_type = PREDICTOR_TYPES[kind or "direct"]
    run = {
        "schema": "frozen_wm_diagnostic_v1",
        "config": {"mode": "dino", "steps": 46},
        "predictor_config": asdict(config),
        "trainable_modules": [model_type.__name__],
    }
    if kind is not None:
        run["config"]["predictor_kind"] = kind
    checkpoint = tmp_path / "step_000046"
    checkpoint.mkdir()
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "COMPLETE").write_text(checkpoint.name)
    (checkpoint / "COMMITTED").write_text("complete")
    torch.save(model_type(config).state_dict(), checkpoint / "predictor.pt")
    torch.save({"schema": run["schema"], "step": 46, "run_identity": run},
               checkpoint / "training_state.pt")
    model, _, _ = _load_predictor(checkpoint, expected_mode="dino", device=torch.device("cpu"))
    assert isinstance(model, model_type)
