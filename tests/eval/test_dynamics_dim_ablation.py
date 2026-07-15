from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.eval.dynamics_dim_ablation import (
    DYNAMICS_DIM_COLUMNS,
    evaluate_dynamics_dims,
    write_dynamics_dim_artifacts,
)
from nimloth.eval.matched_wm_turns import load_turn_spec, prepare_turn_batch
from nimloth.wm.dynamics_dim_heads import DynamicsDimHeadSpec, DynamicsDimWMHeads


def tiny_heads() -> DynamicsDimWMHeads:
    spec = DynamicsDimHeadSpec(external_dim=32, full_dynamics_dim=32, factorized_dynamics_dim=8, predictor_hidden_dim=8, predictor_depth=1, predictor_heads=2, predictor_mlp_dim=16, history_size=2)
    return DynamicsDimWMHeads.create(spec).eval()


def write_cache(path: Path) -> None:
    rows, states = [], []
    for record in range(2):
        for step in range(6):
            rows.append({"id": f"r{record}:{step}", "record_id": f"r{record}", "step_index": step, "action_index": step % 8})
            states.append(torch.full((8, 4), record * 10 + step, dtype=torch.float16))
    torch.save({"state_emb": torch.stack(states), "rows": rows}, path / "shard.pt")
    manifest = {"count": 12, "cond_dim": 32, "state_dtype": "float16", "compression": "none", "shard_size": 12, "shards": [{"file": "shard.pt", "count": 12}], "fingerprint": "fixture", "state_shape": [8, 4], "representation": "frozen_query_state"}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def turn_batch(tmp_path: Path):
    selections, state, positive = [], {}, {}
    actions = [0, 4, 0, 5, 0]
    for run in range(6):
        record_id = f"val/r{run}"
        selections.append({"record_id": record_id, "expected_actions": actions})
        state[record_id], positive[record_id] = {}, {}
        for step in range(6):
            row = {"state_emb": torch.full((8, 4), run + step), "action_index": actions[min(step, 4)], "current_image_path": f"r{run}s{step}.png"}
            state[record_id][step] = row
            positive[record_id][step] = {**row, "state_emb": torch.full((16, 8), run + step)}
    path = tmp_path / "turns.json"
    path.write_text(json.dumps({"selections": selections}), encoding="utf-8")
    return prepare_turn_batch(load_turn_spec(path), state, positive)


def test_full_dynamics_dim_metrics_cover_all_windows(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    write_cache(cache)

    metrics = evaluate_dynamics_dims(tiny_heads(), cache, torch.device("cpu"), batch_size=4)

    assert metrics["one_step_count"] == 10
    assert metrics["one_step_mode"] == "direct_predict_next"
    assert metrics["horizons"]["1"]["mode"] == "autoregressive_rollout"
    assert [metrics["horizons"][str(step)]["count"] for step in range(1, 6)] == [10, 8, 6, 4, 2]
    assert set(metrics["one_step"]) == {"full", "factorized"}
    assert set(metrics["one_step"]["full"]) == {"mse", "cosine", "shuffled_mse", "shuffled_cosine"}


def test_dynamics_dim_artifact_uses_five_fixed_columns(tmp_path: Path) -> None:
    batch = turn_batch(tmp_path)
    generator = torch.Generator().manual_seed(4)
    images = {name: torch.rand(30, 3, 8, 8, generator=generator) for name in DYNAMICS_DIM_COLUMNS}

    metadata = write_dynamics_dim_artifacts(batch, images, tmp_path / "output", seed=4, steps=5, cfg_scale=2.0, noise_fingerprint="noise")

    assert metadata["columns"] == list(DYNAMICS_DIM_COLUMNS)
    assert metadata["num_rows"] == 30
    assert len(metadata["contact_sheets"]) == 6
