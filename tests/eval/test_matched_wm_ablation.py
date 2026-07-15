from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nimloth.eval.matched_wm_ablation import (
    RECONSTRUCTION_COLUMNS,
    evaluate_full_dynamics,
    load_frozen_state_adapter,
    load_turn_spec,
    matched_noise,
    prepare_turn_batch,
    write_turn_artifacts,
)
from nimloth.training.reconstruction.query_bottleneck_probe import QueryBottleneckAdapter
from nimloth.wm.matched_heads import MatchedHeadSpec, MatchedWMHeads


def valid_selections() -> list[dict]:
    patterns = ([0, 4, 0, 5, 0], [0, 4, 0, 5, 4], [0, 0, 4, 0, 5])
    return [{"record_id": f"val/record_{index}", "expected_actions": list(patterns[index % 3])} for index in range(6)]


def write_spec(path: Path, selections: list[dict]) -> None:
    path.write_text(json.dumps({"selections": selections}), encoding="utf-8")


def records(selections: list[dict], shape: tuple[int, int]) -> dict:
    output = {}
    for run, selection in enumerate(selections):
        steps = {}
        for step in range(6):
            actions = selection["expected_actions"]
            steps[step] = {"state_emb": torch.full(shape, run * 10 + step), "action_index": actions[min(step, 4)], "current_image_path": f"r{run}s{step}.png"}
        output[selection["record_id"]] = steps
    return output


def write_cache(path: Path) -> None:
    rows, states = [], []
    for record in range(2):
        for step in range(6):
            rows.append({"id": f"r{record}:{step}", "record_id": f"r{record}", "step_index": step, "action_index": step % 8})
            states.append(torch.full((8, 4), record * 10 + step, dtype=torch.float16))
    torch.save({"state_emb": torch.stack(states), "rows": rows}, path / "shard.pt")
    manifest = {"count": 12, "cond_dim": 32, "state_dtype": "float16", "compression": "none", "shard_size": 12, "shards": [{"file": "shard.pt", "count": 12}], "fingerprint": "fixture", "state_shape": [8, 4], "representation": "frozen_query_state"}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def tiny_heads() -> MatchedWMHeads:
    spec = MatchedHeadSpec(state_tokens=8, token_dim=4, vector_hidden_dim=8, token_hidden_dim=8, depth=1, heads=2, mlp_ratio=2)
    return MatchedWMHeads.create(spec).eval()


def test_turn_spec_requires_six_true_turn_sequences(tmp_path: Path) -> None:
    path = tmp_path / "turns.json"
    selections = valid_selections()
    write_spec(path, selections)
    assert len(load_turn_spec(path)) == 6

    selections[0]["expected_actions"] = [0, 2, 0, 3, 0]
    write_spec(path, selections)
    with pytest.raises(ValueError, match="turn_right.*turn_left"):
        load_turn_spec(path)


def test_turn_batch_is_exactly_six_by_five_and_aligned(tmp_path: Path) -> None:
    path = tmp_path / "turns.json"
    selections = valid_selections()
    write_spec(path, selections)
    state = records(selections, (8, 4))
    positive = records(selections, (16, 8))

    batch = prepare_turn_batch(load_turn_spec(path), state, positive)

    assert len(batch.rows) == 30
    assert batch.initial_state.shape == (6, 8, 4)
    assert batch.target_states.shape == (6, 5, 8, 4)
    assert batch.positive_tokens.shape == (30, 16, 8)
    assert batch.actions.shape == (6, 5)
    assert {row["action_name"] for row in batch.rows} >= {"turn_right", "turn_left"}


def test_full_dynamics_reports_all_horizon_windows(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    write_cache(cache)

    metrics = evaluate_full_dynamics(tiny_heads(), cache, torch.device("cpu"), batch_size=4)

    assert metrics["one_step_count"] == 10
    assert [metrics["horizons"][str(step)]["count"] for step in range(1, 6)] == [10, 8, 6, 4, 2]
    assert set(metrics["one_step"]) == {"vector", "token"}
    assert set(metrics["one_step"]["vector"]) == {"mse", "cosine", "shuffled_mse", "shuffled_cosine"}
    assert all(torch.isfinite(torch.tensor(value)) for branch in metrics["one_step"].values() for value in branch.values())


def test_frozen_adapter_and_noise_reload_contract(tmp_path: Path) -> None:
    model = QueryBottleneckAdapter(input_tokens=8, input_dim=32, bottleneck_dim=4)
    checkpoint = tmp_path / "probe.pt"
    torch.save({"model": model.state_dict(), "step": 9, "invariants": {"input_shape": [8, 32], "bottleneck_shape": [8, 4]}}, checkpoint)

    adapter = load_frozen_state_adapter(checkpoint, torch.device("cpu"))
    first, first_hash = matched_noise(3, seed=9, image_size=8)
    second, second_hash = matched_noise(3, seed=9, image_size=8)

    assert adapter(torch.randn(2, 8, 4)).shape == (2, 16, 512)
    assert all(not parameter.requires_grad for parameter in adapter.parameters())
    assert torch.equal(first, second)
    assert first_hash == second_hash


def test_turn_artifact_has_five_columns_and_thirty_rows(tmp_path: Path) -> None:
    path = tmp_path / "turns.json"
    selections = valid_selections()
    write_spec(path, selections)
    batch = prepare_turn_batch(load_turn_spec(path), records(selections, (8, 4)), records(selections, (16, 8)))
    generator = torch.Generator().manual_seed(9)
    images = {name: torch.rand(30, 3, 8, 8, generator=generator) for name in RECONSTRUCTION_COLUMNS}

    metadata = write_turn_artifacts(batch, images, tmp_path / "output", seed=9, steps=5, cfg_scale=2.0, noise_fingerprint="noise-hash")

    assert metadata["status"] == "completed"
    assert metadata["num_rows"] == 30
    assert metadata["columns"] == list(RECONSTRUCTION_COLUMNS)
    assert metadata["noise_fingerprint"] == "noise-hash"
    assert len(metadata["contact_sheets"]) == 6
    assert len(json.loads((tmp_path / "output/samples.json").read_text())) == 30
    assert all(Path(path).is_file() for path in metadata["contact_sheets"])
