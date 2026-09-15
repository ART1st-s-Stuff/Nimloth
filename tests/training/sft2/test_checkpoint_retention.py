"""Rolling retention excludes incomplete intermediates and all final artifacts."""
import json
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.checkpoint import SFT2CheckpointRuntime


def fixture_checkpoint(root, name, *, step, world_size=2):
    path = root / name
    path.mkdir()
    for filename in ("config.json", "model.safetensors", "state_proj.pt",
                     "wm_predictor/config.json", "wm_predictor/predictor.pt",
                     "value_head/value_head.pt", "selected_token_rows.pt", "outcome_head.pt"):
        target = path / filename
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(b"isolated presence-check fixture")
    torch.save({"step": step, "optimizer": {}, "query_tune": "selected_rows",
                "training_invariants": {"world_size": world_size, "outcome_schema": "test",
                                        "training_unit": "complete_trajectory_v1"}},
               path / "training_state.pt")
    return path


def test_latest_two_complete_steps_retained_without_touching_finals_or_partial(tmp_path):
    for step in (10, 20, 30):
        fixture_checkpoint(tmp_path, f"step_{step:06d}", step=step)
    partial = fixture_checkpoint(tmp_path, "step_000040", step=40)
    (partial / "value_head/value_head.pt").unlink()
    for name in ("final", "best", "epoch_001", "stop_step_000001"):
        fixture_checkpoint(tmp_path, name, step=1)
    runtime = SFT2CheckpointRuntime(manager=SimpleNamespace(output_dir=tmp_path),
                                   rank=0, device=torch.device("cpu"),
                                   interval_steps=10, interval_minutes=0, keep_last=2)
    runtime._prune_step_checkpoints()
    assert not (tmp_path / "step_000010").exists()
    assert {path.name for path in tmp_path.iterdir()} == {
        "step_000020", "step_000030", "step_000040", "final", "best", "epoch_001", "stop_step_000001",
    }


@pytest.mark.parametrize("missing", ["model.safetensors", "state_proj.pt", "selected_token_rows.pt",
                                     "outcome_head.pt", "value_head/value_head.pt"])
def test_incomplete_step_is_not_a_retention_candidate(tmp_path, missing):
    path = fixture_checkpoint(tmp_path, "step_000010", step=10)
    (path / missing).unlink()
    assert not SFT2CheckpointRuntime._complete_step_checkpoint(path)


def test_sharded_checkpoint_requires_every_referenced_weight_file(tmp_path):
    path = fixture_checkpoint(tmp_path, "step_000010", step=10)
    (path / "model.safetensors").unlink()
    (path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"},
    }))
    (path / "part1.safetensors").write_bytes(b"fixture")
    assert not SFT2CheckpointRuntime._complete_step_checkpoint(path)
    (path / "part2.safetensors").write_bytes(b"fixture")
    assert SFT2CheckpointRuntime._complete_step_checkpoint(path)


@pytest.mark.parametrize("unit", [None, "current_step_once_v2_online_cache"])
def test_legacy_window_checkpoint_is_not_pruned(tmp_path, unit):
    path = fixture_checkpoint(tmp_path, "step_000010", step=10)
    state = torch.load(path / "training_state.pt", weights_only=False)
    state["training_invariants"]["training_unit"] = unit
    torch.save(state, path / "training_state.pt")
    assert not SFT2CheckpointRuntime._complete_step_checkpoint(path)
    assert path.exists()
