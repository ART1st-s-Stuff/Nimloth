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


def latest_runtime(root, *, broken=False):
    class Manager:
        output_dir = root

        def save(self, name, **metadata):
            path = fixture_checkpoint(root, name, step=metadata['step'])
            state = torch.load(path / 'training_state.pt', weights_only=False)
            state.update(metadata)
            torch.save(state, path / 'training_state.pt')
            if broken:
                (path / 'model.safetensors').unlink()

    return SFT2CheckpointRuntime(Manager(), 0, torch.device('cpu'), 10, 0, 1,
                                checkpoint_latest_only=True)


def test_latest_only_replaces_epochs_and_steps_without_retaining_best(tmp_path):
    runtime = latest_runtime(tmp_path)
    runtime.save_epoch(step=23, epoch=1, best_val_wm_mse=0.5, improved=True)
    assert {p.name for p in tmp_path.iterdir()} == {'epoch_001'}
    runtime.save_periodic(step=30, epoch=2, micro_step=7, best_val_wm_mse=0.5)
    assert {p.name for p in tmp_path.iterdir()} == {'step_000030'}
    runtime.save_epoch(step=46, epoch=2, best_val_wm_mse=0.5, improved=False)
    assert {p.name for p in tmp_path.iterdir()} == {'epoch_002'}
    runtime.save_final(step=46, epoch=2, best_val_wm_mse=0.5)
    assert (tmp_path / 'final/model.safetensors').stat().st_ino == (
        tmp_path / 'epoch_002/model.safetensors').stat().st_ino
    state = torch.load(tmp_path / 'final/training_state.pt', weights_only=False)
    assert state['optimizer'] == {}
    assert state['best_val_wm_mse'] == 0.5
    assert state['epoch'] == 2


def test_latest_only_incomplete_replacement_preserves_prior(tmp_path):
    previous = fixture_checkpoint(tmp_path, 'epoch_001', step=23)
    runtime = latest_runtime(tmp_path, broken=True)
    with pytest.raises(ValueError, match='retaining prior checkpoints'):
        runtime.save_periodic(step=30, epoch=2, micro_step=7, best_val_wm_mse=0.5)
    assert previous.is_dir()
    assert runtime._complete_checkpoint(previous)


def test_latest_only_leaves_unowned_names_and_symlinks_untouched(tmp_path):
    external = fixture_checkpoint(tmp_path, 'external', step=1)
    (tmp_path / 'epoch_001').symlink_to(external, target_is_directory=True)
    fixture_checkpoint(tmp_path, 'step_000010', step=10)
    runtime = latest_runtime(tmp_path)
    runtime.save_epoch(step=23, epoch=2, best_val_wm_mse=0.5, improved=True)
    assert external.is_dir()
    assert (tmp_path / 'epoch_001').is_symlink()
    assert not (tmp_path / 'step_000010').exists()


def test_latest_only_stop_replaces_last_periodic(tmp_path):
    runtime = latest_runtime(tmp_path)
    runtime.save_periodic(step=10, epoch=1, micro_step=10, best_val_wm_mse=0.5)
    runtime.save_stopped(step=12, epoch=1, micro_step=12, best_val_wm_mse=0.5)
    assert {p.name for p in tmp_path.iterdir()} == {'stop_step_000012'}


def test_latest_only_cli_defaults_to_disabled():
    from nimloth.training.sft.stage3.cli import build_sft2_arg_parser
    parser = build_sft2_arg_parser()
    args = ["--model", "/tmp/model", "--output-dir", "/tmp/out",
            "--train-jsonl", "/tmp/train", "--val-jsonl", "/tmp/val"]
    assert parser.parse_args(args).checkpoint_latest_only is False
    assert parser.parse_args(args + ["--checkpoint-latest-only"]).checkpoint_latest_only is True


def test_latest_only_preserves_checkpoint_from_different_training_identity(tmp_path):
    foreign = fixture_checkpoint(tmp_path, 'step_000010', step=10, world_size=4)
    runtime = latest_runtime(tmp_path)
    runtime.save_epoch(step=23, epoch=1, best_val_wm_mse=0.5, improved=True)
    assert foreign.is_dir()
    assert runtime._complete_checkpoint(tmp_path / 'epoch_001')


def test_latest_only_preserves_malformed_old_checkpoint(tmp_path):
    malformed = fixture_checkpoint(tmp_path, 'step_000010', step=10)
    (malformed / 'training_state.pt').write_bytes(b'not a serialized state')
    runtime = latest_runtime(tmp_path)
    runtime.save_epoch(step=23, epoch=1, best_val_wm_mse=0.5, improved=True)
    assert malformed.is_dir()
    assert runtime._complete_checkpoint(tmp_path / 'epoch_001')


def test_latest_only_final_without_epoch_clone_prunes_previous(tmp_path):
    runtime = latest_runtime(tmp_path)
    runtime.save_periodic(step=10, epoch=1, micro_step=10, best_val_wm_mse=0.5)
    runtime.save_final(step=12, epoch=1, best_val_wm_mse=0.5)
    assert {p.name for p in tmp_path.iterdir()} == {'final'}
    assert runtime._complete_checkpoint(tmp_path / 'final')
