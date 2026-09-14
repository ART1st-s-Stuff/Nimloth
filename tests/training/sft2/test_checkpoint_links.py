"""Linked aliases preserve complete checkpoint contents without weight copies."""
import pytest
import torch

from nimloth.training.sft.stage3.checkpoint import SFT2CheckpointRuntime


def runtime(root, *, deduplicate=True):
    class Manager:
        output_dir = root
        calls = []

        def save(self, name, **state):
            self.calls.append(name)
            directory = root / name
            directory.mkdir(exist_ok=True)
            torch.save(state, directory / "training_state.pt")
            (directory / "weights").mkdir(exist_ok=True)
            (directory / "weights" / "part").write_bytes(str(state["step"]).encode())

    class History:
        def save(self, path):
            path.write_bytes(b"history")

    return SFT2CheckpointRuntime(Manager(), History(), 0, torch.device("cpu"), 10, 0, 2, deduplicate_epoch_checkpoints=deduplicate)


def test_epoch_best_final_share_files_and_only_save_once(tmp_path):
    checkpoint = runtime(tmp_path)
    checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=True)
    checkpoint.save_final(step=10, epoch=1, best_val_wm_mse=.5)
    assert checkpoint.manager.calls == ["epoch_001"]
    for relative in ("weights/part", "training_state.pt", "history_cache_rank_000.pt"):
        files = [tmp_path / name / relative for name in ("epoch_001", "best", "final")]
        assert len({file.stat().st_ino for file in files}) == 1
        assert len({file.read_bytes() for file in files}) == 1


def test_new_best_exchanges_directory_without_mutating_previous_epoch(tmp_path):
    checkpoint = runtime(tmp_path)
    checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=True)
    old = (tmp_path / "epoch_001" / "weights/part").stat().st_ino
    checkpoint.save_epoch(step=20, epoch=2, best_val_wm_mse=.4, improved=True)
    assert (tmp_path / "epoch_001" / "weights/part").read_bytes() == b"10"
    assert (tmp_path / "best" / "weights/part").read_bytes() == b"20"
    assert (tmp_path / "best" / "weights/part").stat().st_ino != old
    assert not list(tmp_path.glob(".best.links-*"))


def test_existing_foreign_best_and_epoch_are_preserved(tmp_path):
    checkpoint = runtime(tmp_path)
    (tmp_path / "best").mkdir()
    (tmp_path / "best" / "foreign").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="unowned"):
        checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=True)
    assert (tmp_path / "best" / "foreign").read_bytes() == b"keep"
    with pytest.raises(FileExistsError, match="immutable epoch"):
        checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=False)


def test_clone_rejects_state_mismatch_and_symlinks(tmp_path):
    checkpoint = runtime(tmp_path)
    checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=False)
    with pytest.raises(ValueError, match="identity mismatch"):
        checkpoint._clone_epoch("best", ("epoch_001", 11, 1, .5))
    (tmp_path / "epoch_001" / "link").symlink_to("training_state.pt")
    with pytest.raises(ValueError, match="symlinks"):
        checkpoint._clone_epoch("best", ("epoch_001", 10, 1, .5))
    assert not (tmp_path / "best").exists()
    assert not list(tmp_path.glob(".best.links-*"))


def test_different_final_boundary_does_not_clone(tmp_path):
    checkpoint = runtime(tmp_path)
    checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=False)
    checkpoint.save_final(step=11, epoch=1, best_val_wm_mse=.5)
    assert checkpoint.manager.calls == ["epoch_001", "final"]
    assert (tmp_path / "final" / "weights/part").read_bytes() == b"11"


def test_default_legacy_saves_separately_and_allows_existing_best(tmp_path):
    checkpoint = runtime(tmp_path, deduplicate=False)
    checkpoint.save_epoch(step=10, epoch=1, best_val_wm_mse=.5, improved=True)
    checkpoint.save_epoch(step=20, epoch=2, best_val_wm_mse=.4, improved=True)
    checkpoint.save_final(step=20, epoch=2, best_val_wm_mse=.4)
    assert checkpoint.manager.calls == ["epoch_001", "best", "epoch_002", "best", "final"]
    assert len({(tmp_path / name / "weights/part").stat().st_ino
                for name in ("epoch_002", "best", "final")}) == 3


def test_dedup_cli_default_and_opt_in():
    from nimloth.training.sft.stage3.cli import build_sft2_arg_parser
    parser = build_sft2_arg_parser()
    args = ["--model", "/tmp/model", "--output-dir", "/tmp/out",
            "--train-jsonl", "/tmp/train", "--val-jsonl", "/tmp/val"]
    assert parser.parse_args(args).deduplicate_epoch_checkpoints is False
    assert parser.parse_args(args + ["--deduplicate-epoch-checkpoints"]).deduplicate_epoch_checkpoints is True
