import json

import pytest
import torch

from nimloth.training.sft.stage1.checkpoint import prune_older_epoch_checkpoints
from nimloth.training.sft.stage1.cli import parse_args

BASE = [
    "--model", "/model",
    "--train-jsonl", "/train.jsonl",
    "--val-jsonl", "/val.jsonl",
    "--output-dir", "/output",
]


def committed_epoch(root, epoch, step, *, identity=None):
    path = root / f"epoch_{epoch:03d}"
    path.mkdir()
    identity = identity or {"stage": "query", "dataset": "fixture"}
    torch.save(
        {"epoch": epoch, "step": step, "identity": identity},
        path / "training_state.pt",
    )
    (path / "COMMITTED").write_text(
        json.dumps({"epoch": epoch, "step": step}) + "\n",
        encoding="utf-8",
    )
    return path


def test_epoch_retention_cli_and_yaml_are_opt_in_positive_counts(tmp_path):
    assert parse_args(BASE)[0].keep_epoch_checkpoints is None
    assert parse_args(BASE + ["--keep-epoch-checkpoints", "1"])[0].keep_epoch_checkpoints == 1

    config = tmp_path / "config.yaml"
    config.write_text(
        "train:\n  keep_epoch_checkpoints: 2\n", encoding="utf-8"
    )
    assert parse_args(BASE + ["--config", str(config)])[0].keep_epoch_checkpoints == 2

    for invalid in ("0", "-1"):
        with pytest.raises(ValueError, match="positive integer"):
            parse_args(BASE + ["--keep-epoch-checkpoints", invalid])

    config.write_text(
        "train:\n  keep_epoch_checkpoints: '1'\n", encoding="utf-8"
    )
    assert parse_args(BASE + ["--config", str(config)])[0].keep_epoch_checkpoints == 1


def test_committed_latest_epoch_prunes_only_older_same_run_epochs(tmp_path):
    old_initial = committed_epoch(tmp_path, 0, 0)
    old = committed_epoch(tmp_path, 1, 27)
    latest = committed_epoch(tmp_path, 2, 54)
    foreign = committed_epoch(
        tmp_path, 3, 81, identity={"stage": "query", "dataset": "foreign"}
    )
    best = tmp_path / "best"
    best.mkdir()
    (best / "weights").write_bytes(b"separate best checkpoint")

    removed = prune_older_epoch_checkpoints(tmp_path, latest, keep=1)

    assert removed == [old_initial, old]
    assert not old_initial.exists()
    assert not old.exists()
    assert latest.is_dir()
    assert foreign.is_dir()
    assert best.is_dir()


def test_retention_count_keeps_the_requested_newest_epochs(tmp_path):
    first = committed_epoch(tmp_path, 1, 27)
    second = committed_epoch(tmp_path, 2, 54)
    latest = committed_epoch(tmp_path, 3, 81)

    assert prune_older_epoch_checkpoints(tmp_path, latest, keep=2) == [first]
    assert not first.exists()
    assert second.is_dir()
    assert latest.is_dir()


def test_incomplete_latest_epoch_never_prunes_previous_epoch(tmp_path):
    previous = committed_epoch(tmp_path, 1, 27)
    latest = committed_epoch(tmp_path, 2, 54)
    (latest / "COMMITTED").unlink()

    with pytest.raises(ValueError, match="not committed"):
        prune_older_epoch_checkpoints(tmp_path, latest, keep=1)

    assert previous.is_dir()


def test_malformed_foreign_and_symlinked_old_epochs_are_preserved(tmp_path):
    malformed = committed_epoch(tmp_path, 1, 27)
    (malformed / "training_state.pt").write_bytes(b"not a torch checkpoint")
    wrong_type = committed_epoch(tmp_path, 0, 0)
    torch.save([], wrong_type / "training_state.pt")
    foreign = committed_epoch(
        tmp_path, 2, 54, identity={"stage": "query", "dataset": "foreign"}
    )
    external = tmp_path / "external"
    external.mkdir()
    linked = tmp_path / "epoch_003"
    linked.symlink_to(external, target_is_directory=True)
    latest = committed_epoch(tmp_path, 4, 108)

    assert prune_older_epoch_checkpoints(tmp_path, latest, keep=1) == []
    assert malformed.is_dir()
    assert wrong_type.is_dir()
    assert foreign.is_dir()
    assert linked.is_symlink()
    assert external.is_dir()


def test_latest_epoch_marker_must_match_state_before_pruning(tmp_path):
    previous = committed_epoch(tmp_path, 1, 27)
    latest = committed_epoch(tmp_path, 2, 54)
    (latest / "COMMITTED").write_text(
        json.dumps({"epoch": 2, "step": 53}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="identity is inconsistent"):
        prune_older_epoch_checkpoints(tmp_path, latest, keep=1)

    assert previous.is_dir()
