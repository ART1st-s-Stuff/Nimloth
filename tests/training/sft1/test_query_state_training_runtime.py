from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nimloth.training.sft1.query_state_checkpoint import QueryStateResumeIdentity
from nimloth.training.sft1.query_state_training_runtime import (
    QueryStateEarlyStoppingCursor,
    QueryStateSegmentStore,
    QueryStateWandbMirror,
    advance_query_state_early_stopping,
    build_training_event_plan,
    consume_pilot_restart_boundary,
    current_process_identity,
    publish_pilot_restart_boundary,
)


_SHA = "a" * 64


def _checkpoint(path: Path, *, control_hash: str = _SHA) -> Path:
    path.mkdir(parents=True)
    (path / "control.json").write_text("{}\n", encoding="utf-8")
    (path / "COMPLETED").write_text(f"control_sha256={control_hash}\n", encoding="utf-8")
    return path


def _bind_checkpoint_control_hash(path: Path) -> str:
    digest = __import__("hashlib").sha256((path / "control.json").read_bytes()).hexdigest()
    (path / "COMPLETED").write_text(
        f"control_sha256={digest}\n",
        encoding="utf-8",
    )
    return digest


def test_resume_identity_explicitly_separates_mechanics_pilot_and_formal() -> None:
    common = dict(
        source_commit="b" * 40,
        source_manifest_identity=_SHA,
        config_identity="c" * 64,
        run_identity="d" * 64,
        world_size=2,
    )
    pilot = QueryStateResumeIdentity(**common, experiment_mode="pilot")
    formal = QueryStateResumeIdentity(**common, experiment_mode="formal")
    assert pilot != formal
    with pytest.raises(ValueError, match="mode"):
        QueryStateResumeIdentity(**common, experiment_mode="legacy_sft1_v2")


def test_event_plan_validates_update0_before_steps_and_commit_boundaries() -> None:
    plan = build_training_event_plan(
        mode="pilot",
        total_updates=8,
        epoch_updates=8,
        checkpoint_cadence=4,
        validation_updates=(0, 8),
        forced_restart_update=4,
    )
    assert plan[0].kind == "validation"
    assert plan[0].update == 0
    kinds_at_four = [event.kind for event in plan if event.update == 4]
    assert kinds_at_four == ["optimizer_update", "commit", "forced_restart"]
    kinds_at_eight = [event.kind for event in plan if event.update == 8]
    assert kinds_at_eight == ["optimizer_update", "validation", "safety_verdict", "commit", "terminal"]
    assert all(event.mode == "pilot" for event in plan)

    with pytest.raises(ValueError, match="commit boundary"):
        build_training_event_plan(
            mode="formal",
            total_updates=8,
            epoch_updates=8,
            checkpoint_cadence=4,
            validation_updates=(0, 6, 8),
            forced_restart_update=0,
        )


def test_formal_sub_epoch_commits_do_not_advance_epoch_validation_or_patience() -> None:
    plan = build_training_event_plan(
        mode="formal",
        total_updates=3210,
        epoch_updates=1605,
        checkpoint_cadence=321,
        validation_updates=(0, 3210),
        forced_restart_update=0,
    )
    kinds_at_sub_epoch = [event.kind for event in plan if event.update == 321]
    assert kinds_at_sub_epoch == ["optimizer_update", "commit"]
    kinds_at_epoch_one = [event.kind for event in plan if event.update == 1605]
    assert kinds_at_epoch_one == [
        "optimizer_update",
        "calibration",
        "safety_verdict",
        "commit",
    ]
    kinds_at_epoch_two = [event.kind for event in plan if event.update == 3210]
    assert kinds_at_epoch_two == [
        "optimizer_update",
        "calibration",
        "validation",
        "safety_verdict",
        "commit",
        "terminal",
    ]


def test_formal_early_stop_is_deterministic_calibration_only_and_resumable() -> None:
    cursor = QueryStateEarlyStoppingCursor.initial()
    first = advance_query_state_early_stopping(
        cursor,
        epoch=1,
        update=1605,
        calibration_dino_mse=2.0,
        calibration_assistant_ce=1.0,
        min_epochs=2,
        max_epochs=10,
        patience_epochs=2,
        min_relative_improvement=0.01,
    )
    assert first.composite == pytest.approx(5.0)
    assert first.cursor.best_composite == pytest.approx(5.0)
    assert first.cursor.last_composite == pytest.approx(5.0)
    assert first.cursor.bad_epochs == 0
    assert first.should_stop is False

    second = advance_query_state_early_stopping(
        first.cursor,
        epoch=2,
        update=3210,
        calibration_dino_mse=1.99,
        calibration_assistant_ce=1.0,
        min_epochs=2,
        max_epochs=10,
        patience_epochs=2,
        min_relative_improvement=0.01,
    )
    assert second.cursor.bad_epochs == 1
    assert second.should_stop is False
    restored_before_terminal = QueryStateEarlyStoppingCursor.from_mapping(
        second.cursor.to_mapping()
    )
    third = advance_query_state_early_stopping(
        restored_before_terminal,
        epoch=3,
        update=4815,
        calibration_dino_mse=1.98,
        calibration_assistant_ce=1.0,
        min_epochs=2,
        max_epochs=10,
        patience_epochs=2,
        min_relative_improvement=0.01,
    )
    assert third.should_stop is True
    assert third.reason == "converged_early_stop"
    assert third.cursor.terminal_epoch == 3
    assert third.cursor.terminal_update == 4815

    restored = QueryStateEarlyStoppingCursor.from_mapping(third.cursor.to_mapping())
    assert restored == third.cursor
    with pytest.raises(TypeError, match="holdout_metric"):
        advance_query_state_early_stopping(
            second.cursor,
            epoch=3,
            update=4815,
            calibration_dino_mse=1.98,
            calibration_assistant_ce=1.0,
            min_epochs=2,
            max_epochs=10,
            patience_epochs=2,
            min_relative_improvement=0.01,
            holdout_metric=0.0,
        )


def test_formal_early_stop_max_epoch_is_terminal_even_with_improvement() -> None:
    cursor = QueryStateEarlyStoppingCursor.initial()
    decision = None
    for epoch in range(1, 11):
        decision = advance_query_state_early_stopping(
            cursor,
            epoch=epoch,
            update=epoch * 1605,
            calibration_dino_mse=10.0 / epoch,
            calibration_assistant_ce=1.0 / epoch,
            min_epochs=2,
            max_epochs=10,
            patience_epochs=2,
            min_relative_improvement=0.01,
        )
        cursor = decision.cursor
    assert decision is not None
    assert decision.should_stop is True
    assert decision.reason == "max_epochs_reached"
    assert cursor.terminal_epoch == 10
    assert cursor.terminal_update == 16050


def test_segment_records_are_pending_until_checkpoint_and_atomic_index(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(tmp_path / "run", run_identity=_SHA, mode="pilot")
    segment = store.begin_segment(start_update=0, end_update=4, process_identity="process-a")
    for update in range(1, 5):
        segment.append_update({"update": update, "loss": float(update)})
    assert store.authoritative_entries() == ()
    assert not (tmp_path / "wandb-called").exists()

    entry = segment.commit(
        checkpoint_path=_checkpoint(tmp_path / "checkpoint-4"),
        checkpoint_control_hash=_SHA,
        data_cursor={"update": 4, "row": 8},
        metric_cursor={"validation": 0, "log": 4, "wandb": 0},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=({"update": 1}, {"update": 2}, {"update": 3}, {"update": 4}),
    )
    assert entry.end_update == 4
    assert store.authoritative_entries() == (entry,)
    indexed = json.loads((tmp_path / "run" / "authoritative_index.json").read_text())
    assert indexed["entries"][0]["checkpoint_control_hash"] == _SHA
    assert Path(indexed["entries"][0]["mirror_batch_path"]).is_file()


def test_formal_authoritative_index_binds_early_stop_and_actual_terminal(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "run", run_identity=_SHA, mode="formal", wandb_run_id="formal-run"
    )
    segment = store.begin_segment(start_update=0, end_update=2, process_identity="process-a")
    segment.append_update({"update": 1})
    segment.append_update({"update": 2})
    cursor = QueryStateEarlyStoppingCursor(
        best_composite=4.0,
        last_composite=4.1,
        best_epoch=2,
        bad_epochs=2,
        last_epoch=4,
        last_update=2,
        terminal_epoch=4,
        terminal_update=2,
        stop_reason="converged_early_stop",
    )
    metric_cursor = {
        "validation": 2,
        "log": 2,
        "wandb": 0,
        "early_stopping": cursor.to_mapping(),
        "actual_terminal": {
            "epoch": 4,
            "update": 2,
            "reason": "converged_early_stop",
            "terminal_primary": True,
        },
    }
    entry = segment.commit(
        checkpoint_path=_checkpoint(tmp_path / "checkpoint-2"),
        checkpoint_control_hash=_SHA,
        data_cursor={"update": 2},
        metric_cursor=metric_cursor,
        validation={"due": True},
        safety={"passed": True},
        mirror_records=({"update": 1}, {"update": 2, "early_stopping": cursor.to_mapping()}),
    )
    assert entry.early_stopping_cursor == cursor.to_mapping()
    assert entry.actual_terminal == metric_cursor["actual_terminal"]
    indexed = json.loads((tmp_path / "run" / "authoritative_index.json").read_text())
    assert indexed["entries"][0]["actual_terminal"]["reason"] == "converged_early_stop"
    mirror = json.loads(Path(entry.mirror_batch_path).read_text())
    assert mirror["records"][-1]["early_stopping"]["bad_epochs"] == 2


def test_index_before_crash_replays_mirror_but_index_before_publication_rolls_back(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "run", run_identity=_SHA, mode="formal", wandb_run_id="formal-run"
    )
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_path = checkpoint_root / "update_00000004"
    segment = store.begin_segment(start_update=0, end_update=4, process_identity="process-a")
    for update in range(1, 5):
        segment.append_update({"update": update})
    with pytest.raises(RuntimeError, match="injected before authoritative index"):
        segment.commit(
            checkpoint_path=_checkpoint(checkpoint_path),
            checkpoint_control_hash=_SHA,
            data_cursor={"update": 4},
            metric_cursor={"validation": 0, "log": 4, "wandb": 0},
            validation={"due": False},
            safety={"passed": True},
            mirror_records=({"update": 1}, {"update": 2}, {"update": 3}, {"update": 4}),
            fail_before_index=True,
        )
    assert store.authoritative_entries() == ()
    recovered = store.recover(checkpoint_root=checkpoint_root)
    assert recovered.resume_update == 0
    assert recovered.abandoned_pending_segments == 1
    assert recovered.abandoned_unindexed_checkpoints == 1
    assert not checkpoint_path.exists()
    assert len(list((tmp_path / "run" / "abandoned" / "checkpoints").iterdir())) == 1

    replay = store.begin_segment(start_update=0, end_update=4, process_identity="process-b")
    for update in range(1, 5):
        replay.append_update({"update": update})
    entry = replay.commit(
        checkpoint_path=_checkpoint(checkpoint_path),
        checkpoint_control_hash=_SHA,
        data_cursor={"update": 4},
        metric_cursor={"validation": 0, "log": 4, "wandb": 0},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=tuple({"update": update} for update in range(1, 5)),
    )
    final_recovery = store.recover(checkpoint_root=checkpoint_root)
    assert final_recovery.resume_update == 4
    assert final_recovery.abandoned_unindexed_checkpoints == 0
    assert store.pending_mirror_batches() == (entry.mirror_batch_path,)


def test_due_validation_failure_preserves_forensic_checkpoint_without_advancing_index(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    store = QueryStateSegmentStore(
        run_root / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run",
    )
    segment = store.begin_segment(start_update=0, end_update=4, process_identity="process-a")
    for update in range(1, 5):
        segment.append_update({"update": update})
    forensic_checkpoint = _checkpoint(
        run_root / "forensics" / "unsafe_update_00000004"
    )
    (forensic_checkpoint / "control.json").write_text(
        json.dumps(
            {
                "identity": {
                    "run_identity": _SHA,
                    "experiment_mode": "formal",
                },
                "global_step": 4,
                "terminal_primary": False,
                "forensic_only": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    control_hash = _bind_checkpoint_control_hash(forensic_checkpoint)
    with pytest.raises(RuntimeError, match="unsafe.*forensic.*non-resumable"):
        segment.commit(
            checkpoint_path=forensic_checkpoint,
            checkpoint_control_hash=control_hash,
            data_cursor={"update": 4},
            metric_cursor={"validation": 4},
            validation={"due": True, "finite": True},
            safety={"passed": False, "reason": "actor"},
            mirror_records=tuple({"update": update} for update in range(1, 5)),
        )
    assert forensic_checkpoint.is_dir()
    assert store.authoritative_entries() == ()
    assert store.pending_mirror_batches() == ()
    failures = list((run_root / "durable" / "failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["forensic_checkpoint"] == {
        "authoritative": False,
        "control_sha256": control_hash,
        "forensic_only": True,
        "path": str(forensic_checkpoint.resolve()),
        "resumable": False,
    }
    assert failure["resumable"] is False
    recovery = store.recover(checkpoint_root=run_root / "checkpoints")
    assert recovery.abandoned_unindexed_checkpoints == 0
    assert forensic_checkpoint.is_dir()


def test_forensic_checkpoint_rejects_cross_run_control_provenance(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    store = QueryStateSegmentStore(
        run_root / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run",
    )
    forensic = _checkpoint(run_root / "forensics" / "unsafe_update_00000004")
    (forensic / "control.json").write_text(
        json.dumps(
            {
                "identity": {
                    "run_identity": "b" * 64,
                    "experiment_mode": "formal",
                },
                "global_step": 4,
                "terminal_primary": False,
                "forensic_only": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    control_hash = _bind_checkpoint_control_hash(forensic)
    with pytest.raises(ValueError, match="control provenance"):
        store.record_unsafe_forensic_checkpoint(
            start_update=0,
            end_update=4,
            checkpoint_path=forensic,
            checkpoint_control_hash=control_hash,
            validation={"due": True},
            safety={"passed": False},
        )
    assert not list((run_root / "durable" / "failures").glob("*.json"))


def test_forensic_save_failure_is_durable_and_non_resumable(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "run" / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run",
    )
    failure_path = store.record_forensic_save_failure(
        update=4,
        validation={"due": True},
        safety={"passed": False},
        error="RuntimeError: rank 1 disk write failed",
    )
    failure = json.loads(failure_path.read_text())
    assert failure["forensic_checkpoint"] is None
    assert failure["forensic_checkpoint_preserved"] is False
    assert failure["resumable"] is False
    assert "disk write failed" in failure["error"]
    assert store.authoritative_entries() == ()


def test_mirror_fallback_is_same_run_all_rank_and_cursor_idempotent(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "run", run_identity=_SHA, mode="formal", wandb_run_id="formal-run"
    )
    segment = store.begin_segment(start_update=0, end_update=2, process_identity="process-a")
    segment.append_update({"update": 1})
    segment.append_update({"update": 2})
    entry = segment.commit(
        checkpoint_path=_checkpoint(tmp_path / "checkpoint-2"),
        checkpoint_control_hash=_SHA,
        data_cursor={"update": 2},
        metric_cursor={"validation": 0, "log": 2, "wandb": 0},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=({"update": 1}, {"update": 2}),
    )
    mirror = QueryStateWandbMirror(
        run_id="formal-run", world_size=2, initial_cursor=0
    )
    mirror.register_authoritative(entry)
    assert mirror.pending_updates() == (1, 2)
    with pytest.raises(ValueError, match="same run"):
        QueryStateWandbMirror(
            run_id="different-run", world_size=2, initial_cursor=0
        ).register_authoritative(entry)
    with pytest.raises(ValueError, match="same run"):
        mirror.replay(run_id="other", updates=(1, 2))
    with pytest.raises(ValueError, match="gap|ordered"):
        mirror.replay(run_id="formal-run", updates=(2,))
    mirror.coordinated_transport_failure((True, True))
    assert mirror.durable_only
    assert mirror.tracking_incomplete
    mirror.replay(run_id="formal-run", updates=(1, 2))
    assert mirror.cursor == 2
    assert mirror.pending_updates() == ()
    mirror.replay(run_id="formal-run", updates=())
    with pytest.raises(ValueError, match="duplicate|cursor"):
        mirror.replay(run_id="formal-run", updates=(2,))
    with pytest.raises(RuntimeError, match="all ranks"):
        QueryStateWandbMirror(
            run_id="formal-run", world_size=2, initial_cursor=0
        ).coordinated_transport_failure((True, False))


def test_forced_pilot_restart_requires_fresh_process_and_exact_cursors(tmp_path: Path) -> None:
    boundary = tmp_path / "restart.json"
    publish_pilot_restart_boundary(
        boundary,
        run_identity=_SHA,
        process_identity=current_process_identity(),
        checkpoint_identity="c" * 64,
        checkpoint_update=4,
        fingerprints={"model": "m", "optimizer": "o", "scheduler": "s", "rng": "r"},
        cursors={"data": 4, "validation": 0, "log": 4, "wandb": 0},
    )
    with pytest.raises(ValueError, match="fresh process"):
        consume_pilot_restart_boundary(
            boundary,
            run_identity=_SHA,
            process_identity=current_process_identity(),
            checkpoint_identity="c" * 64,
            restored_fingerprints={"model": "m", "optimizer": "o", "scheduler": "s", "rng": "r"},
            restored_cursors={"data": 4, "validation": 0, "log": 4, "wandb": 0},
        )

    script = """
from pathlib import Path
from nimloth.training.sft1.query_state_training_runtime import consume_pilot_restart_boundary, current_process_identity
receipt = consume_pilot_restart_boundary(
    Path({path!r}),
    run_identity={run!r},
    process_identity=current_process_identity(),
    checkpoint_identity={checkpoint!r},
    restored_fingerprints={{'model':'m','optimizer':'o','scheduler':'s','rng':'r'}},
    restored_cursors={{'data':4,'validation':0,'log':4,'wandb':0}},
)
print(receipt.fresh_process_verified)
""".format(path=str(boundary), run=_SHA, checkpoint="c" * 64)
    environment = dict(os.environ)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
        "PYTHONPATH": "src:.",
    })
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.stdout.strip() == "True"
