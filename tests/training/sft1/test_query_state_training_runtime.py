from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from nimloth.training.sft1.query_state_checkpoint import QueryStateResumeIdentity
from nimloth.training.sft1.query_state_training_runtime import (
    QueryStateSegmentStore,
    QueryStateWandbMirror,
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
            checkpoint_cadence=4,
            validation_updates=(0, 6, 8),
            forced_restart_update=0,
        )


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


def test_due_validation_failure_is_non_resumable_and_does_not_advance_index(tmp_path: Path) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "run", run_identity=_SHA, mode="formal", wandb_run_id="formal-run"
    )
    segment = store.begin_segment(start_update=0, end_update=4, process_identity="process-a")
    for update in range(1, 5):
        segment.append_update({"update": update})
    with pytest.raises(RuntimeError, match="unsafe.*non-resumable"):
        segment.commit(
            checkpoint_path=_checkpoint(tmp_path / "unsafe-checkpoint"),
            checkpoint_control_hash=_SHA,
            data_cursor={"update": 4},
            metric_cursor={"validation": 4},
            validation={"due": True, "finite": True},
            safety={"passed": False, "reason": "actor"},
            mirror_records=tuple({"update": update} for update in range(1, 5)),
        )
    assert store.authoritative_entries() == ()
    failures = list((tmp_path / "run" / "failures").glob("*.json"))
    assert len(failures) == 1
    assert json.loads(failures[0].read_text())["resumable"] is False


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
