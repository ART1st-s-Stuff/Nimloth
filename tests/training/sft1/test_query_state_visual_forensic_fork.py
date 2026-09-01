from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from nimloth.training.sft1.query_state import DirectSlotProjector
from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateDistributedControl,
    QueryStateResumeIdentity,
    capture_query_state_rank_state,
    finalize_query_state_rank_checkpoint,
    save_query_state_rank_state,
)
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_backend import (
    _authoritative_entries_for_restart,
    _initial_durable_cursor,
)
from nimloth.training.sft1.query_state_training_runtime import (
    QUERY_STATE_SEGMENT_SCHEMA,
    QueryStateSegmentStore,
    QueryStateWandbMirror,
)
from nimloth.training.sft1.query_state_visual_forensic_fork import (
    build_visual_fork_event_plan,
    initialize_visual_fork_from_forensic_model,
)
from tests.training.sft1.test_query_state_training_config import _raw as _training_raw

_SHA = "a" * 64
_ANCESTOR_COMMIT = "a" * 40
_CURRENT_COMMIT = "b" * 40


def _ancestor_identity() -> QueryStateResumeIdentity:
    return QueryStateResumeIdentity(
        source_commit=_ANCESTOR_COMMIT,
        source_manifest_identity="1" * 64,
        config_identity="c" * 64,
        run_identity="d" * 64,
        world_size=8,
        experiment_mode="formal",
    )


def _write_forensic_checkpoint(tmp_path: Path, root: nn.Module) -> tuple[Path, Path, QueryStateResumeIdentity]:
    identity = _ancestor_identity()
    optimizer = torch.optim.AdamW(root.parameters(), lr=1e-4)
    root.objective.projector(torch.ones(1, 16, 2048)).sum().backward()
    optimizer.step()
    checkpoint = tmp_path / "formal38" / "forensics" / "unsafe_update_00001605"
    control = QueryStateDistributedControl(
        identity=identity,
        global_step=1605,
        data_cursor={"next_update": 1606},
        metric_cursor={"validation": 1605},
        forensic_only=True,
    )
    rank_state = capture_query_state_rank_state(
        root,
        optimizer,
        scheduler_state={"last_epoch": 1605},
        identity=identity,
    )
    for rank in range(8):
        save_query_state_rank_state(
            checkpoint,
            rank=rank,
            world_size=8,
            state=rank_state,
        )
    finalize_query_state_rank_checkpoint(checkpoint, control=control)
    failure = tmp_path / "formal38" / "durable" / "failures" / "unsafe_00001284_00001605.json"
    failure.parent.mkdir(parents=True)
    control_sha = hashlib.sha256((checkpoint / "control.json").read_bytes()).hexdigest()
    failure.write_text(json.dumps({
        "schema": "nimloth_sft1_query_state_segment_v1",
        "run_identity": identity.run_identity,
        "mode": "formal",
        "end_update": 1605,
        "forensic_checkpoint": {
            "path": str(checkpoint.resolve()),
            "control_sha256": control_sha,
            "forensic_only": True,
            "resumable": False,
            "authoritative": False,
        },
        "resumable": False,
    }) + "\n", encoding="utf-8")
    return checkpoint, failure, identity


def _checkpoint(
    path: Path,
    *,
    run_identity: str,
    config_identity: str,
    update: int,
) -> tuple[Path, str]:
    path.mkdir(parents=True)
    identity = {
        "source_commit": _CURRENT_COMMIT,
        "source_manifest_identity": "1" * 64,
        "config_identity": config_identity,
        "run_identity": run_identity,
        "world_size": 8,
        "experiment_mode": "visual_only_forensic_fork",
    }
    shard_hashes = {}
    for rank in range(8):
        shard = path / f"rank_{rank:05d}_of_00008.pt"
        shard.write_bytes(f"{path.name}:{rank}".encode())
        shard_hash = hashlib.sha256(shard.read_bytes()).hexdigest()
        shard_hashes[str(rank)] = shard_hash
        (path / f"rank_{rank:05d}_of_00008.json").write_text(
            json.dumps({
                "schema": "nimloth_sft1_query_state_rank_checkpoint_v1",
                "rank": rank,
                "world_size": 8,
                "identity": identity,
                "shard_sha256": shard_hash,
            }, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    finalize_query_state_rank_checkpoint(
        path,
        control=QueryStateDistributedControl(
            identity=QueryStateResumeIdentity(**identity),
            global_step=update,
            data_cursor={"next_update": update + 1},
            metric_cursor={"validation": update},
            terminal_primary=False,
            forensic_only=False,
        ),
    )
    control_hash = hashlib.sha256((path / "control.json").read_bytes()).hexdigest()
    return path, control_hash


def _visual_store(
    root: Path,
    *,
    expected_source_commit: str = _CURRENT_COMMIT,
    run_identity: str = "e" * 64,
    semantic_identity: str | None = None,
) -> QueryStateSegmentStore:
    stable_semantics = semantic_identity or run_identity
    expected_identity = QueryStateResumeIdentity(
        source_commit=expected_source_commit,
        source_manifest_identity="1" * 64,
        config_identity=stable_semantics,
        run_identity=run_identity,
        world_size=8,
        experiment_mode="visual_only_forensic_fork",
    )
    return QueryStateSegmentStore(
        root,
        run_identity=expected_identity.run_identity,
        mode=expected_identity.experiment_mode,
        wandb_run_id="fork-run",
        base_update=1605,
        epoch_updates=1605,
        semantic_identity=stable_semantics,
        expected_checkpoint_identity=expected_identity,
    )


def _commit(store: QueryStateSegmentStore, checkpoint_root: Path, start: int, end: int):
    segment = store.begin_segment(start_update=start, end_update=end, process_identity=f"p-{end}")
    records = tuple({"update": update} for update in range(start + 1, end + 1))
    for record in records:
        segment.append_update(record)
    checkpoint, control_hash = _checkpoint(
        checkpoint_root / f"update_{end:08d}",
        run_identity=store.run_identity,
        config_identity=store.expected_checkpoint_identity.config_identity,
        update=end,
    )
    return segment.commit(
        checkpoint_path=checkpoint,
        checkpoint_control_hash=control_hash,
        data_cursor={"next_update": end + 1},
        metric_cursor={"validation": end},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=records,
    )


def test_visual_fork_plan_has_offset_four_epochs_report_only_validation() -> None:
    plan = build_visual_fork_event_plan(
        schedule_start_update=1605,
        epoch_updates=1605,
        checkpoint_cadence_updates=321,
        fixed_additional_epochs=4,
    )
    assert plan[0].update == 1605 and plan[0].kind == "calibration_parity"
    assert tuple(event.update for event in plan if event.kind == "checkpoint") == tuple(range(1926, 8026, 321))
    assert tuple(event.update for event in plan if event.kind == "calibration") == (3210, 4815, 6420, 8025)
    assert tuple(event.update for event in plan if event.kind == "holdout") == (8025,)
    assert not any(event.kind in {"early_stop", "best", "actor_hard_stop", "generation_hard_stop", "terminal_primary"} for event in plan)


def test_visual_fork_first_checkpoint_and_exact_restart_mirror_from_schedule_offset(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    first = _commit(store, checkpoint_root, 1605, 1926)

    initial_cursor = _initial_durable_cursor(schedule_start_update=1605)
    assert initial_cursor == 1605
    mirror = QueryStateWandbMirror(
        run_id="fork-run",
        world_size=1,
        initial_cursor=initial_cursor,
    )
    mirror.register_authoritative(first)
    assert mirror.pending_updates()[0] == 1606
    assert mirror.pending_updates()[-1] == 1926
    mirror.replay(run_id="fork-run", updates=mirror.pending_updates())
    assert mirror.cursor == 1926

    replay = _authoritative_entries_for_restart(
        store,
        checkpoint_update=1926,
        mirror_cursor=1605,
    )
    restarted_mirror = QueryStateWandbMirror(
        run_id="fork-run",
        world_size=1,
        initial_cursor=initial_cursor,
    )
    restarted_mirror.register_authoritative(replay[0])
    restarted_mirror.replay(
        run_id="fork-run",
        updates=restarted_mirror.pending_updates(),
    )
    assert restarted_mirror.cursor == 1926


def test_visual_fork_durable_store_reopens_on_stable_semantics_and_rejects_drift(
    tmp_path: Path,
) -> None:
    fresh = parse_query_state_training_config(
        _training_raw(mode="visual_only_forensic_fork", resume_mode="fresh")
    )
    restart_raw = _training_raw(
        mode="visual_only_forensic_fork", resume_mode="exact_restart"
    )
    restart_raw["initialization"]["resume_checkpoint"] = (
        "/outputs/visual_only_forensic_fork/checkpoints/update_00001926"
    )
    restart = parse_query_state_training_config(restart_raw)
    assert fresh.identity != restart.identity
    stable_identity = query_state_training_run_identity(fresh)
    assert query_state_training_run_identity(restart) == stable_identity

    root = tmp_path / "durable"
    _visual_store(root, run_identity=stable_identity, semantic_identity=stable_identity)
    _visual_store(root, run_identity=stable_identity, semantic_identity=stable_identity)

    mutations = (
        ("optimizer", "language_learning_rate", 2e-6),
        ("schedule", "max_updates", 6420),
        ("data", "train_manifest_identity", "9" * 64),
        ("source", "commit", "9" * 40),
    )
    for section, field, value in mutations:
        raw = _training_raw(mode="visual_only_forensic_fork", resume_mode="exact_restart")
        raw["initialization"]["resume_checkpoint"] = (
            "/outputs/visual_only_forensic_fork/checkpoints/update_00001926"
        )
        raw[section][field] = value
        with pytest.raises((ValueError, TypeError)):
            drifted = parse_query_state_training_config(raw)
            drift_identity = query_state_training_run_identity(drifted)
            _visual_store(
                root,
                run_identity=drift_identity,
                semantic_identity=drift_identity,
            )


def test_visual_fork_model_initialization_loads_only_model_and_keeps_fresh_state(tmp_path: Path) -> None:
    root = nn.Module()
    root.objective = nn.Module()
    root.objective.projector = DirectSlotProjector()
    checkpoint, failure, identity = _write_forensic_checkpoint(tmp_path, root)
    expected = root.objective.projector.linear.weight.detach().clone()
    with torch.no_grad():
        root.objective.projector.linear.weight.zero_()
    fresh_optimizer = torch.optim.AdamW(root.parameters(), lr=1e-4)
    fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda _: 1.0)
    config_raw = _training_raw(mode="visual_only_forensic_fork")
    assert config_raw["source"]["source_manifest_identity"] != identity.source_manifest_identity
    config_raw["forensic_fork"].update(
        ancestor_source_commit=identity.source_commit,
        ancestor_source_manifest_identity=identity.source_manifest_identity,
        ancestor_checkpoint_path=str(checkpoint.resolve()),
        ancestor_failure_manifest_path=str(failure.resolve()),
        ancestor_control_sha256=hashlib.sha256(
            (checkpoint / "control.json").read_bytes()
        ).hexdigest(),
        ancestor_run_identity=identity.run_identity,
        ancestor_source_config_identity=identity.config_identity,
    )
    config_raw["model"]["initialization_identity"] = (
        "formal38_forensic_model_only:"
        + config_raw["forensic_fork"]["ancestor_control_sha256"]
    )
    config_raw["artifacts"]["file_sha256"].update({
        str(checkpoint / "control.json"): config_raw["forensic_fork"]["ancestor_control_sha256"],
        str(failure): hashlib.sha256(failure.read_bytes()).hexdigest(),
        **{
            str(checkpoint / f"rank_{rank:05d}_of_00008{suffix}"): hashlib.sha256(
                (checkpoint / f"rank_{rank:05d}_of_00008{suffix}").read_bytes()
            ).hexdigest()
            for rank in range(8)
            for suffix in (".pt", ".json")
        },
    })
    config = parse_query_state_training_config(config_raw)
    before_python_rng = random.getstate()
    before_numpy_rng = np.random.get_state()
    before_torch_rng = torch.get_rng_state().clone()
    receipt = initialize_visual_fork_from_forensic_model(
        config,
        root=root,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        rank=0,
        expected_ancestor_identity=identity,
    )
    torch.testing.assert_close(root.objective.projector.linear.weight, expected)
    assert not fresh_optimizer.state
    assert fresh_scheduler.last_epoch == 0
    assert random.getstate() == before_python_rng
    assert np.random.get_state()[0] == before_numpy_rng[0]
    assert np.array_equal(np.random.get_state()[1], before_numpy_rng[1])
    assert np.random.get_state()[2:] == before_numpy_rng[2:]
    torch.testing.assert_close(torch.get_rng_state(), before_torch_rng)
    assert receipt.model_loaded is True
    assert receipt.optimizer_restored is False
    assert receipt.scheduler_restored is False
    assert receipt.rng_restored is False
    assert receipt.data_cursor_restored is False
    assert receipt.wandb_cursor_restored is False
    assert checkpoint.is_dir() and failure.is_file()


def test_successor_first_compaction_keeps_epoch_finals_and_recovery_uses_payload_present_latest(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    first = _commit(store, checkpoint_root, 1605, 1926)
    second = _commit(store, checkpoint_root, 1926, 2247)
    with pytest.raises(ValueError, match="W&B mirror"):
        store.compact_superseded_checkpoint(
            candidate_update=1926,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=1926,
        )
    receipt = store.compact_superseded_checkpoint(
        candidate_update=1926,
        checkpoint_root=checkpoint_root,
        mirrored_through_update=2247,
    )
    assert receipt.candidate_update == 1926
    assert receipt.inventory_hash == hashlib.sha256(
        Path(receipt.inventory_path).read_bytes()
    ).hexdigest()
    assert receipt.tombstone_hash == hashlib.sha256(
        Path(receipt.tombstone_path).read_bytes()
    ).hexdigest()
    assert not (Path(first.checkpoint_path) / "rank_00000_of_00008.pt").exists()
    assert (Path(first.checkpoint_path) / "control.json").is_file()
    assert (Path(second.checkpoint_path) / "rank_00000_of_00008.pt").is_file()
    compacted, latest = store.authoritative_entries()
    assert compacted.checkpoint_payload_present is False
    assert compacted.resumable is False
    assert compacted.compaction_manifest_path is not None
    assert latest.checkpoint_payload_present is True
    assert latest.resumable is True
    assert store.recover(checkpoint_root=checkpoint_root).resume_update == 2247

    _commit(store, checkpoint_root, 2247, 3210)
    _commit(store, checkpoint_root, 3210, 3531)
    with pytest.raises(ValueError, match="epoch-final"):
        store.compact_superseded_checkpoint(
            candidate_update=3210,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=3531,
        )
    with pytest.raises(ValueError, match="latest"):
        store.compact_superseded_checkpoint(
            candidate_update=3531,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=3531,
        )


def test_partial_compaction_failure_marks_candidate_nonresumable_without_harming_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    first = _commit(store, checkpoint_root, 1605, 1926)
    second = _commit(store, checkpoint_root, 1926, 2247)
    original_unlink = Path.unlink

    def fail_after_first(path: Path, *args, **kwargs):
        if path.name == "rank_00001_of_00008.pt":
            raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_after_first)
    with pytest.raises(OSError, match="injected unlink failure"):
        store.compact_superseded_checkpoint(
            candidate_update=1926,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=2247,
        )
    partial, latest = store.authoritative_entries()
    assert partial.checkpoint_payload_present is True
    assert partial.resumable is False
    assert partial.compaction_manifest_path is not None
    failure = json.loads(Path(partial.compaction_manifest_path).read_text(encoding="utf-8"))
    assert failure["removed_payload_count"] == 1
    assert failure["payloads_remaining"] == 7
    assert latest.checkpoint_payload_present is True
    assert latest.resumable is True
    assert (Path(second.checkpoint_path) / "rank_00000_of_00008.pt").is_file()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert store.recover(checkpoint_root=checkpoint_root).resume_update == 2247
    compacted, _latest = store.authoritative_entries()
    assert compacted.checkpoint_payload_present is False
    assert compacted.resumable is False
    assert not tuple(Path(first.checkpoint_path).glob("rank_*_of_*.pt"))


def test_compaction_never_discovers_or_deletes_formal38_ancestor(tmp_path: Path) -> None:
    ancestor = tmp_path / "formal38" / "forensics" / "unsafe_update_00001605"
    ancestor.mkdir(parents=True)
    protected = ancestor / "rank_00000_of_00008.pt"
    protected.write_bytes(b"formal38-protected")
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    _commit(store, checkpoint_root, 1605, 1926)
    _commit(store, checkpoint_root, 1926, 2247)
    store.compact_superseded_checkpoint(
        candidate_update=1926,
        checkpoint_root=checkpoint_root,
        mirrored_through_update=2247,
    )
    assert protected.read_bytes() == b"formal38-protected"
    indexed_paths = {Path(entry.checkpoint_path).resolve() for entry in store.authoritative_entries()}
    assert ancestor.resolve() not in indexed_paths


def _write_interrupted_compaction(
    store: QueryStateSegmentStore,
    *,
    candidate_update: int,
    successor_update: int,
    remove_payloads: int,
) -> Path:
    candidate = next(
        entry for entry in store.authoritative_entries()
        if entry.end_update == candidate_update
    )
    candidate_path = Path(candidate.checkpoint_path)
    payloads = tuple(sorted(candidate_path.glob("rank_*_of_*.pt")))
    compaction_root = store.root / "compactions" / f"update_{candidate_update:08d}"
    compaction_root.mkdir(parents=True)
    inventory = {
        "schema": QUERY_STATE_SEGMENT_SCHEMA,
        "kind": "checkpoint_payload_inventory",
        "run_identity": store.run_identity,
        "mode": store.mode,
        "semantic_identity": store.semantic_identity,
        "candidate_update": candidate_update,
        "successor_update": successor_update,
        "checkpoint_path": str(candidate_path.resolve()),
        "payloads": [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in payloads
        ],
    }
    inventory_path = compaction_root / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True) + "\n", encoding="utf-8")
    tombstone = {
        "schema": QUERY_STATE_SEGMENT_SCHEMA,
        "kind": "checkpoint_payload_tombstone_intent",
        "run_identity": store.run_identity,
        "mode": store.mode,
        "semantic_identity": store.semantic_identity,
        "candidate_update": candidate_update,
        "successor_update": successor_update,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
    }
    tombstone_path = compaction_root / "tombstone_intent.json"
    tombstone_path.write_text(json.dumps(tombstone, sort_keys=True) + "\n", encoding="utf-8")
    for path in payloads[:remove_payloads]:
        path.unlink()
    return compaction_root


@pytest.mark.parametrize("remove_payloads", [0, 1, 8])
def test_recovery_reconciles_every_interrupted_compaction_crash_point(
    tmp_path: Path,
    remove_payloads: int,
) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    _commit(store, checkpoint_root, 1605, 1926)
    successor = _commit(store, checkpoint_root, 1926, 2247)
    compaction_root = _write_interrupted_compaction(
        store,
        candidate_update=1926,
        successor_update=2247,
        remove_payloads=remove_payloads,
    )

    restarted = _visual_store(store.root)
    recovery = restarted.recover(checkpoint_root=checkpoint_root)
    candidate, latest = restarted.authoritative_entries()
    assert recovery.resume_update == 2247
    assert candidate.checkpoint_payload_present is False
    assert candidate.resumable is False
    assert latest.checkpoint_path == successor.checkpoint_path
    assert latest.checkpoint_payload_present is True and latest.resumable is True
    reconciliation = compaction_root / "RECONCILED.json"
    assert reconciliation.is_file()
    evidence = json.loads(reconciliation.read_text(encoding="utf-8"))
    assert evidence["kind"] == "interrupted_checkpoint_payload_compaction_reconciled"
    assert evidence["payloads_missing_before_reconcile"] == remove_payloads
    assert evidence["removed_payload_count"] == 8 - remove_payloads
    assert evidence["total_payload_count"] == 8
    candidate_path = Path(candidate.checkpoint_path)
    assert not tuple(candidate_path.glob("rank_*_of_*.pt"))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rank",
        "tampered_rank",
        "tampered_manifest",
        "manifest_schema",
        "manifest_structure",
        "source",
        "mode",
        "config",
        "update",
    ],
)
def test_compaction_authenticates_complete_successor_inventory_and_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    checkpoint_root = tmp_path / mutation / "checkpoints"
    store = _visual_store(tmp_path / mutation / "durable")
    _commit(store, checkpoint_root, 1605, 1926)
    successor = _commit(store, checkpoint_root, 1926, 2247)
    successor_path = Path(successor.checkpoint_path)
    if mutation == "missing_rank":
        (successor_path / "rank_00007_of_00008.pt").unlink()
    elif mutation == "tampered_rank":
        (successor_path / "rank_00007_of_00008.pt").write_bytes(b"tampered")
    elif mutation == "tampered_manifest":
        (successor_path / "rank_00007_of_00008.json").write_text("{}\n", encoding="utf-8")
    elif mutation in {"manifest_schema", "manifest_structure"}:
        manifest_path = successor_path / "rank_00007_of_00008.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "manifest_schema":
            manifest["schema"] = "rewritten_checkpoint_schema"
        else:
            manifest["untrusted_extra"] = True
        manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    else:
        control_path = successor_path / "control.json"
        control = json.loads(control_path.read_text(encoding="utf-8"))
        if mutation == "source":
            control["identity"]["source_commit"] = "9" * 40
            control["source_commit"] = "9" * 40
        elif mutation == "mode":
            control["identity"]["experiment_mode"] = "formal"
        elif mutation == "config":
            control["identity"]["config_identity"] = "0" * 64
            control["config_identity"] = "0" * 64
        else:
            control["global_step"] = 9999
        control_path.write_text(json.dumps(control, sort_keys=True) + "\n", encoding="utf-8")
        digest = hashlib.sha256(control_path.read_bytes()).hexdigest()
        (successor_path / "COMPLETED").write_text(
            f"control_sha256={digest}\n", encoding="utf-8"
        )
        index_path = store.root / "authoritative_index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["entries"][-1]["checkpoint_control_hash"] = digest
        index_path.write_text(json.dumps(index, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="authenticated rank inventory|identity|schema|contract"):
        store.compact_superseded_checkpoint(
            candidate_update=1926,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=2247,
        )


def test_compaction_rejects_store_expected_identity_mismatch(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "expected-mismatch" / "checkpoints"
    store = _visual_store(
        tmp_path / "expected-mismatch" / "durable",
        expected_source_commit="9" * 40,
    )
    _commit(store, checkpoint_root, 1605, 1926)
    _commit(store, checkpoint_root, 1926, 2247)
    with pytest.raises(ValueError, match="identity"):
        store.compact_superseded_checkpoint(
            candidate_update=1926,
            checkpoint_root=checkpoint_root,
            mirrored_through_update=2247,
        )


def test_visual_fixed_budget_completion_commits_and_recovers_at_update8025(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    segment = store.begin_segment(
        start_update=1605,
        end_update=8025,
        process_identity="fixed-budget-process",
    )
    records = tuple({"update": update} for update in range(1606, 8026))
    for record in records:
        segment.append_update(record)
    checkpoint, control_hash = _checkpoint(
        checkpoint_root / "update_00008025",
        run_identity=store.run_identity,
        config_identity=store.expected_checkpoint_identity.config_identity,
        update=8025,
    )
    completion = {
        "kind": "visual_fixed_budget_diagnostic_complete",
        "epoch": 5,
        "update": 8025,
        "terminal_primary": False,
        "holdout_controls_selection": False,
        "best_checkpoint": None,
    }
    entry = segment.commit(
        checkpoint_path=checkpoint,
        checkpoint_control_hash=control_hash,
        data_cursor={"next_update": 8026},
        metric_cursor={
            "validation": 8025,
            "early_stopping": None,
            "actual_terminal": None,
            "visual_fixed_budget_completion": completion,
        },
        validation={
            "due": True,
            "actual_terminal": None,
            "visual_fixed_budget_completion": completion,
        },
        safety={"passed": True, "report_only": True},
        mirror_records=records,
    )
    assert entry.visual_fixed_budget_completion == completion
    assert entry.actual_terminal is None
    assert store.recover(checkpoint_root=checkpoint_root).resume_update == 8025


def test_recovery_finishes_compaction_completed_before_index_update(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "fork" / "checkpoints"
    store = _visual_store(tmp_path / "fork" / "durable")
    _commit(store, checkpoint_root, 1605, 1926)
    _commit(store, checkpoint_root, 1926, 2247)
    index_path = store.root / "authoritative_index.json"
    before_index = index_path.read_bytes()
    store.compact_superseded_checkpoint(
        candidate_update=1926,
        checkpoint_root=checkpoint_root,
        mirrored_through_update=2247,
    )
    index_path.write_bytes(before_index)

    restarted = _visual_store(store.root)
    assert restarted.recover(checkpoint_root=checkpoint_root).resume_update == 2247
    candidate, successor = restarted.authoritative_entries()
    assert candidate.checkpoint_payload_present is False
    assert candidate.resumable is False
    assert successor.checkpoint_payload_present is True
