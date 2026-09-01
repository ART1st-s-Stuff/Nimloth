from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nimloth.training.sft1 import query_state_training_backend
from nimloth.training.sft1.query_state_training_backend import (
    _FormalTrackingOwner,
    _actor_baseline_path,
    _approved_pause_due,
    _coordinate_early_stopping_decision,
    _forensic_metric_cursor,
    _global_teacher_memo_metric,
    _authoritative_entries_for_restart,
    _index_training_rows,
    _load_actor_baseline,
    _publish_actor_baseline,
    _recover_first_boundary_crash,
    _run_generation_format_probe,
    _validation_boundary_plan,
    _validate_formal_restart_early_stopping_cursor,
    build_query_state_training_updates,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
)
from nimloth.training.sft1.query_state_training_runtime import (
    QueryStateEarlyStoppingCursor,
    QueryStateEarlyStoppingDecision,
    QueryStateSegmentStore,
    QueryStateWandbMirror,
)
from tests.training.sft1.test_query_state_training_config import _raw


_SHA = "a" * 64


def _checkpoint(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "control.json").write_text("{}\n", encoding="utf-8")
    (path / "COMPLETED").write_text(
        f"control_sha256={_SHA}\n", encoding="utf-8"
    )
    return path


def _commit_segment(
    store: QueryStateSegmentStore,
    root: Path,
    *,
    start: int,
    end: int,
):
    segment = store.begin_segment(
        start_update=start,
        end_update=end,
        process_identity=f"process-{start}",
    )
    records = tuple({"update": update, "loss": float(update)} for update in range(start + 1, end + 1))
    for record in records:
        segment.append_update(record)
    return segment.commit(
        checkpoint_path=_checkpoint(root / f"checkpoint-{end}"),
        checkpoint_control_hash=_SHA,
        data_cursor={"next_update": end + 1},
        metric_cursor={"validation": 0, "log": end, "wandb": start},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=records,
    )


class _Run:
    def __init__(self) -> None:
        self.logged: list[tuple[int, dict[str, Any]]] = []

    def log(self, record: dict[str, Any], *, step: int, commit: bool) -> None:
        assert commit is True
        self.logged.append((step, dict(record)))


def _tracking_owner(*, initial_cursor: int = 0) -> tuple[_FormalTrackingOwner, _Run]:
    config = parse_query_state_training_config(_raw(mode="formal"))
    owner = _FormalTrackingOwner(config, rank=0, world_size=1)
    run = _Run()
    owner.run = run
    owner.mirror = QueryStateWandbMirror(
        run_id=config.tracking.run_id,
        world_size=1,
        initial_cursor=initial_cursor,
    )
    return owner, run


def test_formal_tracking_queries_wandb_by_storage_name_not_public_id_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    queries: list[tuple[str, dict[str, str]]] = []

    class _Api:
        def runs(self, path: str, *, filters: dict[str, str]):
            queries.append((path, filters))
            return []

    run = SimpleNamespace(
        entity=config.tracking.entity,
        project=config.tracking.project,
        id=config.tracking.run_id,
        group=config.tracking.group,
        name=config.tracking.run_name,
        url="https://wandb.example/fresh",
    )
    fake_wandb = SimpleNamespace(
        Api=lambda: _Api(),
        init=lambda **_kwargs: run,
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    owner = _FormalTrackingOwner(config, rank=0, world_size=1)
    owner.initialize(0)

    assert queries == [
        (
            f"{config.tracking.entity}/{config.tracking.project}",
            {"name": config.tracking.run_id},
        )
    ]
    assert owner.run is run
    assert owner.mirror is not None


def test_early_stop_verdict_requires_exact_all_rank_consensus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cursor = QueryStateEarlyStoppingCursor(
        best_composite=4.0,
        last_composite=4.1,
        best_epoch=1,
        bad_epochs=1,
        last_epoch=2,
        last_update=3210,
        terminal_epoch=None,
        terminal_update=None,
        stop_reason=None,
    )
    decision = QueryStateEarlyStoppingDecision(
        cursor=cursor,
        composite=4.1,
        improved=False,
        should_stop=False,
        reason=None,
    )
    monkeypatch.setattr(
        query_state_training_backend,
        "_all_gather",
        lambda value, world_size: (value,) * world_size,
    )
    _coordinate_early_stopping_decision(decision, world_size=8)

    monkeypatch.setattr(
        query_state_training_backend,
        "_all_gather",
        lambda value, world_size: (value,) * (world_size - 1) + ({**value, "composite": 3.0},),
    )
    with pytest.raises(RuntimeError, match="differs across ranks"):
        _coordinate_early_stopping_decision(decision, world_size=8)


def test_teacher_memo_metric_gathers_process_local_reports_in_rank_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = {
        "process_identity": "rank-0-process",
        "dino_identity": "facebook/dinov2-large@commit",
        "entries": 7,
        "current_bytes": 7 * 16 * 1024 * 4,
        "peak_bytes": 7 * 16 * 1024 * 4,
    }
    remote = {
        **local,
        "process_identity": "rank-1-process",
        "entries": 6,
        "current_bytes": 6 * 16 * 1024 * 4,
        "peak_bytes": 6 * 16 * 1024 * 4,
    }
    gathered: list[tuple[object, int]] = []

    def all_gather(value: object, world_size: int):
        gathered.append((value, world_size))
        return [value, remote]

    monkeypatch.setattr(query_state_training_backend, "_all_gather", all_gather)

    metric = _global_teacher_memo_metric(local, world_size=2)

    assert gathered == [(local, 2)]
    assert metric == {
        "scope": "process_local_by_rank",
        "reports": [local, remote],
    }


def test_backend_runtime_reindex_uses_data_only_contract_and_strict_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_query_state_training_config(_raw(mode="pilot"))
    expected_rows = (SimpleNamespace(identity="row-a"),)
    expected_audit = object()
    calls: list[tuple[object, bool]] = []

    def index_rows(contract: object, *, enforce_approved_counts: bool):
        calls.append((contract, enforce_approved_counts))
        assert hasattr(contract, "data")
        assert not hasattr(contract, "selection")
        return expected_rows, expected_audit

    validated: list[object] = []
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend.index_early4_rows",
        index_rows,
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend.validate_query_state_row_audit",
        validated.append,
    )

    rows, audit = _index_training_rows(config)

    assert rows == expected_rows
    assert audit is expected_audit
    assert calls and calls[0][1] is False
    assert validated == [expected_audit]


def test_formal_ws8_validation_plan_is_calibration_control_holdout_primary() -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    epoch_one = _validation_boundary_plan(
        config, update=1605, epoch=1, actual_terminal=False
    )
    assert epoch_one == {
        "calibration": True,
        "holdout": False,
        "generation_format": False,
        "actual_terminal": False,
    }
    epoch_two = _validation_boundary_plan(
        config, update=3210, epoch=2, actual_terminal=False
    )
    assert epoch_two["calibration"] is True
    assert epoch_two["holdout"] is True
    assert epoch_two["generation_format"] is True
    early_terminal = _validation_boundary_plan(
        config, update=4815, epoch=3, actual_terminal=True
    )
    assert early_terminal == {
        "calibration": True,
        "holdout": True,
        "generation_format": True,
        "actual_terminal": True,
    }


def test_backend_uses_one_stable_run_identity_across_exact_restart_delta() -> None:
    fresh = parse_query_state_training_config(_raw(mode="formal", resume_mode="fresh"))
    restart = parse_query_state_training_config(
        _raw(mode="formal", resume_mode="exact_restart")
    )
    assert fresh.identity != restart.identity
    assert query_state_training_run_identity(fresh) == query_state_training_run_identity(restart)

    changed = deepcopy(_raw(mode="formal", resume_mode="exact_restart"))
    changed["optimizer"]["language_learning_rate"] *= 2
    with pytest.raises(ValueError, match="learning rate|LR|formal"):
        parse_query_state_training_config(changed)

    early_stop_changed = deepcopy(_raw(mode="formal", resume_mode="exact_restart"))
    early_stop_changed["early_stopping"]["patience_epochs"] = 3
    changed_config = parse_query_state_training_config(early_stop_changed)
    assert query_state_training_run_identity(changed_config) != query_state_training_run_identity(fresh)


def test_backend_schedule_is_deterministic_complete_and_resume_slices_updates() -> None:
    one = build_query_state_training_updates(
        tuple(range(8)),
        epochs=2,
        seed=17,
        rank=0,
        world_size=2,
        rows_per_rank_update=2,
        expected_updates=4,
    )
    two = build_query_state_training_updates(
        tuple(range(8)),
        epochs=2,
        seed=17,
        rank=0,
        world_size=2,
        rows_per_rank_update=2,
        expected_updates=4,
    )
    assert one == two
    assert len(one) == 4
    assert all(len(update) == 2 for update in one)
    consumed = [item.ordinal for update in one for item in update if item.row_valid]
    assert len(consumed) == 8
    assert set(consumed) <= set(range(8))
    peer = build_query_state_training_updates(
        tuple(range(8)),
        epochs=2,
        seed=17,
        rank=1,
        world_size=2,
        rows_per_rank_update=2,
        expected_updates=4,
    )
    global_consumed = consumed + [
        item.ordinal for update in peer for item in update if item.row_valid
    ]
    assert sorted(global_consumed) == sorted(tuple(range(8)) * 2)
    assert tuple(one[2:]) == tuple(two[2:])

    with pytest.raises(ValueError, match="max_updates"):
        build_query_state_training_updates(
            tuple(range(8)),
            epochs=2,
            seed=17,
            rank=0,
            world_size=2,
            rows_per_rank_update=2,
            expected_updates=3,
        )


def test_id176_actor_baseline_is_immutable_identity_bound_and_reloadable(
    tmp_path: Path,
) -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    baseline = {
        "row-a": tuple(float(index) for index in range(8)),
        "row-b": tuple(float(index + 1) for index in range(8)),
    }
    path = _actor_baseline_path(tmp_path)
    identity = _publish_actor_baseline(path, config=config, baseline=baseline)
    restored, restored_identity = _load_actor_baseline(path, config=config)
    assert restored == baseline
    assert restored_identity == identity
    with pytest.raises(FileExistsError):
        _publish_actor_baseline(path, config=config, baseline=baseline)
    raw = __import__("json").loads(path.read_text(encoding="utf-8"))
    raw["rows"][0]["action_logits"][0] += 1.0
    path.write_text(__import__("json").dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_actor_baseline(path, config=config)


def test_backend_generation_format_evidence_is_exact_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_query_state_training_config(_raw(mode="pilot"))
    entry = SimpleNamespace(
        row_identity="row-a", ordinal=1, record_id="record-a", prompt_identity="b" * 64
    )
    manifest = SimpleNamespace(
        entries=(entry,),
        split="calibration",
        identity="c" * 64,
        prompt_protocol_identity="d" * 64,
        turn_generation_spec_identity="e" * 64,
        parser_protocol_identity="f" * 64,
        max_reasoning_tokens=8,
        max_output_tokens=32,
    )
    assembly = SimpleNamespace(
        generation_format_manifest=manifest,
        generation_spec=SimpleNamespace(),
        processor=SimpleNamespace(tokenizer=SimpleNamespace()),
        distributed_worker=SimpleNamespace(root=object()),
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend.validation_mode",
        lambda _root: nullcontext(),
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend._generation_prompt_inputs",
        lambda _assembly, entry: ({"input_ids": torch.tensor([[1]])}, entry.prompt_identity),
    )
    parsed = SimpleNamespace(
        response="<think>real generated</think><queries><action>",
        thought="real generated",
        action_index=2,
        action_token_id=7,
        close_end=3,
        reasoning_truncated=False,
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend.run_fsdp_greedy_turn_probe",
        lambda *_args, **kwargs: SimpleNamespace(
            prompt_identity=kwargs["prompt_identity"],
            checkpoint_identity=kwargs["checkpoint_identity"],
            spec_identity="e" * 64,
            continuation_token_ids=(1, 2, 3),
            parsed=parsed,
            used_current_model_logits=True,
            action_executed=False,
            rollout_persisted=False,
            deployable_materialized=False,
        ),
    )
    evidence = _run_generation_format_probe(
        config, assembly, update=0, world_size=1
    )
    assert evidence["passed"] is True
    assert evidence["parsed_row_count"] == 1
    assert evidence["records"][0]["thought"] == "real generated"
    assert evidence["records"][0]["used_current_model_logits"] is True
    assert evidence["action_execution"] is False
    assert evidence["rollout_persistence"] is False
    assert evidence["deployable_export"] is False

    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_backend.run_fsdp_greedy_turn_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("unparseable")),
    )
    failed = _run_generation_format_probe(
        config, assembly, update=32, world_size=1
    )
    assert failed["passed"] is False
    assert failed["non_resumable_safety_failure"] is True
    assert failed["failure"]["stage"] == "current_fsdp_greedy_parse"


def test_backend_run_identity_binds_network_environment() -> None:
    raw = _raw(mode="pilot")
    first = parse_query_state_training_config(raw)
    changed = deepcopy(raw)
    changed["environment"]["nccl_socket_ifname"] = "test1"
    second = parse_query_state_training_config(changed)
    assert query_state_training_run_identity(first) != query_state_training_run_identity(
        second
    )


def test_backend_validation_wires_global_diagnostics_and_fail_closed_safety() -> None:
    source = Path(query_state_training_backend.__file__).read_text(encoding="utf-8")
    assert "controlled_gather_query_state_diagnostics(" in source
    assert "compute_query_state_diagnostics(" in source
    assert "evaluate_actor_safety(" in source
    assert "VALIDATOR_FAILED" not in source  # controller owns the filename
    assert "not_evaluated_by_p5_backend" not in source
    assert "actor safety failed; forensic checkpoint preserved " in source
    assert "without authoritative index" in source
    assert 'run_root / "forensics" / f"unsafe_update_{segment_end:08d}"' in source
    assert '"forensic_checkpoint_preserved": True' in source
    assert 'checkpoint_path=run_root / "unsafe_checkpoint_not_published"' not in source
    assert 'config.validation["generation_format_updates"]' in source
    assert "run_fsdp_greedy_turn_probe(" in source
    assert "baseline safety failed; forensic checkpoint " in source
    assert 'f"unsafe_update_{baseline_update:08d}"' in source
    assert '"unsafe_update_00000000"' not in source
    assert "record_forensic_save_failure(" in source
    assert "forensic checkpoint save failed" in source
    assert "normalization = query_state_global_normalization(" in source
    assert "device=validation_device" in source
    assert "QueryStateNormalization(1, 1, world_size)" not in source
    assert 'split="calibration"' in source
    assert 'split="holdout"' in source
    assert 'epoch_updates = int(config.schedule["epoch_updates"])' in source
    assert "segment_end % epoch_updates == 0" in source
    assert "segment_end // epoch_updates" in source
    assert "segment_end // cadence" not in source
    assert "no_validation_due_at_sub_epoch_commit" in source
    assert '"holdout_controls_early_stop": False' in source
    assert '"actual_terminal_reason": terminal_reason' in source
    assert "terminal_primary=(config.mode == \"formal\" and actual_terminal is not None)" in source


def test_approved_pause_is_due_only_at_exact_nonterminal_boundary() -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    assert _approved_pause_due(config, segment_end=1605) is False
    paused_raw = _raw(mode="formal")
    paused_raw["schedule"]["approved_pause_update"] = 1605
    paused = parse_query_state_training_config(paused_raw)
    assert _approved_pause_due(paused, segment_end=321) is False
    assert _approved_pause_due(paused, segment_end=1605) is True
    assert _approved_pause_due(paused, segment_end=3210) is False


def test_forensic_metric_cursor_rejects_terminal_candidate_without_losing_evidence() -> None:
    terminal = {"epoch": 2, "update": 3210, "terminal_primary": True}
    early = {
        "last_epoch": 2,
        "last_update": 3210,
        "terminal_epoch": 2,
        "terminal_update": 3210,
        "stop_reason": "converged_early_stop",
    }
    forensic = _forensic_metric_cursor(
        {"actual_terminal": terminal, "early_stopping": early, "log": 3210}
    )
    assert forensic["actual_terminal"] is None
    assert forensic["rejected_terminal_candidate"] == terminal
    assert forensic["early_stopping"] == {
        **early,
        "terminal_epoch": None,
        "terminal_update": None,
        "stop_reason": None,
    }
    assert forensic["rejected_early_stopping_cursor"] == early
    assert forensic["log"] == 3210


def test_formal_restart_accepts_sub_epoch_cursor_without_advancing_patience() -> None:
    initial = QueryStateEarlyStoppingCursor.initial()
    _validate_formal_restart_early_stopping_cursor(
        initial,
        start_update=321,
        epoch_updates=1605,
    )
    epoch_one = QueryStateEarlyStoppingCursor(
        best_composite=4.0,
        last_composite=4.0,
        best_epoch=1,
        bad_epochs=0,
        last_epoch=1,
        last_update=1605,
        terminal_epoch=None,
        terminal_update=None,
        stop_reason=None,
    )
    _validate_formal_restart_early_stopping_cursor(
        epoch_one,
        start_update=1605,
        epoch_updates=1605,
    )
    _validate_formal_restart_early_stopping_cursor(
        epoch_one,
        start_update=1926,
        epoch_updates=1605,
    )
    with pytest.raises(ValueError, match="early-stop cursor is not exact"):
        _validate_formal_restart_early_stopping_cursor(
            epoch_one,
            start_update=321,
            epoch_updates=1605,
        )


def test_backend_first_boundary_crash_replay_quarantines_checkpoint_path(
    tmp_path: Path,
) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "durable",
        run_identity=_SHA,
        mode="pilot",
    )
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "update_00000002"
    segment = store.begin_segment(
        start_update=0,
        end_update=2,
        process_identity="first-process",
    )
    segment.append_update({"update": 1})
    segment.append_update({"update": 2})
    with pytest.raises(RuntimeError, match="before authoritative index"):
        segment.commit(
            checkpoint_path=_checkpoint(checkpoint),
            checkpoint_control_hash=_SHA,
            data_cursor={"next_update": 3},
            metric_cursor={"validation": 0, "log": 2, "wandb": 0},
            validation={"due": False},
            safety={"passed": True},
            mirror_records=({"update": 1}, {"update": 2}),
            fail_before_index=True,
        )

    recovery = _recover_first_boundary_crash(
        store,
        checkpoint_root=checkpoint_root,
    )
    assert recovery.resume_update == 0
    assert recovery.abandoned_pending_segments == 1
    assert recovery.abandoned_unindexed_checkpoints == 1
    assert not checkpoint.exists()

    replay = store.begin_segment(
        start_update=0,
        end_update=2,
        process_identity="replay-process",
    )
    replay.append_update({"update": 1})
    replay.append_update({"update": 2})
    committed = replay.commit(
        checkpoint_path=_checkpoint(checkpoint),
        checkpoint_control_hash=_SHA,
        data_cursor={"next_update": 3},
        metric_cursor={"validation": 0, "log": 2, "wandb": 0},
        validation={"due": False},
        safety={"passed": True},
        mirror_records=({"update": 1}, {"update": 2}),
    )
    assert committed.end_update == 2


def test_pilot_tracking_ignores_authoritative_entries_without_a_wandb_mirror() -> None:
    pilot = _FormalTrackingOwner(
        parse_query_state_training_config(_raw(mode="pilot")),
        rank=0,
        world_size=1,
    )
    entry = SimpleNamespace(end_update=7)

    pilot.publish(entry)

    formal = _FormalTrackingOwner(
        parse_query_state_training_config(_raw(mode="formal")),
        rank=0,
        world_size=1,
    )
    with pytest.raises(RuntimeError, match="authoritative mirror batches"):
        formal.publish(entry)


def test_formal_restart_reloads_authoritative_mirrors_before_next_segment(
    tmp_path: Path,
) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run-id",
    )
    first = _commit_segment(store, tmp_path, start=0, end=2)
    entries = _authoritative_entries_for_restart(
        store,
        checkpoint_update=2,
        mirror_cursor=0,
    )
    assert entries == (first,)

    owner, run = _tracking_owner(initial_cursor=0)
    owner.restore_authoritative(entries)
    assert [step for step, _record in run.logged] == [1, 2]
    assert owner.mirror is not None
    assert owner.mirror.cursor == 2

    second = _commit_segment(store, tmp_path, start=2, end=4)
    owner.publish(second)
    assert [step for step, _record in run.logged] == [1, 2, 3, 4]
    assert owner.mirror.cursor == 4


def test_formal_restart_keeps_pending_batches_when_transport_is_durable_only(
    tmp_path: Path,
) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run-id",
    )
    _commit_segment(store, tmp_path, start=0, end=2)
    entries = _authoritative_entries_for_restart(
        store,
        checkpoint_update=2,
        mirror_cursor=0,
    )
    owner, run = _tracking_owner(initial_cursor=0)
    assert owner.mirror is not None
    owner.mirror.coordinated_transport_failure((True,))
    owner.restore_authoritative(entries)

    assert run.logged == []
    assert owner.mirror.cursor == 0
    assert owner.mirror.pending_updates() == (1, 2)
    assert owner.mirror.tracking_incomplete is True


def test_formal_restart_rejects_cursor_gap_and_missing_mirror_batch(
    tmp_path: Path,
) -> None:
    store = QueryStateSegmentStore(
        tmp_path / "durable",
        run_identity=_SHA,
        mode="formal",
        wandb_run_id="formal-run-id",
    )
    entry = _commit_segment(store, tmp_path, start=0, end=2)
    with pytest.raises(ValueError, match="cursor|boundary"):
        _authoritative_entries_for_restart(
            store,
            checkpoint_update=2,
            mirror_cursor=3,
        )
    Path(entry.mirror_batch_path).unlink()
    entries = _authoritative_entries_for_restart(
        store,
        checkpoint_update=2,
        mirror_cursor=0,
    )
    owner, _run = _tracking_owner(initial_cursor=0)
    with pytest.raises(RuntimeError, match="authoritative mirror registration"):
        owner.restore_authoritative(entries)


def test_thin_entrypoint_is_launchable_but_never_submits_slurm() -> None:
    source = Path("experiments/training/sft1/query_state_train.py").read_text(encoding="utf-8")
    assert "run_query_state_training(" in source
    assert "init_process_group" in source
    assert "validate_query_state_distributed_topology" in source
    assert "all_gather_object" in source
    assert "all_reduce" in source
    assert '"model_or_output_transaction_entered": False' in source
    assert "sbatch" not in source
    assert "subprocess" not in source
    assert '"terminal_epoch": result.terminal_epoch' in source
    assert '"terminal_reason": result.terminal_reason' in source
    assert "automatic_export\": False" in source
