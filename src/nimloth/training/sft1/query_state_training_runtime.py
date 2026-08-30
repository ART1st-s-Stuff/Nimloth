"""Crash-consistent pilot/formal controller primitives.

A resumable commit is the only authority.  Step records remain segment-pending
until validation/safety, the immutable checkpoint control, cursor hashes, and a
same-run W&B mirror batch are all present; only then is the atomic index moved.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

QUERY_STATE_SEGMENT_SCHEMA = "nimloth_sft1_query_state_segment_v1"
QUERY_STATE_RESTART_SCHEMA = "nimloth_sft1_query_state_pilot_restart_v1"
_HEX = frozenset("0123456789abcdef")
_PROCESS_NONCE = os.urandom(16).hex()


def current_process_identity() -> str:
    """Return a process-boot identity that changes across exec/fork boundaries."""

    start_time = "unavailable"
    try:
        start_time = Path("/proc/self/stat").read_text(encoding="utf-8").split()[21]
    except (OSError, IndexError):
        pass
    return _canonical_hash(
        {
            "pid": os.getpid(),
            "start_time": start_time,
            "module_nonce": _PROCESS_NONCE,
        }
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"immutable Query-State runtime artifact exists: {path}")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Query-State runtime JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Query-State runtime JSON must be an object: {path}")
    return value


@dataclass(frozen=True)
class QueryStateEarlyStoppingCursor:
    best_composite: float | None
    last_composite: float | None
    best_epoch: int
    bad_epochs: int
    last_epoch: int
    last_update: int
    terminal_epoch: int | None
    terminal_update: int | None
    stop_reason: str | None

    @classmethod
    def initial(cls) -> QueryStateEarlyStoppingCursor:
        return cls(None, None, 0, 0, 0, 0, None, None, None)

    def __post_init__(self) -> None:
        if (
            any(
                value is not None
                and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                )
                for value in (self.best_composite, self.last_composite)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (
                    self.best_epoch,
                    self.bad_epochs,
                    self.last_epoch,
                    self.last_update,
                )
            )
            or self.best_epoch > self.last_epoch
            or (self.best_composite is None) != (self.best_epoch == 0)
            or (self.last_composite is None) != (self.last_epoch == 0)
        ):
            raise ValueError("Query-State early-stop cursor is invalid")
        terminal_values = (
            self.terminal_epoch,
            self.terminal_update,
            self.stop_reason,
        )
        if any(value is None for value in terminal_values) != all(
            value is None for value in terminal_values
        ):
            raise ValueError("Query-State early-stop terminal cursor is partial")
        if self.stop_reason is not None:
            if (
                self.stop_reason not in {"converged_early_stop", "max_epochs_reached"}
                or isinstance(self.terminal_epoch, bool)
                or not isinstance(self.terminal_epoch, int)
                or self.terminal_epoch != self.last_epoch
                or isinstance(self.terminal_update, bool)
                or not isinstance(self.terminal_update, int)
                or self.terminal_update != self.last_update
            ):
                raise ValueError("Query-State early-stop terminal identity is invalid")

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> QueryStateEarlyStoppingCursor:
        expected = {
            "best_composite",
            "last_composite",
            "best_epoch",
            "bad_epochs",
            "last_epoch",
            "last_update",
            "terminal_epoch",
            "terminal_update",
            "stop_reason",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("Query-State early-stop cursor field set is invalid")
        return cls(**dict(value))


@dataclass(frozen=True)
class QueryStateEarlyStoppingDecision:
    cursor: QueryStateEarlyStoppingCursor
    composite: float
    improved: bool
    should_stop: bool
    reason: str | None


def advance_query_state_early_stopping(
    cursor: QueryStateEarlyStoppingCursor,
    *,
    epoch: int,
    update: int,
    calibration_dino_mse: float,
    calibration_assistant_ce: float,
    min_epochs: int,
    max_epochs: int,
    patience_epochs: int,
    min_relative_improvement: float,
) -> QueryStateEarlyStoppingDecision:
    """Advance only from the globally aggregated calibration objective."""

    if not isinstance(cursor, QueryStateEarlyStoppingCursor):
        raise TypeError("early stopping requires its exact cursor")
    if cursor.stop_reason is not None:
        raise ValueError("completed Query-State early stopping cannot advance")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch != cursor.last_epoch + 1
        or isinstance(update, bool)
        or not isinstance(update, int)
        or update <= cursor.last_update
        or min_epochs < 2
        or max_epochs < min_epochs
        or epoch > max_epochs
        or patience_epochs < 1
        or not math.isfinite(min_relative_improvement)
        or not 0.0 < min_relative_improvement < 1.0
    ):
        raise ValueError("Query-State early-stop schedule is invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (calibration_dino_mse, calibration_assistant_ce)
    ):
        raise ValueError("Query-State calibration convergence metrics are invalid")
    composite = 2.0 * float(calibration_dino_mse) + float(calibration_assistant_ce)
    if cursor.best_composite is None:
        improved = True
    else:
        improvement = (cursor.best_composite - composite) / max(
            abs(cursor.best_composite), 1e-12
        )
        improved = improvement >= min_relative_improvement
    best = composite if improved else cursor.best_composite
    best_epoch = epoch if improved else cursor.best_epoch
    bad_epochs = 0 if improved else cursor.bad_epochs + 1
    reason: str | None = None
    if epoch == max_epochs:
        reason = "max_epochs_reached"
    elif epoch >= min_epochs and bad_epochs >= patience_epochs:
        reason = "converged_early_stop"
    next_cursor = QueryStateEarlyStoppingCursor(
        best_composite=best,
        last_composite=composite,
        best_epoch=best_epoch,
        bad_epochs=bad_epochs,
        last_epoch=epoch,
        last_update=update,
        terminal_epoch=epoch if reason is not None else None,
        terminal_update=update if reason is not None else None,
        stop_reason=reason,
    )
    return QueryStateEarlyStoppingDecision(
        cursor=next_cursor,
        composite=composite,
        improved=improved,
        should_stop=reason is not None,
        reason=reason,
    )


@dataclass(frozen=True)
class QueryStateTrainingEvent:
    mode: str
    update: int
    kind: str


def build_training_event_plan(
    *,
    mode: str,
    total_updates: int,
    checkpoint_cadence: int,
    validation_updates: Sequence[int],
    forced_restart_update: int,
) -> tuple[QueryStateTrainingEvent, ...]:
    if mode not in {"pilot", "formal"}:
        raise ValueError("training event mode must be pilot or formal")
    if total_updates < 1 or checkpoint_cadence < 1 or total_updates % checkpoint_cadence:
        raise ValueError("terminal update must be a commit boundary")
    validations = tuple(validation_updates)
    if (
        not validations
        or validations != tuple(sorted(set(validations)))
        or validations[0] != 0
        or validations[-1] != total_updates
        or any(update and update % checkpoint_cadence for update in validations)
    ):
        raise ValueError("every validation after update 0 must be a commit boundary")
    if mode == "pilot":
        if not 0 < forced_restart_update < total_updates or forced_restart_update % checkpoint_cadence:
            raise ValueError("pilot forced restart must be a commit boundary")
    elif forced_restart_update != 0:
        raise ValueError("formal event plan cannot contain a pilot forced restart")

    events = [QueryStateTrainingEvent(mode, 0, "validation")]
    for update in range(1, total_updates + 1):
        events.append(QueryStateTrainingEvent(mode, update, "optimizer_update"))
        if update in validations:
            events.append(QueryStateTrainingEvent(mode, update, "validation"))
            events.append(QueryStateTrainingEvent(mode, update, "safety_verdict"))
        if update % checkpoint_cadence == 0:
            events.append(QueryStateTrainingEvent(mode, update, "commit"))
        if update == forced_restart_update:
            events.append(QueryStateTrainingEvent(mode, update, "forced_restart"))
        if update == total_updates:
            events.append(QueryStateTrainingEvent(mode, update, "terminal"))
    return tuple(events)


@dataclass(frozen=True)
class QueryStateAuthoritativeEntry:
    schema: str
    run_identity: str
    mode: str
    wandb_run_id: str | None
    start_update: int
    end_update: int
    segment_path: str
    checkpoint_path: str
    checkpoint_control_hash: str
    data_cursor_hash: str
    metric_cursor_hash: str
    mirror_batch_path: str
    mirror_batch_hash: str
    early_stopping_cursor: Mapping[str, Any] | None = None
    actual_terminal: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QueryStateRecovery:
    resume_update: int
    abandoned_pending_segments: int
    abandoned_unindexed_checkpoints: int


class QueryStateSegment:
    def __init__(
        self,
        store: QueryStateSegmentStore,
        *,
        start_update: int,
        end_update: int,
        process_identity: str,
        path: Path,
    ) -> None:
        self.store = store
        self.start_update = start_update
        self.end_update = end_update
        self.process_identity = process_identity
        self.path = path
        self._updates: list[Mapping[str, Any]] = []

    def append_update(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record, Mapping):
            raise ValueError("pending update record must be a mapping")
        expected = self.start_update + len(self._updates) + 1
        if record.get("update") != expected or expected > self.end_update:
            raise ValueError("pending update records must be complete and ordered")
        # JSON serialization here prevents a late commit from discovering an
        # invalid record after GPU updates have already accumulated.
        line = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        with (self.path / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._updates.append(dict(record))

    def commit(
        self,
        *,
        checkpoint_path: Path,
        checkpoint_control_hash: str,
        data_cursor: Mapping[str, Any],
        metric_cursor: Mapping[str, Any],
        validation: Mapping[str, Any],
        safety: Mapping[str, Any],
        mirror_records: Sequence[Mapping[str, Any]],
        fail_before_index: bool = False,
    ) -> QueryStateAuthoritativeEntry:
        if len(self._updates) != self.end_update - self.start_update:
            raise ValueError("segment cannot commit with missing update records")
        if safety.get("passed") is not True:
            failure = {
                "schema": QUERY_STATE_SEGMENT_SCHEMA,
                "run_identity": self.store.run_identity,
                "mode": self.store.mode,
                "start_update": self.start_update,
                "end_update": self.end_update,
                "validation": dict(validation),
                "safety": dict(safety),
                "resumable": False,
            }
            _atomic_json(
                self.store.root / "failures" / f"unsafe_{self.start_update:08d}_{self.end_update:08d}.json",
                failure,
                overwrite=False,
            )
            raise RuntimeError("unsafe validation is non-resumable and cannot advance the index")
        checkpoint = Path(checkpoint_path).resolve()
        marker = checkpoint / "COMPLETED"
        control = checkpoint / "control.json"
        if not marker.is_file() or not control.is_file() or not _is_sha256(checkpoint_control_hash):
            raise ValueError("segment commit requires a complete immutable checkpoint")
        if marker.read_text(encoding="utf-8") != f"control_sha256={checkpoint_control_hash}\n":
            raise ValueError("checkpoint completion marker/control hash mismatch")
        mirror = tuple(dict(record) for record in mirror_records)
        mirror_updates = tuple(record.get("update") for record in mirror)
        expected_updates = tuple(range(self.start_update + 1, self.end_update + 1))
        if mirror_updates != expected_updates:
            raise ValueError("W&B mirror batch must contain every ordered segment update exactly once")
        mirror_payload = {
            "schema": QUERY_STATE_SEGMENT_SCHEMA,
            "run_identity": self.store.run_identity,
            "mode": self.store.mode,
            "wandb_run_id": self.store.wandb_run_id,
            "start_update": self.start_update,
            "end_update": self.end_update,
            "records": list(mirror),
        }
        _atomic_json(self.path / "mirror_batch.json", mirror_payload, overwrite=False)
        _atomic_json(self.path / "validation.json", dict(validation), overwrite=False)
        _atomic_json(self.path / "safety.json", dict(safety), overwrite=False)
        cursor_payload = {
            "data": dict(data_cursor),
            "metric": dict(metric_cursor),
        }
        _atomic_json(self.path / "cursors.json", cursor_payload, overwrite=False)
        early_cursor_raw = metric_cursor.get("early_stopping")
        early_cursor: Mapping[str, Any] | None = None
        if early_cursor_raw is not None:
            early_cursor = QueryStateEarlyStoppingCursor.from_mapping(
                early_cursor_raw
            ).to_mapping()
        actual_terminal_raw = metric_cursor.get("actual_terminal")
        actual_terminal: Mapping[str, Any] | None = None
        if actual_terminal_raw is not None:
            expected_terminal_fields = {
                "epoch",
                "update",
                "reason",
                "terminal_primary",
            }
            if (
                not isinstance(actual_terminal_raw, Mapping)
                or set(actual_terminal_raw) != expected_terminal_fields
                or actual_terminal_raw.get("terminal_primary") is not True
                or isinstance(actual_terminal_raw.get("epoch"), bool)
                or not isinstance(actual_terminal_raw.get("epoch"), int)
                or actual_terminal_raw.get("epoch") < 1
                or isinstance(actual_terminal_raw.get("update"), bool)
                or actual_terminal_raw.get("update") != self.end_update
                or actual_terminal_raw.get("reason")
                not in {"converged_early_stop", "max_epochs_reached"}
            ):
                raise ValueError("actual terminal metric cursor is invalid")
            actual_terminal = dict(actual_terminal_raw)
            if (
                early_cursor is None
                or early_cursor["stop_reason"] != actual_terminal["reason"]
                or early_cursor["terminal_epoch"] != actual_terminal["epoch"]
                or early_cursor["terminal_update"] != actual_terminal["update"]
            ):
                raise ValueError("actual terminal and early-stop cursors disagree")
        segment_name = f"segment_{self.start_update:08d}_{self.end_update:08d}"
        committed_path = self.store.root / "segments" / segment_name
        if committed_path.exists():
            raise FileExistsError("immutable committed segment already exists")
        self.path.replace(committed_path)
        _fsync_directory(committed_path.parent)
        mirror_path = (committed_path / "mirror_batch.json").resolve()
        entry = QueryStateAuthoritativeEntry(
            schema=QUERY_STATE_SEGMENT_SCHEMA,
            run_identity=self.store.run_identity,
            mode=self.store.mode,
            wandb_run_id=self.store.wandb_run_id,
            start_update=self.start_update,
            end_update=self.end_update,
            segment_path=str(committed_path.resolve()),
            checkpoint_path=str(checkpoint),
            checkpoint_control_hash=checkpoint_control_hash,
            data_cursor_hash=_canonical_hash(dict(data_cursor)),
            metric_cursor_hash=_canonical_hash(dict(metric_cursor)),
            mirror_batch_path=str(mirror_path),
            mirror_batch_hash=_canonical_hash(mirror_payload),
            early_stopping_cursor=early_cursor,
            actual_terminal=actual_terminal,
        )
        _atomic_json(committed_path / "commit.json", asdict(entry), overwrite=False)
        if fail_before_index:
            raise RuntimeError("injected before authoritative index publication")
        self.store._append_authoritative(entry)
        return entry


class QueryStateSegmentStore:
    def __init__(
        self,
        root: Path,
        *,
        run_identity: str,
        mode: str,
        wandb_run_id: str | None = None,
    ) -> None:
        if not _is_sha256(run_identity) or mode not in {"pilot", "formal"}:
            raise ValueError("segment store run/mode identity is invalid")
        if mode == "pilot" and wandb_run_id is not None:
            raise ValueError("pilot segment store must keep W&B disabled")
        if mode == "formal" and (not isinstance(wandb_run_id, str) or not wandb_run_id):
            raise ValueError("formal segment store requires the locked W&B run ID")
        self.root = Path(root).resolve()
        self.run_identity = run_identity
        self.mode = mode
        self.wandb_run_id = wandb_run_id
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("pending", "segments", "abandoned", "failures"):
            (self.root / name).mkdir(exist_ok=True)
        if not (self.root / "authoritative_index.json").exists():
            _atomic_json(
                self.root / "authoritative_index.json",
                {"schema": QUERY_STATE_SEGMENT_SCHEMA, "run_identity": run_identity, "mode": mode, "wandb_run_id": wandb_run_id, "entries": []},
                overwrite=False,
            )
        else:
            self._raw_index()

    def _raw_index(self) -> dict[str, Any]:
        raw = _read_json(self.root / "authoritative_index.json")
        if raw.get("schema") != QUERY_STATE_SEGMENT_SCHEMA or raw.get("run_identity") != self.run_identity or raw.get("mode") != self.mode or raw.get("wandb_run_id") != self.wandb_run_id or not isinstance(raw.get("entries"), list):
            raise ValueError("authoritative segment index identity mismatch")
        return raw

    def authoritative_entries(self) -> tuple[QueryStateAuthoritativeEntry, ...]:
        entries = tuple(QueryStateAuthoritativeEntry(**value) for value in self._raw_index()["entries"])
        previous = 0
        for entry in entries:
            if entry.start_update != previous or entry.end_update <= entry.start_update:
                raise ValueError("authoritative segment index is not contiguous")
            previous = entry.end_update
        return entries

    def begin_segment(
        self,
        *,
        start_update: int,
        end_update: int,
        process_identity: str,
    ) -> QueryStateSegment:
        entries = self.authoritative_entries()
        expected = entries[-1].end_update if entries else 0
        if start_update != expected or end_update <= start_update:
            raise ValueError("new segment must start at the authoritative resume cursor")
        if not isinstance(process_identity, str) or not process_identity.strip():
            raise ValueError("segment process identity is required")
        name = f"segment_{start_update:08d}_{end_update:08d}"
        path = self.root / "pending" / name
        if path.exists() or (self.root / "segments" / name).exists():
            raise FileExistsError("segment range already exists")
        path.mkdir()
        _atomic_json(path / "owner.json", {
            "schema": QUERY_STATE_SEGMENT_SCHEMA,
            "run_identity": self.run_identity,
            "mode": self.mode,
            "process_identity": process_identity,
            "start_update": start_update,
            "end_update": end_update,
        }, overwrite=False)
        return QueryStateSegment(
            self,
            start_update=start_update,
            end_update=end_update,
            process_identity=process_identity,
            path=path,
        )

    def _append_authoritative(self, entry: QueryStateAuthoritativeEntry) -> None:
        current = self._raw_index()
        entries = self.authoritative_entries()
        expected = entries[-1].end_update if entries else 0
        if entry.start_update != expected:
            raise ValueError("authoritative segment publication would skip or duplicate updates")
        current["entries"].append(asdict(entry))
        _atomic_json(self.root / "authoritative_index.json", current, overwrite=True)

    def recover(
        self,
        *,
        checkpoint_root: Path | None = None,
    ) -> QueryStateRecovery:
        entries = self.authoritative_entries()
        indexed = {Path(entry.segment_path).resolve() for entry in entries}
        abandoned = 0
        candidates = list((self.root / "pending").iterdir()) + [
            path for path in (self.root / "segments").iterdir() if path.resolve() not in indexed
        ]
        for path in sorted(candidates):
            if not path.is_dir():
                continue
            owner_path = path / "owner.json"
            owner_identity = (
                _read_json(owner_path)
                if owner_path.is_file()
                else {"path": str(path.resolve())}
            )
            suffix = _canonical_hash(owner_identity)[:12]
            destination = self.root / "abandoned" / f"{path.name}_{suffix}"
            if destination.exists():
                raise FileExistsError("abandoned segment identity collision")
            path.replace(destination)
            abandoned += 1

        abandoned_checkpoints = 0
        if checkpoint_root is not None:
            checkpoint_directory = Path(checkpoint_root).resolve()
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            referenced = {Path(entry.checkpoint_path).resolve() for entry in entries}
            if any(path.parent != checkpoint_directory for path in referenced):
                raise ValueError(
                    "authoritative checkpoint lies outside the run checkpoint root"
                )
            quarantine = self.root / "abandoned" / "checkpoints"
            quarantine.mkdir(parents=True, exist_ok=True)
            for path in sorted(checkpoint_directory.iterdir()):
                resolved = path.resolve()
                if resolved in referenced:
                    continue
                if not path.is_dir() or path.is_symlink():
                    raise ValueError(
                        "unindexed checkpoint root contains a non-directory payload"
                    )
                attempt = 0
                while True:
                    destination = quarantine / f"{path.name}_attempt_{attempt:04d}"
                    if not destination.exists():
                        break
                    attempt += 1
                path.replace(destination)
                abandoned_checkpoints += 1
            _fsync_directory(checkpoint_directory)
            _fsync_directory(quarantine)
        return QueryStateRecovery(
            resume_update=entries[-1].end_update if entries else 0,
            abandoned_pending_segments=abandoned,
            abandoned_unindexed_checkpoints=abandoned_checkpoints,
        )

    def pending_mirror_batches(self) -> tuple[str, ...]:
        # The durable index, not directory discovery, determines mirror authority.
        return tuple(entry.mirror_batch_path for entry in self.authoritative_entries())


class QueryStateWandbMirror:
    """Same-run idempotent mirror cursor with coordinated durable-only fallback."""

    def __init__(
        self,
        *,
        run_id: str,
        world_size: int,
        initial_cursor: int,
    ) -> None:
        if (
            not isinstance(run_id, str)
            or not run_id
            or world_size < 1
            or isinstance(initial_cursor, bool)
            or not isinstance(initial_cursor, int)
            or initial_cursor < 0
        ):
            raise ValueError("W&B mirror identity/world size/cursor is invalid")
        self.run_id = run_id
        self.world_size = world_size
        self.cursor = initial_cursor
        self._records: dict[int, Mapping[str, Any]] = {}
        self.durable_only = False
        self.tracking_incomplete = False

    def register_authoritative(self, entry: QueryStateAuthoritativeEntry) -> None:
        payload = _read_json(Path(entry.mirror_batch_path))
        if entry.wandb_run_id != self.run_id or payload.get("wandb_run_id") != self.run_id:
            raise ValueError("W&B mirror batch must use the same run ID")
        if payload.get("run_identity") != entry.run_identity or _canonical_hash(payload) != entry.mirror_batch_hash:
            raise ValueError("mirror batch is not bound to its authoritative segment")
        records = payload.get("records")
        if not isinstance(records, list):
            raise ValueError("mirror batch records are invalid")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("update"), int):
                raise ValueError("mirror record update identity is invalid")
            update = record["update"]
            if update <= self.cursor:
                continue
            if update in self._records:
                raise ValueError("mirror batch duplicates an update")
            self._records[update] = record

    def pending_updates(self) -> tuple[int, ...]:
        return tuple(sorted(update for update in self._records if update > self.cursor))

    def pending_records(self) -> tuple[Mapping[str, Any], ...]:
        """Return authoritative records in exact replay order without acknowledging them."""

        return tuple(self._records[update] for update in self.pending_updates())

    def coordinated_transport_failure(self, rank_failures: Sequence[bool]) -> None:
        if len(rank_failures) != self.world_size or not all(rank_failures):
            raise RuntimeError("W&B durable-only fallback must be coordinated on all ranks")
        self.durable_only = True
        self.tracking_incomplete = True

    def replay(self, *, run_id: str, updates: Sequence[int]) -> None:
        if run_id != self.run_id:
            raise ValueError("W&B delayed replay must use the same run ID")
        values = tuple(updates)
        if not values:
            return
        expected = tuple(range(self.cursor + 1, self.cursor + 1 + len(values)))
        if values != expected:
            if any(value <= self.cursor for value in values):
                raise ValueError("W&B replay duplicates the acknowledged cursor")
            raise ValueError("W&B replay has a gap or is not ordered")
        if any(value not in self._records for value in values):
            raise ValueError("W&B replay references a non-authoritative update")
        self.cursor = values[-1]
        for value in values:
            del self._records[value]


@dataclass(frozen=True)
class QueryStatePilotRestartReceipt:
    run_identity: str
    checkpoint_identity: str
    checkpoint_update: int
    first_process_identity: str
    resumed_process_identity: str
    fresh_process_verified: bool


def _validate_restart_maps(
    fingerprints: Mapping[str, Any],
    cursors: Mapping[str, Any],
) -> None:
    if set(fingerprints) != {"model", "optimizer", "scheduler", "rng"}:
        raise ValueError("pilot restart fingerprint set is incomplete")
    if set(cursors) != {"data", "validation", "log", "wandb"}:
        raise ValueError("pilot restart cursor set is incomplete")


def publish_pilot_restart_boundary(
    path: Path,
    *,
    run_identity: str,
    process_identity: str,
    checkpoint_identity: str,
    checkpoint_update: int,
    fingerprints: Mapping[str, Any],
    cursors: Mapping[str, Any],
) -> None:
    if not _is_sha256(run_identity) or not _is_sha256(checkpoint_identity):
        raise ValueError("pilot restart run/checkpoint identity must be SHA256")
    if checkpoint_update < 1 or cursors.get("data") != checkpoint_update:
        raise ValueError("pilot restart must be published at its exact data cursor")
    if process_identity != current_process_identity():
        raise ValueError("pilot restart process identity must match the current process")
    _validate_restart_maps(fingerprints, cursors)
    _atomic_json(Path(path), {
        "schema": QUERY_STATE_RESTART_SCHEMA,
        "mode": "pilot",
        "run_identity": run_identity,
        "process_identity": process_identity,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_update": checkpoint_update,
        "fingerprints": dict(fingerprints),
        "cursors": dict(cursors),
    }, overwrite=False)


def consume_pilot_restart_boundary(
    path: Path,
    *,
    run_identity: str,
    process_identity: str,
    checkpoint_identity: str,
    restored_fingerprints: Mapping[str, Any],
    restored_cursors: Mapping[str, Any],
) -> QueryStatePilotRestartReceipt:
    raw = _read_json(Path(path))
    if raw.get("schema") != QUERY_STATE_RESTART_SCHEMA or raw.get("mode") != "pilot":
        raise ValueError("restart boundary is not the pilot training owner")
    if raw.get("run_identity") != run_identity or raw.get("checkpoint_identity") != checkpoint_identity:
        raise ValueError("pilot restart run/checkpoint identity mismatch")
    if process_identity != current_process_identity():
        raise ValueError("pilot restart process identity must match the current process")
    if raw.get("process_identity") == process_identity:
        raise ValueError("pilot forced restart must resume in a fresh process")
    _validate_restart_maps(restored_fingerprints, restored_cursors)
    if raw.get("fingerprints") != dict(restored_fingerprints):
        raise ValueError("pilot restart fingerprint mismatch")
    if raw.get("cursors") != dict(restored_cursors):
        raise ValueError("pilot restart cursor mismatch")
    return QueryStatePilotRestartReceipt(
        run_identity=run_identity,
        checkpoint_identity=checkpoint_identity,
        checkpoint_update=int(raw["checkpoint_update"]),
        first_process_identity=str(raw["process_identity"]),
        resumed_process_identity=process_identity,
        fresh_process_verified=True,
    )


__all__ = [
    "QUERY_STATE_RESTART_SCHEMA",
    "QUERY_STATE_SEGMENT_SCHEMA",
    "QueryStateAuthoritativeEntry",
    "QueryStateEarlyStoppingCursor",
    "QueryStateEarlyStoppingDecision",
    "QueryStatePilotRestartReceipt",
    "QueryStateRecovery",
    "QueryStateSegmentStore",
    "QueryStateTrainingEvent",
    "QueryStateWandbMirror",
    "advance_query_state_early_stopping",
    "build_training_event_plan",
    "consume_pilot_restart_boundary",
    "current_process_identity",
    "publish_pilot_restart_boundary",
]
