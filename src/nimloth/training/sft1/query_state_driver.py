"""Deterministic raw-row schedule and DataProto driver primitives.

This module intentionally has no experiment entry point, output directory, GPU
resource choice, epoch budget, or launch command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

from nimloth.training.sft1.query_state_adapter import build_query_state_dataproto
from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateDistributedControl,
    QueryStateResumeIdentity,
    capture_query_state_rank_state,
    finalize_query_state_rank_checkpoint,
    load_query_state_rank_state,
    restore_query_state_rank_state,
    save_query_state_rank_state,
)
from nimloth.training.sft1.query_state_data import (
    FreshQueryStateDINOTeacher,
    prepare_query_state_row,
    render_query_state_row,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row


_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class QueryStateScheduledRow:
    ordinal: int | None
    row_valid: bool

    def __post_init__(self) -> None:
        valid_ordinal = (
            not isinstance(self.ordinal, bool)
            and isinstance(self.ordinal, int)
            and self.ordinal >= 0
        )
        if (
            not isinstance(self.row_valid, bool)
            or (self.row_valid and not valid_ordinal)
            or (not self.row_valid and self.ordinal is not None)
        ):
            raise ValueError("Query-State scheduled row validity/ordinal is inconsistent")


@dataclass(frozen=True)
class QueryStateDataCursor:
    epoch: int
    update_index: int
    consumed_rank_rows: int
    schedule_identity: str
    world_size: int
    rank: int
    metric_cursor: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        integers = (
            self.epoch,
            self.update_index,
            self.consumed_rank_rows,
            self.world_size,
            self.rank,
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
            or self.epoch < 0
            or self.update_index < 0
            or self.consumed_rank_rows < 0
            or self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or len(self.schedule_identity) != 64
            or any(char not in _HEX for char in self.schedule_identity)
            or not isinstance(self.metric_cursor, Mapping)
        ):
            raise ValueError("Query-State data cursor is invalid")


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def deterministic_query_state_schedule(
    ordinals: Sequence[int],
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[tuple[QueryStateScheduledRow, ...], str]:
    """Partition immutable row IDs and pad all ranks to equal row counts."""

    arguments = (epoch, seed, rank, world_size)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in arguments)
        or epoch < 0
        or seed < 0
        or world_size < 1
        or not 0 <= rank < world_size
    ):
        raise ValueError("Query-State schedule arguments are invalid")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in ordinals):
        raise ValueError("Query-State schedule ordinals must be integers")
    values = tuple(ordinals)
    if not values:
        raise ValueError("Query-State schedule requires at least one raw row")
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("Query-State schedule ordinals must be unique and non-negative")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{epoch}:{value}".encode()).digest(),
            value,
        ),
    )
    partitions = [ordered[item_rank::world_size] for item_rank in range(world_size)]
    equal_length = max(len(partition) for partition in partitions)
    local = partitions[rank]
    schedule = tuple(
        QueryStateScheduledRow(
            ordinal=local[index] if index < len(local) else None,
            row_valid=index < len(local),
        )
        for index in range(equal_length)
    )
    identity = _identity(
        {
            "schema": "nimloth_sft1_query_state_schedule_v1",
            "epoch": epoch,
            "seed": seed,
            "world_size": world_size,
            "ordered_ordinals": ordered,
        }
    )
    return schedule, identity


def iter_query_state_updates(
    schedule: Sequence[QueryStateScheduledRow],
    *,
    rows_per_rank_update: int,
) -> Iterator[tuple[QueryStateScheduledRow, ...]]:
    if rows_per_rank_update < 1:
        raise ValueError("Query-State rows_per_rank_update must be positive")
    for start in range(0, len(schedule), rows_per_rank_update):
        yield tuple(schedule[start : start + rows_per_rank_update])


def resume_query_state_schedule(
    schedule: Sequence[QueryStateScheduledRow],
    cursor: QueryStateDataCursor,
    *,
    expected_identity: str,
    expected_epoch: int,
    rank: int,
    world_size: int,
    rows_per_rank_update: int,
) -> tuple[QueryStateScheduledRow, ...]:
    if cursor.schedule_identity != expected_identity:
        raise ValueError("Query-State resume schedule identity mismatch")
    if cursor.epoch != expected_epoch:
        raise ValueError("Query-State resume epoch mismatch")
    if cursor.rank != rank or cursor.world_size != world_size:
        raise ValueError("Query-State resume rank/world-size mismatch")
    if rows_per_rank_update < 1:
        raise ValueError("Query-State resume rows_per_rank_update must be positive")
    consumed = cursor.consumed_rank_rows
    if consumed > len(schedule):
        raise ValueError("Query-State resume cursor is outside the schedule")
    if consumed != len(schedule) and consumed % rows_per_rank_update != 0:
        raise ValueError("Query-State resume cursor is not at an update boundary")
    expected_update_index = (
        (consumed + rows_per_rank_update - 1) // rows_per_rank_update
        if consumed
        else 0
    )
    if cursor.update_index != expected_update_index:
        raise ValueError("Query-State resume update index disagrees with data cursor")
    return tuple(schedule[consumed:])


def save_query_state_distributed_checkpoint(
    path: Path,
    *,
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Mapping[str, Any],
    control: QueryStateDistributedControl,
    rank: int,
) -> Path:
    """Publish all rank shards, then one atomic rank-zero completion marker."""

    world_size = control.identity.world_size
    if not 0 <= rank < world_size:
        raise ValueError("Query-State checkpoint rank is outside its identity")
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed and world_size != 1:
        raise ValueError("multi-rank Query-State checkpoint requires a process group")
    if distributed and (
        torch.distributed.get_world_size() != world_size
        or torch.distributed.get_rank() != rank
    ):
        raise ValueError("Query-State checkpoint process-group rank/world-size mismatch")
    if distributed:
        control_digest: str | None = None
        control_error: Exception | None = None
        try:
            control_digest = hashlib.sha256(
                json.dumps(
                    {
                        "identity": asdict(control.identity),
                        "global_step": control.global_step,
                        "data_cursor": dict(control.data_cursor),
                        "metric_cursor": dict(control.metric_cursor),
                        "terminal_primary": control.terminal_primary,
                        "forensic_only": control.forensic_only,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        except Exception as error:
            control_error = error
        control_status: list[tuple[str | None, str | None]] = [
            (None, None)
        ] * world_size
        torch.distributed.all_gather_object(
            control_status,
            (
                control_digest,
                None
                if control_error is None
                else f"rank {rank}: {type(control_error).__name__}: {control_error}",
            ),
        )
        control_failures = [
            error for _digest, error in control_status if error is not None
        ]
        control_digests = {
            digest for digest, _error in control_status if digest is not None
        }
        if control_failures:
            raise RuntimeError(
                "Query-State rank checkpoint control serialization failed: "
                + "; ".join(control_failures)
            )
        if len(control_digests) != 1:
            raise RuntimeError(
                "Query-State rank checkpoint controls differ across ranks"
            )

    save_error: Exception | None = None
    try:
        state = capture_query_state_rank_state(
            root,
            optimizer,
            scheduler_state=scheduler_state,
            identity=control.identity,
        )
        save_query_state_rank_state(
            path, rank=rank, world_size=world_size, state=state
        )
    except Exception as error:  # Coordinate failure before any rank can finalize.
        save_error = error
    if not distributed:
        if save_error is not None:
            raise save_error
        finalize_query_state_rank_checkpoint(path, control=control)
        return Path(path)

    save_status: list[str | None] = [None] * world_size
    torch.distributed.all_gather_object(
        save_status,
        None if save_error is None else f"rank {rank}: {type(save_error).__name__}: {save_error}",
    )
    failures = [value for value in save_status if value is not None]
    if failures:
        raise RuntimeError("Query-State rank checkpoint save failed: " + "; ".join(failures))

    finalize_error: Exception | None = None
    if rank == 0:
        try:
            finalize_query_state_rank_checkpoint(path, control=control)
        except Exception as error:
            finalize_error = error
    finalize_status = [
        None
        if finalize_error is None
        else f"{type(finalize_error).__name__}: {finalize_error}"
    ]
    torch.distributed.broadcast_object_list(finalize_status, src=0)
    if finalize_status[0] is not None:
        if rank == 0 and finalize_error is not None:
            raise finalize_error
        raise RuntimeError(
            "Query-State rank checkpoint finalization failed: " + finalize_status[0]
        )
    return Path(path)


def restore_query_state_distributed_checkpoint(
    path: Path,
    *,
    root: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: QueryStateResumeIdentity,
    rank: int,
) -> tuple[QueryStateDistributedControl, Mapping[str, Any]]:
    """Restore one exact local shard after validating the complete transaction."""

    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed and expected_identity.world_size != 1:
        raise ValueError("multi-rank Query-State resume requires a process group")
    if distributed and (
        torch.distributed.get_world_size() != expected_identity.world_size
        or torch.distributed.get_rank() != rank
    ):
        raise ValueError("Query-State resume process-group rank/world-size mismatch")
    restore_error: Exception | None = None
    control: QueryStateDistributedControl | None = None
    scheduler: Mapping[str, Any] | None = None
    try:
        state, control = load_query_state_rank_state(
            path, rank=rank, expected_identity=expected_identity
        )
        if control.forensic_only:
            raise ValueError(
                "Query-State forensic checkpoint cannot be used for training resume"
            )
        scheduler = restore_query_state_rank_state(root, optimizer, state)
    except Exception as error:  # Ensure peers do not enter later FSDP collectives alone.
        restore_error = error
    if not distributed:
        if restore_error is not None:
            raise restore_error
    else:
        restore_status: list[str | None] = [None] * expected_identity.world_size
        torch.distributed.all_gather_object(
            restore_status,
            None
            if restore_error is None
            else f"rank {rank}: {type(restore_error).__name__}: {restore_error}",
        )
        failures = [value for value in restore_status if value is not None]
        if failures:
            raise RuntimeError(
                "Query-State rank checkpoint restore failed: " + "; ".join(failures)
            )
    if control is None or scheduler is None:
        raise RuntimeError("Query-State checkpoint restore produced no control state")
    return control, scheduler


def build_query_state_update_dataproto(
    scheduled: Sequence[QueryStateScheduledRow],
    *,
    rows_by_ordinal: Mapping[int, SFT1V2Early4Row],
    padding_row: SFT1V2Early4Row,
    processor: Any,
    dino_teacher: FreshQueryStateDINOTeacher,
    max_length: int,
    source_manifest_identity: str,
) -> Any:
    """Re-render original raw rows and generate fresh DINO targets per update."""

    if not scheduled:
        raise ValueError("Query-State rank update schedule must not be empty")
    rendered = []
    valid = []
    for item in scheduled:
        if item.row_valid:
            if item.ordinal is None or int(item.ordinal) not in rows_by_ordinal:
                raise ValueError("Query-State scheduled raw row is missing")
            row = rows_by_ordinal[int(item.ordinal)]
        else:
            if item.ordinal is not None:
                raise ValueError("Query-State padding row must not own an ordinal")
            row = padding_row
        rendered.append(
            render_query_state_row(row, processor=processor, max_length=max_length)
        )
        valid.append(item.row_valid)
    targets = dino_teacher.build_many(tuple(rendered))
    prepared = tuple(
        prepare_query_state_row(
            item,
            dino_regions=target,
            source_manifest_identity=source_manifest_identity,
        )
        for item, target in zip(rendered, targets, strict=True)
    )
    data = build_query_state_dataproto(prepared)
    data.batch["row_valid"] = torch.tensor(valid, dtype=torch.bool)
    return data


__all__ = [
    "QueryStateDataCursor",
    "QueryStateScheduledRow",
    "build_query_state_update_dataproto",
    "deterministic_query_state_schedule",
    "iter_query_state_updates",
    "restore_query_state_distributed_checkpoint",
    "resume_query_state_schedule",
    "save_query_state_distributed_checkpoint",
]
