"""Strict distributed cache for unsafe forensic Query-State diagnostics.

This owner is intentionally incompatible with the deployable reconstruction cache.
CPU tests exercise the typed collective/extractor protocol only; they are not NCCL
or FSDP evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import torch

from nimloth.training.reconstruction.query_state_cache import (
    QueryStateSourceContract,
    validate_canonical_query_state,
)
from nimloth.training.sft1.query_state_smoke_runtime import (
    build_query_state_source_manifest_identity,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row, index_early4_rows

FORENSIC_QUERY_STATE_CACHE_SCHEMA = (
    "nimloth_query_state_forensic_reconstruction_cache_v1"
)
_FORENSIC_SHARD_SCHEMA = "nimloth_query_state_forensic_reconstruction_cache_shard_v1"
FORENSIC_QUERY_STATE_OWNER_ROLE = "unsafe_forensic_query_state"
FORENSIC_SELECTION_MECHANICS_TRAIN = "mechanics_train"
FORENSIC_SELECTION_MECHANICS_VALIDATION = "mechanics_validation"
FORENSIC_SELECTION_ALL_TRAIN = "all_train"
FORENSIC_SELECTION_EXTERNAL_VALIDATION = "external_validation"
FORENSIC_STAGE_A_SELECTION_ALGORITHM = "sha256_image_group_subset_v1"
FORENSIC_STAGE_B_SELECTION_ALGORITHM = "live_audited_full_roles_v1"
FORENSIC_STAGE_B_ROLES = frozenset(
    {FORENSIC_SELECTION_ALL_TRAIN, FORENSIC_SELECTION_EXTERNAL_VALIDATION}
)
FORENSIC_STAGE_B_TRAIN_COUNT = 12_836
FORENSIC_STAGE_B_EXTERNAL_COUNT = 1_413
FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS = 2_048


class ForensicExperimentStage(str, Enum):
    MECHANICS_ONLY = "mechanics_only"
    STAGE_B_DIAGNOSTIC = "stage_b_diagnostic"
_STATE_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")
_PROVENANCE_FIELDS = (
    "prompt_history_identity",
    "messages_identity",
    "renderer_identity",
    "template_identity",
    "encoded_input_identity",
)
_IDENTITY_FIELDS = (
    "id176_identity",
    "processor_identity",
    "tokenizer_identity",
    "template_identity",
    "data_identity",
)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {owner}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"invalid {owner} mapping: {path}")
    return value


@dataclass(frozen=True)
class ForensicRankShardIdentity:
    rank: int
    file: str
    sha256: str
    count: int


@dataclass(frozen=True)
class ForensicProducerIdentity:
    integrated_repo_root: str
    integrated_source_commit: str
    production_config_identity: str
    formal_config_identity: str


@dataclass(frozen=True)
class ForensicCheckpointIdentity:
    source_commit: str
    config_identity: str
    config_path: str
    config_sha256: str
    run_identity: str
    world_size: int
    rank_topology: tuple[Mapping[str, Any], ...]
    run_root: str
    checkpoint_path: str
    control_sha256: str
    failure_manifest_path: str
    failure_manifest_sha256: str
    rank_shards: tuple[ForensicRankShardIdentity, ...]
    actor_failure: Mapping[str, Any]
    model_data_identities: Mapping[str, str]


@dataclass(frozen=True)
class ForensicSelectionEntry:
    selection_ordinal: int
    role: str
    row: SFT1V2Early4Row


@dataclass(frozen=True)
class ForensicSelection:
    stage: ForensicExperimentStage
    seed: int | None
    algorithm: str
    identity: str
    entries: tuple[ForensicSelectionEntry, ...]


# Public compatibility name for the already-published Stage A owner.
ForensicStageASelection = ForensicSelection


@dataclass(frozen=True)
class PreparedForensicRow:
    row: SFT1V2Early4Row
    provenance: Mapping[str, str]


class ForensicPublicationDurabilityError(RuntimeError):
    """The manifest committed, but final publication durability was not confirmed."""


@dataclass(frozen=True)
class ForensicRankSummary:
    rank: int
    file: str
    count: int
    sha256: str
    ordinal_identity: str


class ForensicStateExtractor(Protocol):
    """Production adapter around exact render + frozen FSDP extraction."""

    def prepare(self, row: SFT1V2Early4Row) -> PreparedForensicRow: ...

    def extract(self, rows: Sequence[PreparedForensicRow]) -> torch.Tensor: ...


class ForensicCollective(Protocol):
    """Typed collective boundary; implementations must preserve call order."""

    rank: int
    world_size: int

    def gate(self, phase: str, *, ready: bool, detail: str) -> tuple[Mapping[str, Any], ...]: ...

    def gather_summaries(
        self, summary: ForensicRankSummary
    ) -> tuple[ForensicRankSummary, ...]: ...

    def teardown(self) -> None: ...


class TorchForensicCollective:
    """Thin production collective adapter. It requires initialized WS8 torch.distributed."""

    def __init__(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("forensic cache requires initialized torch.distributed")
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()
        if self.world_size != 8:
            raise ValueError("forensic cache requires exact Formal38 world_size=8")

    def gate(self, phase: str, *, ready: bool, detail: str) -> tuple[Mapping[str, Any], ...]:
        local = {"rank": self.rank, "phase": phase, "ready": ready, "detail": detail}
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, local)
        return tuple(dict(item) for item in gathered)

    def gather_summaries(
        self, summary: ForensicRankSummary
    ) -> tuple[ForensicRankSummary, ...]:
        gathered: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(gathered, asdict(summary))
        return tuple(ForensicRankSummary(**dict(item)) for item in gathered)

    def teardown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _failed_gate_status(
    statuses: Sequence[Mapping[str, Any]], *, phase: str, world_size: int
) -> Mapping[str, Any] | None:
    if len(statuses) != world_size:
        raise RuntimeError(f"forensic {phase} gate returned incomplete world status")
    ranks = {item.get("rank") for item in statuses}
    if ranks != set(range(world_size)) or any(
        item.get("phase") != phase for item in statuses
    ):
        raise RuntimeError(f"forensic {phase} gate rank/phase identity mismatch")
    failed = [item for item in statuses if item.get("ready") is not True]
    return min(failed, key=lambda item: int(item["rank"])) if failed else None


def _require_gate(
    statuses: Sequence[Mapping[str, Any]], *, phase: str, world_size: int
) -> None:
    failed = _failed_gate_status(statuses, phase=phase, world_size=world_size)
    if failed is not None:
        raise RuntimeError(
            f"forensic {phase} gate failed on rank {failed['rank']}: "
            f"{failed.get('detail', '')}"
        )


def _publication_gate_detail(error: BaseException | None) -> str:
    if error is None:
        payload = {"status": "published"}
    elif isinstance(error, ForensicPublicationDurabilityError):
        payload = {
            "status": "committed_but_durability_unconfirmed",
            "error": str(error),
        }
    else:
        payload = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_publication_gate(
    statuses: Sequence[Mapping[str, Any]], *, world_size: int
) -> None:
    failed = _failed_gate_status(statuses, phase="publish", world_size=world_size)
    if failed is None:
        return
    try:
        detail = json.loads(str(failed.get("detail", "")))
    except json.JSONDecodeError:
        detail = None
    if (
        isinstance(detail, dict)
        and set(detail) == {"status", "error"}
        and detail.get("status") == "committed_but_durability_unconfirmed"
        and isinstance(detail.get("error"), str)
    ):
        raise ForensicPublicationDurabilityError(detail["error"])
    raise RuntimeError(
        f"forensic publish gate failed on rank {failed['rank']}: "
        f"{failed.get('detail', '')}"
    )


def _actor_failure_from_manifest(failure: Mapping[str, Any]) -> dict[str, Any]:
    validation = failure.get("validation")
    global_safety = failure.get("safety")
    calibration = (
        validation.get("calibration") if isinstance(validation, Mapping) else None
    )
    diagnostics = (
        calibration.get("diagnostics") if isinstance(calibration, Mapping) else None
    )
    metrics = diagnostics.get("metrics") if isinstance(diagnostics, Mapping) else None
    validation_safety = (
        calibration.get("safety") if isinstance(calibration, Mapping) else None
    )
    calibration_safety = (
        global_safety.get("calibration")
        if isinstance(global_safety, Mapping)
        else None
    )
    checks = (
        calibration_safety.get("checks")
        if isinstance(calibration_safety, Mapping)
        else None
    )
    kl = metrics.get("actor/kl_baseline_to_current") if isinstance(metrics, Mapping) else None
    top1 = metrics.get("actor/top1_agreement") if isinstance(metrics, Mapping) else None
    if (
        not isinstance(validation, Mapping)
        or not isinstance(calibration, Mapping)
        or not isinstance(global_safety, Mapping)
        or global_safety.get("scope") != "global_id176_actor_generation_safety"
        or global_safety.get("passed") is not False
        or not isinstance(calibration_safety, Mapping)
        or validation_safety != calibration_safety
        or calibration_safety.get("passed") is not False
        or not isinstance(checks, Mapping)
        or checks.get("kl") is not False
        or checks.get("top1") is not False
        or isinstance(kl, bool)
        or not isinstance(kl, (int, float))
        or isinstance(top1, bool)
        or not isinstance(top1, (int, float))
        or not math.isfinite(float(kl))
        or not math.isfinite(float(top1))
    ):
        raise ValueError("forensic failure manifest actor-safety evidence is invalid")
    return {
        "evidence_identity": _identity(
            {"validation": dict(validation), "safety": dict(global_safety)}
        ),
        "kl": float(kl),
        "top1_agreement": float(top1),
        "passed": False,
    }


def actor_failure_evidence_from_manifest(path: str | Path) -> dict[str, Any]:
    """Derive actor failure evidence from the immutable live failure manifest."""

    failure = _read_mapping(Path(path), owner="forensic failure manifest")
    return _actor_failure_from_manifest(failure)


def validate_forensic_checkpoint_identity(
    identity: ForensicCheckpointIdentity,
) -> ForensicCheckpointIdentity:
    """Revalidate every live Formal38 forensic owner without loading a shard."""

    if not isinstance(identity, ForensicCheckpointIdentity):
        raise TypeError("forensic cache requires ForensicCheckpointIdentity")
    if (
        len(identity.source_commit) != 40
        or set(identity.source_commit) - _HEX
        or identity.world_size != 8
        or any(not _is_sha256(value) for value in (
            identity.config_identity,
            identity.config_sha256,
            identity.run_identity,
            identity.control_sha256,
            identity.failure_manifest_sha256,
        ))
    ):
        raise ValueError("forensic source/config/run/world identity is invalid")
    if (
        len(identity.rank_topology) != 8
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"rank", "node_rank", "local_rank"}
            or any(
                isinstance(item[field], bool) or not isinstance(item[field], int)
                for field in ("rank", "node_rank", "local_rank")
            )
            for item in identity.rank_topology
        )
        or {
            (item["rank"], item["node_rank"], item["local_rank"])
            for item in identity.rank_topology
        }
        != {(rank, rank // 4, rank % 4) for rank in range(8)}
    ):
        raise ValueError("forensic exact rank topology is invalid")
    if set(identity.model_data_identities) != set(_IDENTITY_FIELDS) or any(
        not _is_sha256(value) for value in identity.model_data_identities.values()
    ):
        raise ValueError("forensic ID176/processor/tokenizer/template/data identity is invalid")
    actor = identity.actor_failure
    if (
        not isinstance(actor, Mapping)
        or set(actor) != {"evidence_identity", "kl", "top1_agreement", "passed"}
        or not _is_sha256(actor.get("evidence_identity"))
        or actor.get("passed") is not False
        or isinstance(actor.get("kl"), bool)
        or not isinstance(actor.get("kl"), (float, int))
        or isinstance(actor.get("top1_agreement"), bool)
        or not isinstance(actor.get("top1_agreement"), (float, int))
        or not math.isfinite(float(actor["kl"]))
        or not math.isfinite(float(actor["top1_agreement"]))
    ):
        raise ValueError("forensic actor-failure evidence is invalid")
    run_root = Path(identity.run_root)
    config = Path(identity.config_path)
    checkpoint = Path(identity.checkpoint_path)
    failure = Path(identity.failure_manifest_path)
    if (
        not run_root.is_absolute()
        or not config.is_absolute()
        or not config.is_file()
        or config.is_symlink()
        or _sha256_file(config) != identity.config_sha256
        or not checkpoint.is_absolute()
        or not failure.is_absolute()
        or checkpoint.parent != run_root / "forensics"
        or failure.parent != run_root / "durable" / "failures"
        or checkpoint.name != "unsafe_update_00001605"
        or not checkpoint.is_dir()
        or checkpoint.is_symlink()
        or not failure.is_file()
        or failure.is_symlink()
        or _sha256_file(failure) != identity.failure_manifest_sha256
        or not (checkpoint / "control.json").is_file()
        or (checkpoint / "control.json").is_symlink()
        or _sha256_file(checkpoint / "control.json") != identity.control_sha256
    ):
        raise ValueError("forensic run/failure/control live provenance is invalid")
    control_raw = _read_mapping(checkpoint / "control.json", owner="forensic control")
    control_identity = control_raw.get("identity")
    if (
        not isinstance(control_identity, Mapping)
        or control_identity.get("source_commit") != identity.source_commit
        or control_identity.get("config_identity") != identity.config_identity
        or control_identity.get("run_identity") != identity.run_identity
        or control_identity.get("world_size") != 8
        or control_raw.get("forensic_only") is not True
        or control_raw.get("terminal_primary") is not False
        or control_raw.get("global_step") != 1605
    ):
        raise ValueError("forensic control source/config/run/world identity is invalid")
    failure_raw = _read_mapping(failure, owner="forensic failure manifest")
    live_actor_failure = _actor_failure_from_manifest(failure_raw)
    if dict(identity.actor_failure) != live_actor_failure:
        raise ValueError("forensic actor-failure evidence differs from live failure manifest")
    forensic = failure_raw.get("forensic_checkpoint")
    if (
        failure_raw.get("schema") != "nimloth_sft1_query_state_segment_v1"
        or failure_raw.get("run_identity") != identity.run_identity
        or failure_raw.get("mode") != "formal"
        or failure_raw.get("end_update") != 1605
        or failure_raw.get("resumable") is not False
        or not isinstance(forensic, Mapping)
        or forensic.get("path") != str(checkpoint)
        or forensic.get("control_sha256") != identity.control_sha256
        or forensic.get("forensic_only") is not True
        or forensic.get("resumable") is not False
        or forensic.get("authoritative") is not False
    ):
        raise ValueError("forensic failure manifest owner identity is invalid")
    if len(identity.rank_shards) != 8 or {item.rank for item in identity.rank_shards} != set(range(8)):
        raise ValueError("forensic cache requires all 8 rank shard identities")
    expected_rank_files = {f"rank_{rank:05d}_of_00008.pt" for rank in range(8)}
    live_rank_files = {path.name for path in checkpoint.glob("rank_*.pt")}
    if live_rank_files != expected_rank_files:
        raise ValueError("forensic checkpoint live rank shard set is not exact WS8")
    for item in identity.rank_shards:
        expected = f"rank_{item.rank:05d}_of_00008.pt"
        path = checkpoint / item.file
        if (
            item.file != expected
            or not _is_sha256(item.sha256)
            or isinstance(item.count, bool)
            or not isinstance(item.count, int)
            or item.count < 1
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != item.count
            or _sha256_file(path) != item.sha256
        ):
            raise ValueError("forensic rank shard hash/count/live identity is invalid")
    return identity


def _subset_groups_exact(
    groups: Sequence[tuple[str, tuple[SFT1V2Early4Row, ...]]], count: int
) -> tuple[int, ...]:
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_image, rows) in enumerate(groups):
        for current, chosen in sorted(tuple(reachable.items()), reverse=True):
            candidate = current + len(rows)
            if candidate <= count and candidate not in reachable:
                reachable[candidate] = (*chosen, index)
        if count in reachable:
            return reachable[count]
    raise ValueError(f"train exact-image groups cannot form required Stage A count={count}")


def select_forensic_stage_a_rows(
    rows: Sequence[SFT1V2Early4Row], *, seed: int
) -> ForensicStageASelection:
    """Select deterministic 48/16 train-derived rows without splitting image groups."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("forensic Stage A selection seed must be a non-negative integer")
    train = tuple(row for row in rows if row.split == "train")
    if len({row.identity for row in train}) != len(train):
        raise ValueError("forensic Stage A source has duplicate row identity")
    grouped: dict[str, list[SFT1V2Early4Row]] = {}
    for row in train:
        if row.image_content_group != row.original_image_sha256:
            raise ValueError("forensic Stage A exact-image group identity mismatch")
        grouped.setdefault(row.original_image_sha256, []).append(row)
    ordered = sorted(
        ((image, tuple(sorted(items, key=lambda row: row.ordinal))) for image, items in grouped.items()),
        key=lambda item: hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest(),
    )
    train_indices = set(_subset_groups_exact(ordered, 48))
    remaining = tuple(item for index, item in enumerate(ordered) if index not in train_indices)
    validation_indices = set(_subset_groups_exact(remaining, 16))
    train_rows = tuple(row for index, (_image, group) in enumerate(ordered) if index in train_indices for row in group)
    validation_rows = tuple(row for index, (_image, group) in enumerate(remaining) if index in validation_indices for row in group)
    if len(train_rows) != 48 or len(validation_rows) != 16 or (
        {row.original_image_sha256 for row in train_rows}
        & {row.original_image_sha256 for row in validation_rows}
    ):
        raise RuntimeError("forensic Stage A mechanics roles are not exact-image disjoint")
    payload = {
        "algorithm": FORENSIC_STAGE_A_SELECTION_ALGORITHM,
        "seed": seed,
        "roles": {
            FORENSIC_SELECTION_MECHANICS_TRAIN: [row.identity for row in train_rows],
            FORENSIC_SELECTION_MECHANICS_VALIDATION: [row.identity for row in validation_rows],
        },
    }
    entries = tuple(
        ForensicSelectionEntry(ordinal, role, row)
        for ordinal, (role, row) in enumerate(
            (*((FORENSIC_SELECTION_MECHANICS_TRAIN, row) for row in train_rows),
             *((FORENSIC_SELECTION_MECHANICS_VALIDATION, row) for row in validation_rows))
        )
    )
    return ForensicSelection(
        stage=ForensicExperimentStage.MECHANICS_ONLY,
        seed=seed,
        algorithm=FORENSIC_STAGE_A_SELECTION_ALGORITHM,
        identity=_identity(payload),
        entries=entries,
    )


def select_forensic_stage_b_rows(
    rows: Sequence[SFT1V2Early4Row],
) -> ForensicSelection:
    """Select the complete live-audited train/external roles without a caller mask."""

    train = tuple(sorted((row for row in rows if row.split == "train"), key=lambda row: row.ordinal))
    external = tuple(
        sorted(
            (row for row in rows if row.split == "val" and row.external_eligible),
            key=lambda row: row.ordinal,
        )
    )
    if any(not row.external_eligible for row in train) or (len(train), len(external)) != (
        FORENSIC_STAGE_B_TRAIN_COUNT,
        FORENSIC_STAGE_B_EXTERNAL_COUNT,
    ):
        raise ValueError("forensic Stage B live audit must yield exactly 12836/1413 rows")
    identities = [row.identity for row in (*train, *external)]
    if len(set(identities)) != len(identities):
        raise ValueError("forensic Stage B source has duplicate row identity")
    train_images = {row.original_image_sha256 for row in train}
    external_images = {row.original_image_sha256 for row in external}
    if train_images & external_images:
        raise ValueError("forensic Stage B roles must have zero exact-image overlap")
    payload = {
        "algorithm": FORENSIC_STAGE_B_SELECTION_ALGORITHM,
        "roles": {
            FORENSIC_SELECTION_ALL_TRAIN: [row.identity for row in train],
            FORENSIC_SELECTION_EXTERNAL_VALIDATION: [row.identity for row in external],
        },
    }
    entries = tuple(
        ForensicSelectionEntry(ordinal, role, row)
        for ordinal, (role, row) in enumerate(
            (*((FORENSIC_SELECTION_ALL_TRAIN, row) for row in train),
             *((FORENSIC_SELECTION_EXTERNAL_VALIDATION, row) for row in external))
        )
    )
    return ForensicSelection(
        stage=ForensicExperimentStage.STAGE_B_DIAGNOSTIC,
        seed=None,
        algorithm=FORENSIC_STAGE_B_SELECTION_ALGORITHM,
        identity=_identity(payload),
        entries=entries,
    )


def forensic_shard_ranges(count: int, *, max_records: int) -> tuple[tuple[int, int], ...]:
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records < 1
    ):
        raise ValueError("forensic cache shard count and bound must be positive integers")
    return tuple(
        (start, min(start + max_records, count))
        for start in range(0, count, max_records)
    )


def forensic_rank_schedule(
    entries: Sequence[ForensicSelectionEntry], *, rank: int, world_size: int
) -> tuple[tuple[ForensicSelectionEntry, bool], ...]:
    """Return equal-length rank schedules with explicit noncontributing padding."""

    if not entries or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("forensic distributed schedule identity is invalid")
    step_count = (len(entries) + world_size - 1) // world_size
    scheduled: list[tuple[ForensicSelectionEntry, bool]] = []
    for batch in range(step_count):
        index = batch * world_size + rank
        contributing = index < len(entries)
        scheduled.append((entries[index] if contributing else entries[0], contributing))
    return tuple(scheduled)


def _validate_prepared(prepared: PreparedForensicRow, expected: SFT1V2Early4Row) -> None:
    if not isinstance(prepared, PreparedForensicRow) or prepared.row.identity != expected.identity:
        raise ValueError("forensic prepared row identity mismatch")
    provenance = prepared.provenance
    if set(provenance) != {*_PROVENANCE_FIELDS, "response_source"}:
        raise ValueError("forensic rendered provenance is incomplete")
    if provenance.get("response_source") != "archived" or any(
        not _is_sha256(provenance.get(name)) for name in _PROVENANCE_FIELDS
    ):
        raise ValueError("forensic cache requires real archived-response provenance")
    image = Path(expected.original_image_path)
    if not image.is_absolute() or not image.is_file() or _sha256_file(image) != expected.original_image_sha256:
        raise ValueError("forensic original image live identity mismatch")


def _row_payload(entry: ForensicSelectionEntry, prepared: PreparedForensicRow) -> dict[str, Any]:
    response_sha = hashlib.sha256(entry.row.archived_assistant_response.encode()).hexdigest()
    return {
        "selection_ordinal": entry.selection_ordinal,
        "selection_role": entry.role,
        "row_identity": entry.row.identity,
        "record_id": entry.row.record_id,
        "step_index": entry.row.step_index,
        "original_image_path": entry.row.original_image_path,
        "original_image_sha256": entry.row.original_image_sha256,
        "archived_assistant_response_sha256": response_sha,
        **dict(prepared.provenance),
    }


def _write_failure_evidence(staging: Path, *, rank: int, batch: int, error: BaseException) -> None:
    try:
        path = staging / f"failure_rank_{rank:05d}_batch_{batch:05d}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"rank": rank, "batch": batch, "error_type": type(error).__name__, "error": str(error)}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        pass


def _rank_temp_path(staging: Path, rank: int) -> Path:
    return staging / f"rank_cache_{rank:05d}_of_00008.pt"


def _write_rank_payload(
    staging: Path, *, rank: int, records: Sequence[tuple[torch.Tensor, Mapping[str, Any]]]
) -> ForensicRankSummary:
    ordered = sorted(records, key=lambda item: int(item[1]["selection_ordinal"]))
    state = torch.stack([item[0] for item in ordered]).float().contiguous()
    rows = [dict(item[1]) for item in ordered]
    path = _rank_temp_path(staging, rank)
    with path.open("xb") as stream:
        torch.save({"schema": _FORENSIC_SHARD_SCHEMA, "state": state, "rows": rows}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return ForensicRankSummary(
        rank=rank,
        file=path.name,
        count=len(rows),
        sha256=_sha256_file(path),
        ordinal_identity=_identity({"ordinals": [row["selection_ordinal"] for row in rows]}),
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_noreplace(source: Path, destination: Path) -> None:
    """Commit a validated cache with NFS-safe no-overwrite semantics.

    The destination mkdir is the durable ownership claim. Readers treat the
    manifest, moved last, as the validity commit and reject every earlier state.
    """

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"forensic cache output already exists: {destination}")
    manifest = source / "manifest.json"
    if (
        source.is_symlink()
        or not source.is_dir()
        or not manifest.is_file()
        or manifest.is_symlink()
    ):
        raise ValueError("forensic publication source requires a regular manifest.json")
    payloads = sorted(
        path for path in source.iterdir() if path.name != manifest.name
    )
    if not payloads or any(
        not path.is_file() or path.is_symlink() for path in payloads
    ):
        raise ValueError("forensic publication source contains invalid payload entries")

    try:
        destination.mkdir(exist_ok=False)
    except FileExistsError as caught:
        raise FileExistsError(
            f"forensic cache output already exists: {destination}"
        ) from caught
    _fsync_directory(destination.parent)

    committed = False
    try:
        for payload in payloads:
            os.rename(payload, destination / payload.name)
        _fsync_directory(destination)
        os.rename(manifest, destination / manifest.name)
        committed = True
        _fsync_directory(destination)
        source.rmdir()
        _fsync_directory(destination.parent)
    except BaseException as caught:
        if committed:
            raise ForensicPublicationDurabilityError(
                "forensic manifest committed but publication durability "
                "was not confirmed"
            ) from caught
        raise


def _source_payload(source: QueryStateSourceContract) -> dict[str, Any]:
    return {
        "train": {"path": str(Path(source.data.train_jsonl).resolve()), "sha256": source.data.train_sha256, "split": source.data.train_split},
        "validation": {"path": str(Path(source.data.validation_jsonl).resolve()), "sha256": source.data.validation_sha256, "split": source.data.validation_split},
        "source_manifest_identity": source.source_manifest_identity,
    }


def _checkpoint_payload(identity: ForensicCheckpointIdentity) -> dict[str, Any]:
    value = asdict(identity)
    value["rank_shards"] = [asdict(item) for item in identity.rank_shards]
    value["rank_topology"] = [dict(item) for item in identity.rank_topology]
    value["actor_failure"] = dict(identity.actor_failure)
    value["model_data_identities"] = dict(identity.model_data_identities)
    return value


def _validate_local_cache_payload(payload: object) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or set(payload) != {"schema", "state", "rows"} or payload.get("schema") != _FORENSIC_SHARD_SCHEMA:
        raise ValueError("unsupported forensic cache shard schema")
    state, rows = payload.get("state"), payload.get("rows")
    if (
        not isinstance(state, torch.Tensor)
        or state.ndim != 3
        or tuple(state.shape[1:]) != _STATE_SHAPE
        or state.dtype != torch.float32
        or not state.is_contiguous()
        or not torch.isfinite(state).all()
    ):
        raise ValueError("forensic cache shard must preserve contiguous finite float32 [N,16,1024]")
    if not isinstance(rows, list) or len(rows) != state.shape[0]:
        raise ValueError("forensic cache shard row/state count mismatch")
    required_row_fields = {
        "selection_ordinal", "selection_role", "row_identity", "record_id",
        "step_index", "original_image_path", "original_image_sha256",
        "archived_assistant_response_sha256", *_PROVENANCE_FIELDS,
        "response_source",
    }
    ordinals: set[int] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != required_row_fields
            or row.get("selection_role") not in {
                FORENSIC_SELECTION_MECHANICS_TRAIN,
                FORENSIC_SELECTION_MECHANICS_VALIDATION,
                FORENSIC_SELECTION_ALL_TRAIN,
                FORENSIC_SELECTION_EXTERNAL_VALIDATION,
            }
            or not isinstance(row.get("row_identity"), str)
            or not row["row_identity"]
            or not isinstance(row.get("record_id"), str)
            or not row["record_id"]
            or isinstance(row.get("step_index"), bool)
            or not isinstance(row.get("step_index"), int)
            or row["step_index"] < 0
            or not isinstance(row.get("original_image_path"), str)
            or not Path(row["original_image_path"]).is_absolute()
            or row.get("response_source") != "archived"
            or any(not _is_sha256(row.get(name)) for name in (*_PROVENANCE_FIELDS, "original_image_sha256", "archived_assistant_response_sha256"))
        ):
            raise ValueError("forensic cache row provenance is invalid")
        ordinal = row.get("selection_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or ordinal in ordinals:
            raise ValueError("forensic cache selection ordinal is invalid")
        ordinals.add(ordinal)
    return state.detach().contiguous(), [dict(row) for row in rows]


def _validate_producer_identity(
    producer: ForensicProducerIdentity,
) -> ForensicProducerIdentity:
    if (
        not isinstance(producer, ForensicProducerIdentity)
        or not Path(producer.integrated_repo_root).is_absolute()
        or len(producer.integrated_source_commit) != 40
        or set(producer.integrated_source_commit) - _HEX
        or not _is_sha256(producer.production_config_identity)
        or not _is_sha256(producer.formal_config_identity)
    ):
        raise ValueError("forensic production source/config identity is invalid")
    return producer


def _publish_rank_payloads(
    output: Path,
    staging: Path,
    *,
    summaries: Sequence[ForensicRankSummary],
    checkpoint: ForensicCheckpointIdentity,
    source: QueryStateSourceContract,
    selection: ForensicSelection,
    producer: ForensicProducerIdentity,
    max_shard_records: int = FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS,
) -> Mapping[str, Any]:
    if len(summaries) != 8 or {item.rank for item in summaries} != set(range(8)):
        raise ValueError("forensic rank summary world coverage is invalid")
    _validate_producer_identity(producer)
    merged: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for summary in sorted(summaries, key=lambda item: item.rank):
        path = staging / summary.file
        if summary.file != _rank_temp_path(staging, summary.rank).name or not path.is_file() or _sha256_file(path) != summary.sha256:
            raise ValueError("forensic rank temporary shard hash identity mismatch")
        state, rows = _validate_local_cache_payload(torch.load(path, map_location="cpu", weights_only=False))
        if len(rows) != summary.count or _identity({"ordinals": [row["selection_ordinal"] for row in rows]}) != summary.ordinal_identity:
            raise ValueError("forensic rank temporary shard count/ordinal identity mismatch")
        merged.extend((state[index], row) for index, row in enumerate(rows))
    merged.sort(key=lambda item: int(item[1]["selection_ordinal"]))
    if [item[1]["selection_ordinal"] for item in merged] != list(range(len(selection.entries))):
        raise ValueError("forensic global selection-ordinal coverage is invalid")
    expected = {entry.selection_ordinal: entry for entry in selection.entries}
    for _state, row in merged:
        entry = expected[row["selection_ordinal"]]
        if row["row_identity"] != entry.row.identity or row["selection_role"] != entry.role:
            raise ValueError("forensic global row selection identity mismatch")
    publication = staging.with_name(staging.name + ".publish")
    if publication.exists() or publication.is_symlink():
        raise FileExistsError("forensic publication temporary path already exists")
    publication.mkdir()
    try:
        descriptors: list[dict[str, Any]] = []
        for shard_index, (start, stop) in enumerate(
            forensic_shard_ranges(len(merged), max_records=max_shard_records)
        ):
            final_path = publication / f"shard_{shard_index:05d}.pt"
            chunk = merged[start:stop]
            final_payload = {
                "schema": _FORENSIC_SHARD_SCHEMA,
                "state": torch.stack([item[0] for item in chunk]),
                "rows": [item[1] for item in chunk],
            }
            with final_path.open("xb") as stream:
                torch.save(final_payload, stream)
                stream.flush(); os.fsync(stream.fileno())
            descriptors.append({
                "file": final_path.name, "count": stop - start, "start": start,
                "stop": stop, "sha256": _sha256_file(final_path),
                "state_dtype": "float32", "state_shape": list(_STATE_SHAPE),
            })
        roles = {
            role: sum(entry.role == role for entry in selection.entries)
            for role in dict.fromkeys(entry.role for entry in selection.entries)
        }
        manifest: dict[str, Any] = {
            "schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
            "version": 1,
            "owner_role": FORENSIC_QUERY_STATE_OWNER_ROLE,
            "forensic_only": True,
            "authoritative": False,
            "terminal_primary": False,
            "deployable": False,
            "sft2_ready": False,
            "count": len(merged),
            "state_shape": list(_STATE_SHAPE),
            "state_dtype": "float32",
            "checkpoint": _checkpoint_payload(checkpoint),
            "producer": asdict(producer),
            "source_jsonl": _source_payload(source),
            "selection": {"stage": selection.stage.value, "algorithm": selection.algorithm, "seed": selection.seed, "identity": selection.identity, "roles": roles},
            "row_set_identity": _identity({"rows": [item[1] for item in merged]}),
            "rank_cache_summaries": [asdict(item) for item in sorted(summaries, key=lambda item: item.rank)],
            "shards": descriptors,
        }
        manifest["cache_fingerprint"] = _identity(manifest)
        with (publication / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        fd = os.open(publication, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
        _publish_noreplace(publication, output)
        return manifest
    except BaseException:
        if publication.exists():
            shutil.rmtree(publication)
        raise


def build_forensic_query_state_cache_rank(
    output: str | Path,
    *,
    checkpoint: ForensicCheckpointIdentity,
    source: QueryStateSourceContract,
    selection_seed: int | None,
    producer: ForensicProducerIdentity,
    extractor: ForensicStateExtractor,
    collective: ForensicCollective,
    experiment_stage: ForensicExperimentStage = ForensicExperimentStage.MECHANICS_ONLY,
    max_shard_records: int = FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS,
) -> Mapping[str, Any] | None:
    """Run one WS8 rank; rank0 atomically publishes after all rank validation."""

    output_path = Path(output)
    staging = output_path.with_name(f".{output_path.name}.forensic-tmp")
    if collective.world_size != 8 or not 0 <= collective.rank < 8:
        raise ValueError("forensic cache builder requires exact WS8 rank identity")
    identity_error: BaseException | None = None
    rows: tuple[SFT1V2Early4Row, ...] = ()
    selection: ForensicSelection | None = None
    try:
        if not isinstance(experiment_stage, ForensicExperimentStage):
            raise TypeError("forensic cache experiment_stage must be typed")
        if (
            isinstance(max_shard_records, bool)
            or not isinstance(max_shard_records, int)
            or max_shard_records < 1
            or (
                experiment_stage is ForensicExperimentStage.STAGE_B_DIAGNOSTIC
                and max_shard_records != FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS
            )
        ):
            raise ValueError("forensic cache stage/shard bound contract is invalid")
        validate_forensic_checkpoint_identity(checkpoint)
        _validate_producer_identity(producer)
        rows, audit = index_early4_rows(source, enforce_approved_counts=False)
        if build_query_state_source_manifest_identity(rows, audit) != source.source_manifest_identity:
            raise ValueError("forensic live source manifest identity mismatch")
        if experiment_stage is ForensicExperimentStage.STAGE_B_DIAGNOSTIC and (
            audit.train_rows != FORENSIC_STAGE_B_TRAIN_COUNT
            or audit.raw_validation_rows != 1_420
            or audit.external_validation_rows != FORENSIC_STAGE_B_EXTERNAL_COUNT
        ):
            raise ValueError("forensic Stage B live source audit counts are not exact")
        selection = (
            select_forensic_stage_a_rows(rows, seed=selection_seed)
            if experiment_stage is ForensicExperimentStage.MECHANICS_ONLY
            else select_forensic_stage_b_rows(rows)
        )
    except BaseException as caught:
        identity_error = caught
    statuses = collective.gate(
        "identity",
        ready=identity_error is None,
        detail="ready" if identity_error is None else f"{type(identity_error).__name__}: {identity_error}",
    )
    _require_gate(statuses, phase="identity", world_size=8)
    assert selection is not None

    staging_error: BaseException | None = None
    if collective.rank == 0:
        try:
            if output_path.exists() or output_path.is_symlink() or staging.exists() or staging.is_symlink():
                raise FileExistsError("forensic cache output or temporary path already exists")
            staging.mkdir(parents=True)
        except BaseException as caught:
            staging_error = caught
    statuses = collective.gate(
        "staging_init",
        ready=staging_error is None,
        detail="ready" if staging_error is None else f"{type(staging_error).__name__}: {staging_error}",
    )
    _require_gate(statuses, phase="staging_init", world_size=8)
    statuses = collective.gate("staging", ready=staging.is_dir(), detail="staging-ready" if staging.is_dir() else "staging-missing")
    _require_gate(statuses, phase="staging", world_size=8)
    local_records: list[tuple[torch.Tensor, Mapping[str, Any]]] = []
    schedule = forensic_rank_schedule(
        selection.entries, rank=collective.rank, world_size=8
    )
    step_count = len(schedule)
    for batch, (entry, contributing) in enumerate(schedule):
        prepared: PreparedForensicRow | None = None
        error: BaseException | None = None
        try:
            prepared = extractor.prepare(entry.row)
            _validate_prepared(prepared, entry.row)
        except BaseException as caught:
            error = caught
        statuses = collective.gate("pre_forward", ready=error is None, detail="ready" if error is None else f"{type(error).__name__}: {error}")
        _require_gate(statuses, phase="pre_forward", world_size=8)
        assert prepared is not None
        try:
            state = extractor.extract((prepared,))
        except BaseException as caught:
            _write_failure_evidence(staging, rank=collective.rank, batch=batch, error=caught)
            collective.teardown()
            raise
        post_error: BaseException | None = None
        try:
            canonical = validate_canonical_query_state(state)
            if canonical.shape[0] != 1:
                raise ValueError("forensic extractor must return one state per scheduled row")
        except BaseException as caught:
            post_error = caught
        statuses = collective.gate("post_forward", ready=post_error is None, detail="ready" if post_error is None else f"{type(post_error).__name__}: {post_error}")
        _require_gate(statuses, phase="post_forward", world_size=8)
        if contributing:
            local_records.append((canonical[0].detach().cpu().float(), _row_payload(entry, prepared)))
    try:
        summary = _write_rank_payload(staging, rank=collective.rank, records=local_records)
    except BaseException as caught:
        _write_failure_evidence(staging, rank=collective.rank, batch=step_count, error=caught)
        collective.teardown()
        raise
    summaries = collective.gather_summaries(summary)
    publish_error: BaseException | None = None
    manifest: Mapping[str, Any] | None = None
    if collective.rank == 0:
        try:
            manifest = _publish_rank_payloads(
                output_path,
                staging,
                summaries=summaries,
                checkpoint=checkpoint,
                source=source,
                selection=selection,
                producer=producer,
                max_shard_records=(
                    len(selection.entries)
                    if selection.stage is ForensicExperimentStage.MECHANICS_ONLY
                    else max_shard_records
                ),
            )
        except BaseException as caught:
            publish_error = caught
    statuses = collective.gate(
        "publish",
        ready=publish_error is None,
        detail=_publication_gate_detail(publish_error),
    )
    _require_publication_gate(statuses, world_size=8)
    if collective.rank == 0:
        shutil.rmtree(staging)
    return manifest


def _parse_manifest(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema", "version", "owner_role", "forensic_only", "authoritative",
        "terminal_primary", "deployable", "sft2_ready", "count", "state_shape",
        "state_dtype", "checkpoint", "producer", "source_jsonl", "selection", "row_set_identity",
        "rank_cache_summaries", "shards", "cache_fingerprint",
    }
    if set(raw) != required or raw.get("schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA:
        raise ValueError("forensic reader rejects deployable, legacy, or unknown cache schema")
    selection = raw.get("selection")
    shards = raw.get("shards")
    summaries = raw.get("rank_cache_summaries")
    source = raw.get("source_jsonl")
    producer = raw.get("producer")
    stage = selection.get("stage") if isinstance(selection, dict) else None
    expected_count, expected_algorithm, expected_roles = (
        (64, FORENSIC_STAGE_A_SELECTION_ALGORITHM, {
            FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
            FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
        })
        if stage == ForensicExperimentStage.MECHANICS_ONLY.value
        else (FORENSIC_STAGE_B_TRAIN_COUNT + FORENSIC_STAGE_B_EXTERNAL_COUNT,
              FORENSIC_STAGE_B_SELECTION_ALGORITHM, {
                  FORENSIC_SELECTION_ALL_TRAIN: FORENSIC_STAGE_B_TRAIN_COUNT,
                  FORENSIC_SELECTION_EXTERNAL_VALIDATION: FORENSIC_STAGE_B_EXTERNAL_COUNT,
              })
        if stage == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value
        else (-1, "", {})
    )
    valid_shards = isinstance(shards, list) and bool(shards)
    expected_start = 0
    if valid_shards:
        for index, descriptor in enumerate(shards):
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"file", "count", "start", "stop", "sha256", "state_dtype", "state_shape"}
                or descriptor.get("file") != f"shard_{index:05d}.pt"
                or descriptor.get("start") != expected_start
                or isinstance(descriptor.get("stop"), bool)
                or not isinstance(descriptor.get("stop"), int)
                or descriptor["stop"] <= expected_start
                or descriptor.get("count") != descriptor["stop"] - expected_start
                or not _is_sha256(descriptor.get("sha256"))
                or descriptor.get("state_dtype") != "float32"
                or tuple(descriptor.get("state_shape", ())) != _STATE_SHAPE
                or (
                    stage == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value
                    and descriptor["count"] > FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS
                )
            ):
                valid_shards = False
                break
            expected_start = descriptor["stop"]
        valid_shards = valid_shards and expected_start == expected_count
        if valid_shards:
            actual_ranges = tuple((item["start"], item["stop"]) for item in shards)
            expected_ranges = forensic_shard_ranges(
                expected_count,
                max_records=(
                    expected_count
                    if stage == ForensicExperimentStage.MECHANICS_ONLY.value
                    else FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS
                ),
            )
            valid_shards = actual_ranges == expected_ranges
    if (
        raw.get("version") != 1
        or raw.get("owner_role") != FORENSIC_QUERY_STATE_OWNER_ROLE
        or raw.get("forensic_only") is not True
        or any(raw.get(name) is not False for name in ("authoritative", "terminal_primary", "deployable", "sft2_ready"))
        or raw.get("count") != expected_count
        or tuple(raw.get("state_shape", ())) != _STATE_SHAPE
        or raw.get("state_dtype") != "float32"
        or not _is_sha256(raw.get("row_set_identity"))
        or not _is_sha256(raw.get("cache_fingerprint"))
        or not isinstance(selection, dict)
        or set(selection) != {"stage", "algorithm", "seed", "identity", "roles"}
        or stage not in {item.value for item in ForensicExperimentStage}
        or selection.get("algorithm") != expected_algorithm
        or (
            stage == ForensicExperimentStage.MECHANICS_ONLY.value
            and (
                isinstance(selection.get("seed"), bool)
                or not isinstance(selection.get("seed"), int)
                or selection["seed"] < 0
            )
        )
        or (
            stage == ForensicExperimentStage.STAGE_B_DIAGNOSTIC.value
            and selection.get("seed") is not None
        )
        or not _is_sha256(selection.get("identity"))
        or selection.get("roles") != expected_roles
        or not isinstance(producer, dict)
        or set(producer) != {
            "integrated_repo_root", "integrated_source_commit",
            "production_config_identity", "formal_config_identity",
        }
        or not isinstance(producer.get("integrated_repo_root"), str)
        or not Path(producer["integrated_repo_root"]).is_absolute()
        or not isinstance(producer.get("integrated_source_commit"), str)
        or len(producer["integrated_source_commit"]) != 40
        or set(producer["integrated_source_commit"]) - _HEX
        or not _is_sha256(producer.get("production_config_identity"))
        or not _is_sha256(producer.get("formal_config_identity"))
        or not isinstance(source, dict)
        or set(source) != {"train", "validation", "source_manifest_identity"}
        or not _is_sha256(source.get("source_manifest_identity"))
        or not valid_shards
        or not isinstance(summaries, list)
        or len(summaries) != 8
        or {item.get("rank") for item in summaries if isinstance(item, dict)} != set(range(8))
        or any(
            not isinstance(item, dict)
            or set(item) != {"rank", "file", "count", "sha256", "ordinal_identity"}
            or isinstance(item.get("rank"), bool)
            or not isinstance(item.get("rank"), int)
            or item.get("file") != f"rank_cache_{item['rank']:05d}_of_00008.pt"
            or not _is_sha256(item.get("sha256"))
            or not _is_sha256(item.get("ordinal_identity"))
            or isinstance(item.get("count"), bool)
            or not isinstance(item.get("count"), int)
            or item["count"] < 0
            for item in summaries
        )
        or sum(item["count"] for item in summaries) != expected_count
        or _identity({key: value for key, value in raw.items() if key != "cache_fingerprint"}) != raw.get("cache_fingerprint")
    ):
        raise ValueError("forensic cache manifest identity/watermark is invalid")
    for split_name in ("train", "validation"):
        entry = source.get(split_name) if isinstance(source, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "split"}
            or not isinstance(entry.get("path"), str)
            or not Path(entry["path"]).is_absolute()
            or not _is_sha256(entry.get("sha256"))
            or not isinstance(entry.get("split"), str)
            or not entry["split"]
        ):
            raise ValueError("forensic cache source JSONL identity is invalid")
    return raw


class ForensicQueryStateCacheDataset:
    """Strict forensic reader with live source/checkpoint/image/shard validation."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("forensic cache root must not be a symlink")
        self.root = supplied.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("forensic cache manifest must be a regular file")
        raw = _read_mapping(manifest_path, owner="forensic cache manifest")
        self.manifest = _parse_manifest(raw)
        checkpoint_raw = dict(self.manifest["checkpoint"])
        checkpoint_raw["rank_shards"] = tuple(ForensicRankShardIdentity(**item) for item in checkpoint_raw["rank_shards"])
        checkpoint_raw["rank_topology"] = tuple(checkpoint_raw["rank_topology"])
        validate_forensic_checkpoint_identity(ForensicCheckpointIdentity(**checkpoint_raw))
        source_raw = self.manifest["source_jsonl"]
        # Construct through the existing exact source dataclass without weakening it.
        from nimloth.training.reconstruction.query_state_cache import (
            QueryStateSourceData,
        )
        source = QueryStateSourceContract(
            data=QueryStateSourceData(
                train_jsonl=source_raw["train"]["path"], train_sha256=source_raw["train"]["sha256"],
                validation_jsonl=source_raw["validation"]["path"], validation_sha256=source_raw["validation"]["sha256"],
                train_split=source_raw["train"]["split"], validation_split=source_raw["validation"]["split"],
            ),
            source_manifest_identity=source_raw["source_manifest_identity"],
        )
        rows, audit = index_early4_rows(source, enforce_approved_counts=False)
        if build_query_state_source_manifest_identity(rows, audit) != source.source_manifest_identity:
            raise ValueError("forensic cache live source identity mismatch")
        selection_raw = self.manifest["selection"]
        selection = (
            select_forensic_stage_a_rows(rows, seed=selection_raw["seed"])
            if selection_raw["stage"] == ForensicExperimentStage.MECHANICS_ONLY.value
            else select_forensic_stage_b_rows(rows)
        )
        if selection.identity != selection_raw["identity"]:
            raise ValueError("forensic cache live selection identity mismatch")
        self._expected = {entry.selection_ordinal: entry for entry in selection.entries}
        states: list[torch.Tensor] = []
        cache_rows: list[dict[str, Any]] = []
        for descriptor in self.manifest["shards"]:
            path = self.root / descriptor["file"]
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != descriptor["sha256"]:
                raise ValueError("forensic cache shard SHA256/hash mismatch")
            state, shard_rows = _validate_local_cache_payload(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            if len(shard_rows) != descriptor["count"]:
                raise ValueError("forensic cache shard count mismatch")
            states.append(state)
            cache_rows.extend(shard_rows)
        self._state = torch.cat(states, dim=0).contiguous()
        self._rows = cache_rows
        if len(self._rows) != len(selection.entries):
            raise ValueError("forensic cache global shard count mismatch")
        if [row["selection_ordinal"] for row in self._rows] != list(range(len(selection.entries))):
            raise ValueError("forensic cache rows must preserve exact selection ordinal order")
        for row in self._rows:
            expected = self._expected[row["selection_ordinal"]]
            image = Path(row["original_image_path"])
            live_response_sha = hashlib.sha256(
                expected.row.archived_assistant_response.encode()
            ).hexdigest()
            if (
                row["row_identity"] != expected.row.identity
                or row["selection_role"] != expected.role
                or row["record_id"] != expected.row.record_id
                or row["step_index"] != expected.row.step_index
                or row["original_image_path"] != expected.row.original_image_path
                or row["original_image_sha256"] != expected.row.original_image_sha256
                or row["archived_assistant_response_sha256"] != live_response_sha
                or not image.is_file()
                or _sha256_file(image) != row["original_image_sha256"]
            ):
                raise ValueError("forensic cache live row/image/archived-response identity mismatch")
        if _identity({"rows": self._rows}) != self.manifest["row_set_identity"]:
            raise ValueError("forensic cache row-set identity mismatch")

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def cache_fingerprint(self) -> str:
        return str(self.manifest["cache_fingerprint"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("forensic cache index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {"state": self._state[index].detach().clone(), **dict(self._rows[index])}


__all__ = [
    "FORENSIC_QUERY_STATE_CACHE_SCHEMA",
    "FORENSIC_QUERY_STATE_OWNER_ROLE",
    "FORENSIC_SELECTION_ALL_TRAIN",
    "FORENSIC_SELECTION_EXTERNAL_VALIDATION",
    "FORENSIC_SELECTION_MECHANICS_TRAIN",
    "FORENSIC_SELECTION_MECHANICS_VALIDATION",
    "FORENSIC_STAGE_A_SELECTION_ALGORITHM",
    "FORENSIC_STAGE_B_DEFAULT_SHARD_RECORDS",
    "FORENSIC_STAGE_B_EXTERNAL_COUNT",
    "FORENSIC_STAGE_B_ROLES",
    "FORENSIC_STAGE_B_SELECTION_ALGORITHM",
    "FORENSIC_STAGE_B_TRAIN_COUNT",
    "ForensicCheckpointIdentity",
    "ForensicCollective",
    "ForensicExperimentStage",
    "ForensicProducerIdentity",
    "ForensicQueryStateCacheDataset",
    "ForensicRankShardIdentity",
    "ForensicSelection",
    "ForensicStageASelection",
    "ForensicStateExtractor",
    "PreparedForensicRow",
    "TorchForensicCollective",
    "actor_failure_evidence_from_manifest",
    "build_forensic_query_state_cache_rank",
    "forensic_rank_schedule",
    "forensic_shard_ranges",
    "select_forensic_stage_a_rows",
    "select_forensic_stage_b_rows",
    "validate_forensic_checkpoint_identity",
]
