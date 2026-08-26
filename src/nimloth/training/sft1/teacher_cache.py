"""Fresh detached-teacher cache preparation and atomic sharded transactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch

from nimloth.training.sft1.data import SFT1V2TeacherRow, sha256_file
from nimloth.training.sft1.real_rows import SFT1V2RenderedRow


TEACHER_CACHE_SCHEMA = "nimloth_sft1_state_v2_teacher_cache_v1"
TEACHER_CACHE_ROW_SCHEMA = "nimloth_sft1_state_v2_teacher_row_v2"
TEACHER_CACHE_SHARD_SCHEMA = "nimloth_sft1_state_v2_teacher_shard_v1"
COMPLETE_MARKER = "COMPLETED"


def _sha(value: str, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class SFT1V2TeacherCacheIdentity:
    source_commit: str
    actor_checkpoint_sha256: str
    actor_config_sha256: str
    actor_model_index_sha256: str
    actor_action_head_sha256: str
    actor_shards_sha256: tuple[str, ...]
    processor_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    prompt_renderer_sha256: str
    token_table_sha256: str
    query_action_contract_sha256: str
    dino_checkpoint_sha256: str
    dino_processor_sha256: str
    train_trajectory_sha256: str
    validation_trajectory_sha256: str
    query_count: int = 16
    action_count: int = 8
    instruction_dim: int = 2048
    dino_shape: tuple[int, int] = (16, 1024)

    def __post_init__(self) -> None:
        if (
            len(self.source_commit) != 40
            or any(char not in "0123456789abcdef" for char in self.source_commit)
        ):
            raise ValueError("teacher source_commit must be a lowercase Git SHA")
        for name in (
            "actor_checkpoint_sha256", "actor_config_sha256",
            "actor_model_index_sha256", "actor_action_head_sha256",
            "processor_sha256", "tokenizer_sha256",
            "chat_template_sha256", "prompt_renderer_sha256", "token_table_sha256",
            "query_action_contract_sha256", "dino_checkpoint_sha256",
            "dino_processor_sha256", "train_trajectory_sha256",
            "validation_trajectory_sha256",
        ):
            _sha(getattr(self, name), name)
        if not self.actor_shards_sha256:
            raise ValueError("teacher identity requires every actor checkpoint shard")
        for index, digest in enumerate(self.actor_shards_sha256):
            _sha(digest, f"actor_shards_sha256[{index}]")
        if (self.query_count, self.action_count, self.instruction_dim, self.dino_shape) != (16, 8, 2048, (16, 1024)):
            raise ValueError("teacher identity differs from the K16/ID176/DINO contract")

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SFT1V2FreshTargets:
    dino_regions: torch.Tensor
    instruction_teacher: torch.Tensor
    actor_teacher_log_probs: torch.Tensor


class SFT1V2FreshTeacher(Protocol):
    """Production implementation owns frozen ID176 and DINO forwards."""

    def build(self, rendered: SFT1V2RenderedRow) -> SFT1V2FreshTargets: ...

    def build_many(
        self, rendered: Sequence[SFT1V2RenderedRow]
    ) -> Sequence[SFT1V2FreshTargets]: ...


class SFT1V2ParityReference(Protocol):
    """Parity is evidence only; it cannot return or replace a target."""

    def compare(self, rendered: SFT1V2RenderedRow, fresh: SFT1V2FreshTargets) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class SFT1V2CacheShardSummary:
    shard_index: int
    shard_count: int
    row_count: int
    first_ordinal: int | None
    last_ordinal: int | None
    payload_sha256: str
    row_identities_sha256: str
    parity_summary: Mapping[str, float]


@dataclass(frozen=True)
class SFT1V2CacheSummary:
    schema: str
    cache_identity: str
    shard_count: int
    row_count: int
    shard_payload_sha256: tuple[str, ...]
    root_manifest_sha256: str


def deterministic_shard_ownership(ordinal: int, shard_count: int) -> int:
    if ordinal < 0 or shard_count < 1:
        raise ValueError("cache ordinal/shard_count are invalid")
    return ordinal % shard_count


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_dir(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _shard_stem(index: int, count: int) -> str:
    if index < 0 or index >= count:
        raise ValueError("cache shard index is outside world")
    return f"shard_{index:05d}_of_{count:05d}"


def _validate_fresh_targets(targets: SFT1V2FreshTargets) -> SFT1V2FreshTargets:
    shapes = {
        "dino_regions": (16, 1024),
        "instruction_teacher": (2048,),
        "actor_teacher_log_probs": (8,),
    }
    values: dict[str, torch.Tensor] = {}
    for name, shape in shapes.items():
        value = getattr(targets, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"fresh teacher {name} must have shape {shape}")
        if value.requires_grad or not torch.isfinite(value).all():
            raise ValueError(f"fresh teacher {name} must be detached and finite")
        values[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.allclose(
        torch.logsumexp(values["actor_teacher_log_probs"], dim=-1),
        torch.tensor(0.0), atol=1e-5, rtol=0,
    ):
        raise ValueError("fresh actor teacher log probabilities must be normalized")
    return SFT1V2FreshTargets(**values)


def _row_payload(
    rendered: SFT1V2RenderedRow,
    targets: SFT1V2FreshTargets,
    *,
    parity_evidence: Mapping[str, float],
) -> dict[str, Any]:
    row = rendered.row
    if sha256_file(Path(row.original_image_path)) != row.original_image_sha256:
        raise ValueError("original image changed before fresh teacher preparation")
    # No encoded prompt or student state enters the cache.
    return {
        "schema": TEACHER_CACHE_ROW_SCHEMA,
        "row_identity": row.identity,
        "ordinal": row.ordinal,
        "record_id": row.record_id,
        "step_index": row.step_index,
        "split": row.split,
        "source_sha256": row.source_sha256,
        "original_image_sha256": row.original_image_sha256,
        "image_content_group": row.image_content_group,
        "instruction_equivalence_group": row.instruction_equivalence_group,
        "instruction_char_span": row.instruction_char_span,
        "executed_action_index": row.executed_action_index,
        "movement_success": row.movement_success,
        "external_eligible": row.external_eligible,
        "rendered_input_ids_sha256": hashlib.sha256(rendered.input_ids.numpy().tobytes()).hexdigest(),
        "instruction_token_span": rendered.instruction_token_span,
        "action_boundary_index": rendered.action_boundary_index,
        "dino_regions": targets.dino_regions,
        "instruction_teacher": targets.instruction_teacher,
        "actor_teacher_log_probs": targets.actor_teacher_log_probs,
        "parity_evidence": dict(parity_evidence),
    }


def _build_many(
    teacher: SFT1V2FreshTeacher,
    rendered: Sequence[SFT1V2RenderedRow],
) -> tuple[SFT1V2FreshTargets, ...]:
    method = getattr(teacher, "build_many", None)
    values = method(rendered) if callable(method) else [teacher.build(row) for row in rendered]
    if len(values) != len(rendered):
        raise ValueError("fresh teacher batch result count mismatch")
    return tuple(_validate_fresh_targets(value) for value in values)


def _load_partial_chunks(
    partial_dir: Path,
    expected_row_identities: Sequence[str],
    *,
    cache_identity: str,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not partial_dir.exists():
        return rows
    for path in sorted(partial_dir.glob("chunk_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != TEACHER_CACHE_SHARD_SCHEMA
            or payload.get("cache_identity") != cache_identity
            or payload.get("shard_index") != shard_index
            or payload.get("shard_count") != shard_count
            or payload.get("start") != len(rows)
        ):
            raise ValueError("cache partial chunk identity/order mismatch")
        chunk_rows = payload.get("rows")
        if not isinstance(chunk_rows, list) or not chunk_rows:
            raise ValueError("cache partial chunk rows are invalid")
        rows.extend(chunk_rows)
    if len(rows) > len(expected_row_identities):
        raise ValueError("cache partial shard prefix length is invalid")
    for actual, expected_identity in zip(rows, expected_row_identities, strict=False):
        if not isinstance(actual, dict) or actual.get("row_identity") != expected_identity:
            raise ValueError("cache partial shard is not an exact deterministic prefix")
    return rows


def prepare_teacher_cache_shard(
    output_dir: Path,
    rows: Sequence[Any],
    *,
    shard_index: int,
    shard_count: int,
    identity: SFT1V2TeacherCacheIdentity,
    teacher: SFT1V2FreshTeacher,
    parity_reference: SFT1V2ParityReference | None = None,
    teacher_batch_size: int = 1,
    render_row: Callable[[Any], SFT1V2RenderedRow] | None = None,
) -> SFT1V2CacheShardSummary:
    """Generate/resume one deterministic shard and publish it atomically."""

    output = Path(output_dir)
    if (output / COMPLETE_MARKER).exists():
        raise FileExistsError("completed teacher cache is immutable")
    stem = _shard_stem(shard_index, shard_count)
    complete_payload = output / "shards" / f"{stem}.pt"
    complete_metadata = output / "shards" / f"{stem}.json"
    if teacher_batch_size < 1:
        raise ValueError("teacher_batch_size must be positive")
    partial_dir = output / "partial" / stem
    if complete_metadata.exists() and not complete_payload.exists():
        raise ValueError("teacher cache metadata exists without its payload")
    if complete_payload.exists() and not complete_metadata.exists():
        # The payload rename is not committed until metadata exists. Chunk files
        # remain the authoritative exact prefix, so discard only this orphaned
        # transaction file and deterministically republish it.
        complete_payload.unlink()
    if complete_payload.exists() and complete_metadata.exists():
        metadata = json.loads(complete_metadata.read_text(encoding="utf-8"))
        if metadata.get("cache_identity") != identity.identity or sha256_file(complete_payload) != metadata.get("payload_sha256"):
            raise ValueError("completed teacher cache shard identity/hash mismatch")
        if partial_dir.exists():
            for path in partial_dir.glob("chunk_*.pt"):
                path.unlink()
            partial_dir.rmdir()
        return SFT1V2CacheShardSummary(**{
            key: metadata[key]
            for key in SFT1V2CacheShardSummary.__dataclass_fields__
        })

    def source_row(value: Any) -> Any:
        return value.row if isinstance(value, SFT1V2RenderedRow) else value

    owned = tuple(sorted(
        (
            row for row in rows
            if deterministic_shard_ownership(source_row(row).ordinal, shard_count)
            == shard_index
        ),
        key=lambda row: source_row(row).ordinal,
    ))
    expected_identities = tuple(source_row(row).identity for row in owned)
    payload_rows = _load_partial_chunks(
        partial_dir, expected_identities, cache_identity=identity.identity,
        shard_index=shard_index, shard_count=shard_count,
    )

    for start in range(len(payload_rows), len(owned), teacher_batch_size):
        source_batch = owned[start : start + teacher_batch_size]
        if render_row is None and any(
            not isinstance(row, SFT1V2RenderedRow) for row in source_batch
        ):
            raise TypeError("raw cache rows require render_row")
        rendered_batch = tuple(
            row
            if isinstance(row, SFT1V2RenderedRow)
            else render_row(row)  # type: ignore[misc]
            for row in source_batch
        )
        if any(not isinstance(row, SFT1V2RenderedRow) for row in rendered_batch):
            raise TypeError("cache row renderer must return SFT1V2RenderedRow")
        fresh_batch = _build_many(teacher, rendered_batch)
        chunk_rows: list[dict[str, Any]] = []
        for rendered, fresh in zip(rendered_batch, fresh_batch, strict=True):
            evidence: dict[str, float] = {}
            if parity_reference is not None:
                for name, value in parity_reference.compare(rendered, fresh).items():
                    numeric = float(value)
                    if not torch.isfinite(torch.tensor(numeric)):
                        raise ValueError("parity evidence must be finite")
                    evidence[str(name)] = numeric
            chunk_rows.append(_row_payload(
                rendered, fresh, parity_evidence=evidence
            ))
        partial_dir.mkdir(parents=True, exist_ok=True)
        stop = start + len(chunk_rows)
        _atomic_torch(partial_dir / f"chunk_{start:06d}_{stop:06d}.pt", {
            "schema": TEACHER_CACHE_SHARD_SCHEMA,
            "cache_identity": identity.identity,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "start": start,
            "rows": chunk_rows,
        })
        payload_rows.extend(chunk_rows)

    final_payload = {
        "schema": TEACHER_CACHE_SHARD_SCHEMA,
        "cache_identity": identity.identity,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "rows": payload_rows,
    }
    _atomic_torch(complete_payload, final_payload)
    payload_sha = sha256_file(complete_payload)
    row_ids = [row["row_identity"] for row in payload_rows]
    row_ids_sha = hashlib.sha256("\n".join(row_ids).encode()).hexdigest()
    parity_values: dict[str, list[float]] = {}
    for row in payload_rows:
        evidence = row.get("parity_evidence", {})
        if not isinstance(evidence, dict):
            raise ValueError("cache row parity evidence is invalid")
        for name, value in evidence.items():
            parity_values.setdefault(str(name), []).append(float(value))
    parity_summary = {
        name: sum(values) / len(values)
        for name, values in sorted(parity_values.items())
        if values
    }
    summary = SFT1V2CacheShardSummary(
        shard_index=shard_index, shard_count=shard_count,
        row_count=len(payload_rows),
        first_ordinal=payload_rows[0]["ordinal"] if payload_rows else None,
        last_ordinal=payload_rows[-1]["ordinal"] if payload_rows else None,
        payload_sha256=payload_sha, row_identities_sha256=row_ids_sha,
        parity_summary=parity_summary,
    )
    _atomic_json(complete_metadata, {
        "schema": TEACHER_CACHE_SHARD_SCHEMA,
        "cache_identity": identity.identity,
        **asdict(summary),
    })
    if partial_dir.exists():
        for path in partial_dir.glob("chunk_*.pt"):
            path.unlink()
        partial_dir.rmdir()
    return summary


def finalize_teacher_cache(
    output_dir: Path,
    *,
    identity: SFT1V2TeacherCacheIdentity,
    shard_count: int,
    expected_row_count: int,
) -> SFT1V2CacheSummary:
    """Verify every shard and publish root manifest then completion marker last."""

    output = Path(output_dir)
    if (output / COMPLETE_MARKER).exists():
        raise FileExistsError("completed teacher cache is immutable")
    if expected_row_count < 1:
        raise ValueError("teacher cache expected row count must be positive")
    summaries: list[dict[str, Any]] = []
    for index in range(shard_count):
        stem = _shard_stem(index, shard_count)
        payload = output / "shards" / f"{stem}.pt"
        metadata_path = output / "shards" / f"{stem}.json"
        if not payload.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"teacher cache shard {index} is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_identity") != identity.identity or metadata.get("shard_index") != index:
            raise ValueError("teacher cache shard metadata identity mismatch")
        if sha256_file(payload) != metadata.get("payload_sha256"):
            raise ValueError("teacher cache shard payload hash mismatch")
        summaries.append(metadata)
    if sum(int(item["row_count"]) for item in summaries) != expected_row_count:
        raise ValueError("teacher cache total row count mismatch")
    root = {
        "schema": TEACHER_CACHE_SCHEMA,
        "row_schema": TEACHER_CACHE_ROW_SCHEMA,
        "cache_identity": identity.identity,
        "identity": asdict(identity),
        "shard_count": shard_count,
        "row_count": expected_row_count,
        "shards": summaries,
        "forbidden_cache_fields": [
            "hidden", "query_hidden", "student_hidden", "state",
            "projected_state", "pixel_values", "encoded_tensors",
        ],
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("teacher cache root manifest already exists")
    _atomic_json(manifest_path, root)
    root_sha = sha256_file(manifest_path)
    marker = output / COMPLETE_MARKER
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, f"{root_sha}\n".encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_dir(output)
    return SFT1V2CacheSummary(
        schema=TEACHER_CACHE_SCHEMA, cache_identity=identity.identity,
        shard_count=shard_count, row_count=expected_row_count,
        shard_payload_sha256=tuple(str(item["payload_sha256"]) for item in summaries),
        root_manifest_sha256=root_sha,
    )


def inspect_teacher_cache(output_dir: Path) -> SFT1V2CacheSummary:
    """Inspect hashes/metadata on login CPU without importing teacher models."""

    output = Path(output_dir)
    marker = output / COMPLETE_MARKER
    manifest_path = output / "manifest.json"
    if not marker.is_file() or not manifest_path.is_file():
        raise ValueError("teacher cache is not atomically complete")
    root_sha = sha256_file(manifest_path)
    if marker.read_text(encoding="utf-8").strip() != root_sha:
        raise ValueError("teacher cache completion marker does not bind root manifest")
    root = json.loads(manifest_path.read_text(encoding="utf-8"))
    if root.get("schema") != TEACHER_CACHE_SCHEMA:
        raise ValueError("unsupported teacher cache schema")
    shards = root.get("shards")
    if not isinstance(shards, list) or len(shards) != root.get("shard_count"):
        raise ValueError("teacher cache root shard list is invalid")
    for item in shards:
        stem = _shard_stem(int(item["shard_index"]), int(root["shard_count"]))
        path = output / "shards" / f"{stem}.pt"
        if not path.is_file() or sha256_file(path) != item.get("payload_sha256"):
            raise ValueError("teacher cache inspection found a shard hash mismatch")
    return SFT1V2CacheSummary(
        schema=TEACHER_CACHE_SCHEMA,
        cache_identity=str(root["cache_identity"]),
        shard_count=int(root["shard_count"]), row_count=int(root["row_count"]),
        shard_payload_sha256=tuple(str(item["payload_sha256"]) for item in shards),
        root_manifest_sha256=root_sha,
    )


class SFT1V2TeacherCacheReader:
    """Validate once and retain an ordinal index for repeated training reads."""

    def __init__(self, output_dir: Path, *, manifest_identity: str) -> None:
        self.output_dir = Path(output_dir)
        self.summary = inspect_teacher_cache(self.output_dir)
        self.manifest_identity = _sha(manifest_identity, "manifest_identity")
        rows: dict[int, Mapping[str, Any]] = {}
        for index in range(self.summary.shard_count):
            stem = _shard_stem(index, self.summary.shard_count)
            payload = torch.load(
                self.output_dir / "shards" / f"{stem}.pt",
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            if (
                not isinstance(payload, dict)
                or payload.get("cache_identity") != self.summary.cache_identity
                or not isinstance(payload.get("rows"), list)
            ):
                raise ValueError("teacher cache shard payload identity is invalid")
            for row in payload["rows"]:
                ordinal = int(row["ordinal"])
                if ordinal in rows:
                    raise ValueError("teacher cache contains a duplicated ordinal")
                rows[ordinal] = row
        if len(rows) != self.summary.row_count:
            raise ValueError("teacher cache ordinal index count mismatch")
        self._rows = rows

    def load(self, ordinal: int) -> SFT1V2TeacherRow:
        try:
            row = self._rows[int(ordinal)]
        except KeyError as error:
            raise ValueError("teacher cache row ordinal is missing") from error
        forbidden = {
            "hidden", "query_hidden", "student_hidden", "state",
            "projected_state", "encoded_tensors", "pixel_values",
        }
        if forbidden & set(row):
            raise ValueError("teacher cache row contains forbidden student state")
        targets = _validate_fresh_targets(SFT1V2FreshTargets(
            dino_regions=row["dino_regions"],
            instruction_teacher=row["instruction_teacher"],
            actor_teacher_log_probs=row["actor_teacher_log_probs"],
        ))
        return SFT1V2TeacherRow(
            manifest_identity=self.manifest_identity,
            record_id=str(row["record_id"]),
            step_index=int(row["step_index"]),
            original_image_sha256=str(row["original_image_sha256"]),
            image_content_group=str(row["image_content_group"]),
            instruction_equivalence_group=str(row["instruction_equivalence_group"]),
            dino_regions=targets.dino_regions,
            instruction_teacher=targets.instruction_teacher,
            actor_teacher_log_probs=targets.actor_teacher_log_probs,
        )


def load_teacher_cache_row(
    output_dir: Path,
    *,
    ordinal: int,
    shard_count: int,
    manifest_identity: str,
) -> SFT1V2TeacherRow:
    """Compatibility convenience for one-off inspection, not training loops."""

    reader = SFT1V2TeacherCacheReader(
        output_dir, manifest_identity=manifest_identity
    )
    if reader.summary.shard_count != shard_count:
        raise ValueError("teacher cache shard count mismatch")
    return reader.load(ordinal)


__all__ = [
    "COMPLETE_MARKER", "SFT1V2CacheShardSummary", "SFT1V2CacheSummary",
    "SFT1V2FreshTargets", "SFT1V2FreshTeacher", "SFT1V2ParityReference",
    "SFT1V2TeacherCacheIdentity", "SFT1V2TeacherCacheReader",
    "TEACHER_CACHE_ROW_SCHEMA",
    "TEACHER_CACHE_SCHEMA", "deterministic_shard_ownership",
    "finalize_teacher_cache", "inspect_teacher_cache", "load_teacher_cache_row",
    "prepare_teacher_cache_shard",
]
