"""Manifest-last cache owner for the unsafe authoritative update6420 producer.

The schema is intentionally incompatible with both deployable Query-State caches and
Formal38 forensic caches.  A reader re-authenticates the live checkpoint, baseline
manifest, every state/row tensor payload, and the locked ordered row identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from nimloth.training.reconstruction.forensic_query_state_cache import (
    ForensicQueryStateCacheDataset,
)
from nimloth.training.reconstruction.update6420_forensic_comparison import (
    LOCKED_SELECTION_DIGESTS,
    LOCKED_UPDATE6420_EXPECTED,
    UPDATE6420_CACHE_SCHEMA,
    canonical_identity,
    validate_checkpoint_evidence,
    validate_matched_rows,
)

UPDATE6420_CACHE_SHARD_SCHEMA = "nimloth_update6420_unsafe_query_state_cache_shard_v1"
BASELINE_CACHE_MANIFEST_SHA256 = "76b36e10edd666136ccaa115f14b2fa36156a0bcd56e94b9d1425f6f4dc1d083"
BASELINE_CACHE_FINGERPRINT = "9bd942267140aede839087b09bb0f755bd023ec69d216a8ad77ee845fc120899"
BASELINE_CACHE_PATH = Path("/project/peilab/atst/nimloth/outputs/experiments/evaluation/reconstruction/2026-09-02/197_formal38_unsafe1605_qstate_stageb_full_cachefeat_128px_normal/cache")
MAX_SHARD_RECORDS = 2_048
_STATE_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a mapping")
    return value


def derive_matched_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Add comparison identities absent from the native Formal38 cache schema."""

    required = {
        "selection_ordinal", "selection_role", "row_identity", "record_id",
        "step_index", "original_image_path", "original_image_sha256",
        "archived_assistant_response_sha256", "response_source",
        "encoded_input_identity", "messages_identity", "prompt_history_identity",
        "renderer_identity", "template_identity",
    }
    if not isinstance(row, Mapping) or set(row) != required:
        raise ValueError("baseline row schema is not the native Formal38 Stage B schema")
    result = dict(row)
    result["observation_identity"] = canonical_identity({
        key: result[key]
        for key in ("record_id", "step_index", "original_image_path", "original_image_sha256")
    })
    result["archived_response_identity"] = result["archived_assistant_response_sha256"]
    return result


def load_locked_baseline_rows(
    root: str | Path = BASELINE_CACHE_PATH,
    *,
    expected_manifest_sha256: str = BASELINE_CACHE_MANIFEST_SHA256,
    expected_fingerprint: str = BASELINE_CACHE_FINGERPRINT,
    expected_digests: Mapping[str, str] = LOCKED_SELECTION_DIGESTS,
) -> tuple[dict[str, Any], ...]:
    """Load native baseline rows and derive observation/response identities."""

    supplied = Path(root)
    manifest_path = supplied / "manifest.json"
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Formal38 baseline cache manifest hash mismatch")
    dataset = ForensicQueryStateCacheDataset(supplied)
    if dataset.cache_fingerprint != expected_fingerprint:
        raise ValueError("Formal38 baseline cache fingerprint mismatch")
    rows = tuple(derive_matched_row({key: value for key, value in dataset[index].items() if key != "state"}) for index in range(len(dataset)))
    validate_matched_rows(rows, baseline_rows=rows, expected_digests=expected_digests)
    return rows


def _validate_state(state: torch.Tensor, *, count: int) -> torch.Tensor:
    if (
        not isinstance(state, torch.Tensor)
        or state.shape != (count, *_STATE_SHAPE)
        or state.dtype != torch.float32
        or not state.is_contiguous()
        or not torch.isfinite(state).all()
    ):
        raise ValueError("update6420 cache state must be contiguous finite float32 [N,16,1024]")
    return state.detach().cpu().contiguous()


def _write_rank_payload(path: Path, *, rank: int, state: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"update6420 rank cache payload exists: {path}")
    normalized = [dict(row) for row in rows]
    canonical = _validate_state(state, count=len(normalized))
    ordinals = [row.get("selection_ordinal") for row in normalized]
    payload = {"schema": UPDATE6420_CACHE_SHARD_SCHEMA, "state": canonical, "rows": normalized}
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "rank": rank,
        "file": path.name,
        "count": len(normalized),
        "file_sha256": _sha256_file(path),
        "state_sha256": _tensor_sha256(canonical),
        "rows_sha256": canonical_identity(normalized),
        "ordinals_sha256": canonical_identity(ordinals),
    }


def _load_shard(path: Path, descriptor: Mapping[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if path.is_symlink() or not path.is_file() or _sha256_file(path) != descriptor.get("file_sha256"):
        raise ValueError("update6420 cache shard file hash mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("update6420 cache shard is unreadable") from error
    if not isinstance(payload, dict) or set(payload) != {"schema", "state", "rows"} or payload.get("schema") != UPDATE6420_CACHE_SHARD_SCHEMA:
        raise ValueError("update6420 cache shard schema mismatch")
    rows = payload["rows"]
    if not isinstance(rows, list) or descriptor.get("count") != len(rows):
        raise ValueError("update6420 cache shard row count mismatch")
    state = _validate_state(payload["state"], count=len(rows))
    if _tensor_sha256(state) != descriptor.get("state_sha256") or canonical_identity(rows) != descriptor.get("rows_sha256"):
        raise ValueError("update6420 cache shard tensor/row content hash mismatch")
    if canonical_identity([row.get("selection_ordinal") for row in rows]) != descriptor.get("ordinals_sha256"):
        raise ValueError("update6420 cache shard ordinal hash mismatch")
    return state, [dict(row) for row in rows]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_update6420_cache_from_rank_payloads(
    *,
    staging: Path,
    output: Path,
    rank_descriptors: Sequence[Mapping[str, Any]],
    checkpoint_evidence: Mapping[str, Any],
    baseline_rows: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    expected_checkpoint: Mapping[str, Any] | None = None,
    expected_digests: Mapping[str, str] = LOCKED_SELECTION_DIGESTS,
    _expected_counts: Mapping[str, int] = {"all_train": 12_836, "external_validation": 1_413},
) -> Mapping[str, Any]:
    """Merge rank payloads and atomically commit bounded shards, manifest last."""

    validate_checkpoint_evidence(checkpoint_evidence, expected=expected_checkpoint or LOCKED_UPDATE6420_EXPECTED)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"update6420 cache output exists: {output}")
    if len(rank_descriptors) != 8 or {item.get("rank") for item in rank_descriptors} != set(range(8)):
        raise ValueError("update6420 cache requires exactly eight rank payloads")
    merged: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for descriptor in sorted(rank_descriptors, key=lambda item: int(item["rank"])):
        state, rows = _load_shard(staging / str(descriptor["file"]), descriptor)
        merged.extend((state[index], row) for index, row in enumerate(rows))
    merged.sort(key=lambda item: int(item[1]["selection_ordinal"]))
    rows = [item[1] for item in merged]
    digests = validate_matched_rows(
        rows, baseline_rows=baseline_rows, expected_counts=_expected_counts,
        expected_digests=expected_digests,
    )
    if Counter(row["selection_role"] for row in rows) != Counter(_expected_counts):
        raise ValueError("update6420 cache role boundary mismatch")
    publication = staging.with_name(staging.name + ".publish")
    if publication.exists() or publication.is_symlink():
        raise FileExistsError("update6420 publication staging exists")
    publication.mkdir()
    try:
        shards: list[dict[str, Any]] = []
        for shard_index, start in enumerate(range(0, len(merged), MAX_SHARD_RECORDS)):
            stop = min(start + MAX_SHARD_RECORDS, len(merged))
            chunk = merged[start:stop]
            path = publication / f"shard_{shard_index:05d}.pt"
            descriptor = _write_rank_payload(
                path,
                rank=shard_index,
                state=torch.stack([item[0] for item in chunk]).float().contiguous(),
                rows=[item[1] for item in chunk],
            )
            descriptor.update({"start": start, "stop": stop})
            shards.append(descriptor)
        checkpoint = {
            "checkpoint_identity": checkpoint_evidence["control_sha256"],
            "run_identity": checkpoint_evidence["run_identity"],
            "update": 6420,
            "forensic_only": False,
            "actor_unsafe": True,
            "deployable": False,
            "evidence": json.loads(json.dumps(dict(checkpoint_evidence))),
        }
        manifest: dict[str, Any] = {
            "schema": UPDATE6420_CACHE_SCHEMA,
            "version": 1,
            "owner_role": "unsafe_authoritative_update6420_query_state",
            "actor_unsafe": True,
            "deployable": False,
            "sft2_ready": False,
            "forensic_only": False,
            "count": len(rows),
            "state_shape": list(_STATE_SHAPE),
            "state_dtype": "float32",
            "roles": dict(_expected_counts),
            "ordered_identity_digests": digests,
            "row_set_sha256": canonical_identity(rows),
            "checkpoint": checkpoint,
            "baseline": {
                "manifest_path": str(BASELINE_CACHE_PATH / "manifest.json"),
                "manifest_sha256": BASELINE_CACHE_MANIFEST_SHA256,
                "cache_fingerprint": BASELINE_CACHE_FINGERPRINT,
            },
            "producer": dict(producer),
            "rank_payloads": [dict(item) for item in sorted(rank_descriptors, key=lambda item: int(item["rank"]))],
            "shards": shards,
        }
        manifest["cache_fingerprint"] = canonical_identity(manifest)
        manifest_path = publication / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(publication)
        output.mkdir(parents=True, exist_ok=False)
        _fsync_directory(output.parent)
        for path in sorted(publication.iterdir(), key=lambda item: item.name == "manifest.json"):
            os.rename(path, output / path.name)
        _fsync_directory(output)
        publication.rmdir()
        return manifest
    except BaseException:
        if publication.exists():
            shutil.rmtree(publication)
        raise


class Update6420QueryStateCacheDataset:
    """Strict consumer that rejects Formal38, deployable, and rehashed drift."""

    def __init__(
        self,
        root: str | Path,
        *,
        _expected_checkpoint: Mapping[str, Any] | None = None,
        _baseline_rows: Sequence[Mapping[str, Any]] | None = None,
        _expected_digests: Mapping[str, str] = LOCKED_SELECTION_DIGESTS,
        _expected_counts: Mapping[str, int] = {"all_train": 12_836, "external_validation": 1_413},
    ) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("update6420 cache root must not be a symlink")
        self.root = supplied.resolve()
        raw = _read_json(self.root / "manifest.json", label="update6420 cache manifest")
        required = {
            "schema", "version", "owner_role", "actor_unsafe", "deployable",
            "sft2_ready", "forensic_only", "count", "state_shape", "state_dtype",
            "roles", "ordered_identity_digests", "row_set_sha256", "checkpoint",
            "baseline", "producer", "rank_payloads", "shards", "cache_fingerprint",
        }
        if (
            set(raw) != required
            or raw.get("schema") != UPDATE6420_CACHE_SCHEMA
            or raw.get("version") != 1
            or raw.get("owner_role") != "unsafe_authoritative_update6420_query_state"
            or raw.get("actor_unsafe") is not True
            or any(raw.get(field) is not False for field in ("deployable", "sft2_ready"))
            or raw.get("forensic_only") is not False
            or raw.get("count") != sum(_expected_counts.values())
            or raw.get("state_shape") != [16, 1024]
            or raw.get("state_dtype") != "float32"
            or raw.get("roles") != dict(_expected_counts)
            or raw.get("ordered_identity_digests") != dict(_expected_digests)
            or not _is_sha256(raw.get("row_set_sha256"))
            or not _is_sha256(raw.get("cache_fingerprint"))
            or canonical_identity({key: value for key, value in raw.items() if key != "cache_fingerprint"}) != raw.get("cache_fingerprint")
        ):
            raise ValueError("update6420 cache manifest schema/classification/fingerprint mismatch")
        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, Mapping) or set(checkpoint) != {"checkpoint_identity", "run_identity", "update", "forensic_only", "actor_unsafe", "deployable", "evidence"}:
            raise ValueError("update6420 cache checkpoint binding is malformed")
        if checkpoint.get("checkpoint_identity") != (_expected_checkpoint or LOCKED_UPDATE6420_EXPECTED)["control_sha256"] or checkpoint.get("run_identity") != (_expected_checkpoint or LOCKED_UPDATE6420_EXPECTED)["run_identity"] or checkpoint.get("update") != 6420 or checkpoint.get("forensic_only") is not False or checkpoint.get("actor_unsafe") is not True or checkpoint.get("deployable") is not False:
            raise ValueError("update6420 cache checkpoint/consumer classification drift")
        evidence = checkpoint.get("evidence")
        if not isinstance(evidence, Mapping):
            raise TypeError("update6420 cache checkpoint evidence is absent")
        validate_checkpoint_evidence(evidence, expected=_expected_checkpoint or LOCKED_UPDATE6420_EXPECTED)
        baseline = raw.get("baseline")
        if not isinstance(baseline, Mapping) or baseline != {
            "manifest_path": str(BASELINE_CACHE_PATH / "manifest.json"),
            "manifest_sha256": BASELINE_CACHE_MANIFEST_SHA256,
            "cache_fingerprint": BASELINE_CACHE_FINGERPRINT,
        }:
            raise ValueError("update6420 cache baseline owner binding drift")
        baseline_rows = tuple(_baseline_rows) if _baseline_rows is not None else load_locked_baseline_rows(expected_digests=_expected_digests)
        rank_payloads = raw.get("rank_payloads")
        if (
            not isinstance(rank_payloads, list)
            or len(rank_payloads) != 8
            or {item.get("rank") for item in rank_payloads if isinstance(item, Mapping)} != set(range(8))
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"rank", "file", "count", "file_sha256", "state_sha256", "rows_sha256", "ordinals_sha256"}
                or item.get("file") != f"rank_{int(item.get('rank', -1)):05d}_of_00008.pt"
                or any(not _is_sha256(item.get(field)) for field in ("file_sha256", "state_sha256", "rows_sha256", "ordinals_sha256"))
                or isinstance(item.get("count"), bool)
                or not isinstance(item.get("count"), int)
                or item["count"] < 0
                for item in rank_payloads
            )
            or sum(item["count"] for item in rank_payloads) != sum(_expected_counts.values())
        ):
            raise ValueError("update6420 rank-payload fingerprint binding is invalid")
        descriptors = raw.get("shards")
        if not isinstance(descriptors, list) or not descriptors:
            raise ValueError("update6420 cache shard manifest is absent")
        states: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        expected_start = 0
        for index, descriptor in enumerate(descriptors):
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor) != {"rank", "file", "count", "file_sha256", "state_sha256", "rows_sha256", "ordinals_sha256", "start", "stop"}
                or descriptor.get("rank") != index
                or descriptor.get("file") != f"shard_{index:05d}.pt"
                or descriptor.get("start") != expected_start
                or descriptor.get("stop") != expected_start + descriptor.get("count", -1)
                or isinstance(descriptor.get("count"), bool)
                or not isinstance(descriptor.get("count"), int)
                or not 0 < descriptor["count"] <= MAX_SHARD_RECORDS
                or any(not _is_sha256(descriptor.get(field)) for field in ("file_sha256", "state_sha256", "rows_sha256", "ordinals_sha256"))
            ):
                raise ValueError("update6420 cache shard ranges/hashes are invalid")
            state, shard_rows = _load_shard(self.root / str(descriptor["file"]), descriptor)
            states.append(state)
            rows.extend(shard_rows)
            expected_start = int(descriptor["stop"])
        if expected_start != sum(_expected_counts.values()):
            raise ValueError("update6420 cache shard coverage is incomplete")
        validate_matched_rows(
            rows, baseline_rows=baseline_rows, expected_counts=_expected_counts,
            expected_digests=_expected_digests,
        )
        if canonical_identity(rows) != raw["row_set_sha256"]:
            raise ValueError("update6420 cache row-set hash mismatch")
        self.manifest = raw
        self._state = torch.cat(states).contiguous()
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def cache_fingerprint(self) -> str:
        return str(self.manifest["cache_fingerprint"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("update6420 cache index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {"state": self._state[index].clone(), **dict(self._rows[index])}


__all__ = [
    "BASELINE_CACHE_FINGERPRINT", "BASELINE_CACHE_MANIFEST_SHA256", "BASELINE_CACHE_PATH",
    "MAX_SHARD_RECORDS", "UPDATE6420_CACHE_SHARD_SCHEMA", "Update6420QueryStateCacheDataset",
    "derive_matched_row", "load_locked_baseline_rows", "publish_update6420_cache_from_rank_payloads",
]
