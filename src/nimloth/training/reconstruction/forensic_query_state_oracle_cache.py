"""Exact frozen-DINO oracle cache for the Formal38 reconstruction ladder.

The cache is a separate forensic condition owner.  It never rewrites the
Query-State cache and never treats decoder-resized RGB as the SFT1 teacher
input.  Production extraction loads only original archived observations through
the pinned SFT1 DINO path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
    _processor_fingerprint,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_ALL_TRAIN,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION,
    FORENSIC_STAGE_B_EXTERNAL_COUNT,
    FORENSIC_STAGE_B_SELECTION_ALGORITHM,
    FORENSIC_STAGE_B_TRAIN_COUNT,
    ForensicQueryStateCacheDataset,
    _publish_noreplace,
)

FORENSIC_DINO_ORACLE_CACHE_SCHEMA = "nimloth_query_state_dino_oracle_cache_v1"
FORENSIC_DINO_ORACLE_SHARD_SCHEMA = (
    "nimloth_query_state_dino_oracle_cache_shard_v1"
)
FORENSIC_DINO_ORACLE_OWNER_ROLE = "unsafe_forensic_dino_oracle_condition"
_CONDITION_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")
_ROW_FIELDS = {
    "selection_ordinal",
    "selection_role",
    "row_identity",
    "record_id",
    "step_index",
    "original_image_path",
    "original_image_sha256",
    "archived_assistant_response_sha256",
    "prompt_history_identity",
    "messages_identity",
    "renderer_identity",
    "template_identity",
    "encoded_input_identity",
    "response_source",
}


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_DINO_PAYLOAD = {
    "source": DINOV2_LARGE_IDENTITY.source,
    "revision": DINOV2_LARGE_IDENTITY.revision,
    "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
    "hidden_size": DINOV2_LARGE_IDENTITY.hidden_size,
    "grid_size": 4,
    "feature_identity": _identity(
        {
            "source": DINOV2_LARGE_IDENTITY.source,
            "revision": DINOV2_LARGE_IDENTITY.revision,
            "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
            "hidden_size": DINOV2_LARGE_IDENTITY.hidden_size,
            "grid_size": 4,
        }
    ),
    "input_owner": "original_archived_observation",
    "resize_before_processor": False,
    "pooling": "final_patch_tokens_adaptive_avg_pool2d_4x4_row_major",
}
EXACT_DINO_FEATURE_IDENTITY = str(_DINO_PAYLOAD["feature_identity"])


def _read_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {owner}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"invalid {owner} mapping: {path}")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_teacher(teacher: FrozenDINOGridTargets) -> Mapping[str, Any]:
    model = getattr(teacher, "model", None)
    processor = getattr(teacher, "image_processor", None)
    if (
        not isinstance(teacher, FrozenDINOGridTargets)
        or getattr(teacher, "identity", None) != DINOV2_LARGE_IDENTITY
        or getattr(teacher, "grid_size", None) != 4
        or not isinstance(model, torch.nn.Module)
        or processor is None
        or _processor_fingerprint(processor)
        != DINOV2_LARGE_IDENTITY.processor_fingerprint
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError(
            "oracle cache requires concrete pinned frozen/eval DINOv2-large 4x4 teacher"
        )
    model_parameters = tuple(model.parameters())
    if not model_parameters:
        raise ValueError("oracle cache DINO teacher must expose model dtype evidence")
    dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in model_parameters}
    if len(dtypes) != 1:
        raise ValueError("oracle cache DINO teacher model dtype must be uniform")
    return {
        **_DINO_PAYLOAD,
        "model_dtype": next(iter(dtypes)),
        "output_dtype": "float32",
        "batch_size": teacher.batch_size,
    }


def _current_source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[4],
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("oracle cache cannot authenticate implementation commit") from error
    commit = result.stdout.strip()
    if len(commit) != 40 or set(commit) - _HEX:
        raise RuntimeError("oracle cache implementation commit is invalid")
    return commit


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(
        candidate.is_symlink()
        for candidate in (absolute, *absolute.parents)
    )


def _reject_protected_output(output: Path, protected: Sequence[Path]) -> None:
    if not output.is_absolute() or _has_symlink_component(output):
        raise ValueError("oracle cache output must be an absolute non-symlink path")
    candidate = output.parent.resolve() / output.name
    for raw_root in protected:
        if not raw_root.is_absolute() or _has_symlink_component(raw_root):
            raise ValueError("oracle cache protected input must be absolute and non-symlinked")
        root = raw_root.resolve()
        if candidate == root or root in candidate.parents:
            raise ValueError("oracle cache output must not be inside an immutable input root")


def _validate_condition_batch(
    value: object,
    *,
    expected_count: int,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or tuple(value.shape) != (expected_count, *_CONDITION_SHAPE)
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            "DINO oracle condition batch must be contiguous finite float32 "
            "[B,16,1024] without gradients"
        )
    return value.detach().cpu().contiguous()


def _row_without_state(item: Mapping[str, Any], *, expected_ordinal: int) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != {*_ROW_FIELDS, "state"}:
        raise ValueError("oracle source state-cache row schema is invalid")
    row = {key: item[key] for key in _ROW_FIELDS}
    path = Path(str(row.get("original_image_path", "")))
    if (
        row.get("selection_ordinal") != expected_ordinal
        or row.get("selection_role")
        not in {FORENSIC_SELECTION_ALL_TRAIN, FORENSIC_SELECTION_EXTERNAL_VALIDATION}
        or not isinstance(row.get("row_identity"), str)
        or not row["row_identity"]
        or not isinstance(row.get("record_id"), str)
        or not row["record_id"]
        or isinstance(row.get("step_index"), bool)
        or not isinstance(row.get("step_index"), int)
        or row["step_index"] < 0
        or not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or row.get("response_source") != "archived"
        or any(
            not _is_sha256(row.get(field))
            for field in (
                "original_image_sha256",
                "archived_assistant_response_sha256",
                "prompt_history_identity",
                "messages_identity",
                "renderer_identity",
                "template_identity",
                "encoded_input_identity",
            )
        )
        or _sha256_file(path) != row["original_image_sha256"]
    ):
        raise ValueError(
            "oracle source rows must preserve ordered original-image/archived-CoT identity"
        )
    return row


def _validate_source_state_cache(
    dataset: ForensicQueryStateCacheDataset,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    manifest = dataset.manifest
    selection = manifest.get("selection")
    expected_roles = {
        FORENSIC_SELECTION_ALL_TRAIN: FORENSIC_STAGE_B_TRAIN_COUNT,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: FORENSIC_STAGE_B_EXTERNAL_COUNT,
    }
    expected_count = sum(expected_roles.values())
    if (
        manifest.get("schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA
        or manifest.get("owner_role") != FORENSIC_QUERY_STATE_OWNER_ROLE
        or manifest.get("forensic_only") is not True
        or any(
            manifest.get(field) is not False
            for field in ("authoritative", "terminal_primary", "deployable", "sft2_ready")
        )
        or manifest.get("count") != expected_count
        or tuple(manifest.get("state_shape", ())) != _CONDITION_SHAPE
        or manifest.get("state_dtype") != "float32"
        or not _is_sha256(manifest.get("cache_fingerprint"))
        or not _is_sha256(manifest.get("row_set_identity"))
        or not isinstance(selection, Mapping)
        or selection.get("stage") != "stage_b_diagnostic"
        or selection.get("algorithm") != FORENSIC_STAGE_B_SELECTION_ALGORITHM
        or selection.get("seed") is not None
        or selection.get("roles") != expected_roles
        or not _is_sha256(selection.get("identity"))
        or len(dataset) != expected_count
    ):
        raise ValueError(
            "oracle cache requires the exact complete Formal38 Stage B state cache"
        )

    rows: list[dict[str, Any]] = []
    roles = {role: 0 for role in expected_roles}
    seen_rows: set[str] = set()
    train_images: set[str] = set()
    external_images: set[str] = set()
    for ordinal in range(len(dataset)):
        item = dataset[ordinal]
        state = item.get("state")
        if (
            not isinstance(state, torch.Tensor)
            or tuple(state.shape) != _CONDITION_SHAPE
            or state.dtype != torch.float32
            or not bool(torch.isfinite(state).all())
        ):
            raise ValueError("oracle source state cache contains an invalid canonical state")
        row = _row_without_state(item, expected_ordinal=ordinal)
        identity = str(row["row_identity"])
        if identity in seen_rows:
            raise ValueError("oracle source state-cache row identity must be unique")
        seen_rows.add(identity)
        role = str(row["selection_role"])
        roles[role] += 1
        if role == FORENSIC_SELECTION_ALL_TRAIN:
            train_images.add(str(row["original_image_sha256"]))
        else:
            external_images.add(str(row["original_image_sha256"]))
        rows.append(row)
    if roles != expected_roles or train_images & external_images:
        raise ValueError("oracle Stage B role counts/images are not exact and disjoint")
    if _identity({"rows": rows}) != manifest["row_set_identity"]:
        raise ValueError("oracle source state-cache ordered row-set identity mismatch")
    return manifest, rows


def _source_state_cache_payload(
    root: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = root / "manifest.json"
    return {
        "path": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "schema": manifest["schema"],
        "owner_role": manifest["owner_role"],
        "cache_fingerprint": manifest["cache_fingerprint"],
        "row_set_identity": manifest["row_set_identity"],
        "count": manifest["count"],
        "condition_shape": manifest["state_shape"],
        "condition_dtype": manifest["state_dtype"],
        "checkpoint": manifest["checkpoint"],
        "source_jsonl": manifest["source_jsonl"],
        "selection": manifest["selection"],
    }


def _write_shard(
    path: Path,
    *,
    index: int,
    start: int,
    condition: torch.Tensor,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    validated = _validate_condition_batch(condition, expected_count=len(rows))
    if not rows:
        raise ValueError("DINO oracle shard cannot be empty")
    payload = {
        "schema": FORENSIC_DINO_ORACLE_SHARD_SCHEMA,
        "condition": validated,
        "rows": [dict(row) for row in rows],
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    stop = start + len(rows)
    return {
        "file": path.name,
        "count": len(rows),
        "start": start,
        "stop": stop,
        "sha256": _sha256_file(path),
        "condition_dtype": "float32",
        "condition_shape": list(_CONDITION_SHAPE),
    }


def _parse_shard(payload: object) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "condition", "rows"}
        or payload.get("schema") != FORENSIC_DINO_ORACLE_SHARD_SCHEMA
        or not isinstance(payload.get("rows"), list)
    ):
        raise ValueError("unsupported DINO oracle cache shard schema")
    rows = [dict(row) for row in payload["rows"] if isinstance(row, Mapping)]
    if len(rows) != len(payload["rows"]):
        raise ValueError("DINO oracle cache shard row schema is invalid")
    condition = _validate_condition_batch(
        payload.get("condition"), expected_count=len(rows)
    )
    return condition, rows


def _parse_manifest(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {
        "schema",
        "version",
        "owner_role",
        "forensic_only",
        "unsafe_actor_checkpoint",
        "authoritative",
        "terminal_primary",
        "deployable",
        "sft2_ready",
        "count",
        "condition_shape",
        "condition_dtype",
        "source_state_cache",
        "dino",
        "selection",
        "row_set_identity",
        "original_image_dimensions",
        "producer",
        "max_shard_records",
        "shards",
        "cache_fingerprint",
    }
    source = raw.get("source_state_cache")
    selection = raw.get("selection")
    shards = raw.get("shards")
    dino = raw.get("dino")
    dimensions = raw.get("original_image_dimensions")
    producer = raw.get("producer")
    expected_roles = {
        FORENSIC_SELECTION_ALL_TRAIN: FORENSIC_STAGE_B_TRAIN_COUNT,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: FORENSIC_STAGE_B_EXTERNAL_COUNT,
    }
    expected_count = sum(expected_roles.values())
    max_records = raw.get("max_shard_records")
    source_required = {
        "path",
        "manifest_path",
        "manifest_sha256",
        "schema",
        "owner_role",
        "cache_fingerprint",
        "row_set_identity",
        "count",
        "condition_shape",
        "condition_dtype",
        "checkpoint",
        "source_jsonl",
        "selection",
    }
    valid_max_records = (
        not isinstance(max_records, bool)
        and isinstance(max_records, int)
        and max_records >= 1
    )
    valid_shards = isinstance(shards, list) and bool(shards) and valid_max_records
    expected_start = 0
    if valid_shards:
        for index, descriptor in enumerate(shards):
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor)
                != {
                    "file",
                    "count",
                    "start",
                    "stop",
                    "sha256",
                    "condition_dtype",
                    "condition_shape",
                }
                or descriptor.get("file") != f"shard_{index:05d}.pt"
                or isinstance(descriptor.get("start"), bool)
                or not isinstance(descriptor.get("start"), int)
                or descriptor["start"] != expected_start
                or isinstance(descriptor.get("stop"), bool)
                or not isinstance(descriptor.get("stop"), int)
                or descriptor["stop"] <= expected_start
                or isinstance(descriptor.get("count"), bool)
                or not isinstance(descriptor.get("count"), int)
                or descriptor["count"] != descriptor["stop"] - expected_start
                or descriptor["count"] > int(max_records)
                or not _is_sha256(descriptor.get("sha256"))
                or descriptor.get("condition_dtype") != "float32"
                or tuple(descriptor.get("condition_shape", ())) != _CONDITION_SHAPE
            ):
                valid_shards = False
                break
            expected_start = descriptor["stop"]
        valid_shards = valid_shards and expected_start == expected_count
    if (
        set(raw) != required
        or raw.get("schema") != FORENSIC_DINO_ORACLE_CACHE_SCHEMA
        or raw.get("version") != 1
        or raw.get("owner_role") != FORENSIC_DINO_ORACLE_OWNER_ROLE
        or raw.get("forensic_only") is not True
        or raw.get("unsafe_actor_checkpoint") is not True
        or any(
            raw.get(field) is not False
            for field in ("authoritative", "terminal_primary", "deployable", "sft2_ready")
        )
        or raw.get("count") != expected_count
        or tuple(raw.get("condition_shape", ())) != _CONDITION_SHAPE
        or raw.get("condition_dtype") != "float32"
        or not isinstance(dino, Mapping)
        or {
            key: dino.get(key)
            for key in _DINO_PAYLOAD
        }
        != _DINO_PAYLOAD
        or dino.get("model_dtype") not in {"float32", "float16", "bfloat16"}
        or dino.get("output_dtype") != "float32"
        or isinstance(dino.get("batch_size"), bool)
        or not isinstance(dino.get("batch_size"), int)
        or dino["batch_size"] < 1
        or set(dino) != {*_DINO_PAYLOAD, "model_dtype", "output_dtype", "batch_size"}
        or not isinstance(dimensions, list)
        or len(dimensions) != expected_count
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"row_identity", "width", "height"}
            or not isinstance(item.get("row_identity"), str)
            or isinstance(item.get("width"), bool)
            or not isinstance(item.get("width"), int)
            or item["width"] < 1
            or isinstance(item.get("height"), bool)
            or not isinstance(item.get("height"), int)
            or item["height"] < 1
            for item in dimensions
        )
        or not isinstance(producer, Mapping)
        or set(producer) != {"module", "source_commit", "resolved_config", "identity"}
        or producer.get("module")
        != "nimloth.training.reconstruction.forensic_query_state_oracle_cache"
        or not isinstance(producer.get("source_commit"), str)
        or len(producer["source_commit"]) != 40
        or set(producer["source_commit"]) - _HEX
        or not isinstance(producer.get("resolved_config"), Mapping)
        or set(producer["resolved_config"])
        != {
            "state_cache",
            "output",
            "device",
            "dtype",
            "batch_size",
            "max_shard_records",
        }
        or not isinstance(producer["resolved_config"].get("state_cache"), str)
        or not isinstance(producer["resolved_config"].get("output"), str)
        or not isinstance(producer["resolved_config"].get("device"), str)
        or producer["resolved_config"].get("dtype")
        not in {"float32", "float16", "bfloat16"}
        or not _is_sha256(producer.get("identity"))
        or _identity(
            {key: value for key, value in producer.items() if key != "identity"}
        )
        != producer.get("identity")
        or not isinstance(source, Mapping)
        or set(source) != source_required
        or source.get("schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA
        or source.get("owner_role") != FORENSIC_QUERY_STATE_OWNER_ROLE
        or source.get("count") != expected_count
        or tuple(source.get("condition_shape", ())) != _CONDITION_SHAPE
        or source.get("condition_dtype") != "float32"
        or not all(
            _is_sha256(source.get(field))
            for field in ("manifest_sha256", "cache_fingerprint", "row_set_identity")
        )
        or not isinstance(selection, Mapping)
        or selection != source.get("selection")
        or selection.get("stage") != "stage_b_diagnostic"
        or selection.get("algorithm") != FORENSIC_STAGE_B_SELECTION_ALGORITHM
        or selection.get("seed") is not None
        or selection.get("roles") != expected_roles
        or not _is_sha256(selection.get("identity"))
        or not _is_sha256(raw.get("row_set_identity"))
        or raw.get("row_set_identity") != source.get("row_set_identity")
        or isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records < 1
        or not valid_shards
        or not _is_sha256(raw.get("cache_fingerprint"))
        or _identity(
            {key: value for key, value in raw.items() if key != "cache_fingerprint"}
        )
        != raw.get("cache_fingerprint")
    ):
        raise ValueError("DINO oracle cache manifest identity/owner/schema is invalid")
    for field in ("path", "manifest_path"):
        if not isinstance(source.get(field), str) or not Path(source[field]).is_absolute():
            raise ValueError("DINO oracle source state-cache path identity is invalid")
    resolved = producer["resolved_config"]
    if (
        resolved["state_cache"] != source["path"]
        or resolved["dtype"] != dino["model_dtype"]
        or resolved["batch_size"] != dino["batch_size"]
        or resolved["max_shard_records"] != max_records
        or not Path(resolved["output"]).is_absolute()
        or not resolved["device"]
    ):
        raise ValueError("DINO oracle resolved producer configuration is inconsistent")
    return raw


@torch.inference_mode()
def build_forensic_dino_oracle_cache(
    output: str | Path,
    *,
    state_cache: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_shard_records: int = 2_048,
) -> Mapping[str, Any]:
    """Extract exact DINO4×4 conditions and publish one immutable cache."""

    output_path = Path(output)
    source_supplied = Path(state_cache)
    _reject_protected_output(output_path, (source_supplied,))
    temporary = output_path.with_name(f".{output_path.name}.dino-oracle-tmp")
    if (
        output_path.exists()
        or output_path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError("DINO oracle cache output or temporary path already exists")
    if (
        source_supplied.is_symlink()
        or not source_supplied.is_dir()
        or not source_supplied.is_absolute()
    ):
        raise ValueError("DINO oracle source state cache must be an absolute regular directory")
    source_root = source_supplied.resolve()
    source_manifest_path = source_root / "manifest.json"
    if not source_manifest_path.is_file() or source_manifest_path.is_symlink():
        raise ValueError("DINO oracle source state cache requires a regular manifest")
    if (
        not isinstance(device, torch.device)
        or dtype not in {torch.float32, torch.float16, torch.bfloat16}
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(max_shard_records, bool)
        or not isinstance(max_shard_records, int)
        or max_shard_records < 1
    ):
        raise ValueError("DINO oracle device/batch/shard configuration is invalid")

    teacher = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=dtype,
        grid_size=4,
        batch_size=batch_size,
    )
    dino = _validate_teacher(teacher)
    dataset = ForensicQueryStateCacheDataset(source_root)
    source_manifest, rows = _validate_source_state_cache(dataset)
    temporary.mkdir(parents=True)
    descriptors: list[Mapping[str, Any]] = []
    shard_conditions: list[torch.Tensor] = []
    shard_rows: list[Mapping[str, Any]] = []
    image_dimensions: list[Mapping[str, Any]] = []
    shard_start = 0

    def flush() -> None:
        nonlocal shard_start
        if not shard_rows:
            return
        condition = torch.cat(shard_conditions, dim=0).contiguous()
        index = len(descriptors)
        descriptor = _write_shard(
            temporary / f"shard_{index:05d}.pt",
            index=index,
            start=shard_start,
            condition=condition,
            rows=shard_rows,
        )
        descriptors.append(descriptor)
        shard_start += len(shard_rows)
        shard_conditions.clear()
        shard_rows.clear()

    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            paths = [str(row["original_image_path"]) for row in batch_rows]
            for row, path in zip(batch_rows, paths, strict=True):
                with Image.open(path) as image:
                    width, height = image.size
                image_dimensions.append(
                    {
                        "row_identity": row["row_identity"],
                        "width": width,
                        "height": height,
                    }
                )
            condition = _validate_condition_batch(
                teacher.load(paths, device=device),
                expected_count=len(batch_rows),
            )
            cursor = 0
            while cursor < len(batch_rows):
                room = max_shard_records - len(shard_rows)
                take = min(room, len(batch_rows) - cursor)
                shard_conditions.append(condition[cursor : cursor + take])
                shard_rows.extend(batch_rows[cursor : cursor + take])
                cursor += take
                if len(shard_rows) == max_shard_records:
                    flush()
        flush()
        if shard_start != len(rows):
            raise RuntimeError("DINO oracle shard publication lost rows")
        resolved_config = {
            "state_cache": str(source_root),
            "output": str(output_path),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "batch_size": batch_size,
            "max_shard_records": max_shard_records,
        }
        producer: dict[str, Any] = {
            "module": (
                "nimloth.training.reconstruction."
                "forensic_query_state_oracle_cache"
            ),
            "source_commit": _current_source_commit(),
            "resolved_config": resolved_config,
        }
        producer["identity"] = _identity(producer)
        manifest: dict[str, Any] = {
            "schema": FORENSIC_DINO_ORACLE_CACHE_SCHEMA,
            "version": 1,
            "owner_role": FORENSIC_DINO_ORACLE_OWNER_ROLE,
            "forensic_only": True,
            "unsafe_actor_checkpoint": True,
            "authoritative": False,
            "terminal_primary": False,
            "deployable": False,
            "sft2_ready": False,
            "count": len(rows),
            "condition_shape": list(_CONDITION_SHAPE),
            "condition_dtype": "float32",
            "source_state_cache": _source_state_cache_payload(
                source_root, source_manifest
            ),
            "dino": dino,
            "selection": source_manifest["selection"],
            "row_set_identity": _identity({"rows": rows}),
            "original_image_dimensions": image_dimensions,
            "producer": producer,
            "max_shard_records": max_shard_records,
            "shards": descriptors,
        }
        manifest["cache_fingerprint"] = _identity(manifest)
        manifest_path = temporary / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(temporary)
        _publish_noreplace(temporary, output_path)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


class ForensicDinoOracleCacheDataset:
    """Strict reader that revalidates the live Stage B state-cache owner."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("DINO oracle cache root must not be a symlink")
        self.root = supplied.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("DINO oracle cache requires a regular manifest")
        self.manifest = _parse_manifest(
            _read_mapping(manifest_path, owner="DINO oracle cache manifest")
        )
        if self.manifest["producer"]["resolved_config"]["output"] != str(
            self.root
        ):
            raise ValueError("DINO oracle cache producer output/root identity mismatch")
        source = self.manifest["source_state_cache"]
        source_root = Path(source["path"])
        source_manifest = Path(source["manifest_path"])
        if (
            source_root.is_symlink()
            or source_manifest != source_root / "manifest.json"
            or not source_manifest.is_file()
            or source_manifest.is_symlink()
            or _sha256_file(source_manifest) != source["manifest_sha256"]
        ):
            raise ValueError("DINO oracle live source state-cache manifest drift")
        live_dataset = ForensicQueryStateCacheDataset(source_root)
        live_manifest, live_rows = _validate_source_state_cache(live_dataset)
        for field in (
            "schema",
            "owner_role",
            "cache_fingerprint",
            "row_set_identity",
            "count",
            "state_shape",
            "state_dtype",
            "checkpoint",
            "source_jsonl",
            "selection",
        ):
            source_field = {
                "state_shape": "condition_shape",
                "state_dtype": "condition_dtype",
            }.get(field, field)
            if live_manifest[field] != source[source_field]:
                raise ValueError("DINO oracle live source state-cache identity drift")

        conditions: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        for descriptor in self.manifest["shards"]:
            path = self.root / descriptor["file"]
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != descriptor["sha256"]
            ):
                raise ValueError("DINO oracle shard SHA256/hash mismatch")
            condition, shard_rows = _parse_shard(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            if len(shard_rows) != descriptor["count"]:
                raise ValueError("DINO oracle shard row count mismatch")
            conditions.append(condition)
            rows.extend(shard_rows)
        self._condition = torch.cat(conditions, dim=0).contiguous()
        self._rows = rows
        if len(rows) != len(live_rows):
            raise ValueError("DINO oracle row count differs from live state cache")
        dimensions = self.manifest["original_image_dimensions"]
        for ordinal, (row, live_row, dimension) in enumerate(
            zip(rows, live_rows, dimensions, strict=True)
        ):
            if row != live_row or row.get("selection_ordinal") != ordinal:
                raise ValueError("DINO oracle ordered row identity differs from state cache")
            image_path = Path(str(row["original_image_path"]))
            with Image.open(image_path) as image:
                width, height = image.size
            if dimension != {
                "row_identity": row["row_identity"],
                "width": width,
                "height": height,
            }:
                raise ValueError("DINO oracle original-image dimension identity drift")
        if (
            _identity({"rows": rows}) != self.manifest["row_set_identity"]
            or self.manifest["row_set_identity"] != live_manifest["row_set_identity"]
        ):
            raise ValueError("DINO oracle row-set identity mismatch")

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def cache_fingerprint(self) -> str:
        return str(self.manifest["cache_fingerprint"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("DINO oracle cache index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return {
            "condition": self._condition[index].detach().clone(),
            **dict(self._rows[index]),
        }


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported DINO dtype: {name}") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the exact Formal38 Stage B DINO4x4 oracle cache"
    )
    parser.add_argument("--state-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype", required=True, choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-shard-records", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    manifest = build_forensic_dino_oracle_cache(
        Path(args.output),
        state_cache=Path(args.state_cache),
        device=device,
        dtype=dtype,
        batch_size=args.batch_size,
        max_shard_records=args.max_shard_records,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EXACT_DINO_FEATURE_IDENTITY",
    "FORENSIC_DINO_ORACLE_CACHE_SCHEMA",
    "FORENSIC_DINO_ORACLE_OWNER_ROLE",
    "FORENSIC_DINO_ORACLE_SHARD_SCHEMA",
    "ForensicDinoOracleCacheDataset",
    "build_forensic_dino_oracle_cache",
]
