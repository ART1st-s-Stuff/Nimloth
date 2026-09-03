"""Immutable direct-DINO grid8/grid16 cache for reconstruction-ceiling tests.

Every stored view is independently pooled from the same native 37×37 DINO
feature map of the original archived observation.  The direct 4×4 view is not
stored; it must exactly match the immutable parent grid4 oracle cache for every
row before publication can succeed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOMultigridTargets,
    _processor_fingerprint,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_SELECTION_ALL_TRAIN,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION,
    FORENSIC_STAGE_B_EXTERNAL_COUNT,
    FORENSIC_STAGE_B_TRAIN_COUNT,
    _publish_noreplace,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    EXACT_DINO_FEATURE_IDENTITY,
    FORENSIC_DINO_ORACLE_CACHE_SCHEMA,
    FORENSIC_DINO_ORACLE_OWNER_ROLE,
    _current_source_commit,
    _fsync_directory,
    _identity,
    _is_sha256,
    _read_mapping,
    _reject_protected_output,
    _sha256_file,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    _parse_manifest as _parse_grid4_manifest,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    _parse_shard as _parse_grid4_shard,
)

DINO_GRID_CEILING_CACHE_SCHEMA = "nimloth_dino_grid_reconstruction_ceiling_cache_v1"
DINO_GRID_CEILING_SHARD_SCHEMA = (
    "nimloth_dino_grid_reconstruction_ceiling_cache_shard_v1"
)
DINO_GRID_CEILING_OWNER_ROLE = "unsafe_forensic_direct_dino_multigrid_condition"
GRID4_FEATURE_IDENTITY = EXACT_DINO_FEATURE_IDENTITY
_NATIVE_GRID_SIZE = 37
_GRID_SIZES = (8, 16)
_VIEW_KEYS = {8: "grid8", 16: "grid16"}
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


def _view_payload(grid_size: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "grid_size": grid_size,
        "condition_shape": [grid_size**2, DINOV2_LARGE_IDENTITY.hidden_size],
        "condition_dtype": "float32",
        "pooling": (
            f"native37_direct_adaptive_avg_pool2d_"
            f"{grid_size}x{grid_size}_row_major"
        ),
        "input_owner": "original_archived_observation",
        "resize_before_processor": False,
    }
    payload["feature_identity"] = _identity(
        {
            "source": DINOV2_LARGE_IDENTITY.source,
            "revision": DINOV2_LARGE_IDENTITY.revision,
            "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
            "hidden_size": DINOV2_LARGE_IDENTITY.hidden_size,
            "native_grid_size": _NATIVE_GRID_SIZE,
            **payload,
        }
    )
    return payload


_VIEW_PAYLOADS = {_VIEW_KEYS[size]: _view_payload(size) for size in _GRID_SIZES}
DINO_GRID_CEILING_FEATURE_IDENTITIES = {
    size: str(_VIEW_PAYLOADS[_VIEW_KEYS[size]]["feature_identity"])
    for size in _GRID_SIZES
}


def _validate_teacher(teacher: FrozenDINOMultigridTargets) -> dict[str, Any]:
    model = getattr(teacher, "model", None)
    processor = getattr(teacher, "image_processor", None)
    if (
        not isinstance(teacher, FrozenDINOMultigridTargets)
        or teacher.identity != DINOV2_LARGE_IDENTITY
        or tuple(getattr(teacher, "grid_sizes", ())) != (4, 8, 16)
        or getattr(teacher, "native_grid_size", None) != _NATIVE_GRID_SIZE
        or not isinstance(model, torch.nn.Module)
        or processor is None
        or _processor_fingerprint(processor)
        != DINOV2_LARGE_IDENTITY.processor_fingerprint
        or model.training
        or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError(
            "multigrid cache requires concrete pinned frozen/eval DINOv2-large "
            "native37 direct-pooling teacher"
        )
    parameters = tuple(model.parameters())
    if not parameters:
        raise ValueError("multigrid DINO teacher must expose model dtype evidence")
    dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in parameters}
    if len(dtypes) != 1:
        raise ValueError("multigrid DINO teacher model dtype must be uniform")
    return {
        "source": DINOV2_LARGE_IDENTITY.source,
        "revision": DINOV2_LARGE_IDENTITY.revision,
        "processor_fingerprint": DINOV2_LARGE_IDENTITY.processor_fingerprint,
        "hidden_size": DINOV2_LARGE_IDENTITY.hidden_size,
        "native_grid_size": _NATIVE_GRID_SIZE,
        "native_tokens": _NATIVE_GRID_SIZE**2,
        "input_owner": "original_archived_observation",
        "resize_before_processor": False,
        "model_dtype": next(iter(dtypes)),
        "output_dtype": "float32",
        "batch_size": teacher.batch_size,
    }


def _validate_grid_batch(
    value: object,
    *,
    grid_size: int,
    expected_count: int,
) -> torch.Tensor:
    expected_shape = (
        expected_count,
        grid_size**2,
        DINOV2_LARGE_IDENTITY.hidden_size,
    )
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != expected_shape
        or value.dtype != torch.float32
        or value.requires_grad
        or value.grad_fn is not None
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(
            f"direct DINO grid{grid_size} batch must be contiguous finite float32 "
            f"{expected_shape} without gradients"
        )
    return value.detach().cpu().contiguous()


def _row_without_condition(item: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(item, Mapping) or set(item) != {*_ROW_FIELDS, "condition"}:
        raise ValueError("grid4 oracle row schema is invalid")
    return {key: item[key] for key in _ROW_FIELDS}


class _MetadataOnlyGrid4CacheDataset:
    """Read immutable grid4 features/rows without opening the SFT1 state cache."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("grid4 metadata-only cache root must not be a symlink")
        self.root = supplied.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("grid4 metadata-only cache requires a regular manifest")
        self.manifest = _parse_grid4_manifest(
            _read_mapping(manifest_path, owner="grid4 metadata-only cache manifest")
        )
        if self.manifest["producer"]["resolved_config"]["output"] != str(self.root):
            raise ValueError("grid4 metadata-only producer output/root mismatch")
        conditions: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        for descriptor in self.manifest["shards"]:
            path = self.root / descriptor["file"]
            if (
                not path.is_file()
                or path.is_symlink()
                or _sha256_file(path) != descriptor["sha256"]
            ):
                raise ValueError("grid4 metadata-only shard SHA256/hash mismatch")
            condition, shard_rows = _parse_grid4_shard(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            if len(shard_rows) != descriptor["count"]:
                raise ValueError("grid4 metadata-only shard row count mismatch")
            conditions.append(condition)
            rows.extend(shard_rows)
        self._condition = torch.cat(conditions, dim=0).contiguous()
        self._rows = rows
        dimensions = self.manifest["original_image_dimensions"]
        if len(rows) != self.manifest["count"] or len(dimensions) != len(rows):
            raise ValueError("grid4 metadata-only row/dimension count mismatch")
        for ordinal, (row, dimension) in enumerate(zip(rows, dimensions, strict=True)):
            image_path = Path(str(row.get("original_image_path", "")))
            if (
                set(row) != _ROW_FIELDS
                or row.get("selection_ordinal") != ordinal
                or row.get("response_source") != "archived"
                or not image_path.is_absolute()
                or not image_path.is_file()
                or image_path.is_symlink()
                or _sha256_file(image_path) != row.get("original_image_sha256")
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
            ):
                raise ValueError("grid4 metadata-only original row identity mismatch")
            with Image.open(image_path) as image:
                width, height = image.size
            if dimension != {
                "row_identity": row["row_identity"],
                "width": width,
                "height": height,
            }:
                raise ValueError("grid4 metadata-only original image dimension drift")
        if _identity({"rows": rows}) != self.manifest["row_set_identity"]:
            raise ValueError("grid4 metadata-only ordered row-set identity mismatch")

    def __len__(self) -> int:
        return len(self._rows)

    def row_metadata(self, index: int) -> dict[str, Any]:
        return dict(self._rows[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "condition": self._condition[index].detach().clone(),
            **self.row_metadata(index),
        }


def _validate_grid4_cache(
    dataset: _MetadataOnlyGrid4CacheDataset,
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    manifest = dataset.manifest
    dino = manifest.get("dino")
    if (
        manifest.get("schema") != FORENSIC_DINO_ORACLE_CACHE_SCHEMA
        or manifest.get("owner_role") != FORENSIC_DINO_ORACLE_OWNER_ROLE
        or manifest.get("count") != len(rows)
        or manifest.get("row_set_identity") != _identity({"rows": list(rows)})
        or tuple(manifest.get("condition_shape", ())) != (16, 1024)
        or manifest.get("condition_dtype") != "float32"
        or not _is_sha256(manifest.get("cache_fingerprint"))
        or not isinstance(dino, Mapping)
        or dino.get("feature_identity") != GRID4_FEATURE_IDENTITY
        or dino.get("input_owner") != "original_archived_observation"
        or dino.get("resize_before_processor") is not False
        or dino.get("pooling")
        != "final_patch_tokens_adaptive_avg_pool2d_4x4_row_major"
        or len(dataset) != len(rows)
    ):
        raise ValueError("multigrid cache requires the exact immutable direct-DINO grid4 cache")
    for ordinal, expected in enumerate(rows):
        item = dataset[ordinal]
        raw_condition = item.get("condition")
        condition = _validate_grid_batch(
            raw_condition.unsqueeze(0) if isinstance(raw_condition, torch.Tensor) else raw_condition,
            grid_size=4,
            expected_count=1,
        )[0]
        if _row_without_condition(item) != dict(expected) or condition.shape != (16, 1024):
            raise ValueError("immutable grid4 cache ordered row identity mismatch")
    return manifest


def _source_grid4_payload(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
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
        "condition_shape": manifest["condition_shape"],
        "condition_dtype": manifest["condition_dtype"],
        "feature_identity": manifest["dino"]["feature_identity"],
        "embedded_state_cache_fingerprint": manifest["source_state_cache"][
            "cache_fingerprint"
        ],
    }


def _write_shard(
    path: Path,
    *,
    start: int,
    features: Mapping[int, torch.Tensor],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows or set(features) != set(_GRID_SIZES):
        raise ValueError("multigrid shard requires nonempty direct grid8/grid16 features")
    stored = {
        _VIEW_KEYS[size]: _validate_grid_batch(
            features[size], grid_size=size, expected_count=len(rows)
        )
        for size in _GRID_SIZES
    }
    payload = {
        "schema": DINO_GRID_CEILING_SHARD_SCHEMA,
        "features": stored,
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
        "views": {
            key: {
                "condition_shape": list(payload["features"][key].shape[1:]),
                "condition_dtype": "float32",
            }
            for key in ("grid8", "grid16")
        },
    }


def _parse_shard(payload: object) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "features", "rows"}
        or payload.get("schema") != DINO_GRID_CEILING_SHARD_SCHEMA
        or not isinstance(payload.get("features"), Mapping)
        or set(payload["features"]) != {"grid8", "grid16"}
        or not isinstance(payload.get("rows"), list)
        or any(not isinstance(row, Mapping) for row in payload["rows"])
    ):
        raise ValueError("unsupported direct-DINO multigrid cache shard schema")
    rows = [dict(row) for row in payload["rows"]]
    features = {
        key: _validate_grid_batch(
            payload["features"][key],
            grid_size=size,
            expected_count=len(rows),
        )
        for size, key in _VIEW_KEYS.items()
    }
    return features, rows


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
        "native_grid",
        "views",
        "source_grid4_cache",
        "dino",
        "lineage_audit",
        "selection",
        "row_set_identity",
        "original_image_dimensions",
        "producer",
        "max_shard_records",
        "shards",
        "cache_fingerprint",
    }
    expected_count = FORENSIC_STAGE_B_TRAIN_COUNT + FORENSIC_STAGE_B_EXTERNAL_COUNT
    expected_roles = {
        FORENSIC_SELECTION_ALL_TRAIN: FORENSIC_STAGE_B_TRAIN_COUNT,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: FORENSIC_STAGE_B_EXTERNAL_COUNT,
    }
    source_grid4 = raw.get("source_grid4_cache")
    dino = raw.get("dino")
    lineage = raw.get("lineage_audit")
    grid4 = lineage.get("grid4") if isinstance(lineage, Mapping) else None
    selection = raw.get("selection")
    dimensions = raw.get("original_image_dimensions")
    producer = raw.get("producer")
    resolved = producer.get("resolved_config") if isinstance(producer, Mapping) else None
    max_records = raw.get("max_shard_records")
    valid_max_records = (
        not isinstance(max_records, bool)
        and isinstance(max_records, int)
        and max_records >= 1
    )
    shards = raw.get("shards")
    valid_shards = isinstance(shards, list) and bool(shards) and valid_max_records
    expected_start = 0
    expected_view_descriptors = {
        "grid8": {"condition_shape": [64, 1024], "condition_dtype": "float32"},
        "grid16": {"condition_shape": [256, 1024], "condition_dtype": "float32"},
    }
    if valid_shards:
        for index, descriptor in enumerate(shards):
            if (
                not isinstance(descriptor, Mapping)
                or set(descriptor)
                != {"file", "count", "start", "stop", "sha256", "views"}
                or descriptor.get("file") != f"shard_{index:05d}.pt"
                or descriptor.get("start") != expected_start
                or isinstance(descriptor.get("count"), bool)
                or not isinstance(descriptor.get("count"), int)
                or descriptor["count"] < 1
                or descriptor.get("stop") != expected_start + descriptor["count"]
                or descriptor["count"] > max_records
                or not _is_sha256(descriptor.get("sha256"))
                or descriptor.get("views") != expected_view_descriptors
            ):
                valid_shards = False
                break
            expected_start = descriptor["stop"]
        valid_shards = valid_shards and expected_start == expected_count
    source_grid4_fields = {
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
        "feature_identity",
        "embedded_state_cache_fingerprint",
    }
    dino_fields = {
        "source",
        "revision",
        "processor_fingerprint",
        "hidden_size",
        "native_grid_size",
        "native_tokens",
        "input_owner",
        "resize_before_processor",
        "model_dtype",
        "output_dtype",
        "batch_size",
    }
    if (
        set(raw) != required
        or raw.get("schema") != DINO_GRID_CEILING_CACHE_SCHEMA
        or raw.get("version") != 1
        or raw.get("owner_role") != DINO_GRID_CEILING_OWNER_ROLE
        or raw.get("forensic_only") is not True
        or raw.get("unsafe_actor_checkpoint") is not True
        or any(
            raw.get(key) is not False
            for key in ("authoritative", "terminal_primary", "deployable", "sft2_ready")
        )
        or raw.get("count") != expected_count
        or raw.get("native_grid") != {"height": 37, "width": 37, "tokens": 1369}
        or raw.get("views") != _VIEW_PAYLOADS
        or not isinstance(source_grid4, Mapping)
        or set(source_grid4) != source_grid4_fields
        or source_grid4.get("schema") != FORENSIC_DINO_ORACLE_CACHE_SCHEMA
        or source_grid4.get("owner_role") != FORENSIC_DINO_ORACLE_OWNER_ROLE
        or source_grid4.get("count") != expected_count
        or source_grid4.get("condition_shape") != [16, 1024]
        or source_grid4.get("condition_dtype") != "float32"
        or source_grid4.get("feature_identity") != GRID4_FEATURE_IDENTITY
        or not all(
            _is_sha256(source_grid4.get(key))
            for key in (
                "manifest_sha256",
                "cache_fingerprint",
                "row_set_identity",
                "embedded_state_cache_fingerprint",
            )
        )
        or not isinstance(dino, Mapping)
        or set(dino) != dino_fields
        or dino.get("source") != DINOV2_LARGE_IDENTITY.source
        or dino.get("revision") != DINOV2_LARGE_IDENTITY.revision
        or dino.get("processor_fingerprint")
        != DINOV2_LARGE_IDENTITY.processor_fingerprint
        or dino.get("hidden_size") != 1024
        or dino.get("native_grid_size") != 37
        or dino.get("native_tokens") != 1369
        or dino.get("input_owner") != "original_archived_observation"
        or dino.get("resize_before_processor") is not False
        or dino.get("model_dtype") not in {"float32", "float16", "bfloat16"}
        or dino.get("output_dtype") != "float32"
        or isinstance(dino.get("batch_size"), bool)
        or not isinstance(dino.get("batch_size"), int)
        or dino["batch_size"] < 1
        or not isinstance(lineage, Mapping)
        or set(lineage) != {"grid4"}
        or not isinstance(grid4, Mapping)
        or set(grid4)
        != {
            "feature_identity",
            "comparison",
            "compared_rows",
            "all_rows_equal",
            "max_abs_error",
        }
        or grid4.get("feature_identity") != GRID4_FEATURE_IDENTITY
        or grid4.get("comparison") != "torch.equal_float32_all_rows"
        or grid4.get("compared_rows") != expected_count
        or grid4.get("all_rows_equal") is not True
        or grid4.get("max_abs_error") != 0.0
        or not isinstance(selection, Mapping)
        or selection.get("stage") != "stage_b_diagnostic"
        or selection.get("algorithm") != "live_audited_full_roles_v1"
        or selection.get("seed") is not None
        or selection.get("roles") != expected_roles
        or not _is_sha256(selection.get("identity"))
        or raw.get("row_set_identity") != source_grid4.get("row_set_identity")
        or not _is_sha256(raw.get("row_set_identity"))
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
        != "nimloth.training.reconstruction.dino_grid_ceiling_cache"
        or not isinstance(producer.get("source_commit"), str)
        or len(producer["source_commit"]) != 40
        or not isinstance(resolved, Mapping)
        or set(resolved)
        != {
            "grid4_cache",
            "output",
            "device",
            "dtype",
            "batch_size",
            "max_shard_records",
        }
        or resolved.get("grid4_cache") != source_grid4.get("path")
        or resolved.get("dtype") != dino.get("model_dtype")
        or resolved.get("batch_size") != dino.get("batch_size")
        or resolved.get("max_shard_records") != max_records
        or not isinstance(resolved.get("output"), str)
        or not Path(resolved["output"]).is_absolute()
        or not isinstance(resolved.get("device"), str)
        or not resolved["device"]
        or not _is_sha256(producer.get("identity"))
        or _identity({key: value for key, value in producer.items() if key != "identity"})
        != producer.get("identity")
        or not valid_shards
        or not _is_sha256(raw.get("cache_fingerprint"))
        or _identity({key: value for key, value in raw.items() if key != "cache_fingerprint"})
        != raw.get("cache_fingerprint")
    ):
        raise ValueError(
            "direct-DINO multigrid cache manifest identity/owner/pooling is invalid"
        )
    for source in (source_grid4,):
        if any(
            not isinstance(source.get(key), str)
            or not Path(source[key]).is_absolute()
            for key in ("path", "manifest_path")
        ):
            raise ValueError("multigrid source paths must be absolute")
    return raw


@torch.inference_mode()
def build_dino_grid_ceiling_cache(
    output: str | Path,
    *,
    grid4_cache: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_shard_records: int = 512,
) -> Mapping[str, Any]:
    """Build direct grid8/grid16 views and prove exact immutable-grid4 lineage."""

    output_path = Path(output)
    grid4_supplied = Path(grid4_cache)
    _reject_protected_output(output_path, (grid4_supplied,))
    temporary = output_path.with_name(f".{output_path.name}.dino-grid-ceiling-tmp")
    if output_path.exists() or output_path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError("multigrid cache output or temporary path already exists")
    for supplied, owner in ((grid4_supplied, "grid4"),):
        if supplied.is_symlink() or not supplied.is_absolute() or not supplied.is_dir():
            raise ValueError(f"multigrid {owner} input must be an absolute regular directory")
        manifest = supplied / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError(f"multigrid {owner} input requires a regular manifest")
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
        raise ValueError("multigrid cache device/dtype/batch/shard configuration is invalid")

    grid4_root = grid4_supplied.resolve()
    grid4_dataset = _MetadataOnlyGrid4CacheDataset(grid4_root)
    rows = [
        _row_without_condition(grid4_dataset[index])
        for index in range(len(grid4_dataset))
    ]
    grid4_manifest = _validate_grid4_cache(grid4_dataset, rows)
    teacher = FrozenDINOMultigridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    dino = _validate_teacher(teacher)

    temporary.mkdir(parents=True)
    descriptors: list[dict[str, Any]] = []
    shard_features: dict[int, list[torch.Tensor]] = {8: [], 16: []}
    shard_rows: list[Mapping[str, Any]] = []
    dimensions: list[Mapping[str, Any]] = []
    shard_start = 0
    compared_rows = 0
    max_abs_error = 0.0

    def flush() -> None:
        nonlocal shard_start
        if not shard_rows:
            return
        descriptor = _write_shard(
            temporary / f"shard_{len(descriptors):05d}.pt",
            start=shard_start,
            features={size: torch.cat(shard_features[size], dim=0) for size in _GRID_SIZES},
            rows=shard_rows,
        )
        descriptors.append(descriptor)
        shard_start += len(shard_rows)
        shard_rows.clear()
        for values in shard_features.values():
            values.clear()

    try:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            paths = [str(row["original_image_path"]) for row in batch_rows]
            for row, path in zip(batch_rows, paths, strict=True):
                with Image.open(path) as image:
                    width, height = image.size
                dimensions.append({"row_identity": row["row_identity"], "width": width, "height": height})
            raw_grids = teacher.load_grids(paths, device=device)
            if not isinstance(raw_grids, Mapping) or set(raw_grids) != {4, 8, 16}:
                raise ValueError("multigrid teacher must return exact direct grid4/grid8/grid16 views")
            grids = {
                size: _validate_grid_batch(raw_grids[size], grid_size=size, expected_count=len(batch_rows))
                for size in (4, 8, 16)
            }
            expected_grid4 = torch.stack(
                [grid4_dataset[start + index]["condition"] for index in range(len(batch_rows))]
            ).contiguous()
            difference = (grids[4] - expected_grid4).abs().max().item()
            max_abs_error = max(max_abs_error, float(difference))
            if not torch.equal(grids[4], expected_grid4):
                raise ValueError(
                    "direct native37 grid4 lineage must be exactly equal to immutable grid4 cache"
                )
            compared_rows += len(batch_rows)
            cursor = 0
            while cursor < len(batch_rows):
                room = max_shard_records - len(shard_rows)
                take = min(room, len(batch_rows) - cursor)
                for size in _GRID_SIZES:
                    shard_features[size].append(grids[size][cursor : cursor + take])
                shard_rows.extend(batch_rows[cursor : cursor + take])
                cursor += take
                if len(shard_rows) == max_shard_records:
                    flush()
        flush()
        if shard_start != len(rows) or compared_rows != len(rows):
            raise RuntimeError("multigrid cache publication or lineage audit lost rows")

        resolved_config = {
            "grid4_cache": str(grid4_root),
            "output": str(output_path),
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "batch_size": batch_size,
            "max_shard_records": max_shard_records,
        }
        producer: dict[str, Any] = {
            "module": "nimloth.training.reconstruction.dino_grid_ceiling_cache",
            "source_commit": _current_source_commit(),
            "resolved_config": resolved_config,
        }
        producer["identity"] = _identity(producer)
        manifest: dict[str, Any] = {
            "schema": DINO_GRID_CEILING_CACHE_SCHEMA,
            "version": 1,
            "owner_role": DINO_GRID_CEILING_OWNER_ROLE,
            "forensic_only": True,
            "unsafe_actor_checkpoint": True,
            "authoritative": False,
            "terminal_primary": False,
            "deployable": False,
            "sft2_ready": False,
            "count": len(rows),
            "native_grid": {"height": 37, "width": 37, "tokens": 1369},
            "views": _VIEW_PAYLOADS,
            "source_grid4_cache": _source_grid4_payload(grid4_root, grid4_manifest),
            "dino": dino,
            "lineage_audit": {
                "grid4": {
                    "feature_identity": GRID4_FEATURE_IDENTITY,
                    "comparison": "torch.equal_float32_all_rows",
                    "compared_rows": compared_rows,
                    "all_rows_equal": True,
                    "max_abs_error": max_abs_error,
                }
            },
            "selection": grid4_manifest["selection"],
            "row_set_identity": _identity({"rows": rows}),
            "original_image_dimensions": dimensions,
            "producer": producer,
            "max_shard_records": max_shard_records,
            "shards": descriptors,
        }
        manifest["cache_fingerprint"] = _identity(manifest)
        with (temporary / "manifest.json").open("x", encoding="utf-8") as stream:
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


class DinoGridCeilingCacheDataset:
    """Strict reader for one stored direct-pooled grid8 or grid16 view."""

    def __init__(self, root: str | Path, *, grid_size: int) -> None:
        if grid_size not in _GRID_SIZES:
            raise ValueError("multigrid cache stores only grid8 and grid16 views")
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("multigrid cache root must not be a symlink")
        self.root = supplied.resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("multigrid cache requires a regular manifest")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid multigrid cache manifest") from error
        if not isinstance(raw, dict):
            raise TypeError("invalid multigrid cache manifest mapping")
        self.manifest = _parse_manifest(raw)
        if self.manifest["producer"]["resolved_config"]["output"] != str(self.root):
            raise ValueError("multigrid producer output/root identity mismatch")
        self.grid_size = grid_size
        self.view_key = _VIEW_KEYS[grid_size]

        source_grid4 = self.manifest["source_grid4_cache"]
        for source, owner in ((source_grid4, "grid4"),):
            source_root = Path(source["path"])
            source_manifest = Path(source["manifest_path"])
            if (
                source_root.is_symlink()
                or source_manifest != source_root / "manifest.json"
                or not source_manifest.is_file()
                or source_manifest.is_symlink()
                or _sha256_file(source_manifest) != source["manifest_sha256"]
            ):
                raise ValueError(f"multigrid live {owner} source manifest drift")
        grid4_dataset = _MetadataOnlyGrid4CacheDataset(Path(source_grid4["path"]))
        live_rows = [
            _row_without_condition(grid4_dataset[index])
            for index in range(len(grid4_dataset))
        ]
        grid4_manifest = _validate_grid4_cache(grid4_dataset, live_rows)
        if grid4_manifest["cache_fingerprint"] != source_grid4["cache_fingerprint"]:
            raise ValueError("multigrid live grid4 source cache identity drift")

        conditions: list[torch.Tensor] = []
        rows: list[dict[str, Any]] = []
        for descriptor in self.manifest["shards"]:
            path = self.root / descriptor["file"]
            if not path.is_file() or path.is_symlink() or _sha256_file(path) != descriptor["sha256"]:
                raise ValueError("multigrid shard SHA256/hash mismatch")
            features, shard_rows = _parse_shard(
                torch.load(path, map_location="cpu", weights_only=False)
            )
            if len(shard_rows) != descriptor["count"]:
                raise ValueError("multigrid shard row count mismatch")
            conditions.append(features[self.view_key])
            rows.extend(shard_rows)
        self._condition = torch.cat(conditions, dim=0).contiguous()
        self._rows = rows
        dimensions = self.manifest["original_image_dimensions"]
        if len(rows) != len(live_rows) or len(dimensions) != len(rows):
            raise ValueError("multigrid row/dimension count differs from live source")
        for ordinal, (row, live_row, dimension) in enumerate(zip(rows, live_rows, dimensions, strict=True)):
            if row != live_row or row.get("selection_ordinal") != ordinal:
                raise ValueError("multigrid ordered row identity differs from live source")
            with Image.open(str(row["original_image_path"])) as image:
                width, height = image.size
            if dimension != {"row_identity": row["row_identity"], "width": width, "height": height}:
                raise ValueError("multigrid original-image dimension identity drift")
        if _identity({"rows": rows}) != self.manifest["row_set_identity"]:
            raise ValueError("multigrid ordered row-set identity mismatch")

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def cache_fingerprint(self) -> str:
        return str(self.manifest["cache_fingerprint"])

    @property
    def feature_identity(self) -> str:
        return str(self.manifest["views"][self.view_key]["feature_identity"])

    def row_metadata(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("multigrid cache index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return dict(self._rows[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "condition": self._condition[index].detach().clone(),
            **self.row_metadata(index),
        }


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]
    except KeyError as error:
        raise ValueError(f"unsupported DINO dtype: {name}") from error


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build direct native37-to-grid8/grid16 DINO ceiling cache"
    )
    parser.add_argument("--grid4-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", required=True, choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--max-shard-records", required=True, type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = build_dino_grid_ceiling_cache(
        Path(args.output),
        grid4_cache=Path(args.grid4_cache),
        device=torch.device(args.device),
        dtype=_dtype(args.dtype),
        batch_size=args.batch_size,
        max_shard_records=args.max_shard_records,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DINO_GRID_CEILING_CACHE_SCHEMA",
    "DINO_GRID_CEILING_FEATURE_IDENTITIES",
    "DINO_GRID_CEILING_OWNER_ROLE",
    "DINO_GRID_CEILING_SHARD_SCHEMA",
    "GRID4_FEATURE_IDENTITY",
    "DinoGridCeilingCacheDataset",
    "build_dino_grid_ceiling_cache",
]
