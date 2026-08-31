"""Auditable direct Query-State/DINO feature diagnostics.

Formal APIs consume canonical state only through the strict reconstruction cache
reader and internally load the pinned frozen DINO owner. They never invent a
response or participate in a training loss. Its PCA colorization is a Nimloth-defined
reproducible method; DeepSight's exact
Figure 7 colorization is not public and is deliberately not claimed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    DINOGridTargets,
    FrozenDINOGridTargets,
)
from nimloth.training.reconstruction.query_state_cache import (
    QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
    QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
    QueryStateReconstructionCacheDataset,
)

NIMLOTH_SHARED_BASIS_METHOD = "nimloth_shared_basis"
_BASIS_SCHEMA = "nimloth_query_state_shared_feature_basis_v1"
_REPORT_SCHEMA = "nimloth_query_state_feature_report_v1"
_STATE_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")
_NORMALIZATION = "shared_global"


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


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(_canonical_json({"shape": list(tensor.shape)}))
    digest.update(bytes(tensor.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def _validate_feature_batch(value: torch.Tensor, *, owner: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or tuple(value.shape[1:]) != _STATE_SHAPE
    ):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"{owner} must have shape [N,16,1024] (K16), got {shape}")
    if value.shape[0] < 1 or not value.is_floating_point():
        raise ValueError(f"{owner} must be a nonempty floating [N,16,1024] tensor")
    if value.requires_grad or value.grad_fn is not None:
        raise ValueError(f"{owner} must be detached frozen features")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{owner} must contain only finite features")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _validate_identity(identity: SharedFeatureBasisIdentity) -> None:
    if not isinstance(identity, SharedFeatureBasisIdentity):
        raise TypeError("shared basis identity must use SharedFeatureBasisIdentity")
    if identity.method != NIMLOTH_SHARED_BASIS_METHOD:
        raise ValueError("colorization method must be nimloth_shared_basis, not DeepSight exact")
    if identity.fit_split != "train":
        raise ValueError("shared PCA basis fit split must be train; validation refit is forbidden")
    for field in (
        "bundle_fingerprint",
        "source_jsonl_sha256",
        "source_manifest_identity",
        "fit_split_identity",
        "fit_row_set_identity",
        "dino_identity",
    ):
        if not _is_sha256(getattr(identity, field)):
            raise ValueError(f"shared basis identity {field} must be SHA256")
    if tuple(identity.state_shape) != _STATE_SHAPE:
        raise ValueError("shared basis identity must preserve K16 [16,1024]")
    if identity.interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("shared basis interpolation identity is unsupported")


@dataclass(frozen=True)
class SharedFeatureBasisIdentity:
    method: str
    bundle_fingerprint: str
    source_jsonl_sha256: str
    source_manifest_identity: str
    fit_split: str
    fit_split_identity: str
    fit_row_set_identity: str
    dino_identity: str
    state_shape: tuple[int, int]
    interpolation: str


@dataclass(frozen=True)
class SharedFeatureBasis:
    identity: SharedFeatureBasisIdentity
    center: torch.Tensor
    components: torch.Tensor
    global_scale: torch.Tensor
    feature_norm_scale: float
    rmse_scale: float
    artifact_sha256: str
    global_scale_sha256: str

    def transform(
        self,
        features: torch.Tensor,
        *,
        normalization: str = _NORMALIZATION,
    ) -> torch.Tensor:
        if normalization != _NORMALIZATION:
            raise ValueError(
                "normalization must be shared_global; per-image min-max is forbidden"
            )
        value = _validate_feature_batch(features, owner="shared-basis features")
        projected = (value - self.center) @ self.components
        lower = self.global_scale[:, 0]
        upper = self.global_scale[:, 1]
        denominator = (upper - lower).clamp_min(1e-12)
        rgb = ((projected - lower) / denominator).clamp(0.0, 1.0)
        return rgb.reshape(value.shape[0], 4, 4, 3).to(torch.float32)


def _basis_content_sha256(
    *,
    identity: SharedFeatureBasisIdentity,
    center: torch.Tensor,
    components: torch.Tensor,
    global_scale: torch.Tensor,
    feature_norm_scale: float,
    rmse_scale: float,
) -> str:
    payload = {
        "schema": _BASIS_SCHEMA,
        "identity": asdict(identity),
        "center_sha256": _tensor_sha256(center),
        "components_sha256": _tensor_sha256(components),
        "global_scale_sha256": _tensor_sha256(global_scale),
        "feature_norm_scale": feature_norm_scale,
        "rmse_scale": rmse_scale,
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _fit_shared_feature_basis_from_records(
    state_records: Sequence[QueryStateFeatureRecord],
    dino_records: Sequence[DinoFeatureRecord],
    *,
    interpolation: str,
    output_path: str | Path,
) -> SharedFeatureBasis:
    """Test-only numeric helper; supplied records are not authoritative provenance."""

    identity, left, right = _derive_train_basis_inputs(
        state_records,
        dino_records,
        interpolation=interpolation,
    )
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"shared basis output already exists; overwrite forbidden: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    flat_domains = (left.reshape(-1, 1024), right.reshape(-1, 1024))
    sample_count = sum(value.shape[0] for value in flat_domains)
    center = sum(value.sum(dim=0, dtype=torch.float64) for value in flat_domains) / sample_count
    covariance = torch.zeros((1024, 1024), dtype=torch.float64)
    chunk_size = 4096
    for domain in flat_domains:
        for start in range(0, domain.shape[0], chunk_size):
            centered = domain[start : start + chunk_size].to(torch.float64) - center
            covariance.add_(centered.transpose(0, 1) @ centered)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    top_values = eigenvalues[-3:].flip(0)
    if bool((top_values <= 1e-18).any()):
        raise ValueError("train features need at least three non-collapsed PCA directions")
    components = eigenvectors[:, -3:].flip(1).contiguous()
    # SVD signs are arbitrary; pin each component to a deterministic orientation.
    for column in range(3):
        pivot = int(torch.argmax(torch.abs(components[:, column])).item())
        if float(components[pivot, column]) < 0.0:
            components[:, column].neg_()
    projected_domains = tuple(
        (domain.to(torch.float64) - center) @ components for domain in flat_domains
    )
    global_scale = torch.stack(
        (
            torch.stack([value.amin(dim=0) for value in projected_domains]).amin(dim=0),
            torch.stack([value.amax(dim=0) for value in projected_domains]).amax(dim=0),
        ),
        dim=1,
    )
    norms = torch.cat(
        [torch.linalg.vector_norm(domain, dim=-1) for domain in flat_domains]
    )
    rmses = torch.sqrt(torch.mean(torch.square(left - right), dim=-1))
    feature_norm_scale = max(float(norms.max()), 1e-12)
    rmse_scale = max(float(rmses.max()), 1e-12)

    center = center.to(torch.float32)
    components = components.to(torch.float32)
    global_scale = global_scale.to(torch.float32)
    artifact_sha256 = _basis_content_sha256(
        identity=identity,
        center=center,
        components=components,
        global_scale=global_scale,
        feature_norm_scale=feature_norm_scale,
        rmse_scale=rmse_scale,
    )
    scale_sha256 = _tensor_sha256(global_scale)
    payload = {
        "schema": _BASIS_SCHEMA,
        "identity": asdict(identity),
        "center": center,
        "components": components,
        "global_scale": global_scale,
        "feature_norm_scale": feature_norm_scale,
        "rmse_scale": rmse_scale,
        "artifact_sha256": artifact_sha256,
        "global_scale_sha256": scale_sha256,
    }
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return SharedFeatureBasis(
        identity=identity,
        center=center,
        components=components,
        global_scale=global_scale,
        feature_norm_scale=feature_norm_scale,
        rmse_scale=rmse_scale,
        artifact_sha256=artifact_sha256,
        global_scale_sha256=scale_sha256,
    )


def load_shared_feature_basis(
    path: str | Path,
    *,
    expected_identity: SharedFeatureBasisIdentity,
    expected_artifact_sha256: str | None = None,
) -> SharedFeatureBasis:
    """Load a basis only when all source/split/teacher/render identities match exactly."""

    _validate_identity(expected_identity)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("shared basis artifact is missing or is a symlink")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("shared basis artifact is unreadable") from error
    required = {
        "schema",
        "identity",
        "center",
        "components",
        "global_scale",
        "feature_norm_scale",
        "rmse_scale",
        "artifact_sha256",
        "global_scale_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema") != _BASIS_SCHEMA:
        raise ValueError("shared basis artifact schema is invalid")
    try:
        stored_identity = SharedFeatureBasisIdentity(
            **{
                **payload["identity"],
                "state_shape": tuple(payload["identity"]["state_shape"]),
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("shared basis artifact identity is invalid") from error
    _validate_identity(stored_identity)
    if stored_identity != expected_identity:
        raise ValueError("shared basis identity/source/split/hash/interpolation mismatch")
    center = payload["center"]
    components = payload["components"]
    scale = payload["global_scale"]
    if (
        not isinstance(center, torch.Tensor)
        or center.shape != (1024,)
        or not isinstance(components, torch.Tensor)
        or components.shape != (1024, 3)
        or not isinstance(scale, torch.Tensor)
        or scale.shape != (3, 2)
        or not all(item.is_floating_point() and bool(torch.isfinite(item).all()) for item in (center, components, scale))
    ):
        raise ValueError("shared basis tensor shape/integrity is invalid")
    norm_scale = payload["feature_norm_scale"]
    rmse_scale = payload["rmse_scale"]
    if not all(isinstance(item, (int, float)) and math.isfinite(item) and item > 0 for item in (norm_scale, rmse_scale)):
        raise ValueError("shared basis global scale integrity is invalid")
    computed = _basis_content_sha256(
        identity=stored_identity,
        center=center,
        components=components,
        global_scale=scale,
        feature_norm_scale=float(norm_scale),
        rmse_scale=float(rmse_scale),
    )
    if not _is_sha256(payload["artifact_sha256"]) or computed != payload["artifact_sha256"]:
        raise ValueError("shared basis artifact SHA256 integrity mismatch")
    if expected_artifact_sha256 is not None and computed != expected_artifact_sha256:
        raise ValueError("shared basis expected artifact SHA256 mismatch")
    scale_hash = _tensor_sha256(scale)
    if scale_hash != payload["global_scale_sha256"]:
        raise ValueError("shared basis global scale SHA256 integrity mismatch")
    return SharedFeatureBasis(
        identity=stored_identity,
        center=center.detach().contiguous(),
        components=components.detach().contiguous(),
        global_scale=scale.detach().contiguous(),
        feature_norm_scale=float(norm_scale),
        rmse_scale=float(rmse_scale),
        artifact_sha256=computed,
        global_scale_sha256=scale_hash,
    )


@dataclass(frozen=True)
class QueryStateFeatureRecord:
    row_identity: str
    split: str
    image_path: str
    image_sha256: str
    archived_response_sha256: str
    bundle_fingerprint: str
    source_jsonl_sha256: str
    source_manifest_identity: str
    selection_role: str
    cache_split_identity: str
    state: torch.Tensor


@dataclass(frozen=True)
class DinoFeatureRecord:
    """Frozen teacher output paired to the exact original observation identity."""

    row_identity: str
    split: str
    image_sha256: str
    dino_identity: str
    features: torch.Tensor


def _feature_row_set_identity(records: Sequence[QueryStateFeatureRecord]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "rows": [
                    {
                        "row_identity": row.row_identity,
                        "image_sha256": row.image_sha256,
                        "archived_response_sha256": row.archived_response_sha256,
                    }
                    for row in records
                ]
            }
        )
    ).hexdigest()


def _feature_split_identity(
    records: Sequence[QueryStateFeatureRecord], *, source_manifest_identity: str
) -> str:
    values = tuple(records)
    if not values:
        raise ValueError("feature split identity requires records")
    expected_role = (
        QUERY_STATE_CACHE_SELECTION_ALL_TRAIN
        if values[0].split == "train"
        else QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION
    )
    identities = {row.cache_split_identity for row in values}
    if (
        any(row.source_manifest_identity != source_manifest_identity for row in values)
        or any(row.selection_role != expected_role for row in values)
        or len(identities) != 1
        or not _is_sha256(next(iter(identities)))
    ):
        raise ValueError("feature records do not share an identity-bound cache selection")
    return next(iter(identities))


def _derive_train_basis_inputs(
    state_records: Sequence[QueryStateFeatureRecord],
    dino_records: Sequence[DinoFeatureRecord],
    *,
    interpolation: str,
) -> tuple[SharedFeatureBasisIdentity, torch.Tensor, torch.Tensor]:
    """Recompute every basis identity from the paired train-record manifest."""

    states = tuple(state_records)
    targets = tuple(dino_records)
    if len(states) != len(targets) or not states:
        raise ValueError("shared PCA fitting requires nonempty paired train records")
    if interpolation not in {"nearest", "bilinear", "bicubic"}:
        raise ValueError("shared basis interpolation identity is unsupported")
    seen: set[str] = set()
    bundle_ids: set[str] = set()
    source_ids: set[str] = set()
    source_manifest_ids: set[str] = set()
    dino_ids: set[str] = set()
    state_values: list[torch.Tensor] = []
    target_values: list[torch.Tensor] = []
    for state_record, target_record in zip(states, targets, strict=True):
        if not isinstance(state_record, QueryStateFeatureRecord) or not isinstance(target_record, DinoFeatureRecord):
            raise TypeError("basis inputs use QueryStateFeatureRecord/DinoFeatureRecord")
        if state_record.split != "train" or target_record.split != "train":
            raise ValueError("shared PCA fitting accepts train rows only; validation is transform-only")
        if not state_record.row_identity or state_record.row_identity != target_record.row_identity:
            raise ValueError("train paired row identity mismatch")
        if state_record.row_identity in seen:
            raise ValueError("train paired row identity must be unique")
        seen.add(state_record.row_identity)
        if state_record.image_sha256 != target_record.image_sha256 or not _is_sha256(state_record.image_sha256):
            raise ValueError("train paired original image identity mismatch")
        image_path = Path(state_record.image_path)
        if not image_path.is_absolute() or not image_path.is_file() or _sha256_file(image_path) != state_record.image_sha256:
            raise ValueError("train original image SHA256 identity mismatch")
        if not _is_sha256(state_record.archived_response_sha256):
            raise ValueError("train archived response/CoT identity must be SHA256")
        bundle_ids.add(state_record.bundle_fingerprint)
        source_ids.add(state_record.source_jsonl_sha256)
        source_manifest_ids.add(state_record.source_manifest_identity)
        dino_ids.add(target_record.dino_identity)
        state_values.append(_validate_feature_batch(state_record.state.unsqueeze(0), owner="train Query-State").squeeze(0))
        target_values.append(_validate_feature_batch(target_record.features.unsqueeze(0), owner="train DINO target").squeeze(0))
    if not all(
        len(values) == 1 and _is_sha256(next(iter(values)))
        for values in (bundle_ids, source_ids, source_manifest_ids, dino_ids)
    ):
        raise ValueError("train records must resolve one valid bundle/source/DINO identity")
    source_manifest_identity = next(iter(source_manifest_ids))
    identity = SharedFeatureBasisIdentity(
        method=NIMLOTH_SHARED_BASIS_METHOD,
        bundle_fingerprint=next(iter(bundle_ids)),
        source_jsonl_sha256=next(iter(source_ids)),
        source_manifest_identity=source_manifest_identity,
        fit_split="train",
        fit_split_identity=_feature_split_identity(
            states, source_manifest_identity=source_manifest_identity
        ),
        fit_row_set_identity=_feature_row_set_identity(states),
        dino_identity=next(iter(dino_ids)),
        state_shape=_STATE_SHAPE,
        interpolation=interpolation,
    )
    _validate_identity(identity)
    return identity, torch.stack(state_values), torch.stack(target_values)


def dino_feature_identity(teacher: DINOGridTargets) -> str:
    """Bind the exact frozen teacher, processor revision, and 4x4 grid contract."""

    identity = getattr(teacher, "identity", None)
    grid_size = getattr(teacher, "grid_size", None)
    if identity != DINOV2_LARGE_IDENTITY or grid_size != 4:
        raise ValueError(
            "feature visualization requires pinned frozen DINOv2-large 4x4x1024"
        )
    model = getattr(teacher, "model", None)
    if isinstance(model, torch.nn.Module) and (
        model.training or any(parameter.requires_grad for parameter in model.parameters())
    ):
        raise ValueError("feature visualization DINO teacher must be frozen and in eval mode")
    return hashlib.sha256(
        _canonical_json(
            {
                "source": identity.source,
                "revision": identity.revision,
                "processor_fingerprint": identity.processor_fingerprint,
                "hidden_size": identity.hidden_size,
                "grid_size": grid_size,
            }
        )
    ).hexdigest()


@torch.inference_mode()
def extract_dino_feature_records(
    state_records: Sequence[QueryStateFeatureRecord],
    *,
    teacher: DINOGridTargets,
    device: torch.device,
    batch_size: int = 32,
) -> list[DinoFeatureRecord]:
    """Stream exact original observations through a frozen DINO teacher."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("DINO feature extraction batch size must be positive")
    records = tuple(state_records)
    if not records:
        raise ValueError("DINO feature extraction requires state records")
    identity = dino_feature_identity(teacher)
    outputs: list[DinoFeatureRecord] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        paths: list[str] = []
        for record in chunk:
            path = Path(record.image_path)
            if (
                not path.is_absolute()
                or not path.is_file()
                or not _is_sha256(record.image_sha256)
                or _sha256_file(path) != record.image_sha256
            ):
                raise ValueError("DINO feature extraction original image identity mismatch")
            paths.append(str(path))
        features = teacher.load(paths, device=device).detach().to(
            device="cpu", dtype=torch.float32
        )
        validated = _validate_feature_batch(features, owner="frozen DINO target")
        if validated.shape[0] != len(chunk):
            raise ValueError("DINO feature extraction batch cardinality mismatch")
        outputs.extend(
            DinoFeatureRecord(
                row_identity=record.row_identity,
                split=record.split,
                image_sha256=record.image_sha256,
                dino_identity=identity,
                features=feature.detach().clone(),
            )
            for record, feature in zip(chunk, validated, strict=True)
        )
    return outputs


def deterministic_global_derangement(count: int, *, seed: int) -> list[int]:
    """Create one full-split, deterministic non-identity row mapping."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("global derangement needs at least two rows for non-identity mapping")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("derangement seed must be an integer")
    indices = list(range(count))
    generator = random.Random(seed)
    # Sattolo's cycle is a permutation with no fixed points for every N >= 2.
    for index in range(count - 1, 0, -1):
        other = generator.randrange(index)
        indices[index], indices[other] = indices[other], indices[index]
    return indices


def _effective_rank(value: torch.Tensor) -> float:
    matrix = value.reshape(-1, value.shape[-1]).to(torch.float64)
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    weights = singular.square()
    total = float(weights.sum())
    if total <= 1e-24:
        return 0.0
    probabilities = weights / total
    entropy = -torch.sum(probabilities * torch.log(probabilities.clamp_min(1e-24)))
    return float(torch.exp(entropy))


def _collapse_fraction(value: torch.Tensor) -> float:
    centered = value - value.mean(dim=1, keepdim=True)
    distances = torch.linalg.vector_norm(centered, dim=-1)
    reference = torch.linalg.vector_norm(value, dim=-1).mean().clamp_min(1e-12)
    return float((distances <= reference * 1e-6).to(torch.float64).mean())


def _comparison_metrics(state: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = state - target
    cosine = torch.nn.functional.cosine_similarity(
        state.reshape(-1, 1024), target.reshape(-1, 1024), dim=-1, eps=1e-12
    )
    return {
        "mse": float(torch.mean(torch.square(difference))),
        "cosine": float(cosine.mean()),
    }


def _pair_metrics(state: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    state_flat = state.reshape(-1, 1024)
    target_flat = target.reshape(-1, 1024)
    return {
        **_comparison_metrics(state, target),
        "state_norm_mean": float(torch.linalg.vector_norm(state_flat, dim=-1).mean()),
        "target_norm_mean": float(torch.linalg.vector_norm(target_flat, dim=-1).mean()),
        "state_variance": float(torch.var(state, unbiased=False)),
        "target_variance": float(torch.var(target, unbiased=False)),
        "state_effective_rank": _effective_rank(state),
        "target_effective_rank": _effective_rank(target),
        "state_collapse_fraction": _collapse_fraction(state),
        "target_collapse_fraction": _collapse_fraction(target),
    }


def aggregate_direct_feature_metrics(
    state: torch.Tensor,
    target: torch.Tensor,
    *,
    shuffle_seed: int,
) -> dict[str, Any]:
    """Aggregate the whole split before applying a guaranteed-global baseline."""

    left = _validate_feature_batch(state, owner="metric Query-State")
    right = _validate_feature_batch(target, owner="metric DINO target")
    if left.shape != right.shape:
        raise ValueError("metric Query-State and DINO target shapes must match")
    mapping = deterministic_global_derangement(left.shape[0], seed=shuffle_seed)
    direct = _pair_metrics(left, right)
    shuffled_pair = _comparison_metrics(left[mapping], right)
    mapping_hash = hashlib.sha256(
        _canonical_json({"algorithm": "sattolo_cycle_v1", "seed": shuffle_seed, "mapping": mapping})
    ).hexdigest()
    return {
        "count": left.shape[0],
        "direct": direct,
        "shuffled_row_baseline": {
            "mse": shuffled_pair["mse"],
            "cosine": shuffled_pair["cosine"],
            "mapping": mapping,
            "mapping_sha256": mapping_hash,
            "seed": shuffle_seed,
            "count": left.shape[0],
            "algorithm": "sattolo_cycle_v1",
        },
    }


def _validate_records(
    state_records: Sequence[QueryStateFeatureRecord],
    dino_records: Sequence[DinoFeatureRecord],
    *,
    basis: SharedFeatureBasis,
) -> tuple[tuple[QueryStateFeatureRecord, ...], tuple[DinoFeatureRecord, ...], torch.Tensor, torch.Tensor]:
    states = tuple(state_records)
    targets = tuple(dino_records)
    if len(states) != len(targets) or len(states) < 2:
        raise ValueError("paired state/DINO records need equal counts and at least two rows")
    seen: set[str] = set()
    split: str | None = None
    evaluation_source_ids: set[str] = set()
    state_tensors: list[torch.Tensor] = []
    target_tensors: list[torch.Tensor] = []
    for state_record, target_record in zip(states, targets, strict=True):
        if not isinstance(state_record, QueryStateFeatureRecord) or not isinstance(target_record, DinoFeatureRecord):
            raise TypeError("paired feature records use QueryStateFeatureRecord/DinoFeatureRecord")
        if not state_record.row_identity or state_record.row_identity != target_record.row_identity:
            raise ValueError("paired row identity mismatch")
        if state_record.row_identity in seen:
            raise ValueError("paired row identity must be unique")
        seen.add(state_record.row_identity)
        if not state_record.split or state_record.split != target_record.split:
            raise ValueError("paired split identity mismatch")
        if split is None:
            split = state_record.split
        elif split != state_record.split:
            raise ValueError("report may contain only one split identity")
        if state_record.image_sha256 != target_record.image_sha256 or not _is_sha256(state_record.image_sha256):
            raise ValueError("paired original image SHA256 identity mismatch")
        image_path = Path(state_record.image_path)
        if not image_path.is_absolute() or not image_path.is_file() or _sha256_file(image_path) != state_record.image_sha256:
            raise ValueError("original image SHA256/hash identity mismatch")
        if not _is_sha256(state_record.archived_response_sha256):
            raise ValueError("archived response/CoT identity must be SHA256")
        if state_record.bundle_fingerprint != basis.identity.bundle_fingerprint or not _is_sha256(state_record.bundle_fingerprint):
            raise ValueError("bundle identity does not match shared basis")
        if not _is_sha256(state_record.source_jsonl_sha256):
            raise ValueError("evaluation source JSONL identity must be SHA256")
        evaluation_source_ids.add(state_record.source_jsonl_sha256)
        if (
            state_record.source_manifest_identity
            != basis.identity.source_manifest_identity
            or not _is_sha256(state_record.source_manifest_identity)
        ):
            raise ValueError("source manifest identity does not match shared basis")
        if target_record.dino_identity != basis.identity.dino_identity or not _is_sha256(target_record.dino_identity):
            raise ValueError("DINO frozen teacher identity does not match shared basis")
        state_tensors.append(
            _validate_feature_batch(state_record.state.unsqueeze(0), owner="record Query-State").squeeze(0)
        )
        target_tensors.append(
            _validate_feature_batch(target_record.features.unsqueeze(0), owner="record DINO target").squeeze(0)
        )
    if len(evaluation_source_ids) != 1:
        raise ValueError("evaluation records must resolve one source JSONL identity")
    return states, targets, torch.stack(state_tensors), torch.stack(target_tensors)


def _rgb_image(value: torch.Tensor) -> Image.Image:
    pixels = (value.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    return Image.frombytes("RGB", (pixels.shape[1], pixels.shape[0]), bytes(pixels.reshape(-1).tolist()))


def _heatmap(value: torch.Tensor) -> Image.Image:
    scalar = value.to(torch.float32).clamp(0.0, 1.0)
    red = scalar
    green = 1.0 - torch.abs(2.0 * scalar - 1.0)
    blue = 1.0 - scalar
    return _rgb_image(torch.stack((red, green, blue), dim=-1))


def _resample(name: str) -> Image.Resampling:
    return {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[name]


def _display_tile(image: Image.Image, *, interpolation: str, size: int = 128) -> Image.Image:
    return image.convert("RGB").resize((size, size), resample=_resample(interpolation))


def _horizontal(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("image strip cannot be empty")
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    output = Image.new("RGB", (width, height), color=(0, 0, 0))
    cursor = 0
    for image in images:
        output.paste(image.convert("RGB"), (cursor, 0))
        cursor += image.width
    return output


def _vertical(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("contact sheet cannot be empty")
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    output = Image.new("RGB", (width, height), color=(0, 0, 0))
    cursor = 0
    for image in images:
        output.paste(image.convert("RGB"), (0, cursor))
        cursor += image.height
    return output


def _render_query_state_feature_report_from_records(
    state_records: Sequence[QueryStateFeatureRecord],
    dino_records: Sequence[DinoFeatureRecord],
    *,
    basis: SharedFeatureBasis,
    output_dir: str | Path,
    interpolation: str,
    normalization: str,
    shuffle_seed: int,
    colorization_method: str = NIMLOTH_SHARED_BASIS_METHOD,
    authoritative_provenance: bool = False,
) -> dict[str, Any]:
    """Numeric renderer; only cache-owned wrappers may label provenance authoritative."""

    if colorization_method != NIMLOTH_SHARED_BASIS_METHOD:
        raise ValueError("DeepSight exact colorization is unpublished; use nimloth_shared_basis")
    if normalization != _NORMALIZATION:
        raise ValueError("normalization must be shared_global; per-image min-max is forbidden")
    if interpolation != basis.identity.interpolation:
        raise ValueError("interpolation does not match shared basis identity")
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"feature report output already exists: {destination}")
    states, _targets, state_batch, target_batch = _validate_records(
        state_records, dino_records, basis=basis
    )
    metrics = aggregate_direct_feature_metrics(state_batch, target_batch, shuffle_seed=shuffle_seed)
    state_rgb = basis.transform(state_batch, normalization=normalization)
    target_rgb = basis.transform(target_batch, normalization=normalization)
    row_set_identity = _feature_row_set_identity(states)
    evaluation_split_identity = _feature_split_identity(
        states, source_manifest_identity=basis.identity.source_manifest_identity
    )
    metadata = {
        "schema": _REPORT_SCHEMA,
        "colorization_method": NIMLOTH_SHARED_BASIS_METHOD,
        "deep_sight_exact_colorization": False,
        "diagnostic_role": (
            "primary_direct_feature_space_post_hoc_only"
            if authoritative_provenance
            else "test_only_supplied_records_non_authoritative"
        ),
        "authoritative_cache_provenance": authoritative_provenance,
        "training_or_checkpoint_selection": False,
        "basis_sha256": basis.artifact_sha256,
        "global_scale_sha256": basis.global_scale_sha256,
        "basis_fit_split": basis.identity.fit_split,
        "basis_fit_split_identity": basis.identity.fit_split_identity,
        "basis_fit_row_set_identity": basis.identity.fit_row_set_identity,
        "interpolation": interpolation,
        "normalization": normalization,
        "bundle_fingerprint": basis.identity.bundle_fingerprint,
        "basis_train_source_jsonl_sha256": basis.identity.source_jsonl_sha256,
        "evaluation_source_jsonl_sha256": states[0].source_jsonl_sha256,
        "source_manifest_identity": basis.identity.source_manifest_identity,
        "dino_identity": basis.identity.dino_identity,
        "evaluation_split": states[0].split,
        "evaluation_split_identity": evaluation_split_identity,
        "evaluation_row_set_identity": row_set_identity,
        "state_shape": list(_STATE_SHAPE),
        "feature_norm_scale": basis.feature_norm_scale,
        "rmse_scale": basis.rmse_scale,
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    rows_json: list[dict[str, Any]] = []
    strips: list[Image.Image] = []
    try:
        for index, (state_record, state_value, target_value) in enumerate(
            zip(states, state_batch, target_batch, strict=True)
        ):
            stem = f"row_{index:05d}_{hashlib.sha256(state_record.row_identity.encode()).hexdigest()[:12]}"
            row_dir = temporary / stem
            row_dir.mkdir()
            with Image.open(state_record.image_path) as source_image:
                original = source_image.convert("RGB")
            target_pca = _rgb_image(target_rgb[index])
            state_pca = _rgb_image(state_rgb[index])
            target_norm = _heatmap(
                torch.linalg.vector_norm(target_value, dim=-1).reshape(4, 4) / basis.feature_norm_scale
            )
            state_norm = _heatmap(
                torch.linalg.vector_norm(state_value, dim=-1).reshape(4, 4) / basis.feature_norm_scale
            )
            cosine = torch.nn.functional.cosine_similarity(
                state_value, target_value, dim=-1, eps=1e-12
            ).reshape(4, 4)
            cosine_image = _heatmap((cosine + 1.0) / 2.0)
            rmse = torch.sqrt(torch.mean(torch.square(state_value - target_value), dim=-1)).reshape(4, 4)
            rmse_image = _heatmap(rmse / basis.rmse_scale)
            named = {
                "original": original,
                "target_pca_rgb": target_pca,
                "state_pca_rgb": state_pca,
                "target_feature_norm": target_norm,
                "state_feature_norm": state_norm,
                "slot_cosine": cosine_image,
                "slot_rmse": rmse_image,
            }
            artifacts: dict[str, str] = {}
            for name, image in named.items():
                path = row_dir / f"{name}.png"
                image.convert("RGB").save(path)
                artifacts[name] = path.relative_to(temporary).as_posix()
            strip = _horizontal(
                [_display_tile(image, interpolation=interpolation) for image in named.values()]
            )
            strip_path = row_dir / "strip.png"
            strip.save(strip_path)
            artifacts["strip"] = strip_path.relative_to(temporary).as_posix()
            strips.append(strip)
            rows_json.append(
                {
                    "row_identity": state_record.row_identity,
                    "split": state_record.split,
                    "image_sha256": state_record.image_sha256,
                    "archived_response_sha256": state_record.archived_response_sha256,
                    "artifacts": artifacts,
                }
            )
        contact = _vertical(strips)
        contact_path = temporary / "contact_sheet.png"
        contact.save(contact_path)
        report = {
            "metadata": metadata,
            "metrics": metrics,
            "rows": rows_json,
            "contact_sheet": contact_path.relative_to(temporary).as_posix(),
        }
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def _state_records_from_cache(
    cache_dir: str | Path,
    *,
    expected_role: str,
) -> list[QueryStateFeatureRecord]:
    """Load every state/identity through the strict live-revalidating cache reader."""

    required_selection = {
        "train": QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
        "evaluation": QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
    }
    if expected_role not in required_selection:
        raise ValueError("feature cache role must be train or evaluation")
    dataset = QueryStateReconstructionCacheDataset(cache_dir)
    manifest = dataset.manifest
    split = str(manifest.split["name"])
    selection_role = str(manifest.selection["role"])
    train_split = str(manifest.source_jsonl["train"]["split"])
    validation_split = str(manifest.source_jsonl["validation"]["split"])
    if selection_role != required_selection[expected_role]:
        raise ValueError(
            f"feature cache requires {required_selection[expected_role]} selection"
        )
    if expected_role == "train" and split != train_split:
        raise ValueError("shared basis requires a strict all_train Query-State cache")
    if expected_role == "evaluation" and split != validation_split:
        raise ValueError(
            "formal feature rendering requires external_validation, not raw validation"
        )
    source_role = "train" if split == train_split else "validation"
    if train_split == validation_split:
        raise ValueError("Query-State source train/validation splits must differ")
    bundle_fingerprint = hashlib.sha256(
        _canonical_json(dict(manifest.bundle))
    ).hexdigest()
    source_sha256 = str(manifest.source_jsonl[source_role]["sha256"])
    source_manifest_identity = str(manifest.source_jsonl["source_manifest_identity"])
    records: list[QueryStateFeatureRecord] = []
    raw_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        state = item.pop("state")
        raw_rows.append(dict(item))
        records.append(
            QueryStateFeatureRecord(
                row_identity=str(item["row_identity"]),
                split=str(item["split"]),
                image_path=str(item["original_image_path"]),
                image_sha256=str(item["original_image_sha256"]),
                archived_response_sha256=str(
                    item["archived_assistant_response_sha256"]
                ),
                bundle_fingerprint=bundle_fingerprint,
                source_jsonl_sha256=source_sha256,
                source_manifest_identity=source_manifest_identity,
                selection_role=selection_role,
                cache_split_identity=str(manifest.split["identity"]),
                state=state,
            )
        )
    if hashlib.sha256(_canonical_json({"rows": raw_rows})).hexdigest() != manifest.row_set_identity:
        raise ValueError("feature cache row-set identity differs from strict cache manifest")
    expected_split_identity = _feature_split_identity(
        records, source_manifest_identity=source_manifest_identity
    )
    if expected_split_identity != manifest.split["identity"]:
        raise ValueError("feature cache split identity differs from strict cache manifest")
    return records


def _build_pinned_dino_teacher(
    *,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> FrozenDINOGridTargets:
    if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise ValueError("DINO dtype must be float32, float16, or bfloat16")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("DINO batch size must be positive")
    teacher = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=dtype,
        grid_size=4,
        batch_size=batch_size,
    )
    dino_feature_identity(teacher)
    return teacher


def fit_shared_feature_basis(
    train_cache: str | Path,
    *,
    interpolation: str,
    output_path: str | Path,
    dino_device: torch.device,
    dino_dtype: torch.dtype,
    dino_batch_size: int,
) -> SharedFeatureBasis:
    """Formal fit API: strict train cache plus internally owned pinned DINO."""

    states = _state_records_from_cache(train_cache, expected_role="train")
    teacher = _build_pinned_dino_teacher(
        device=dino_device, dtype=dino_dtype, batch_size=dino_batch_size
    )
    targets = extract_dino_feature_records(
        states, teacher=teacher, device=dino_device, batch_size=dino_batch_size
    )
    return _fit_shared_feature_basis_from_records(
        states, targets, interpolation=interpolation, output_path=output_path
    )


def render_query_state_feature_report(
    *,
    train_cache: str | Path,
    evaluation_cache: str | Path,
    basis_path: str | Path,
    output_dir: str | Path,
    interpolation: str,
    normalization: str,
    shuffle_seed: int,
    dino_device: torch.device,
    dino_dtype: torch.dtype,
    dino_batch_size: int,
) -> dict[str, Any]:
    """Formal report API requiring train-basis and evaluation cache provenance."""

    train_states = _state_records_from_cache(train_cache, expected_role="train")
    evaluation_states = _state_records_from_cache(
        evaluation_cache, expected_role="evaluation"
    )
    teacher = _build_pinned_dino_teacher(
        device=dino_device, dtype=dino_dtype, batch_size=dino_batch_size
    )
    dino_identity = dino_feature_identity(teacher)
    expected_identity = SharedFeatureBasisIdentity(
        method=NIMLOTH_SHARED_BASIS_METHOD,
        bundle_fingerprint=train_states[0].bundle_fingerprint,
        source_jsonl_sha256=train_states[0].source_jsonl_sha256,
        source_manifest_identity=train_states[0].source_manifest_identity,
        fit_split="train",
        fit_split_identity=_feature_split_identity(
            train_states,
            source_manifest_identity=train_states[0].source_manifest_identity,
        ),
        fit_row_set_identity=_feature_row_set_identity(train_states),
        dino_identity=dino_identity,
        state_shape=_STATE_SHAPE,
        interpolation=interpolation,
    )
    basis = load_shared_feature_basis(
        basis_path, expected_identity=expected_identity
    )
    targets = extract_dino_feature_records(
        evaluation_states,
        teacher=teacher,
        device=dino_device,
        batch_size=dino_batch_size,
    )
    return _render_query_state_feature_report_from_records(
        evaluation_states,
        targets,
        basis=basis,
        output_dir=output_dir,
        interpolation=interpolation,
        normalization=normalization,
        shuffle_seed=shuffle_seed,
        authoritative_provenance=True,
    )


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dino-device", choices=("cpu", "cuda"), required=True)
    common.add_argument(
        "--dino-dtype", choices=("float32", "float16", "bfloat16"), required=True
    )
    common.add_argument("--dino-batch-size", type=int, required=True)
    fit = subparsers.add_parser(
        "fit-basis",
        parents=[common],
        help="fit from a strict train Query-State cache with pinned frozen DINO",
    )
    fit.add_argument("--train-cache", type=Path, required=True)
    fit.add_argument("--interpolation", choices=("nearest", "bilinear", "bicubic"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    render = subparsers.add_parser(
        "render-report",
        parents=[common],
        help="render from strict train/evaluation caches with pinned frozen DINO",
    )
    render.add_argument("--basis", type=Path, required=True)
    render.add_argument("--train-cache", type=Path, required=True)
    render.add_argument("--evaluation-cache", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument(
        "--interpolation", choices=("nearest", "bilinear", "bicubic"), required=True
    )
    render.add_argument("--normalization", choices=(_NORMALIZATION,), required=True)
    render.add_argument("--shuffle-seed", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.dino_device)
    dtype = _dtype(args.dino_dtype)
    if args.command == "fit-basis":
        basis = fit_shared_feature_basis(
            args.train_cache,
            interpolation=args.interpolation,
            output_path=args.output,
            dino_device=device,
            dino_dtype=dtype,
            dino_batch_size=args.dino_batch_size,
        )
        result: Mapping[str, Any] = {
            "method": basis.identity.method,
            "artifact_sha256": basis.artifact_sha256,
            "global_scale_sha256": basis.global_scale_sha256,
        }
    else:
        report = render_query_state_feature_report(
            train_cache=args.train_cache,
            evaluation_cache=args.evaluation_cache,
            basis_path=args.basis,
            output_dir=args.output_dir,
            interpolation=args.interpolation,
            normalization=args.normalization,
            shuffle_seed=args.shuffle_seed,
            dino_device=device,
            dino_dtype=dtype,
            dino_batch_size=args.dino_batch_size,
        )
        result = {
            "method": NIMLOTH_SHARED_BASIS_METHOD,
            "basis_sha256": report["metadata"]["basis_sha256"],
            "report": str(args.output_dir),
            "row_count": len(report["rows"]),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "NIMLOTH_SHARED_BASIS_METHOD",
    "DinoFeatureRecord",
    "QueryStateFeatureRecord",
    "SharedFeatureBasis",
    "SharedFeatureBasisIdentity",
    "aggregate_direct_feature_metrics",
    "deterministic_global_derangement",
    "dino_feature_identity",
    "extract_dino_feature_records",
    "fit_shared_feature_basis",
    "load_shared_feature_basis",
    "render_query_state_feature_report",
]
