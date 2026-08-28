"""Distinct DataProto transport for ``nimloth_sft1_query_state_v1``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from nimloth.backbone.base import BackboneBatch, BackboneInputBuilder
from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingBatch
from nimloth.training.sft1.manifest import (
    PINNED_VAGEN_COMMIT,
    PINNED_VERL_COMMIT,
    verify_pinned_vagen_verl_source,
)
from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
    QueryStateTargets,
)
from nimloth.training.sft1.query_state_data import (
    QUERY_STATE_PREPARED_ROW_SCHEMA,
    QueryStatePreparedRow,
)
from nimloth.training.verl.source import require_pinned_verl_import


QUERY_STATE_DATAPROTO_SCHEMA = "nimloth_sft1_query_state_dataproto_v1"
_FORBIDDEN_KEYS = frozenset(
    {"hidden", "query_hidden", "student_hidden", "state", "projected_state"}
)
_FORBIDDEN_KEY_FRAGMENTS = (
    "hidden",
    "student_state",
    "projected_state",
    "state_cache",
    "cached_state",
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class QueryStateUpdateInputs:
    student_batch: QwenStateTrainingBatch
    targets: QueryStateTargets
    record_ids: tuple[str, ...]
    step_indices: tuple[int, ...]
    splits: tuple[str, ...]
    token_counts: tuple[int, ...]
    original_image_paths: tuple[str, ...]
    original_image_sha256: tuple[str, ...]
    source_manifest_identity: str


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _forbidden_student_keys(values: Sequence[object]) -> list[str]:
    forbidden: list[str] = []
    for value in values:
        name = str(value).lower()
        if name in _FORBIDDEN_KEYS or any(
            fragment in name for fragment in _FORBIDDEN_KEY_FRAGMENTS
        ):
            forbidden.append(str(value))
    return sorted(forbidden)


def _object_array(values: Sequence[Any]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = list(values)
    return result


def _runtime_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_pinned_dataproto_type() -> type[Any]:
    return require_pinned_verl_import(_runtime_repo_root())


def build_query_state_dataproto(rows: Sequence[QueryStatePreparedRow]) -> Any:
    """Carry raw encoded rows and detached DINO targets, never student state."""

    verify_pinned_vagen_verl_source(_runtime_repo_root())
    if not rows:
        raise ValueError("Query-State DataProto batch must not be empty")
    if any(row.schema != QUERY_STATE_PREPARED_ROW_SCHEMA for row in rows):
        raise ValueError("Query-State prepared row schema mismatch")
    identities = {row.source_manifest_identity for row in rows}
    if len(identities) != 1 or not _is_sha256(next(iter(identities))):
        raise ValueError("Query-State rows contain mixed/invalid source manifest identities")
    for row in rows:
        forbidden = _forbidden_student_keys(tuple(row.encoded_tensors))
        if forbidden:
            raise ValueError(
                "Query-State DataProto rejects cached student hidden/state: "
                + forbidden[0]
            )
        encoded = row.encoded_tensors
        input_ids = encoded.get("input_ids")
        labels = encoded.get("labels")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype != torch.long
            or input_ids.ndim != 1
            or not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long
            or labels.shape != input_ids.shape
            or row.token_count < 1
        ):
            raise ValueError("Query-State DataProto requires a supervised 1D token row")
        label_valid = labels != -100
        if (
            labels[0].item() != -100
            or not torch.any(label_valid)
            or torch.any(label_valid & labels.ne(input_ids))
        ):
            raise ValueError("Query-State DataProto labels are not exact shifted-CE targets")
        if (
            row.dino_regions.shape != (16, 1024)
            or row.dino_regions.requires_grad
            or not torch.isfinite(row.dino_regions).all()
        ):
            raise ValueError("Query-State DINO targets must be detached finite (16,1024)")
        if (
            isinstance(row.step_index, bool)
            or row.step_index < 0
            or isinstance(row.executed_action_index, bool)
            or not 0 <= row.executed_action_index < 8
            or not row.record_id
            or not row.split
            or not row.original_image_path
            or not _is_sha256(row.original_image_sha256)
        ):
            raise ValueError("Query-State prepared-row provenance is invalid")
    DataProto = _load_pinned_dataproto_type()
    tensors = {
        "dino_regions": torch.stack(
            tuple(row.dino_regions.detach().float().cpu() for row in rows)
        ),
        "token_counts": torch.tensor([row.token_count for row in rows], dtype=torch.long),
        "step_indices": torch.tensor([row.step_index for row in rows], dtype=torch.long),
        "row_valid": torch.ones(len(rows), dtype=torch.bool),
        "executed_action_indices": torch.tensor(
            [row.executed_action_index for row in rows], dtype=torch.long
        ),
    }
    if not torch.isfinite(tensors["dino_regions"]).all():
        raise ValueError("Query-State DataProto contains non-finite DINO targets")
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors={
            "encoded_rows": _object_array([dict(row.encoded_tensors) for row in rows]),
            "archived_assistant_responses": _object_array(
                [row.archived_assistant_response for row in rows]
            ),
            "response_sources": _object_array(["archived"] * len(rows)),
            "record_ids": _object_array([row.record_id for row in rows]),
            "splits": _object_array([row.split for row in rows]),
            "original_image_paths": _object_array(
                [row.original_image_path for row in rows]
            ),
            "original_image_sha256": _object_array(
                [row.original_image_sha256 for row in rows]
            ),
        },
        meta_info={
            "schema": QUERY_STATE_DATAPROTO_SCHEMA,
            "row_schema": QUERY_STATE_PREPARED_ROW_SCHEMA,
            "training_schema": QUERY_STATE_SCHEMA,
            "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
            "state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
            "source_manifest_identity": next(iter(identities)),
            "query_count": 16,
            "state_dim": 1024,
            "vagen_commit": PINNED_VAGEN_COMMIT,
            "verl_commit": PINNED_VERL_COMMIT,
        },
    )


def query_state_update_inputs(
    data: Any,
    *,
    input_builder: BackboneInputBuilder,
) -> QueryStateUpdateInputs:
    """Validate and collate one worker-local Query-State DataProto chunk."""

    expected_meta = {
        "schema": QUERY_STATE_DATAPROTO_SCHEMA,
        "row_schema": QUERY_STATE_PREPARED_ROW_SCHEMA,
        "training_schema": QUERY_STATE_SCHEMA,
        "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
        "state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
        "query_count": 16,
        "state_dim": 1024,
        "vagen_commit": PINNED_VAGEN_COMMIT,
        "verl_commit": PINNED_VERL_COMMIT,
    }
    for name, expected in expected_meta.items():
        if data.meta_info.get(name) != expected:
            raise ValueError(f"Query-State DataProto {name} mismatch")
    manifest_identity = data.meta_info.get("source_manifest_identity")
    if not _is_sha256(manifest_identity):
        raise ValueError("Query-State DataProto source manifest identity is invalid")
    required_tensors = {
        "dino_regions",
        "token_counts",
        "step_indices",
        "row_valid",
        "executed_action_indices",
    }
    missing = sorted(required_tensors - set(data.batch))
    if missing:
        raise ValueError("Query-State DataProto is missing tensor: " + missing[0])
    forbidden = _forbidden_student_keys(tuple(data.batch))
    if forbidden:
        raise ValueError("Query-State DataProto contains cached student tensor: " + forbidden[0])
    required_non_tensors = {
        "encoded_rows",
        "archived_assistant_responses",
        "response_sources",
        "record_ids",
        "splits",
        "original_image_paths",
        "original_image_sha256",
    }
    missing_non_tensor = sorted(required_non_tensors - set(data.non_tensor_batch))
    if missing_non_tensor:
        raise ValueError("Query-State DataProto is missing metadata: " + missing_non_tensor[0])
    size = len(data)
    for name in required_non_tensors:
        if len(data.non_tensor_batch[name]) != size:
            raise ValueError(f"Query-State DataProto metadata does not align: {name}")
    dino = data.batch["dino_regions"]
    token_counts_tensor = data.batch["token_counts"]
    step_indices_tensor = data.batch["step_indices"]
    row_valid_tensor = data.batch["row_valid"]
    action_indices_tensor = data.batch["executed_action_indices"]
    if (
        not isinstance(dino, torch.Tensor)
        or dino.shape != (size, 16, 1024)
        or dino.requires_grad
        or not torch.isfinite(dino).all()
        or not isinstance(token_counts_tensor, torch.Tensor)
        or token_counts_tensor.dtype != torch.long
        or token_counts_tensor.shape != (size,)
        or torch.any(token_counts_tensor < 1)
        or not isinstance(step_indices_tensor, torch.Tensor)
        or step_indices_tensor.dtype != torch.long
        or step_indices_tensor.shape != (size,)
        or torch.any(step_indices_tensor < 0)
        or not isinstance(row_valid_tensor, torch.Tensor)
        or row_valid_tensor.dtype != torch.bool
        or row_valid_tensor.shape != (size,)
        or not isinstance(action_indices_tensor, torch.Tensor)
        or action_indices_tensor.dtype != torch.long
        or action_indices_tensor.shape != (size,)
        or torch.any((action_indices_tensor < 0) | (action_indices_tensor >= 8))
    ):
        raise ValueError("Query-State DataProto tensor contract is invalid")
    if any(
        str(source) != "archived"
        for source in data.non_tensor_batch["response_sources"]
    ) or any(
        not _is_sha256(str(value))
        for value in data.non_tensor_batch["original_image_sha256"]
    ):
        raise ValueError("Query-State DataProto archived/image provenance is invalid")
    encoded_rows = tuple(data.non_tensor_batch["encoded_rows"])
    for index, encoded in enumerate(encoded_rows):
        if not isinstance(encoded, dict):
            raise ValueError("Query-State encoded row is not a supervised raw tensor mapping")
        forbidden = _forbidden_student_keys(tuple(encoded))
        if forbidden:
            raise ValueError(
                "Query-State encoded row contains cached student hidden/state: "
                + forbidden[0]
            )
        input_ids = encoded.get("input_ids")
        labels = encoded.get("labels")
        if (
            not isinstance(input_ids, torch.Tensor)
            or input_ids.dtype != torch.long
            or input_ids.ndim != 1
            or not isinstance(labels, torch.Tensor)
            or labels.dtype != torch.long
            or labels.shape != input_ids.shape
            or int(token_counts_tensor[index].item()) != int(input_ids.numel())
        ):
            raise ValueError("Query-State encoded row token/label contract is invalid")
        label_valid = labels != -100
        if labels[0].item() != -100 or torch.any(label_valid & labels.ne(input_ids)):
            raise ValueError("Query-State encoded row labels changed rendered token IDs")
    backbone_batch = input_builder.collate_encoded(
        list(encoded_rows),
        include_labels=True,
    )
    labels = backbone_batch.tensors.get("labels")
    input_ids = backbone_batch.tensors.get("input_ids")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or not isinstance(input_ids, torch.Tensor)
        or labels.shape != input_ids.shape
    ):
        raise ValueError("Query-State collated labels/input_ids are invalid")
    row_valid = row_valid_tensor
    # Explicit schedule/FSDP padding must execute the same forward but contribute
    # neither state MSE nor LM CE.  Mask after collation so a copied raw archived
    # response cannot silently add valid shifted-CE targets on a padding rank.
    labels = labels.clone()
    labels[~row_valid.to(device=labels.device)] = -100
    backbone_batch = BackboneBatch(
        {**dict(backbone_batch.tensors), "labels": labels}
    )
    targets = QueryStateTargets(
        dino_regions=data.batch["dino_regions"].detach(),
        sample_valid=row_valid.detach(),
    )
    return QueryStateUpdateInputs(
        student_batch=QwenStateTrainingBatch(
            backbone_batch=backbone_batch,
            archived_assistant_responses=tuple(
                str(value)
                for value in data.non_tensor_batch["archived_assistant_responses"]
            ),
            response_sources=tuple(
                str(value) for value in data.non_tensor_batch["response_sources"]
            ),
        ),
        targets=targets,
        record_ids=tuple(str(value) for value in data.non_tensor_batch["record_ids"]),
        step_indices=tuple(int(value) for value in data.batch["step_indices"].tolist()),
        splits=tuple(str(value) for value in data.non_tensor_batch["splits"]),
        token_counts=tuple(int(value) for value in data.batch["token_counts"].tolist()),
        original_image_paths=tuple(
            str(value) for value in data.non_tensor_batch["original_image_paths"]
        ),
        original_image_sha256=tuple(
            str(value) for value in data.non_tensor_batch["original_image_sha256"]
        ),
        source_manifest_identity=manifest_identity,
    )


__all__ = [
    "QUERY_STATE_DATAPROTO_SCHEMA",
    "QueryStateUpdateInputs",
    "build_query_state_dataproto",
    "query_state_update_inputs",
]
