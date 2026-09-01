"""Strict direct Query-State producer and reconstruction-cache contract.

The cache stores only the frozen canonical K16 state and row provenance.  It is
schema-distinct from historical SFT2/RCDM caches and deliberately has no legacy
checkpoint or world-model compatibility path.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from nimloth.backbone.qwen25vl.input import Qwen25VLInputBuilder
from nimloth.backbone.qwen25vl.loading import qwen_hidden_size
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    forward_qwen_state_training,
    require_archived_assistant_response,
)
from nimloth.latent import LatentActionTokens, latent_state_tokens, special_token_ids
from nimloth.training.sft1.query_state import (
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
)
from nimloth.training.sft1.query_state_checkpoint import (
    QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA,
    QUERY_STATE_DEPLOYABLE_SCHEMA,
    QueryStateResumeIdentity,
    load_direct_query_state_artifact,
)
from nimloth.training.sft1.query_state_data import (
    QueryStateRenderedRow,
    render_query_state_row,
)
from nimloth.training.sft1.query_state_smoke_runtime import (
    build_query_state_source_manifest_identity,
)
from nimloth.training.sft1.real_rows import (
    SFT1V2Early4Row,
    SFT1V2RowAudit,
    index_early4_rows,
)
from nimloth.wm.grid import DirectSlotProjector

QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA = (
    "nimloth_query_state_reconstruction_cache_v1"
)
_QUERY_STATE_RECONSTRUCTION_SHARD_SCHEMA = (
    "nimloth_query_state_reconstruction_cache_shard_v1"
)
_STATE_SHAPE = (16, 1024)
_STATE_ORDERING = "row_major"
QUERY_STATE_CACHE_SELECTION_ALL_TRAIN = "all_train"
QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION = "external_validation"
_SELECTION_ROLES = {
    QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
    QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
}
_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_BUNDLE_TERMS = (
    "optimizer",
    "scheduler",
    "rng",
    "rank",
    "resume",
    "legacy",
    "world_model",
    "worldmodel",
    "wm_checkpoint",
    "value_head",
    "valuehead",
    "state_projector",
    "stateprojector",
    "shared_slot_projector",
    "sharedslotprojector",
    "grid_encoder",
    "gridencoder",
)
_EXPECTED_OWNERS = {
    "actor": "full_qwen_actor",
    "processor": "qwen_processor_tokenizer",
    "direct_state": "no_bias_linear_2048_to_1024",
}
_DTYPE_BY_NAME = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _tree_sha256(root: Path) -> str:
    """Hash owner-relative names and bytes while rejecting links/non-files."""

    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Query-State bundle owner is incomplete: {root.name}")
    files = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    regular = [item for item in files if item.is_file() and not item.is_symlink()]
    if any(item.is_symlink() or (not item.is_dir() and not item.is_file()) for item in files):
        raise ValueError(f"Query-State bundle owner contains unsupported links: {root.name}")
    if not regular:
        raise ValueError(f"Query-State bundle owner is incomplete: {root.name}")
    digest = hashlib.sha256()
    for item in regular:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _forbidden_owner_component(path: Path) -> str | None:
    for component in path.parts:
        normalized = component.casefold().replace("-", "_").replace(".", "_")
        if any(term in normalized for term in _FORBIDDEN_BUNDLE_TERMS):
            return component
    return None


def _require_clean_owner_path(path: Path, *, relative_to: Path | None = None) -> None:
    candidate = path if relative_to is None else path.relative_to(relative_to)
    forbidden = _forbidden_owner_component(candidate)
    if forbidden is not None:
        raise ValueError(
            "Query-State bundle contains rank-local, resume, or legacy owner path: "
            f"{candidate.as_posix()} (component={forbidden})"
        )


def _read_json_mapping(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {owner} JSON: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"invalid {owner} mapping: {path}")
    return raw


@dataclass(frozen=True)
class QueryStateBundleIdentity:
    bundle_path: str
    bundle_schema: str
    source_commit: str
    source_identity: Mapping[str, Any]
    owners: Mapping[str, str]
    owner_sha256: Mapping[str, str]
    processor_identity: str
    tokenizer_identity: str
    template_identity: str
    checkpoint_identity: str
    human_gate_receipt_sha256: str
    state_shape: tuple[int, int] = _STATE_SHAPE
    state_dtype_contract: str = "floating"
    state_ordering: str = _STATE_ORDERING


@dataclass(frozen=True)
class QueryStateSourceData:
    """Exact immutable pre-RL sources consumed by ``index_early4_rows``."""

    train_jsonl: str
    train_sha256: str
    validation_jsonl: str
    validation_sha256: str
    train_split: str
    validation_split: str


@dataclass(frozen=True)
class QueryStateSourceContract:
    """Audited source boundary cryptographically owned by the bundle identity."""

    data: QueryStateSourceData
    source_manifest_identity: str


@dataclass(frozen=True)
class _LoadedQueryStateBundleOwners:
    """Internally loaded full bundle owners; never accepted from a caller."""

    actor: Qwen2_5_VLForConditionalGeneration
    processor: Any
    input_builder: Qwen25VLInputBuilder
    projector: DirectSlotProjector
    token_id_map: Mapping[str, int]
    device: torch.device


@dataclass(frozen=True)
class _QueryStateCacheRecord:
    row: SFT1V2Early4Row
    state: torch.Tensor
    provenance: Mapping[str, str] | None = None


@dataclass(frozen=True)
class _ExtractedQueryStateBatch:
    state: torch.Tensor
    rendered: tuple[QueryStateRenderedRow, ...]


@dataclass(frozen=True)
class QueryStateCacheShard:
    file: str
    count: int
    start: int
    stop: int
    sha256: str
    row_metadata_sha256: str
    state_dtype: str
    state_shape: tuple[int, int]


@dataclass(frozen=True)
class QueryStateCacheManifest:
    schema: str
    version: int
    count: int
    state_shape: tuple[int, int]
    state_ordering: str
    state_dtype: str
    bundle: Mapping[str, Any]
    source_jsonl: Mapping[str, Any]
    split: Mapping[str, Any]
    selection: Mapping[str, Any]
    row_set_identity: str
    shards: tuple[QueryStateCacheShard, ...]
    cache_fingerprint: str


def require_real_archived_response(response: str | None, *, source: str) -> str:
    """Expose the canonical archived-response gate under cache ownership."""

    return require_archived_assistant_response(response, source=source)


def _validate_export_metadata(metadata: object) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("Query-State bundle metadata is absent")
    sha_fields = (
        "checkpoint_identity",
        "checkpoint_control_identity",
        "human_gate_receipt_sha256",
        "export_approval_sha256",
        "export_command_identity",
        "processor_identity",
        "tokenizer_identity",
        "template_identity",
        "materialization_process_identity",
    )
    for field in sha_fields:
        if not _is_sha256(metadata.get(field)):
            raise ValueError(f"Query-State bundle metadata {field} must be SHA256")
    terminal_update = metadata.get("terminal_update")
    if (
        isinstance(terminal_update, bool)
        or not isinstance(terminal_update, int)
        or terminal_update < 1
    ):
        raise ValueError("Query-State bundle requires a terminal formal checkpoint")
    if (
        not isinstance(metadata.get("export_approval_id"), str)
        or not metadata["export_approval_id"].strip()
        or metadata.get("automatic_sft2_authorization") is not False
    ):
        raise ValueError("Query-State bundle human gate/export boundary is invalid")
    return metadata


def validate_query_state_bundle(path: str | Path) -> QueryStateBundleIdentity:
    """Validate a complete human-gated full-owner Query-State bundle."""

    supplied = Path(path)
    _require_clean_owner_path(Path(supplied.name))
    if supplied.is_symlink():
        raise ValueError("Query-State deployable bundle must not be a symlink")
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError("Query-State deployable bundle is incomplete")
    manifest_path = root / "bundle.json"
    actor_path = root / "actor"
    processor_path = root / "processor"
    direct_path = root / "direct_state.pt"
    if not manifest_path.is_file() or not direct_path.is_file():
        raise ValueError("Query-State bundle is incomplete: bundle/direct-state owner missing")
    if manifest_path.is_symlink() or direct_path.is_symlink():
        raise ValueError("Query-State bundle owners must not be symlinks")

    all_items = list(root.rglob("*"))
    for item in all_items:
        _require_clean_owner_path(item, relative_to=root)

    manifest = _read_json_mapping(manifest_path, owner="Query-State bundle")
    required = {
        "schema",
        "training_schema",
        "objective_version",
        "direct_state_schema",
        "source_identity",
        "metadata",
        "owners",
    }
    if set(manifest) != required:
        raise ValueError("Query-State bundle manifest is incomplete or has unknown owners")
    if (
        manifest.get("schema") != QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA
        or manifest.get("training_schema") != QUERY_STATE_SCHEMA
        or manifest.get("objective_version") != QUERY_STATE_OBJECTIVE_VERSION
        or manifest.get("direct_state_schema") != QUERY_STATE_DEPLOYABLE_SCHEMA
    ):
        raise ValueError("unsupported, legacy, or mismatched Query-State bundle schema")
    owners = manifest.get("owners")
    if owners != _EXPECTED_OWNERS:
        raise ValueError(
            "Query-State bundle owner mismatch; legacy StateProjector/WM owners are forbidden"
        )
    source_raw = manifest.get("source_identity")
    if not isinstance(source_raw, dict):
        raise ValueError("Query-State bundle source identity is absent")
    source = QueryStateResumeIdentity(**source_raw)
    if source.experiment_mode != "formal":
        raise ValueError("Query-State bundle source must be a human-gated formal run")
    metadata = _validate_export_metadata(manifest.get("metadata"))

    actor_hash = _tree_sha256(actor_path)
    processor_hash = _tree_sha256(processor_path)
    actor_config = _read_json_mapping(actor_path / "config.json", owner="full Qwen actor")
    architectures = actor_config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not any(isinstance(item, str) and "Qwen" in item for item in architectures)
    ):
        raise ValueError("Query-State bundle actor is not a full Qwen actor")
    actor_weights = [
        item
        for item in actor_path.rglob("*")
        if item.is_file() and item.suffix in {".safetensors", ".bin"}
    ]
    if not actor_weights:
        raise ValueError("Query-State bundle full Qwen actor weights are incomplete")
    if not (processor_path / "tokenizer.json").is_file():
        raise ValueError("Query-State bundle processor/tokenizer owner is incomplete")
    if not any(
        item.is_file() and "processor" in item.name.lower()
        for item in processor_path.rglob("*")
    ):
        raise ValueError("Query-State bundle processor owner is incomplete")

    _, artifact_metadata = load_direct_query_state_artifact(
        direct_path,
        expected_source_identity=source,
    )
    if artifact_metadata != {"bundle_role": "direct_state_only"}:
        raise ValueError("Query-State direct-state owner metadata mismatch")
    owner_hashes = {
        "actor": actor_hash,
        "processor": processor_hash,
        "direct_state": _sha256_file(direct_path),
        "bundle": _sha256_file(manifest_path),
    }
    return QueryStateBundleIdentity(
        bundle_path=str(root),
        bundle_schema=QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA,
        source_commit=source.source_commit,
        source_identity=asdict(source),
        owners=dict(_EXPECTED_OWNERS),
        owner_sha256=owner_hashes,
        processor_identity=metadata["processor_identity"],
        tokenizer_identity=metadata["tokenizer_identity"],
        template_identity=metadata["template_identity"],
        checkpoint_identity=metadata["checkpoint_identity"],
        human_gate_receipt_sha256=metadata["human_gate_receipt_sha256"],
    )


def validate_frozen_query_state_producer(
    *, actor: nn.Module, projector: nn.Module
) -> None:
    """Require every actor/vision/projector descendant to remain frozen/eval."""

    if not isinstance(actor, nn.Module):
        raise TypeError("Query-State actor must be a torch module")
    if not isinstance(projector, DirectSlotProjector):
        raise TypeError("Query-State producer requires DirectSlotProjector")
    for owner_name, owner in (("actor", actor), ("direct projector", projector)):
        training = [name for name, module in owner.named_modules() if module.training]
        if training:
            raise ValueError(
                f"Query-State {owner_name} must be recursively in eval mode: {training[0]}"
            )
        trainable = [
            name
            for name, parameter in owner.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise ValueError(
                f"Query-State {owner_name} must be recursively frozen; "
                f"requires_grad=True: {trainable[0]}"
            )


def validate_canonical_query_state(state: torch.Tensor) -> torch.Tensor:
    """Validate ordered detached ``[B,16,1024]`` state without reshaping it."""

    if (
        not isinstance(state, torch.Tensor)
        or state.ndim != 3
        or tuple(state.shape[1:]) != _STATE_SHAPE
    ):
        shape = tuple(state.shape) if isinstance(state, torch.Tensor) else type(state).__name__
        raise ValueError(f"canonical Query-State must have shape [B,16,1024] (K16), got {shape}")
    if not state.is_floating_point():
        raise ValueError("canonical Query-State dtype must be floating point")
    if state.requires_grad or state.grad_fn is not None:
        raise ValueError("canonical Query-State must be detached and carry no grad graph")
    if not torch.isfinite(state).all():
        raise ValueError("canonical Query-State must be finite")
    return state.detach().contiguous()


def _extract_canonical_query_states(
    rows: Sequence[SFT1V2Early4Row],
    *,
    processor: Any,
    input_builder: Any,
    actor: nn.Module,
    projector: DirectSlotProjector,
    token_id_map: Mapping[str, int],
    device: torch.device,
    max_length: int,
) -> _ExtractedQueryStateBatch:
    """Render archived rows and run the unique one-forward frozen state path."""

    if not rows:
        raise ValueError("Query-State extraction batch must not be empty")
    validate_frozen_query_state_producer(actor=actor, projector=projector)
    rendered = tuple(
        render_query_state_row(row, processor=processor, max_length=max_length)
        for row in rows
    )
    responses = tuple(
        require_real_archived_response(
            item.row.archived_assistant_response,
            source="archived",
        )
        for item in rendered
    )
    backbone_batch = input_builder.collate_encoded(
        [dict(item.encoded_tensors) for item in rendered],
        include_labels=False,
    )
    batch = QwenStateTrainingBatch(
        backbone_batch=backbone_batch,
        archived_assistant_responses=responses,
        response_sources=("archived",) * len(rendered),
        diagnostic_image_token_indices=tuple(
            item.diagnostic_image_token_indices for item in rendered
        ),
        diagnostic_instruction_token_spans=tuple(
            item.diagnostic_instruction_token_span for item in rendered
        ),
    )
    with torch.inference_mode():
        student = forward_qwen_state_training(
            actor,
            batch,
            dict(token_id_map),
            device,
            latent_token_count=16,
        )
        state = projector(student.query_hidden)
    validate_frozen_query_state_producer(actor=actor, projector=projector)
    canonical = validate_canonical_query_state(state.detach()).cpu().clone()
    return _ExtractedQueryStateBatch(state=canonical, rendered=rendered)


_PROVENANCE_IDENTITY_FIELDS = (
    "prompt_history_identity",
    "messages_identity",
    "renderer_identity",
    "template_identity",
    "encoded_input_identity",
)


def _rendered_row_provenance(rendered: QueryStateRenderedRow) -> dict[str, str]:
    if rendered.response_source != "archived":
        raise ValueError("Query-State cache requires archived rendered-row provenance")
    provenance = {
        name: getattr(rendered, name)
        for name in _PROVENANCE_IDENTITY_FIELDS
    }
    if any(not _is_sha256(value) for value in provenance.values()):
        raise ValueError("Query-State rendered-row provenance identity is invalid")
    return {**provenance, "response_source": "archived"}


def _metadata_for_source_row(row: SFT1V2Early4Row) -> dict[str, Any]:
    response = require_real_archived_response(
        row.archived_assistant_response,
        source="archived",
    )
    return {
        "row_identity": row.identity,
        "record_id": row.record_id,
        "step_index": row.step_index,
        "split": row.split,
        "executed_action_index": row.executed_action_index,
        "original_image_path": row.original_image_path,
        "original_image_sha256": row.original_image_sha256,
        "archived_assistant_response_sha256": _sha256_bytes(response.encode("utf-8")),
    }


def _row_set_identity_from_shards(shards: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "shards": [
                    {
                        "start": shard["start"],
                        "stop": shard["stop"],
                        "row_metadata_sha256": shard["row_metadata_sha256"],
                    }
                    for shard in shards
                ]
            }
        )
    )


def _validate_record(
    record: _QueryStateCacheRecord,
    *,
    split: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not isinstance(record, _QueryStateCacheRecord) or not isinstance(
        record.row, SFT1V2Early4Row
    ):
        raise TypeError("Query-State cache records must own audited SFT1V2Early4Row")
    row = record.row
    if row.split != split:
        raise ValueError("Query-State row split identity mismatch")
    image_path = Path(row.original_image_path)
    if not image_path.is_absolute() or not image_path.is_file():
        raise ValueError("Query-State original image path is missing or not absolute")
    if _sha256_file(image_path) != row.original_image_sha256:
        raise ValueError("Query-State original image SHA256 identity mismatch")
    state = validate_canonical_query_state(record.state.unsqueeze(0)).squeeze(0)
    if record.provenance is None:
        raise ValueError(
            "Query-State cache requires actual rendered prompt/template/encoding provenance"
        )
    provenance = dict(record.provenance)
    if set(provenance) != {*_PROVENANCE_IDENTITY_FIELDS, "response_source"}:
        raise ValueError("Query-State cache row provenance fields are incomplete")
    if provenance["response_source"] != "archived" or any(
        not _is_sha256(provenance[name]) for name in _PROVENANCE_IDENTITY_FIELDS
    ):
        raise ValueError("Query-State cache row provenance identity is invalid")
    return state, {**_metadata_for_source_row(row), **provenance}


def _index_audited_sources(
    source: QueryStateSourceContract,
) -> tuple[tuple[SFT1V2Early4Row, ...], SFT1V2RowAudit, str]:
    if not isinstance(source, QueryStateSourceContract):
        raise TypeError("Query-State cache source must use QueryStateSourceContract")
    if not _is_sha256(source.source_manifest_identity):
        raise ValueError("Query-State source manifest identity must be SHA256")
    rows, audit = index_early4_rows(source, enforce_approved_counts=False)
    identity = build_query_state_source_manifest_identity(rows, audit)
    if identity != source.source_manifest_identity:
        raise ValueError("Query-State audited source manifest identity mismatch")
    return rows, audit, identity


def _selection_for_role(
    rows: Sequence[SFT1V2Early4Row],
    audit: SFT1V2RowAudit,
    *,
    source: QueryStateSourceContract,
    selection_role: str,
) -> tuple[tuple[SFT1V2Early4Row, ...], str, dict[str, Any]]:
    """Reconstruct one audited role without accepting a caller-supplied row mask."""

    if selection_role not in _SELECTION_ROLES:
        raise ValueError(
            "Query-State cache selection role must be all_train or external_validation"
        )
    if source.data.train_split != "train" or source.data.validation_split != "val":
        raise ValueError("Query-State audited source split labels must be train/val")
    raw = tuple(
        row for row in rows
        if row.split == (
            source.data.train_split
            if selection_role == QUERY_STATE_CACHE_SELECTION_ALL_TRAIN
            else source.data.validation_split
        )
    )
    selected = (
        raw
        if selection_role == QUERY_STATE_CACHE_SELECTION_ALL_TRAIN
        else tuple(row for row in raw if row.external_eligible)
    )
    if not selected:
        raise ValueError("Query-State audited cache selection has no eligible rows")
    if selection_role == QUERY_STATE_CACHE_SELECTION_ALL_TRAIN:
        if len(raw) != audit.train_rows or any(not row.external_eligible for row in selected):
            raise ValueError("Query-State all_train selection disagrees with source audit")
    elif (
        len(raw) != audit.raw_validation_rows
        or len(selected) != audit.external_validation_rows
        or any(not row.external_eligible for row in selected)
    ):
        raise ValueError(
            "Query-State external_validation selection disagrees with source audit"
        )
    audit_payload = asdict(audit)
    selection_payload: dict[str, Any] = {
        "role": selection_role,
        "raw_row_count": len(raw),
        "selected_row_count": len(selected),
        "excluded_row_count": len(raw) - len(selected),
        "source_audit": audit_payload,
        "ordered_row_identities": [row.identity for row in selected],
    }
    selection_identity = _sha256_bytes(_canonical_json(selection_payload))
    return selected, selection_identity, {
        key: value
        for key, value in selection_payload.items()
        if key != "ordered_row_identities"
    } | {"identity": selection_identity}


def _split_identity(
    rows: Sequence[SFT1V2Early4Row],
    *,
    source_identity: str,
    split: str,
    selection_role: str,
    selection_identity: str,
) -> str:
    return _sha256_bytes(_canonical_json({
        "source_manifest_identity": source_identity,
        "split": split,
        "selection_role": selection_role,
        "selection_identity": selection_identity,
        "ordered_row_identities": [row.identity for row in rows],
    }))


def _manifest_without_fingerprint(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(manifest)
    value.pop("cache_fingerprint", None)
    return value


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(_manifest_without_fingerprint(manifest)))


def _write_bytes_fsynced(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing a concurrent owner."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace cache publication requires renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(
            f"Query-State cache destination appeared during publication: {destination}"
        )
    raise OSError(error, os.strerror(error), str(destination))


def _dtype_name(dtype: torch.dtype) -> str:
    for name, expected in _DTYPE_BY_NAME.items():
        if dtype == expected:
            return name
    raise ValueError(f"unsupported Query-State cache dtype: {dtype}")


def _validate_shard_payload(
    payload: object,
    descriptor: Mapping[str, Any],
    *,
    expected_dtype: torch.dtype,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or set(payload) != {"schema", "state", "rows"}:
        raise ValueError("Query-State shard payload schema is invalid")
    if payload.get("schema") != _QUERY_STATE_RECONSTRUCTION_SHARD_SCHEMA:
        raise ValueError("unsupported or legacy Query-State shard schema")
    state = payload.get("state")
    rows = payload.get("rows")
    count = descriptor.get("count")
    if not isinstance(count, int) or count < 1:
        raise ValueError("Query-State shard count is invalid")
    if not isinstance(state, torch.Tensor) or state.shape != (count, *_STATE_SHAPE):
        raise ValueError("Query-State shard state shape/count must preserve [N,16,1024]")
    if state.dtype != expected_dtype:
        raise ValueError("Query-State shard state dtype mismatch")
    if state.requires_grad or state.grad_fn is not None or not torch.isfinite(state).all():
        raise ValueError("Query-State shard state must be detached and finite")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("Query-State shard row count mismatch")
    validated_rows: list[dict[str, Any]] = []
    required_row_fields = {
        "row_identity",
        "record_id",
        "step_index",
        "split",
        "executed_action_index",
        "original_image_path",
        "original_image_sha256",
        "archived_assistant_response_sha256",
        *_PROVENANCE_IDENTITY_FIELDS,
        "response_source",
    }
    for row in rows:
        if not isinstance(row, dict) or set(row) != required_row_fields:
            raise ValueError("Query-State shard row provenance schema mismatch")
        if (
            not isinstance(row["row_identity"], str)
            or not row["row_identity"]
            or not isinstance(row["record_id"], str)
            or not row["record_id"]
            or isinstance(row["step_index"], bool)
            or not isinstance(row["step_index"], int)
            or row["step_index"] < 0
            or not isinstance(row["split"], str)
            or not row["split"]
            or isinstance(row["executed_action_index"], bool)
            or not isinstance(row["executed_action_index"], int)
            or not 0 <= row["executed_action_index"] < 8
            or not isinstance(row["original_image_path"], str)
            or not Path(row["original_image_path"]).is_absolute()
            or not _is_sha256(row["original_image_sha256"])
            or not _is_sha256(row["archived_assistant_response_sha256"])
            or row["response_source"] != "archived"
            or any(not _is_sha256(row[name]) for name in _PROVENANCE_IDENTITY_FIELDS)
        ):
            raise ValueError("Query-State shard row/image/CoT identity is invalid")
        image_path = Path(row["original_image_path"])
        if (
            not image_path.is_file()
            or _sha256_file(image_path) != row["original_image_sha256"]
        ):
            raise ValueError("Query-State shard original image identity mismatch")
        validated_rows.append(dict(row))
    if len({row["row_identity"] for row in validated_rows}) != count:
        raise ValueError("Query-State shard has duplicate row identity")
    if _sha256_bytes(_canonical_json({"rows": validated_rows})) != descriptor.get(
        "row_metadata_sha256"
    ):
        raise ValueError("Query-State shard row/image/CoT identity hash mismatch")
    return state.detach().contiguous(), validated_rows


def _require_bundle_owner_hashes(
    root: Path,
    identity: QueryStateBundleIdentity,
) -> None:
    actual = {
        "actor": _tree_sha256(root / "actor"),
        "processor": _tree_sha256(root / "processor"),
        "direct_state": _sha256_file(root / "direct_state.pt"),
        "bundle": _sha256_file(root / "bundle.json"),
    }
    if actual != dict(identity.owner_sha256):
        raise ValueError("Query-State bundle owner hashes changed during production load")


def _validate_actor_config_contract(config: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
    vocab_size = config.get("vocab_size")
    action_token_ids = config.get("nimloth_action_token_ids")
    text_config = config.get("text_config")
    hidden_size = config.get("hidden_size")
    if hidden_size is None and isinstance(text_config, Mapping):
        hidden_size = text_config.get("hidden_size")
    if (
        hidden_size != 2048
        or config.get("nimloth_latent_token_count") != 16
        or config.get("nimloth_latent_query_mode") != "inject"
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
        or not isinstance(action_token_ids, (list, tuple))
        or len(action_token_ids) != 8
        or any(isinstance(value, bool) or not isinstance(value, int) for value in action_token_ids)
        or len(set(action_token_ids)) != 8
    ):
        raise ValueError("Query-State full actor K16/hidden/vocab/action-token contract mismatch")
    return vocab_size, tuple(action_token_ids)


def _load_query_state_bundle_owners(
    root: Path,
    identity: QueryStateBundleIdentity,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    attention_implementation: str,
    max_length: int,
) -> _LoadedQueryStateBundleOwners:
    """Load every production owner from its validated bundle path exactly once."""

    if not isinstance(device, torch.device):
        raise TypeError("Query-State production device must be an explicit torch.device")
    if model_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("Query-State production dtype must be float32 or bfloat16")
    if attention_implementation not in {"sdpa", "flash_attention_2"}:
        raise ValueError("Query-State attention implementation is unsupported")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("Query-State max length must be positive")

    actor_path = root / "actor"
    processor_path = root / "processor"
    direct_path = root / "direct_state.pt"
    _require_bundle_owner_hashes(root, identity)
    try:
        actor_config = _read_json_mapping(
            actor_path / "config.json",
            owner="full Qwen actor",
        )
        vocab_size, configured_action_ids = _validate_actor_config_contract(actor_config)
        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("Query-State bundle processor has no tokenizer")
        token_id_map = special_token_ids(tokenizer, latent_token_count=16)
        latent_ids = tuple(token_id_map[token] for token in latent_state_tokens(16))
        action_ids = tuple(
            token_id_map[token] for token in LatentActionTokens().action_tokens
        )
        if (
            len(tokenizer) != vocab_size
            or len(latent_ids) != 16
            or len(set(token_id_map.values())) != len(token_id_map)
            or action_ids != configured_action_ids
            or max(token_id_map.values()) >= vocab_size
        ):
            raise ValueError("Query-State processor token/vocabulary/K16 identity mismatch")

        actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            actor_path,
            torch_dtype=model_dtype,
            attn_implementation=attention_implementation,
            trust_remote_code=True,
            local_files_only=True,
        )
        if not isinstance(actor, Qwen2_5_VLForConditionalGeneration):
            raise TypeError("Query-State bundle actor loader did not return full Qwen2.5-VL")
        loaded_config = getattr(actor, "config", None)
        if (
            loaded_config is None
            or int(getattr(loaded_config, "vocab_size", -1)) != vocab_size
            or qwen_hidden_size(loaded_config) != 2048
            or int(getattr(loaded_config, "nimloth_latent_token_count", -1)) != 16
            or getattr(loaded_config, "nimloth_latent_query_mode", None) != "inject"
            or tuple(getattr(loaded_config, "nimloth_action_token_ids", ()))
            != configured_action_ids
        ):
            raise ValueError("loaded Query-State actor changed K16/vocabulary identity")
        input_embeddings = actor.get_input_embeddings()
        output_embeddings = actor.get_output_embeddings()
        if (
            input_embeddings is None
            or output_embeddings is None
            or tuple(input_embeddings.weight.shape) != (vocab_size, 2048)
            or tuple(output_embeddings.weight.shape) != (vocab_size, 2048)
        ):
            raise ValueError("Query-State actor embedding/LM-head vocabulary shape mismatch")

        resume_identity = QueryStateResumeIdentity(**dict(identity.source_identity))
        projector, artifact_metadata = load_direct_query_state_artifact(
            direct_path,
            expected_source_identity=resume_identity,
        )
        if artifact_metadata != {"bundle_role": "direct_state_only"}:
            raise ValueError("Query-State direct-state owner metadata mismatch")

        actor.to(device=device)
        projector.to(device=device, dtype=model_dtype)
        actor.eval().requires_grad_(False)
        projector.eval().requires_grad_(False)
        validate_frozen_query_state_producer(actor=actor, projector=projector)
        floating = [parameter for parameter in actor.parameters() if parameter.is_floating_point()]
        if (
            not floating
            or any(parameter.device != device for parameter in floating)
            or any(parameter.dtype != model_dtype for parameter in floating)
        ):
            raise ValueError("Query-State actor device/dtype placement mismatch")
        input_builder = Qwen25VLInputBuilder(
            processor=processor,
            max_length=max_length,
            latent_token_count=16,
            mask_latent_query_labels=True,
        )
        return _LoadedQueryStateBundleOwners(
            actor=actor,
            processor=processor,
            input_builder=input_builder,
            projector=projector,
            token_id_map=token_id_map,
            device=device,
        )
    finally:
        _require_bundle_owner_hashes(root, identity)


def build_query_state_reconstruction_cache(
    output: str | Path,
    *,
    bundle_path: str | Path,
    source: QueryStateSourceContract,
    selection_role: str,
    device: torch.device,
    model_dtype: torch.dtype,
    attention_implementation: str,
    max_length: int,
    extraction_batch_size: int,
    state_dtype: str,
    shard_size: int,
) -> QueryStateCacheManifest:
    """Build only from internally loaded bundle owners and audited pre-RL rows.

    Callers provide runtime placement but cannot inject actor, processor, direct
    state, extracted tensors, or row provenance.
    """

    destination = Path(output)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Query-State cache output already exists: {destination}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Query-State cache temporary path already exists: {temporary}")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 1
        or isinstance(extraction_batch_size, bool)
        or not isinstance(extraction_batch_size, int)
        or extraction_batch_size < 1
    ):
        raise ValueError("Query-State max length and extraction batch size must be positive")
    bundle = validate_query_state_bundle(bundle_path)
    source_rows, source_audit, source_identity = _index_audited_sources(source)
    resume_identity = QueryStateResumeIdentity(**dict(bundle.source_identity))
    if resume_identity.source_manifest_identity != source_identity:
        raise ValueError("Query-State bundle/source manifest identity mismatch")
    selected, _selection_identity, _selection = _selection_for_role(
        source_rows,
        source_audit,
        source=source,
        selection_role=selection_role,
    )

    loaded = _load_query_state_bundle_owners(
        Path(bundle.bundle_path),
        bundle,
        device=device,
        model_dtype=model_dtype,
        attention_implementation=attention_implementation,
        max_length=max_length,
    )
    rebound = validate_query_state_bundle(bundle.bundle_path)
    if rebound != bundle:
        raise ValueError("Query-State bundle owners changed while loading")
    validate_frozen_query_state_producer(
        actor=loaded.actor,
        projector=loaded.projector,
    )

    records: list[_QueryStateCacheRecord] = []
    for start in range(0, len(selected), extraction_batch_size):
        rows = selected[start : start + extraction_batch_size]
        extracted = _extract_canonical_query_states(
            rows,
            processor=loaded.processor,
            input_builder=loaded.input_builder,
            actor=loaded.actor,
            projector=loaded.projector,
            token_id_map=loaded.token_id_map,
            device=loaded.device,
            max_length=max_length,
        )
        if extracted.state.shape[0] != len(rows) or len(extracted.rendered) != len(rows):
            raise ValueError("Query-State extraction row/state count mismatch")
        records.extend(
            _QueryStateCacheRecord(
                row=row,
                state=state,
                provenance=_rendered_row_provenance(rendered),
            )
            for row, state, rendered in zip(
                rows,
                extracted.state,
                extracted.rendered,
                strict=True,
            )
        )
    return _write_query_state_reconstruction_cache(
        output,
        bundle=bundle,
        source=source,
        selection_role=selection_role,
        records=records,
        state_dtype=state_dtype,
        shard_size=shard_size,
    )


def _bundle_manifest_payload(bundle: QueryStateBundleIdentity) -> dict[str, Any]:
    return {
        "schema": bundle.bundle_schema,
        "path": bundle.bundle_path,
        "source_commit": bundle.source_commit,
        "source_identity": dict(bundle.source_identity),
        "owners": dict(bundle.owners),
        "owner_sha256": dict(bundle.owner_sha256),
        "processor_identity": bundle.processor_identity,
        "tokenizer_identity": bundle.tokenizer_identity,
        "template_identity": bundle.template_identity,
        "checkpoint_identity": bundle.checkpoint_identity,
        "human_gate_receipt_sha256": bundle.human_gate_receipt_sha256,
    }


def _write_query_state_reconstruction_cache(
    output: str | Path,
    *,
    bundle: QueryStateBundleIdentity,
    source: QueryStateSourceContract,
    selection_role: str,
    records: Sequence[_QueryStateCacheRecord],
    state_dtype: str,
    shard_size: int,
) -> QueryStateCacheManifest:
    """Atomically publish owner-produced K16 shards without overwrite semantics."""

    destination = Path(output)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Query-State cache output already exists: {destination}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"Query-State cache temporary path already exists: {temporary}")
    if not isinstance(bundle, QueryStateBundleIdentity):
        raise TypeError("Query-State cache requires a validated bundle identity")
    if not isinstance(source, QueryStateSourceContract):
        raise TypeError("Query-State cache requires an audited source contract")
    source_rows, source_audit, source_identity = _index_audited_sources(source)
    bundle_source = QueryStateResumeIdentity(**dict(bundle.source_identity))
    if bundle_source.source_manifest_identity != source_identity:
        raise ValueError("Query-State cache bundle/source manifest identity mismatch")
    selected, selection_identity, selection = _selection_for_role(
        source_rows,
        source_audit,
        source=source,
        selection_role=selection_role,
    )
    split = selected[0].split
    split_identity = _split_identity(
        selected,
        source_identity=source_identity,
        split=split,
        selection_role=selection_role,
        selection_identity=selection_identity,
    )
    if state_dtype not in _DTYPE_BY_NAME:
        raise ValueError("unsupported Query-State cache state dtype")
    if isinstance(shard_size, bool) or not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("Query-State cache shard size must be positive")
    record_items = tuple(records)
    if not record_items:
        raise ValueError("Query-State cache requires at least one record")
    if tuple(record.row.identity for record in record_items) != tuple(
        row.identity for row in selected
    ):
        raise ValueError(
            "Query-State cache records must exactly cover the audited selection role"
        )
    validated: list[tuple[torch.Tensor, dict[str, Any]]] = []
    seen: set[str] = set()
    for record in record_items:
        state, metadata = _validate_record(record, split=split)
        if metadata["row_identity"] in seen:
            raise ValueError("duplicate Query-State row identity")
        seen.add(metadata["row_identity"])
        validated.append((state, metadata))
    temporary.mkdir(parents=True)
    try:
        descriptors: list[dict[str, Any]] = []
        target_dtype = _DTYPE_BY_NAME[state_dtype]
        for shard_index, start in enumerate(range(0, len(validated), shard_size)):
            chunk = validated[start : start + shard_size]
            states = torch.stack([state for state, _ in chunk], dim=0).to(
                dtype=target_dtype,
                device="cpu",
            ).contiguous()
            rows_payload = [metadata for _, metadata in chunk]
            payload = {
                "schema": _QUERY_STATE_RECONSTRUCTION_SHARD_SCHEMA,
                "state": states,
                "rows": rows_payload,
            }
            file_name = f"shard_{shard_index:05d}.pt"
            shard_path = temporary / file_name
            with shard_path.open("xb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            descriptor = {
                "file": file_name,
                "count": len(chunk),
                "start": start,
                "stop": start + len(chunk),
                "sha256": _sha256_file(shard_path),
                "row_metadata_sha256": _sha256_bytes(
                    _canonical_json({"rows": rows_payload})
                ),
                "state_dtype": state_dtype,
                "state_shape": list(_STATE_SHAPE),
            }
            loaded = torch.load(shard_path, map_location="cpu", weights_only=False)
            _validate_shard_payload(loaded, descriptor, expected_dtype=target_dtype)
            descriptors.append(descriptor)

        manifest: dict[str, Any] = {
            "schema": QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
            "version": 1,
            "count": len(validated),
            "state_shape": list(_STATE_SHAPE),
            "state_ordering": _STATE_ORDERING,
            "state_dtype": state_dtype,
            "bundle": _bundle_manifest_payload(bundle),
            "source_jsonl": {
                "train": {
                    "path": str(Path(source.data.train_jsonl).resolve()),
                    "sha256": source.data.train_sha256,
                    "split": source.data.train_split,
                },
                "validation": {
                    "path": str(Path(source.data.validation_jsonl).resolve()),
                    "sha256": source.data.validation_sha256,
                    "split": source.data.validation_split,
                },
                "source_manifest_identity": source.source_manifest_identity,
            },
            "split": {"name": split, "identity": split_identity},
            "selection": selection,
            "row_set_identity": _row_set_identity_from_shards(descriptors),
            "shards": descriptors,
        }
        manifest["cache_fingerprint"] = _manifest_fingerprint(manifest)
        parsed_manifest = _parse_manifest(manifest)
        _write_bytes_fsynced(
            temporary / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
        )
        _fsync_directory(temporary)
        _publish_directory_noreplace(temporary, destination)
        _fsync_directory(destination.parent)
        return parsed_manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _parse_manifest(raw: Mapping[str, Any]) -> QueryStateCacheManifest:
    required = {
        "schema",
        "version",
        "count",
        "state_shape",
        "state_ordering",
        "state_dtype",
        "bundle",
        "source_jsonl",
        "split",
        "selection",
        "row_set_identity",
        "shards",
        "cache_fingerprint",
    }
    if set(raw) != required or raw.get("schema") != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA:
        raise ValueError("unsupported legacy/schema cache; direct Query-State cache required")
    if raw.get("version") != 1:
        raise ValueError("unsupported Query-State cache manifest version")
    count = raw.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Query-State cache manifest count is invalid")
    if tuple(raw.get("state_shape", ())) != _STATE_SHAPE:
        raise ValueError("Query-State cache manifest shape must preserve K16 [16,1024]")
    if raw.get("state_ordering") != _STATE_ORDERING:
        raise ValueError("Query-State cache state ordering mismatch")
    dtype_name = raw.get("state_dtype")
    if dtype_name not in _DTYPE_BY_NAME:
        raise ValueError("Query-State cache manifest dtype is invalid")
    if not _is_sha256(raw.get("row_set_identity")) or not _is_sha256(raw.get("cache_fingerprint")):
        raise ValueError("Query-State cache manifest identity is invalid")
    bundle = raw.get("bundle")
    source = raw.get("source_jsonl")
    split = raw.get("split")
    selection = raw.get("selection")
    bundle_fields = {
        "schema",
        "path",
        "source_commit",
        "source_identity",
        "owners",
        "owner_sha256",
        "processor_identity",
        "tokenizer_identity",
        "template_identity",
        "checkpoint_identity",
        "human_gate_receipt_sha256",
    }
    owner_hashes = bundle.get("owner_sha256") if isinstance(bundle, dict) else None
    source_identity_valid = False
    if isinstance(bundle, dict) and isinstance(bundle.get("source_identity"), dict):
        try:
            parsed_source_identity = QueryStateResumeIdentity(**bundle["source_identity"])
        except (TypeError, ValueError):
            pass
        else:
            source_identity_valid = (
                parsed_source_identity.experiment_mode == "formal"
                and parsed_source_identity.source_commit == bundle.get("source_commit")
            )
    source_entries_valid = False
    if isinstance(source, dict) and set(source) == {
        "train", "validation", "source_manifest_identity"
    }:
        entries = (source.get("train"), source.get("validation"))
        source_entries_valid = all(
            isinstance(entry, dict)
            and set(entry) == {"path", "sha256", "split"}
            and isinstance(entry.get("path"), str)
            and Path(entry["path"]).is_absolute()
            and _is_sha256(entry.get("sha256"))
            and isinstance(entry.get("split"), str)
            and bool(entry["split"])
            for entry in entries
        ) and _is_sha256(source.get("source_manifest_identity"))
    if (
        not isinstance(bundle, dict)
        or set(bundle) != bundle_fields
        or bundle.get("schema") != QUERY_STATE_DEPLOYABLE_BUNDLE_SCHEMA
        or not isinstance(bundle.get("path"), str)
        or not Path(bundle["path"]).is_absolute()
        or not isinstance(bundle.get("source_commit"), str)
        or len(bundle["source_commit"]) != 40
        or set(bundle["source_commit"]) - _HEX
        or not source_identity_valid
        or bundle.get("owners") != _EXPECTED_OWNERS
        or not isinstance(owner_hashes, dict)
        or set(owner_hashes) != {"actor", "processor", "direct_state", "bundle"}
        or not all(_is_sha256(value) for value in owner_hashes.values())
        or not all(
            _is_sha256(bundle.get(field))
            for field in (
                "processor_identity",
                "tokenizer_identity",
                "template_identity",
                "checkpoint_identity",
                "human_gate_receipt_sha256",
            )
        )
        or not source_entries_valid
        or not isinstance(split, dict)
        or set(split) != {"name", "identity"}
        or not isinstance(split.get("name"), str)
        or not split["name"]
        or not _is_sha256(split.get("identity"))
        or not isinstance(selection, dict)
        or set(selection) != {
            "role", "raw_row_count", "selected_row_count", "excluded_row_count",
            "source_audit", "identity",
        }
        or selection.get("role") not in _SELECTION_ROLES
        or any(
            isinstance(selection.get(field), bool)
            or not isinstance(selection.get(field), int)
            or selection[field] < 0
            for field in ("raw_row_count", "selected_row_count", "excluded_row_count")
        )
        or selection.get("selected_row_count") != count
        or selection.get("raw_row_count")
        != selection.get("selected_row_count") + selection.get("excluded_row_count")
        or not isinstance(selection.get("source_audit"), dict)
        or not _is_sha256(selection.get("identity"))
    ):
        raise ValueError("Query-State cache bundle/source/split/selection identity is invalid")
    raw_shards = raw.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Query-State cache manifest shards are absent")
    shards: list[QueryStateCacheShard] = []
    shard_files: set[str] = set()
    cursor = 0
    for item in raw_shards:
        if not isinstance(item, dict) or set(item) != {
            "file",
            "count",
            "start",
            "stop",
            "sha256",
            "row_metadata_sha256",
            "state_dtype",
            "state_shape",
        }:
            raise ValueError("Query-State cache shard descriptor is invalid")
        if (
            not isinstance(item["file"], str)
            or Path(item["file"]).name != item["file"]
            or item["start"] != cursor
            or isinstance(item["count"], bool)
            or not isinstance(item["count"], int)
            or item["count"] < 1
            or item["stop"] != cursor + item["count"]
            or not _is_sha256(item["sha256"])
            or not _is_sha256(item["row_metadata_sha256"])
            or item["state_dtype"] != dtype_name
            or tuple(item["state_shape"]) != _STATE_SHAPE
        ):
            raise ValueError("Query-State cache shard count/dtype/shape identity mismatch")
        if item["file"] in shard_files:
            raise ValueError("Query-State cache manifest repeats a shard file")
        shard_files.add(item["file"])
        shards.append(QueryStateCacheShard(
            file=item["file"],
            count=item["count"],
            start=item["start"],
            stop=item["stop"],
            sha256=item["sha256"],
            row_metadata_sha256=item["row_metadata_sha256"],
            state_dtype=item["state_dtype"],
            state_shape=tuple(item["state_shape"]),
        ))
        cursor = item["stop"]
    if cursor != count:
        raise ValueError("Query-State cache manifest count disagrees with shards")
    if _row_set_identity_from_shards(raw_shards) != raw["row_set_identity"]:
        raise ValueError("Query-State cache manifest row-set identity mismatch")
    return QueryStateCacheManifest(
        schema=raw["schema"],
        version=raw["version"],
        count=count,
        state_shape=tuple(raw["state_shape"]),
        state_ordering=raw["state_ordering"],
        state_dtype=dtype_name,
        bundle=dict(bundle),
        source_jsonl=dict(source),
        split=dict(split),
        selection=dict(selection),
        row_set_identity=raw["row_set_identity"],
        shards=tuple(shards),
        cache_fingerprint=raw["cache_fingerprint"],
    )


class QueryStateReconstructionCacheDataset:
    """Strict lazy reader; each shard is hash-checked on first access."""

    def __init__(self, root: str | Path) -> None:
        supplied = Path(root)
        if supplied.is_symlink():
            raise ValueError("Query-State cache root must not be a symlink")
        self.root = supplied.resolve()
        manifest_path = self.root / "manifest.json"
        if not self.root.is_dir() or not manifest_path.is_file():
            raise ValueError("Query-State cache manifest is missing")
        self._raw_manifest = _read_json_mapping(
            manifest_path,
            owner="Query-State cache manifest",
        )
        self.manifest = _parse_manifest(self._raw_manifest)
        if _manifest_fingerprint(self._raw_manifest) != self.manifest.cache_fingerprint:
            raise ValueError("Query-State cache manifest fingerprint mismatch")
        live_bundle = validate_query_state_bundle(self.manifest.bundle["path"])
        if _bundle_manifest_payload(live_bundle) != self.manifest.bundle:
            raise ValueError("Query-State cache live bundle owner identity mismatch")
        source_raw = self.manifest.source_jsonl
        source = QueryStateSourceContract(
            data=QueryStateSourceData(
                train_jsonl=source_raw["train"]["path"],
                train_sha256=source_raw["train"]["sha256"],
                validation_jsonl=source_raw["validation"]["path"],
                validation_sha256=source_raw["validation"]["sha256"],
                train_split=source_raw["train"]["split"],
                validation_split=source_raw["validation"]["split"],
            ),
            source_manifest_identity=source_raw["source_manifest_identity"],
        )
        rows, audit, source_identity = _index_audited_sources(source)
        bundle_source = QueryStateResumeIdentity(**dict(live_bundle.source_identity))
        if bundle_source.source_manifest_identity != source_identity:
            raise ValueError("Query-State cache source/bundle resume identity mismatch")
        selected, selection_identity, live_selection = _selection_for_role(
            rows,
            audit,
            source=source,
            selection_role=str(self.manifest.selection["role"]),
        )
        if _canonical_json(live_selection) != _canonical_json(self.manifest.selection):
            raise ValueError("Query-State cache live audited selection identity mismatch")
        if (
            selected[0].split != self.manifest.split["name"]
            or _split_identity(
                selected,
                source_identity=source_identity,
                split=self.manifest.split["name"],
                selection_role=str(self.manifest.selection["role"]),
                selection_identity=selection_identity,
            ) != self.manifest.split["identity"]
        ):
            raise ValueError("Query-State cache live split/selection identity mismatch")
        self._source_rows = {row.identity: row for row in selected}
        self._ordered_source_row_ids = tuple(row.identity for row in selected)
        if len(self._source_rows) != self.manifest.count:
            raise ValueError("Query-State cache row set differs from live audited source")
        self._loaded: dict[int, tuple[torch.Tensor, list[dict[str, Any]]]] = {}

    def __len__(self) -> int:
        return self.manifest.count

    @property
    def cache_fingerprint(self) -> str:
        return self.manifest.cache_fingerprint

    def _load_shard(self, shard_index: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        cached = self._loaded.get(shard_index)
        if cached is not None:
            return cached
        if _manifest_fingerprint(self._raw_manifest) != self.manifest.cache_fingerprint:
            raise ValueError("Query-State cache manifest fingerprint mismatch")
        descriptor = self.manifest.shards[shard_index]
        path = self.root / descriptor.file
        if not path.is_file() or path.is_symlink():
            raise ValueError("Query-State cache shard file is missing")
        if _sha256_file(path) != descriptor.sha256:
            raise ValueError("Query-State cache shard SHA256/hash mismatch")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError("Query-State cache shard payload is unreadable") from error
        state, rows = _validate_shard_payload(
            payload,
            asdict(descriptor),
            expected_dtype=_DTYPE_BY_NAME[self.manifest.state_dtype],
        )
        expected_ids = self._ordered_source_row_ids[
            descriptor.start : descriptor.stop
        ]
        actual_ids = tuple(metadata["row_identity"] for metadata in rows)
        if actual_ids != expected_ids:
            raise ValueError("Query-State cache shard row order/coverage mismatch")
        for metadata in rows:
            source_row = self._source_rows.get(metadata["row_identity"])
            if source_row is None:
                raise ValueError("Query-State cache row is absent from live audited source")
            expected = _metadata_for_source_row(source_row)
            if any(metadata.get(name) != value for name, value in expected.items()):
                raise ValueError("Query-State cache row/source provenance identity mismatch")
            if metadata.get("response_source") != "archived" or any(
                not _is_sha256(metadata.get(name))
                for name in _PROVENANCE_IDENTITY_FIELDS
            ):
                raise ValueError("Query-State cache rendered provenance identity mismatch")
        self._loaded[shard_index] = (state, rows)
        return state, rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Query-State cache index must be an integer")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for shard_index, descriptor in enumerate(self.manifest.shards):
            if descriptor.start <= index < descriptor.stop:
                state, rows = self._load_shard(shard_index)
                local = index - descriptor.start
                return {
                    "state": state[local].detach().clone(),
                    **dict(rows[local]),
                }
        raise RuntimeError("Query-State cache index is not owned by any shard")


__all__ = [
    "QUERY_STATE_CACHE_SELECTION_ALL_TRAIN",
    "QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION",
    "QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA",
    "QueryStateBundleIdentity",
    "QueryStateCacheManifest",
    "QueryStateCacheShard",
    "QueryStateReconstructionCacheDataset",
    "QueryStateSourceContract",
    "QueryStateSourceData",
    "build_query_state_reconstruction_cache",
    "require_real_archived_response",
    "validate_canonical_query_state",
    "validate_frozen_query_state_producer",
    "validate_query_state_bundle",
]
