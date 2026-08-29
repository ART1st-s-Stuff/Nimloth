"""Fail-closed preparation and evidence boundaries for Query-State GPU smoke.

The helpers in this module do not initialize a process group, load model
weights, choose resources, or submit work.  They bind exact archived rows,
mechanics-only evidence, and immutable fresh/resume phase ownership around the
existing Query-State update/checkpoint primitives.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from nimloth.training.sft1.query_state import (
    SFT1QueryStateTrainingRoot,
    query_state_trainable_parameter_groups,
)
from nimloth.training.sft1.query_state_data import (
    QueryStateRenderedRow,
    render_query_state_row,
)
from nimloth.training.sft1.query_state_smoke_config import (
    QueryStateSmokeConfig,
    QueryStateSmokeRowDescriptor,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row, SFT1V2RowAudit


_SMOKE_EVIDENCE_KIND = "production_path_checkpoint_resume_smoke_not_model_quality_evidence"
_SOURCE_MANIFEST_SCHEMA = "nimloth_sft1_query_state_source_manifest_v1"
_PHASE_RECORD_SCHEMA = "nimloth_sft1_query_state_smoke_phase_v1"
_CONFIG_RECORD_SCHEMA = "nimloth_sft1_query_state_smoke_resolved_config_v1"
_COMPLETE_RECORD_SCHEMA = "nimloth_sft1_query_state_smoke_complete_v1"


@dataclass(frozen=True)
class QueryStateSmokePhaseContext:
    config: QueryStateSmokeConfig
    phase: str
    rank: int
    world_size: int
    process_identity: str
    approved_command_manifest: str
    phase_root: Path
    checkpoint_path: Path
    previous_checkpoint_path: Path | None
    previous_process_identity: str | None
    expected_global_step: int
    row_descriptor: QueryStateSmokeRowDescriptor


@dataclass(frozen=True)
class QueryStateSmokePhaseOutcome:
    global_step: int
    checkpoint_path: Path
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class QueryStateSmokeParameterRecord:
    name: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    requires_grad: bool
    optimizer_group: str | None


@dataclass(frozen=True)
class QueryStateSmokeInventoryEvidence:
    parameters: tuple[QueryStateSmokeParameterRecord, ...]
    trainable_parameter_count: int
    trainable_numel: int
    frozen_parameter_count: int
    frozen_numel: int
    optimizer_group_parameter_names: Mapping[str, tuple[str, ...]]
    embedding_lm_head_tied: bool
    visual_trainable_names: tuple[str, ...]
    query_adapter_names: tuple[str, ...]
    direct_state_names: tuple[str, ...]


@dataclass(frozen=True)
class QueryStateSmokeRuntimeFingerprint:
    trainable_model_sha256: str
    optimizer_sha256: str
    scheduler_sha256: str
    rng_sha256: str
    identity: str


@dataclass(frozen=True)
class QueryStateSmokeGroupGradientEvidence:
    group_norms: Mapping[str, float]
    all_finite_nonzero: bool


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"immutable Query-State smoke JSON exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_query_state_source_manifest_identity(
    rows: Sequence[SFT1V2Early4Row],
    audit: SFT1V2RowAudit,
) -> str:
    """Bind the complete ordered row index and audited source/split semantics."""

    if not isinstance(audit, SFT1V2RowAudit):
        raise TypeError("Query-State source manifest requires SFT1V2RowAudit")
    if not rows:
        raise ValueError("Query-State source manifest requires indexed rows")
    ordinals = [row.ordinal for row in rows]
    if any(not isinstance(row, SFT1V2Early4Row) for row in rows):
        raise TypeError("Query-State source manifest rows use the wrong schema")
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("Query-State source manifest ordinals must be unique")
    payload = {
        "schema": _SOURCE_MANIFEST_SCHEMA,
        "audit": asdict(audit),
        "ordered_rows": [
            {
                "ordinal": row.ordinal,
                "identity": row.identity,
                "split": row.split,
                "record_id": row.record_id,
                "step_index": row.step_index,
                "original_image_sha256": row.original_image_sha256,
            }
            for row in rows
        ],
    }
    return _sha256_json(payload)


def _descriptor_for_rendered(
    rendered: QueryStateRenderedRow,
    *,
    phase: str,
    rank: int,
) -> QueryStateSmokeRowDescriptor:
    labels = rendered.encoded_tensors.get("labels")
    input_ids = rendered.encoded_tensors.get("input_ids")
    if (
        not isinstance(labels, torch.Tensor)
        or not isinstance(input_ids, torch.Tensor)
        or labels.shape != input_ids.shape
        or labels.ndim != 1
    ):
        raise ValueError("Query-State smoke rendered row token/label tensors are invalid")
    row = rendered.row
    return QueryStateSmokeRowDescriptor(
        phase=phase,
        rank=rank,
        ordinal=row.ordinal,
        record_id=row.record_id,
        step_index=row.step_index,
        row_identity=row.identity,
        original_image_sha256=row.original_image_sha256,
        rendered_token_count=int(input_ids.numel()),
        valid_lm_token_count=int((labels != -100).sum().item()),
        split=row.split,
    )


def verify_query_state_smoke_rows(
    config: QueryStateSmokeConfig,
    *,
    rows: Sequence[SFT1V2Early4Row],
    processor: Any,
) -> tuple[QueryStateRenderedRow, ...]:
    """Re-render exactly the locked descriptors; never select rows dynamically."""

    if not isinstance(config, QueryStateSmokeConfig) or not config.launch_locked:
        raise PermissionError("exact Query-State smoke row verification requires launch-locked config")
    descriptors = config.data.smoke_rows
    by_ordinal: dict[int, SFT1V2Early4Row] = {}
    for row in rows:
        if not isinstance(row, SFT1V2Early4Row):
            raise TypeError("Query-State smoke rows must be audited early-4 rows")
        if row.ordinal in by_ordinal:
            raise ValueError("Query-State smoke input rows contain duplicate ordinals")
        by_ordinal[row.ordinal] = row
    expected_ordinals = {descriptor.ordinal for descriptor in descriptors}
    if set(by_ordinal) != expected_ordinals:
        raise ValueError("Query-State smoke exact row ordinal set mismatch")
    max_length = config.runtime.max_sequence_length
    if not isinstance(max_length, int):
        raise ValueError("Query-State smoke max sequence length remains unresolved")
    verified: list[QueryStateRenderedRow] = []
    for descriptor in descriptors:
        row = by_ordinal[descriptor.ordinal]
        rendered = render_query_state_row(
            row,
            processor=processor,
            max_length=max_length,
        )
        actual = _descriptor_for_rendered(
            rendered,
            phase=descriptor.phase,
            rank=descriptor.rank,
        )
        if actual != descriptor:
            changed = [
                name
                for name in QueryStateSmokeRowDescriptor.__dataclass_fields__
                if getattr(actual, name) != getattr(descriptor, name)
            ]
            raise ValueError(
                "Query-State smoke descriptor mismatch: " + (changed[0] if changed else "unknown")
            )
        verified.append(rendered)
    return tuple(verified)


def build_query_state_inventory_evidence(
    root: SFT1QueryStateTrainingRoot,
    optimizer: torch.optim.Optimizer | None = None,
) -> QueryStateSmokeInventoryEvidence:
    """Record exhaustive pre-wrap ownership without loading or guessing names."""

    if not isinstance(root, SFT1QueryStateTrainingRoot):
        raise TypeError("Query-State smoke inventory requires the complete training root")
    inventory = root.assert_trainable_contract()
    named = dict(root.named_parameters())
    by_id = {id(parameter): name for name, parameter in named.items()}
    group_names: dict[str, tuple[str, ...]] = {}
    optimizer_seen: set[int] = set()
    if optimizer is None:
        raw_groups = tuple(
            {
                "group_name": group.name,
                "params": group.parameters,
            }
            for group in query_state_trainable_parameter_groups(root)
        )
    else:
        raw_groups = tuple(optimizer.param_groups)
    for group in raw_groups:
        group_name = group.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            raise ValueError("Query-State smoke optimizer group name is absent")
        if group_name in group_names:
            raise ValueError("Query-State smoke optimizer group name is duplicated")
        names: list[str] = []
        for parameter in group["params"]:
            identity = id(parameter)
            if identity in optimizer_seen or identity not in by_id:
                raise ValueError("Query-State smoke optimizer ownership is duplicate/unowned")
            optimizer_seen.add(identity)
            names.append(by_id[identity])
        group_names[group_name] = tuple(names)
    trainable_ids = {id(parameter) for parameter in root.parameters() if parameter.requires_grad}
    if optimizer_seen != trainable_ids or tuple(group_names) != ("language", "direct_state"):
        raise ValueError("Query-State smoke optimizer inventory is not exhaustive/disjoint")

    parameter_group = {
        name: group_name
        for group_name, names in group_names.items()
        for name in names
    }
    records = tuple(
        QueryStateSmokeParameterRecord(
            name=name,
            shape=tuple(int(value) for value in parameter.shape),
            dtype=str(parameter.dtype),
            numel=int(parameter.numel()),
            requires_grad=bool(parameter.requires_grad),
            optimizer_group=parameter_group.get(name),
        )
        for name, parameter in named.items()
    )
    trainable = tuple(record for record in records if record.requires_grad)
    frozen = tuple(record for record in records if not record.requires_grad)

    # ``remove_duplicate=False`` exposes actual tying if supported.  Tying is
    # evidence, not an assumed requirement; the exact result is persisted.
    all_named = list(root.named_parameters(remove_duplicate=False))
    embedding_ids = {
        id(parameter)
        for name, parameter in all_named
        if "embed" in name.lower()
    }
    lm_head_ids = {
        id(parameter)
        for name, parameter in all_named
        if name.startswith("backbone.lm_head.") or ".lm_head." in name
    }
    tied = bool(embedding_ids & lm_head_ids)
    visual_trainable = tuple(
        name for name in inventory.visual_frozen if named[name].requires_grad
    )
    query_adapter = tuple(name for name in named if "query_embedding_adapter" in name)
    direct_names = tuple(inventory.direct_state_trainable)
    if visual_trainable or query_adapter or direct_names != (
        "objective.projector.linear.weight",
    ):
        raise ValueError("Query-State smoke state/vision/query ownership changed")
    return QueryStateSmokeInventoryEvidence(
        parameters=records,
        trainable_parameter_count=len(trainable),
        trainable_numel=sum(record.numel for record in trainable),
        frozen_parameter_count=len(frozen),
        frozen_numel=sum(record.numel for record in frozen),
        optimizer_group_parameter_names=group_names,
        embedding_lm_head_tied=tied,
        visual_trainable_names=visual_trainable,
        query_adapter_names=query_adapter,
        direct_state_names=direct_names,
    )


def collect_query_state_group_gradient_evidence(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
) -> QueryStateSmokeGroupGradientEvidence:
    """All-reduce detached norm/status scalars; never synchronize gradients.

    A local non-finite shard must not raise before its peers enter the same
    collectives.  Otherwise one rank would exit while the remaining ranks hang
    in the norm reduction.  Reduce the finite/saw flags first, then fail
    coherently on every rank.
    """

    names = tuple(group.get("group_name") for group in optimizer.param_groups)
    if names != ("language", "direct_state"):
        raise ValueError("Query-State smoke requires language/direct_state gradient evidence")
    values: dict[str, float] = {}
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    for group, name in zip(optimizer.param_groups, names, strict=True):
        if not isinstance(name, str):
            raise ValueError("Query-State smoke gradient group identity is invalid")
        square = torch.zeros((), dtype=torch.float64, device=device)
        saw_gradient = False
        local_finite = True
        for parameter in group["params"]:
            gradient = parameter.grad
            if gradient is None:
                continue
            saw_gradient = True
            detached = gradient.detach()
            finite = bool(torch.isfinite(detached).all().item())
            local_finite = local_finite and finite
            if finite:
                square += detached.double().square().sum().to(device=device)
        saw = torch.tensor(
            1 if saw_gradient else 0,
            dtype=torch.long,
            device=device,
        )
        finite_flag = torch.tensor(
            1 if local_finite else 0,
            dtype=torch.long,
            device=device,
        )
        if distributed:
            torch.distributed.all_reduce(square, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(saw, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(
                finite_flag, op=torch.distributed.ReduceOp.MIN
            )
        norm = math.sqrt(float(square.item()))
        if int(finite_flag.item()) != 1:
            raise RuntimeError(f"Query-State smoke gradient is non-finite: {name}")
        if int(saw.item()) != 1 or not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(
                "Query-State smoke optimizer group has no finite nonzero "
                f"gradient: {name}"
            )
        values[name] = norm
    return QueryStateSmokeGroupGradientEvidence(
        group_norms=values,
        all_finite_nonzero=True,
    )


def _hash_value(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda entry: repr(entry)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"\0")
            for value_item in item:
                update(value_item)
        elif isinstance(item, np.ndarray):
            digest.update(b"numpy\0" + str(item.dtype).encode() + b"\0")
            digest.update(json.dumps(list(item.shape)).encode() + b"\0")
            digest.update(item.tobytes())
        else:
            digest.update(b"scalar\0")
            digest.update(pickle.dumps(item, protocol=5))

    update(value)
    return digest.hexdigest()


def build_query_state_runtime_fingerprint(
    root: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler_state: Mapping[str, Any],
) -> QueryStateSmokeRuntimeFingerprint:
    """Fingerprint model/optimizer/scheduler/RNG for exact fresh-process restore."""

    model = {
        name: parameter.detach()
        for name, parameter in root.named_parameters()
        if parameter.requires_grad
    }
    if not model or not any(name.endswith("objective.projector.linear.weight") for name in model):
        raise ValueError("Query-State smoke fingerprint lacks the direct state owner")
    model_hash = _hash_value(model)
    optimizer_hash = _hash_value(optimizer.state_dict())
    scheduler_hash = _hash_value(dict(scheduler_state))
    rng: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    rng_hash = _hash_value(rng)
    identity = _sha256_json(
        {
            "schema": "nimloth_sft1_query_state_runtime_fingerprint_v1",
            "model": model_hash,
            "optimizer": optimizer_hash,
            "scheduler": scheduler_hash,
            "rng": rng_hash,
        }
    )
    return QueryStateSmokeRuntimeFingerprint(
        trainable_model_sha256=model_hash,
        optimizer_sha256=optimizer_hash,
        scheduler_sha256=scheduler_hash,
        rng_sha256=rng_hash,
        identity=identity,
    )


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _coherent_precondition(error: Exception | None, *, world_size: int) -> None:
    if not _distributed():
        if error is not None:
            raise error
        return
    statuses: list[str | None] = [None] * world_size
    torch.distributed.all_gather_object(
        statuses,
        None if error is None else f"{type(error).__name__}: {error}",
    )
    failures = [status for status in statuses if status is not None]
    if failures:
        raise RuntimeError("Query-State smoke phase precondition failed: " + "; ".join(failures))


def _validate_outcome(
    outcome: QueryStateSmokePhaseOutcome,
    *,
    expected_step: int,
    checkpoint_path: Path,
    world_size: int,
    config_identity: str,
    approved_command_sha256: str,
) -> None:
    if not isinstance(outcome, QueryStateSmokePhaseOutcome):
        raise TypeError("Query-State smoke execute callback returned the wrong type")
    if outcome.global_step != expected_step or Path(outcome.checkpoint_path) != checkpoint_path:
        raise ValueError("Query-State smoke phase outcome step/checkpoint mismatch")
    marker = checkpoint_path / "COMPLETED"
    if not marker.is_file():
        raise ValueError("Query-State smoke phase checkpoint is incomplete")
    if (
        not isinstance(outcome.evidence, Mapping)
        or outcome.evidence.get("kind") != _SMOKE_EVIDENCE_KIND
        or outcome.evidence.get("config_identity") != config_identity
        or outcome.evidence.get("approved_command_sha256")
        != approved_command_sha256
    ):
        raise ValueError("Query-State smoke phase evidence identity is invalid")
    forbidden = {"automatic_model_quality_pass", "automatic_sft2_authorization"} & set(outcome.evidence)
    if forbidden:
        raise ValueError("Query-State smoke phase may not publish an automatic quality decision")
    per_rank = outcome.evidence.get("per_rank_mechanics")
    if (
        not isinstance(per_rank, Mapping)
        or set(per_rank) != {str(rank) for rank in range(world_size)}
        or any(not isinstance(value, Mapping) for value in per_rank.values())
    ):
        raise ValueError("Query-State smoke phase evidence does not cover every rank")


def orchestrate_query_state_smoke_phase(
    config: QueryStateSmokeConfig,
    *,
    phase: str,
    rank: int,
    world_size: int,
    process_identity: str,
    approved_command_manifest: str,
    execute: Callable[[QueryStateSmokePhaseContext], QueryStateSmokePhaseOutcome],
) -> QueryStateSmokePhaseOutcome:
    """Own immutable fresh/resume output boundaries around one phase callback."""

    if not isinstance(config, QueryStateSmokeConfig) or not config.launch_locked:
        raise PermissionError("Query-State smoke orchestration requires launch-locked config")
    if phase not in {"fresh", "resume"}:
        raise ValueError("Query-State smoke phase must be fresh or resume")
    if (
        isinstance(rank, bool) or not isinstance(rank, int)
        or isinstance(world_size, bool) or not isinstance(world_size, int)
        or world_size != config.runtime.world_size
        or not 0 <= rank < world_size
    ):
        raise ValueError("Query-State smoke rank/world-size mismatch")
    if not isinstance(process_identity, str) or not process_identity.strip():
        raise ValueError("Query-State smoke process identity must be non-empty")
    if (
        not isinstance(approved_command_manifest, str)
        or not approved_command_manifest.endswith("\n")
        or len(approved_command_manifest.splitlines()) != 2
        or any(not line.strip() for line in approved_command_manifest.splitlines())
        or hashlib.sha256(approved_command_manifest.encode()).hexdigest()
        != config.authorization.approved_command_sha256
    ):
        raise ValueError("Query-State smoke approved command manifest mismatch")
    if _distributed() and (
        torch.distributed.get_rank() != rank
        or torch.distributed.get_world_size() != world_size
    ):
        raise ValueError("Query-State smoke process-group identity mismatch")

    run_root = Path(config.output.run_root)
    fresh_root = run_root / config.output.fresh_child
    resume_root = run_root / config.output.resume_child
    process_record = run_root / "fresh_process.json"
    config_record = run_root / "resolved_config.json"
    fresh_checkpoint = fresh_root / config.checkpoint.fresh_checkpoint_name
    resume_checkpoint = resume_root / config.checkpoint.resume_checkpoint_name
    fresh_phase_record = fresh_root / "phase_complete.json"
    phase_root = fresh_root if phase == "fresh" else resume_root
    checkpoint = fresh_checkpoint if phase == "fresh" else resume_checkpoint

    precondition_error: Exception | None = None
    prior_process: str | None = None
    try:
        if phase == "fresh":
            if run_root.exists():
                raise FileExistsError(f"Query-State smoke fresh run root exists: {run_root}")
        else:
            if not (fresh_checkpoint / config.checkpoint.completion_marker).is_file():
                raise FileNotFoundError("Query-State smoke fresh checkpoint is incomplete")
            if resume_root.exists():
                raise FileExistsError(f"Query-State smoke resume child exists: {resume_root}")
            try:
                config_payload = json.loads(config_record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("Query-State smoke resolved config record is invalid") from error
            if (
                not isinstance(config_payload, dict)
                or set(config_payload) != {"schema", "config_identity", "config"}
                or config_payload["schema"] != _CONFIG_RECORD_SCHEMA
                or config_payload["config_identity"] != config.identity
                or _sha256_json(config_payload["config"]) != config.identity
            ):
                raise ValueError("Query-State smoke resolved config record mismatch")
            try:
                phase_payload = json.loads(
                    fresh_phase_record.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("Query-State smoke fresh phase record is invalid") from error
            phase_evidence = (
                phase_payload.get("evidence")
                if isinstance(phase_payload, dict)
                else None
            )
            if (
                not isinstance(phase_payload, dict)
                or phase_payload.get("schema") != _PHASE_RECORD_SCHEMA
                or phase_payload.get("phase") != "fresh"
                or phase_payload.get("global_step") != 1
                or phase_payload.get("checkpoint") != str(fresh_checkpoint)
                or phase_payload.get("automatic_model_quality_pass") is not None
                or phase_payload.get("automatic_sft2_authorization") is not False
                or not isinstance(phase_evidence, dict)
                or phase_evidence.get("kind") != _SMOKE_EVIDENCE_KIND
                or phase_evidence.get("config_identity") != config.identity
                or phase_evidence.get("approved_command_sha256")
                != config.authorization.approved_command_sha256
            ):
                raise ValueError("Query-State smoke fresh phase record mismatch")
            try:
                record = json.loads(process_record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("Query-State smoke fresh process record is invalid") from error
            if not isinstance(record, dict) or set(record) != {
                "schema", "config_identity", "process_identity", "record_identity"
            }:
                raise ValueError("Query-State smoke fresh process record contract is invalid")
            process_payload = {
                "schema": record["schema"],
                "config_identity": record["config_identity"],
                "process_identity": record["process_identity"],
            }
            if (
                record["schema"] != _PHASE_RECORD_SCHEMA
                or record["config_identity"] != config.identity
                or record["record_identity"] != _sha256_json(process_payload)
            ):
                raise ValueError("Query-State smoke fresh process/config identity mismatch")
            prior_process = record["process_identity"]
            if not isinstance(prior_process, str) or not prior_process.strip():
                raise ValueError("Query-State smoke fresh process identity is invalid")
            if prior_process == process_identity:
                raise ValueError("Query-State smoke resume requires a fresh process")
    except Exception as error:
        precondition_error = error
    _coherent_precondition(precondition_error, world_size=world_size)

    config_publish_error: Exception | None = None
    if phase == "fresh" and rank == 0:
        try:
            _atomic_json(
                {
                    "schema": _CONFIG_RECORD_SCHEMA,
                    "config_identity": config.identity,
                    "config": asdict(config),
                },
                config_record,
            )
        except Exception as error:
            config_publish_error = error
    if _distributed():
        status = [
            None
            if config_publish_error is None
            else f"{type(config_publish_error).__name__}: {config_publish_error}"
        ]
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(
                "Query-State smoke resolved config publication failed: " + status[0]
            )
    elif config_publish_error is not None:
        raise config_publish_error

    descriptors = tuple(
        row for row in config.data.smoke_rows
        if row.phase == phase and row.rank == rank
    )
    if len(descriptors) != 1:
        raise ValueError("Query-State smoke phase requires one exact real row per rank")
    context = QueryStateSmokePhaseContext(
        config=config,
        phase=phase,
        rank=rank,
        world_size=world_size,
        process_identity=process_identity,
        approved_command_manifest=approved_command_manifest,
        phase_root=phase_root,
        checkpoint_path=checkpoint,
        previous_checkpoint_path=None if phase == "fresh" else fresh_checkpoint,
        previous_process_identity=prior_process,
        expected_global_step=1 if phase == "fresh" else 2,
        row_descriptor=descriptors[0],
    )
    outcome: QueryStateSmokePhaseOutcome | None = None
    execute_error: Exception | None = None
    try:
        outcome = execute(context)
        _validate_outcome(
            outcome,
            expected_step=context.expected_global_step,
            checkpoint_path=checkpoint,
            world_size=world_size,
            config_identity=config.identity,
            approved_command_sha256=config.authorization.approved_command_sha256,
        )
    except Exception as error:
        execute_error = error
    # Do not let a rank-local callback/outcome error strand peers at the later
    # publication barrier.  Inner FSDP/checkpoint collectives retain their own
    # framework/coordinated failure boundaries.  Once a phase owns output,
    # preserve an immutable mechanics-failure record without promoting it to a
    # completion marker or model-quality decision.
    try:
        _coherent_precondition(execute_error, world_size=world_size)
    except Exception as phase_error:
        failure_publish_error: Exception | None = None
        if rank == 0:
            try:
                _atomic_json(
                    {
                        "schema": _PHASE_RECORD_SCHEMA,
                        "kind": _SMOKE_EVIDENCE_KIND,
                        "phase": phase,
                        "config_identity": config.identity,
                        "process_identity": process_identity,
                        "approved_command_manifest": approved_command_manifest,
                        "error_type": type(phase_error).__name__,
                        "error": str(phase_error),
                        "automatic_model_quality_pass": None,
                        "automatic_sft2_authorization": False,
                    },
                    phase_root / "phase_failed.json",
                )
            except Exception as error:
                failure_publish_error = error
        if _distributed():
            status = [
                None
                if failure_publish_error is None
                else f"{type(failure_publish_error).__name__}: {failure_publish_error}"
            ]
            torch.distributed.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(
                    "Query-State smoke failure publication failed: " + status[0]
                ) from phase_error
        elif failure_publish_error is not None:
            raise failure_publish_error from phase_error
        raise
    if outcome is None:
        raise RuntimeError("Query-State smoke phase produced no validated outcome")
    if _distributed():
        torch.distributed.barrier()

    publish_error: Exception | None = None
    if rank == 0:
        try:
            if phase == "fresh":
                process_payload = {
                    "schema": _PHASE_RECORD_SCHEMA,
                    "config_identity": config.identity,
                    "process_identity": process_identity,
                }
                _atomic_json(
                    {
                        **process_payload,
                        "record_identity": _sha256_json(process_payload),
                    },
                    process_record,
                )
                _atomic_json(
                    {
                        "schema": _PHASE_RECORD_SCHEMA,
                        "phase": phase,
                        "global_step": outcome.global_step,
                        "checkpoint": str(outcome.checkpoint_path),
                        "evidence": dict(outcome.evidence),
                        "automatic_model_quality_pass": None,
                        "automatic_sft2_authorization": False,
                    },
                    fresh_root / "phase_complete.json",
                )
            else:
                resume_phase_record = resume_root / "phase_complete.json"
                _atomic_json(
                    {
                        "schema": _PHASE_RECORD_SCHEMA,
                        "phase": phase,
                        "global_step": outcome.global_step,
                        "checkpoint": str(outcome.checkpoint_path),
                        "evidence": dict(outcome.evidence),
                        "automatic_model_quality_pass": None,
                        "automatic_sft2_authorization": False,
                    },
                    resume_phase_record,
                )
                _atomic_json(
                    {
                        "schema": _COMPLETE_RECORD_SCHEMA,
                        "kind": _SMOKE_EVIDENCE_KIND,
                        "config_identity": config.identity,
                        "fresh_process_identity": prior_process,
                        "resume_process_identity": process_identity,
                        "fresh_checkpoint": str(fresh_checkpoint),
                        "resume_checkpoint": str(resume_checkpoint),
                        "global_step": 2,
                        "evidence_sha256": {
                            "resolved_config": _sha256_file(config_record),
                            "fresh_process": _sha256_file(process_record),
                            "fresh_phase": _sha256_file(fresh_phase_record),
                            "resume_phase": _sha256_file(resume_phase_record),
                            "fresh_checkpoint_control": _sha256_file(
                                fresh_checkpoint / "control.json"
                            ),
                            "resume_checkpoint_control": _sha256_file(
                                resume_checkpoint / "control.json"
                            ),
                        },
                        "automatic_model_quality_pass": None,
                        "automatic_sft2_authorization": False,
                    },
                    run_root / "smoke_complete.json",
                )
        except Exception as error:
            publish_error = error
    if _distributed():
        status = [
            None if publish_error is None else f"{type(publish_error).__name__}: {publish_error}"
        ]
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError("Query-State smoke publication failed: " + status[0])
    elif publish_error is not None:
        raise publish_error
    return outcome


__all__ = [
    "QueryStateSmokeGroupGradientEvidence",
    "QueryStateSmokeInventoryEvidence",
    "QueryStateSmokeParameterRecord",
    "QueryStateSmokePhaseContext",
    "QueryStateSmokePhaseOutcome",
    "QueryStateSmokeRuntimeFingerprint",
    "build_query_state_inventory_evidence",
    "build_query_state_runtime_fingerprint",
    "build_query_state_source_manifest_identity",
    "collect_query_state_group_gradient_evidence",
    "orchestrate_query_state_smoke_phase",
    "verify_query_state_smoke_rows",
]
