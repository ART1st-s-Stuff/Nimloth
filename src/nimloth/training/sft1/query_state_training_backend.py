"""Launchable torchrun backend for Query-State pilot/formal training.

This module binds the reviewed production constructor, deterministic raw-row
schedule, real update transaction, detached validation, immutable rank
checkpoint, durable segment/log cursor, exact restart, and formal tracking
owner.  It never submits Slurm and never materializes a deployable bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, FrozenDINOGridTargets
from nimloth.backbone.qwen25vl.factory import build_input_builder, load_backbone
from nimloth.backbone.qwen25vl.policy import collect_policy_images, render_policy_messages
from nimloth.backbone.qwen25vl.turn_generation import (
    TurnGenerationSpec,
    build_turn_generation_spec,
    response_policy_prompt_identity,
    run_fsdp_greedy_turn_probe,
    turn_generation_spec_identity,
)
from nimloth.latent import special_token_ids
from nimloth.training.sft1.query_state import QueryStateValidationForwardOutput
from nimloth.training.sft1.query_state_adapter import query_state_update_inputs
from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateDistributedControl,
    QueryStateResumeIdentity,
)
from nimloth.training.sft1.query_state_data import FreshQueryStateDINOTeacher
from nimloth.training.sft1.query_state_distributed import (
    QueryStateDistributedWorkerAssembly,
    build_query_state_distributed_worker,
    query_state_global_normalization,
)
from nimloth.training.sft1.query_state_driver import (
    QueryStateScheduledRow,
    build_query_state_update_dataproto,
    deterministic_query_state_schedule,
    iter_query_state_updates,
    restore_query_state_distributed_checkpoint,
    save_query_state_distributed_checkpoint,
)
from nimloth.training.sft1.query_state_runtime import (
    QueryStateConstructedRoot,
    QueryStateWorkerAssembly,
    assemble_query_state_training_root,
    construct_query_state_production_root,
)
from nimloth.training.sft1.query_state_training_config import (
    QueryStateTrainingConfig,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_manifest import (
    QueryStateGenerationFormatManifest,
    build_generation_response_policy_prompt,
    deserialize_generation_format_manifest,
    deserialize_query_state_validation_split,
    validate_query_state_row_audit,
)
from nimloth.training.sft1.query_state_training_controller import (
    QueryStateTrainingController,
)
from nimloth.training.sft1.query_state_training_runtime import (
    QueryStateAuthoritativeEntry,
    QueryStateEarlyStoppingCursor,
    QueryStateRecovery,
    QueryStateSegmentStore,
    QueryStateWandbMirror,
    advance_query_state_early_stopping,
    current_process_identity,
)
from nimloth.training.sft1.query_state_training_validation import (
    QueryStateValidationMetadata,
    compute_query_state_diagnostics,
    controlled_gather_query_state_diagnostics,
    evaluate_actor_safety,
    validation_mode,
)
from nimloth.training.sft1.real_rows import (
    SFT1V2Early4Row,
    SFT1V2RowAudit,
    index_early4_rows,
)
from nimloth.training.verl.runtime import MixedPrecisionConfig


@dataclass(frozen=True)
class QueryStateTrainingBackendAssembly:
    constructed: QueryStateConstructedRoot
    worker: QueryStateWorkerAssembly
    distributed_worker: QueryStateDistributedWorkerAssembly
    scheduler: torch.optim.lr_scheduler.LRScheduler
    processor: Any
    rows_by_ordinal: Mapping[int, SFT1V2Early4Row]
    padding_row: SFT1V2Early4Row
    training_ordinals: tuple[int, ...]
    calibration_ordinals: tuple[int, ...]
    holdout_ordinals: tuple[int, ...]
    generation_format_manifest: QueryStateGenerationFormatManifest
    generation_spec: TurnGenerationSpec
    dino_teacher: FreshQueryStateDINOTeacher


@dataclass(frozen=True)
class QueryStateTrainingRunResult:
    mode: str
    final_update: int
    final_checkpoint: str
    validation_cursor: int
    log_cursor: int
    tracking_cursor: int
    tracking_incomplete: bool
    terminal_epoch: int | None
    terminal_reason: str | None


@dataclass(frozen=True)
class QueryStateDetachedValidationResult:
    publication: Mapping[str, Any]
    baseline_action_logits: Mapping[str, tuple[float, ...]]


def _dtype(value: str) -> torch.dtype:
    values = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    if value not in values:
        raise ValueError(f"unsupported Query-State dtype: {value}")
    return values[value]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _backbone_args(config: QueryStateTrainingConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model=config.initialization["actor_checkpoint"],
        max_pixels=config.runtime["max_pixels"],
        gradient_checkpointing=config.runtime["gradient_checkpointing"],
        attn_implementation=config.runtime["attention_implementation"],
        llm_tune=config.model["llm_tune"],
        vision_tune=config.model["vision_tune"],
        query_tune=config.model["query_tune"],
        lora=False,
        resume=False,
    )


def _index_contract(config: QueryStateTrainingConfig) -> SimpleNamespace:
    hashes = config.artifacts["file_sha256"]
    return SimpleNamespace(data=SimpleNamespace(
        train_jsonl=config.data["train_source_path"],
        validation_jsonl=config.data["validation_source_path"],
        train_sha256=hashes[config.data["train_source_path"]],
        validation_sha256=hashes[config.data["validation_source_path"]],
        train_split="train",
        validation_split="val",
    ))


def _index_training_rows(
    config: QueryStateTrainingConfig,
) -> tuple[tuple[SFT1V2Early4Row, ...], SFT1V2RowAudit]:
    """Rebuild rows from the data-only contract and enforce the canonical audit."""

    rows, audit = index_early4_rows(
        _index_contract(config),
        enforce_approved_counts=False,
    )
    validate_query_state_row_audit(audit)
    return rows, audit


def _manifest_ordinals(
    path: Path,
    *,
    identity: str,
    expected_rows: int,
    rows_by_identity: Mapping[str, SFT1V2Early4Row],
    mode: str,
) -> tuple[int, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Query-State {mode} manifest") from error
    entries = raw.get("entries") if isinstance(raw, dict) else None
    expected_kind = "coverage_first_pilot" if mode == "pilot" else "full_train_once_per_epoch"
    if (
        not isinstance(raw, dict)
        or raw.get("identity") != identity
        or raw.get("kind") != expected_kind
        or not isinstance(entries, list)
        or len(entries) != expected_rows
    ):
        raise ValueError("Query-State training manifest identity/kind/count mismatch")
    ordinals: list[int] = []
    identities: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Query-State training manifest entry is invalid")
        row_identity, ordinal = entry.get("row_identity"), entry.get("ordinal")
        row = rows_by_identity.get(str(row_identity))
        if row is None or isinstance(ordinal, bool) or not isinstance(ordinal, int) or row.ordinal != ordinal:
            raise ValueError("Query-State training manifest row identity/ordinal mismatch")
        identities.append(str(row_identity))
        ordinals.append(ordinal)
    if len(set(identities)) != len(identities) or len(set(ordinals)) != len(ordinals):
        raise ValueError("Query-State training manifest duplicates a row")
    return tuple(ordinals)


def _scheduler(
    optimizer: torch.optim.Optimizer,
    config: QueryStateTrainingConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    total = int(config.schedule["max_updates"])
    warmup = int(config.optimizer["warmup_updates"])
    name = str(config.optimizer["scheduler"])

    def scale(step: int) -> float:
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
        if name == "constant":
            return 1.0
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError("unsupported Query-State scheduler")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def construct_query_state_training_backend(
    config: QueryStateTrainingConfig,
    *,
    repo_root: Path,
    device: torch.device,
) -> QueryStateTrainingBackendAssembly:
    """Build the real ID176→direct-root→FULL_SHARD→online-DINO backend."""

    if config.lifecycle_state != "launch_locked" or device.type != "cuda":
        raise PermissionError("Query-State production backend requires launch lock and CUDA")
    _seed(int(config.schedule["seed"]))
    loaded = load_backbone(
        _backbone_args(config),
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
        resume_dir=None,
        resume_state_path=None,
    )
    constructed = construct_query_state_production_root(loaded)
    worker = assemble_query_state_training_root(
        constructed=constructed,
        device=device,
        repo_root=Path(repo_root),
        wrap_policy=config.runtime["fsdp_wrap_policy"],
        mixed_precision=MixedPrecisionConfig(
            param_dtype=_dtype(str(config.runtime["model_dtype"])),
            reduce_dtype=_dtype(str(config.runtime["model_dtype"])),
            buffer_dtype=_dtype(str(config.runtime["model_dtype"])),
        ),
        language_learning_rate=float(config.optimizer["language_learning_rate"]),
        direct_state_learning_rate=float(config.optimizer["direct_state_learning_rate"]),
        weight_decay=float(config.optimizer["weight_decay"]),
        adam_betas=tuple(float(value) for value in config.optimizer["betas"]),
        adam_epsilon=float(config.optimizer["epsilon"]),
    )
    scheduler = _scheduler(worker.optimizer, config)
    input_builder = build_input_builder(
        loaded,
        max_length=int(config.runtime["max_sequence_length"]),
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    rows, _audit = _index_training_rows(config)
    by_identity = {row.identity: row for row in rows}
    by_ordinal = {row.ordinal: row for row in rows}
    training_ordinals = _manifest_ordinals(
        Path(str(config.data["train_manifest_path"])),
        identity=str(config.data["train_manifest_identity"]),
        expected_rows=int(config.data["train_rows"]),
        rows_by_identity=by_identity,
        mode=config.mode,
    )
    validation_split = deserialize_query_state_validation_split(
        Path(str(config.data["validation_manifest_path"])),
        rows=rows,
        expected_identity=str(config.data["validation_manifest_identity"]),
    )
    calibration_ordinals = tuple(
        by_identity[identity].ordinal
        for identity in validation_split.calibration_row_identities
    )
    holdout_ordinals = tuple(
        by_identity[identity].ordinal
        for identity in validation_split.holdout_row_identities
    )
    if len(calibration_ordinals) != 80 or len(holdout_ordinals) != 1333:
        raise ValueError("Query-State validation split must remain calibration80/holdout1333")
    generation_format = deserialize_generation_format_manifest(
        Path(str(config.validation["generation_format_manifest_path"])),
        rows=rows,
        validation_split=validation_split,
        expected_identity=str(config.validation["generation_format_manifest_identity"]),
        expected_mode=config.mode,
    )
    tokenizer = loaded.processor.tokenizer
    generation_spec = build_turn_generation_spec(
        tokenizer=tokenizer,
        token_id_map=special_token_ids(tokenizer, latent_token_count=16),
        action_token_ids=tuple(int(value) for value in config.model["action_token_ids"]),
        latent_token_count=16,
        max_response_tokens=generation_format.max_output_tokens,
    )
    if (
        generation_spec.max_reasoning_tokens != generation_format.max_reasoning_tokens
        or generation_spec.max_output_tokens != generation_format.max_output_tokens
        or turn_generation_spec_identity(generation_spec)
        != generation_format.turn_generation_spec_identity
    ):
        raise ValueError(
            "Query-State production TurnGenerationSpec identity/budget differs from manifest"
        )
    train_rows = tuple(row for row in rows if row.split == "train")
    if not train_rows:
        raise ValueError("Query-State source has no train padding row")
    dino = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=_dtype(str(config.runtime["dino_dtype"])),
        grid_size=4,
        batch_size=int(config.runtime["dino_batch_size"]),
    )
    teacher = FreshQueryStateDINOTeacher(
        dino,
        process_identity=current_process_identity(),
    )
    distributed = build_query_state_distributed_worker(
        worker=worker,
        input_builder=input_builder,
        device=device,
        max_padded_tokens=int(config.runtime["max_padded_tokens"]),
        max_rows=int(config.runtime["max_rows_per_micro_batch"]),
        max_grad_norm=float(config.runtime["max_grad_norm"]),
        scheduler=scheduler,
    )
    return QueryStateTrainingBackendAssembly(
        constructed=constructed,
        worker=worker,
        distributed_worker=distributed,
        scheduler=scheduler,
        processor=loaded.processor,
        rows_by_ordinal=by_ordinal,
        padding_row=train_rows[0],
        training_ordinals=training_ordinals,
        calibration_ordinals=calibration_ordinals,
        holdout_ordinals=holdout_ordinals,
        generation_format_manifest=generation_format,
        generation_spec=generation_spec,
        dino_teacher=teacher,
    )


def build_query_state_training_updates(
    ordinals: Sequence[int],
    *,
    epochs: int,
    seed: int,
    rank: int,
    world_size: int,
    rows_per_rank_update: int,
    expected_updates: int,
) -> tuple[tuple[QueryStateScheduledRow, ...], ...]:
    updates: list[tuple[QueryStateScheduledRow, ...]] = []
    for epoch in range(epochs):
        local, _identity = deterministic_query_state_schedule(
            ordinals, epoch=epoch, seed=seed, rank=rank, world_size=world_size
        )
        updates.extend(iter_query_state_updates(local, rows_per_rank_update=rows_per_rank_update))
    if len(updates) != expected_updates:
        raise ValueError(
            f"Query-State max_updates disagrees with exact schedule: {expected_updates} != {len(updates)}"
        )
    return tuple(updates)


def _validation_boundary_plan(
    config: QueryStateTrainingConfig,
    *,
    update: int,
    epoch: int,
    actual_terminal: bool,
) -> dict[str, bool]:
    if config.mode != "formal":
        raise ValueError("dynamic dual-split validation is formal-only")
    cadence = int(config.validation["calibration_cadence_updates"])
    if (
        update != epoch * cadence
        or epoch < 1
        or update < 1
        or not isinstance(actual_terminal, bool)
    ):
        raise ValueError("formal validation boundary is not an epoch commit")
    holdout_due = (
        update in {int(value) for value in config.validation["holdout_updates"]}
        or actual_terminal
    )
    generation_due = holdout_due and (
        update
        in {
            int(value)
            for value in config.validation["generation_format_updates"]
        }
        or actual_terminal
    )
    return {
        "calibration": True,
        "holdout": holdout_due,
        "generation_format": generation_due,
        "actual_terminal": actual_terminal,
    }


def _resume_identity(config: QueryStateTrainingConfig) -> QueryStateResumeIdentity:
    run_identity = query_state_training_run_identity(config)
    return QueryStateResumeIdentity(
        source_commit=str(config.source["commit"]),
        source_manifest_identity=str(config.source["source_manifest_identity"]),
        config_identity=run_identity,
        run_identity=run_identity,
        world_size=int(config.resources["world_size"]),
        experiment_mode=config.mode,
    )


def _all_gather(value: Any, world_size: int) -> tuple[Any, ...]:
    if world_size == 1:
        return (value,)
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, value)
    return tuple(gathered)


def _coordinate_early_stopping_decision(
    decision: Any,
    *,
    world_size: int,
) -> None:
    payload = asdict(decision)
    gathered = _all_gather(payload, world_size)
    if len(gathered) != world_size or any(
        not isinstance(value, Mapping) or value != payload for value in gathered
    ):
        raise RuntimeError("formal early-stop verdict differs across ranks")


def _global_teacher_memo_metric(
    local_report: Mapping[str, Any],
    *,
    world_size: int,
) -> Mapping[str, Any]:
    """Gather process-local memo telemetry into one rank-identical metric value."""

    expected = {
        "process_identity",
        "dino_identity",
        "entries",
        "current_bytes",
        "peak_bytes",
    }
    gathered = _all_gather(dict(local_report), world_size)
    if len(gathered) != world_size or any(
        not isinstance(report, Mapping) or set(report) != expected
        for report in gathered
    ):
        raise RuntimeError("Query-State teacher memo rank telemetry is invalid")
    reports = [dict(report) for report in gathered]
    dino_identities = {report["dino_identity"] for report in reports}
    if len(dino_identities) != 1:
        raise RuntimeError("Query-State teacher memo DINO identity differs across ranks")
    return {
        "scope": "process_local_by_rank",
        "reports": reports,
    }


def _coordinated_rank0_status(
    error: BaseException | None,
    *,
    rank: int,
    world_size: int,
    operation: str,
) -> None:
    status = [
        None if error is None else f"{type(error).__name__}: {error}"
    ] if rank == 0 else [None]
    if world_size > 1:
        torch.distributed.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(f"Query-State rank-zero {operation} failed: {status[0]}")


def _actor_baseline_path(run_root: Path) -> Path:
    return Path(run_root) / "actor_baseline_id176.json"


def _baseline_payload(
    config: QueryStateTrainingConfig,
    baseline: Mapping[str, tuple[float, ...]],
) -> dict[str, Any]:
    rows = [
        {"row_identity": identity, "action_logits": list(logits)}
        for identity, logits in sorted(baseline.items())
    ]
    body = {
        "schema": "nimloth_sft1_query_state_id176_actor_baseline_v1",
        "actor_checkpoint_identity": config.initialization["actor_checkpoint_identity"],
        "validation_manifest_identity": config.data["validation_manifest_identity"],
        "rows": rows,
    }
    return {
        **body,
        "identity": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }


def _publish_actor_baseline(
    path: Path,
    *,
    config: QueryStateTrainingConfig,
    baseline: Mapping[str, tuple[float, ...]],
) -> str:
    payload = _baseline_payload(config, baseline)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return str(payload["identity"])


def _load_actor_baseline(
    path: Path,
    *,
    config: QueryStateTrainingConfig,
) -> tuple[dict[str, tuple[float, ...]], str]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query-State exact restart lacks an immutable ID176 actor baseline") from error
    rows = raw.get("rows") if isinstance(raw, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Query-State ID176 actor baseline rows are invalid")
    baseline: dict[str, tuple[float, ...]] = {}
    for item in rows:
        if not isinstance(item, dict) or set(item) != {"row_identity", "action_logits"}:
            raise ValueError("Query-State ID176 actor baseline row schema is invalid")
        identity = item["row_identity"]
        logits = item["action_logits"]
        if (
            not isinstance(identity, str)
            or not isinstance(logits, list)
            or len(logits) != 8
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in logits)
            or identity in baseline
        ):
            raise ValueError("Query-State ID176 actor baseline row is invalid")
        baseline[identity] = tuple(float(value) for value in logits)
    expected = _baseline_payload(config, baseline)
    if raw != expected:
        raise ValueError("Query-State ID176 actor baseline identity mismatch")
    return baseline, str(expected["identity"])


def _verify_replayed_update_zero_evidence(
    run_root: Path,
    *,
    actor_baseline_identity: str,
) -> None:
    try:
        raw = json.loads(
            (Path(run_root) / "validation_update_00000000.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Query-State crash replay lacks immutable update-zero validation"
        ) from error
    safety = raw.get("safety") if isinstance(raw, dict) else None
    if (
        raw.get("update") != 0
        or raw.get("actor_baseline_identity") != actor_baseline_identity
        or not isinstance(safety, dict)
        or safety.get("passed") is not True
    ):
        raise ValueError(
            "Query-State crash replay update-zero validation identity/safety mismatch"
        )


def _generation_prompt_inputs(
    assembly: QueryStateTrainingBackendAssembly,
    *,
    entry: Any,
) -> tuple[dict[str, torch.Tensor], str]:
    row = assembly.rows_by_ordinal.get(entry.ordinal)
    if row is None or row.identity != entry.row_identity:
        raise ValueError("generation-format runtime row is not registered exactly")
    prompt = build_generation_response_policy_prompt(row)
    prompt_identity = response_policy_prompt_identity(prompt)
    if prompt_identity != entry.prompt_identity:
        raise ValueError("generation-format production prompt identity changed")
    bound_messages = prompt.bound_messages()
    text = render_policy_messages(
        bound_messages,
        assembly.processor,
        latent_token_count=16,
        continue_final_message=True,
    )
    images = collect_policy_images(bound_messages)
    encoded = assembly.processor(
        text=[text],
        images=[images] if images else None,
        padding=True,
        return_tensors="pt",
    )
    device = next(assembly.distributed_worker.root.parameters()).device
    inputs = {
        str(name): value.to(device)
        for name, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    if "input_ids" not in inputs:
        raise ValueError("generation-format production prompt omitted input_ids")
    return inputs, prompt_identity


def _run_generation_format_probe(
    config: QueryStateTrainingConfig,
    assembly: QueryStateTrainingBackendAssembly,
    *,
    update: int,
    world_size: int,
) -> Mapping[str, Any]:
    manifest = assembly.generation_format_manifest
    snapshot_identity = hashlib.sha256(
        f"{query_state_training_run_identity(config)}:{update}:current-fsdp-logits".encode()
    ).hexdigest()
    records: list[Mapping[str, Any]] = []
    failure: Mapping[str, Any] | None = None
    with validation_mode(assembly.distributed_worker.root):
        for entry in manifest.entries:
            prepared: tuple[dict[str, torch.Tensor], str] | None = None
            prepare_error: str | None = None
            try:
                prepared = _generation_prompt_inputs(assembly, entry=entry)
            except Exception as error:
                prepare_error = f"{type(error).__name__}: {error}"
            prepare_errors = _all_gather(prepare_error, world_size)
            if any(error is not None for error in prepare_errors):
                failure = {
                    "row_identity": entry.row_identity,
                    "stage": "production_prompt",
                    "all_rank_errors": prepare_errors,
                }
                break
            assert prepared is not None
            prompt_inputs, prompt_identity = prepared
            local_record: Mapping[str, Any] | None = None
            local_error: str | None = None
            try:
                result = run_fsdp_greedy_turn_probe(
                    assembly.distributed_worker.root,
                    prompt_inputs=prompt_inputs,
                    tokenizer=assembly.processor.tokenizer,
                    spec=assembly.generation_spec,
                    checkpoint_identity=snapshot_identity,
                    prompt_identity=prompt_identity,
                )
                local_record = {
                    "row_identity": entry.row_identity,
                    "ordinal": entry.ordinal,
                    "record_id": entry.record_id,
                    "prompt_identity": result.prompt_identity,
                    "checkpoint_identity": result.checkpoint_identity,
                    "spec_identity": result.spec_identity,
                    "continuation_token_ids": list(result.continuation_token_ids),
                    "response": result.parsed.response,
                    "thought": result.parsed.thought,
                    "action_index": result.parsed.action_index,
                    "action_token_id": result.parsed.action_token_id,
                    "close_end": result.parsed.close_end,
                    "reasoning_truncated": result.parsed.reasoning_truncated,
                    "used_current_model_logits": result.used_current_model_logits,
                    "action_executed": result.action_executed,
                    "rollout_persisted": result.rollout_persisted,
                    "deployable_materialized": result.deployable_materialized,
                    "parsed_exactly": True,
                }
            except Exception as error:
                local_error = f"{type(error).__name__}: {error}"
            rank_results = _all_gather((local_record, local_error), world_size)
            errors = tuple(value[1] for value in rank_results)
            rank_records = tuple(value[0] for value in rank_results)
            if any(error is not None for error in errors):
                failure = {
                    "row_identity": entry.row_identity,
                    "stage": "current_fsdp_greedy_parse",
                    "all_rank_errors": errors,
                }
                break
            if any(record != rank_records[0] for record in rank_records[1:]):
                failure = {
                    "row_identity": entry.row_identity,
                    "stage": "all_rank_exact_evidence",
                    "all_rank_errors": ("generated evidence differs by rank",),
                }
                break
            assert rank_records[0] is not None
            records.append(rank_records[0])
    return {
        "schema": "nimloth_sft1_query_state_generation_format_evidence_v1",
        "due": True,
        "update": update,
        "split": manifest.split,
        "manifest_identity": manifest.identity,
        "checkpoint_identity": snapshot_identity,
        "prompt_protocol_identity": manifest.prompt_protocol_identity,
        "turn_generation_spec_identity": manifest.turn_generation_spec_identity,
        "parser_protocol_identity": manifest.parser_protocol_identity,
        "max_reasoning_tokens": manifest.max_reasoning_tokens,
        "max_output_tokens": manifest.max_output_tokens,
        "registered_row_count": len(manifest.entries),
        "parsed_row_count": len(records),
        "records": records,
        "failure": failure,
        "passed": failure is None and len(records) == len(manifest.entries),
        "non_resumable_safety_failure": failure is not None,
        "current_fsdp_logits": True,
        "fixed_or_repaired_cot": False,
        "action_execution": False,
        "rollout_persistence": False,
        "deployable_export": False,
    }


def _run_detached_validation(
    config: QueryStateTrainingConfig,
    assembly: QueryStateTrainingBackendAssembly,
    *,
    update: int,
    split: str,
    generation_format_due: bool,
    rank: int,
    world_size: int,
    baseline_action_logits: Mapping[str, tuple[float, ...]] | None,
) -> QueryStateDetachedValidationResult:
    if split not in {"calibration", "holdout"}:
        raise ValueError("Query-State detached validation split is invalid")
    if config.mode == "pilot" and split != "calibration":
        raise ValueError("pilot must never open holdout validation rows")
    if generation_format_due and assembly.generation_format_manifest.split != split:
        raise ValueError("generation-format manifest and validation split disagree")
    ordinals = (
        assembly.calibration_ordinals
        if split == "calibration"
        else assembly.holdout_ordinals
    )
    schedule, _identity = deterministic_query_state_schedule(
        ordinals,
        epoch=0,
        seed=int(config.schedule["seed"]),
        rank=rank,
        world_size=world_size,
    )
    tensor_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "raw_query_hidden", "canonical_state", "dino_regions",
            "action_logits", "fused_image_features", "instruction_features",
        )
    }
    metadata: list[QueryStateValidationMetadata] = []
    validation_device = next(assembly.distributed_worker.root.parameters()).device
    ce = torch.zeros(4, dtype=torch.float64, device=validation_device)
    with validation_mode(assembly.distributed_worker.root), torch.no_grad():
        for scheduled in iter_query_state_updates(
            schedule,
            rows_per_rank_update=int(config.schedule["rows_per_rank_update"]),
        ):
            data = build_query_state_update_dataproto(
                scheduled,
                rows_by_ordinal=assembly.rows_by_ordinal,
                padding_row=assembly.padding_row,
                processor=assembly.processor,
                dino_teacher=assembly.dino_teacher,
                max_length=int(config.runtime["max_sequence_length"]),
                source_manifest_identity=str(config.source["source_manifest_identity"]),
            )
            inputs = query_state_update_inputs(
                data,
                input_builder=assembly.distributed_worker.core.input_builder,
                include_diagnostics=True,
            )
            normalization = query_state_global_normalization(
                data,
                device=validation_device,
            )
            output = assembly.distributed_worker.root(
                inputs.student_batch,
                inputs.targets,
                normalization,
                diagnostic=True,
            )
            if not isinstance(output, QueryStateValidationForwardOutput):
                raise RuntimeError("Query-State validation did not return same-forward diagnostics")
            valid_indices = torch.nonzero(
                inputs.targets.sample_valid.to(device=output.objective.state.device),
                as_tuple=False,
            ).flatten()
            student = output.student
            views = {
                "raw_query_hidden": output.objective.raw_query_hidden,
                "canonical_state": output.objective.state,
                "dino_regions": inputs.targets.dino_regions.to(output.objective.state.device),
                "action_logits": output.objective.action_logits,
                "fused_image_features": student.fused_image_features,
                "instruction_features": student.instruction_features,
            }
            if views["fused_image_features"] is None or views["instruction_features"] is None:
                raise RuntimeError("Query-State validation omitted real-row diagnostic features")
            for name, value in views.items():
                assert isinstance(value, torch.Tensor)
                tensor_parts[name].append(value.index_select(0, valid_indices).detach().cpu())
            for item in scheduled:
                if item.row_valid:
                    assert item.ordinal is not None
                    metadata.append(QueryStateValidationMetadata.from_row(
                        assembly.rows_by_ordinal[item.ordinal]
                    ))
            if student.lm_loss_sum is None or student.action_lm_loss_sum is None:
                raise RuntimeError("Query-State validation omitted archived CE sums")
            ce += torch.tensor(
                [
                    float(student.lm_loss_sum.item()),
                    float(student.lm_valid_token_count),
                    float(student.action_lm_loss_sum.item()),
                    float(student.action_lm_valid_token_count),
                ],
                dtype=torch.float64,
                device=ce.device,
            )
    if world_size > 1:
        torch.distributed.all_reduce(ce, op=torch.distributed.ReduceOp.SUM)
    if ce[1].item() <= 0 or ce[3].item() <= 0:
        raise RuntimeError("Query-State global validation CE denominators are empty")
    local = {name: torch.cat(parts, dim=0) for name, parts in tensor_parts.items()}
    if baseline_action_logits is None:
        local_baseline = local["action_logits"]
    else:
        local_baseline = torch.tensor(
            [baseline_action_logits[item.row_identity] for item in metadata],
            dtype=local["action_logits"].dtype,
        )
    gathered, gathered_metadata = controlled_gather_query_state_diagnostics(
        {**local, "baseline_action_logits": local_baseline},
        metadata,
        max_global_rows=int(config.data["external_rows"]),
    )
    current_baseline = {
        item.row_identity: tuple(float(value) for value in logits.tolist())
        for item, logits in zip(
            gathered_metadata,
            gathered["baseline_action_logits"],
            strict=True,
        )
    }
    report = compute_query_state_diagnostics(
        raw_query_hidden=gathered["raw_query_hidden"],
        canonical_state=gathered["canonical_state"],
        dino_regions=gathered["dino_regions"],
        action_logits=gathered["action_logits"],
        baseline_action_logits=gathered["baseline_action_logits"],
        fused_image_features=gathered["fused_image_features"],
        instruction_features=gathered["instruction_features"],
        archived_assistant_ce=float((ce[0] / ce[1]).item()),
        archived_action_ce=float((ce[2] / ce[3]).item()),
        metadata=gathered_metadata,
        effective_rank_collapse_threshold=float(
            config.validation["effective_rank_collapse_threshold"]
        ),
        bootstrap_seed=int(config.validation["bootstrap_seed"]),
        bootstrap_resamples=int(config.validation["bootstrap_resamples"]),
        ordinary_cluster_unit=str(config.validation["ordinary_cluster_unit"]),
        ordinary_bootstrap_formula=str(
            config.validation["ordinary_bootstrap_formula"]
        ),
        natural_pair_unit=str(config.validation["natural_pair_unit"]),
        natural_pair_formula=str(config.validation["natural_pair_formula"]),
        globally_aggregated=True,
    )
    actor_safety = evaluate_actor_safety(
        report,
        tolerances=config.validation["actor_tolerances"],
    )
    generation_due = generation_format_due
    generation_format: Mapping[str, Any]
    if generation_due:
        generation_format = _run_generation_format_probe(
            config,
            assembly,
            update=update,
            world_size=world_size,
        )
    else:
        generation_format = {
            "due": False,
            "update": update,
            "passed": None,
            "reason": "not_in_explicit_generation_format_cadence",
        }
    format_passed = generation_format["passed"] is not False
    safety = {
        "passed": actor_safety.passed and format_passed,
        "checks": {
            **dict(actor_safety.checks),
            "generation_format": format_passed,
        },
        "tolerances": dict(actor_safety.tolerances),
        "generation_format_due": generation_due,
    }
    publication = {
        "update": update,
        "split": split,
        "diagnostics": asdict(report),
        "generation_format": generation_format,
        "safety": safety,
        "global_rank_coverage": world_size,
        "diagnostic_only": True,
        "automatic_checkpoint_selection": False,
        "automatic_state_gate": False,
        "human_terminal_state_gate": (
            dict(config.validation["terminal_state_gates"])
            if split == "holdout"
            else None
        ),
    }
    return QueryStateDetachedValidationResult(publication, current_baseline)


def _checkpoint_control_hash(path: Path) -> str:
    digest = hashlib.sha256((path / "control.json").read_bytes()).hexdigest()
    marker = (path / "COMPLETED").read_text(encoding="utf-8")
    if marker != f"control_sha256={digest}\n":
        raise ValueError("Query-State checkpoint marker/control hash mismatch")
    return digest


def _recover_first_boundary_crash(
    store: QueryStateSegmentStore,
    *,
    checkpoint_root: Path,
) -> QueryStateRecovery:
    """Quarantine a pre-index transaction so update zero can replay exactly."""

    recovery = store.recover(checkpoint_root=checkpoint_root)
    if (
        recovery.resume_update != 0
        or (
            recovery.abandoned_pending_segments == 0
            and recovery.abandoned_unindexed_checkpoints == 0
        )
    ):
        raise ValueError(
            "Query-State crash replay requires an empty authoritative cursor "
            "and an abandoned pre-index transaction"
        )
    return recovery


def _validate_formal_restart_early_stopping_cursor(
    cursor: QueryStateEarlyStoppingCursor,
    *,
    start_update: int,
    epoch_updates: int,
) -> None:
    if start_update < 0 or epoch_updates < 1:
        raise ValueError("formal restart update/epoch cadence is invalid")
    completed_epoch = start_update // epoch_updates
    expected_early_stop_update = completed_epoch * epoch_updates
    if (
        cursor.stop_reason is not None
        or cursor.last_epoch != completed_epoch
        or cursor.last_update != expected_early_stop_update
    ):
        raise ValueError("formal restored early-stop cursor is not exact")


def _approved_pause_due(
    config: QueryStateTrainingConfig,
    *,
    segment_end: int,
) -> bool:
    approved_pause = int(config.schedule["approved_pause_update"])
    return config.mode == "formal" and approved_pause > 0 and segment_end == approved_pause


def _forensic_metric_cursor(
    metric_cursor: Mapping[str, Any],
) -> Mapping[str, Any]:
    forensic = dict(metric_cursor)
    actual_terminal = forensic.get("actual_terminal")
    early = forensic.get("early_stopping")
    if actual_terminal is not None:
        forensic["rejected_terminal_candidate"] = actual_terminal
        forensic["actual_terminal"] = None
    if isinstance(early, Mapping) and (
        early.get("terminal_epoch") is not None
        or early.get("terminal_update") is not None
        or early.get("stop_reason") is not None
    ):
        forensic["rejected_early_stopping_cursor"] = dict(early)
        forensic["early_stopping"] = {
            **dict(early),
            "terminal_epoch": None,
            "terminal_update": None,
            "stop_reason": None,
        }
    return forensic


def _authoritative_entries_for_restart(
    store: QueryStateSegmentStore,
    *,
    checkpoint_update: int,
    mirror_cursor: int,
) -> tuple[QueryStateAuthoritativeEntry, ...]:
    """Select durable mirror batches newer than a checkpoint's pre-mirror cursor."""

    if (
        isinstance(checkpoint_update, bool)
        or not isinstance(checkpoint_update, int)
        or checkpoint_update < 0
        or isinstance(mirror_cursor, bool)
        or not isinstance(mirror_cursor, int)
        or mirror_cursor < 0
        or mirror_cursor > checkpoint_update
    ):
        raise ValueError("Query-State W&B cursor exceeds the restart checkpoint boundary")
    entries = store.authoritative_entries()
    durable_update = entries[-1].end_update if entries else 0
    if durable_update != checkpoint_update:
        raise ValueError("Query-State durable index/checkpoint restart boundary mismatch")
    selected = tuple(entry for entry in entries if entry.end_update > mirror_cursor)
    if selected:
        if selected[0].start_update > mirror_cursor or selected[-1].end_update != checkpoint_update:
            raise ValueError("Query-State authoritative mirror cursor has a gap")
    elif mirror_cursor != checkpoint_update:
        raise ValueError("Query-State authoritative mirror cursor lacks its checkpoint boundary")
    return selected


class _FormalTrackingOwner:
    def __init__(self, config: QueryStateTrainingConfig, *, rank: int, world_size: int) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.run: Any | None = None
        self.mirror: QueryStateWandbMirror | None = None

    def initialize(self, cursor: int) -> None:
        if self.config.mode == "pilot":
            return
        result: dict[str, Any] | None = None
        if self.rank == 0:
            try:
                import wandb

                from nimloth.training.sft1.query_state_training_config import (
                    resolve_wandb_start,
                )

                tracking = self.config.tracking
                remote_runs = list(wandb.Api().runs(
                    f"{tracking.entity}/{tracking.project}",
                    filters={"name": tracking.run_id},
                ))
                if len(remote_runs) > 1:
                    raise RuntimeError("W&B query returned duplicate locked run IDs")
                remote_identity = (
                    None
                    if not remote_runs
                    else (
                        f"{remote_runs[0].entity}/{remote_runs[0].project}/"
                        f"{remote_runs[0].group}/{remote_runs[0].name}/"
                        f"{remote_runs[0].id}"
                    )
                )
                start = resolve_wandb_start(
                    self.config,
                    remote_exists=bool(remote_runs),
                    remote_identity=remote_identity,
                )
                self.run = wandb.init(
                    entity=tracking.entity,
                    project=tracking.project,
                    group=tracking.group,
                    name=tracking.run_name,
                    id=tracking.run_id,
                    resume=start.resume,
                    config={"query_state_config_identity": self.config.identity},
                )
                actual = (
                    f"{self.run.entity}/{self.run.project}/{self.run.group}/"
                    f"{self.run.name}/{self.run.id}"
                )
                if actual != tracking.identity:
                    raise RuntimeError("initialized W&B identity differs from launch lock")
                result = {"ok": True, "identity": actual, "url": self.run.url}
            except Exception as error:
                result = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        values = [result]
        if self.world_size > 1:
            torch.distributed.broadcast_object_list(values, src=0)
        if not isinstance(values[0], dict) or values[0].get("ok") is not True:
            raise RuntimeError("Query-State coordinated W&B initialization failed: " + str(values[0]))
        self.mirror = QueryStateWandbMirror(
            run_id=self.config.tracking.run_id,
            world_size=self.world_size,
            initial_cursor=cursor,
        )

    def _flush_pending(self) -> None:
        if self.mirror is None or self.mirror.durable_only:
            return
        failure = False
        if self.rank == 0:
            try:
                if self.run is None:
                    raise RuntimeError("W&B run is absent after coordinated initialization")
                for record in self.mirror.pending_records():
                    self.run.log(record, step=int(record["update"]), commit=True)
            except Exception:
                failure = True
        failures = _all_gather(failure, self.world_size)
        if any(failures):
            self.mirror.coordinated_transport_failure(tuple(True for _ in failures))
            return
        pending = self.mirror.pending_updates()
        if pending:
            self.mirror.replay(run_id=self.config.tracking.run_id, updates=pending)

    def restore_authoritative(
        self,
        entries: Sequence[QueryStateAuthoritativeEntry],
    ) -> None:
        if self.config.mode == "pilot":
            if self.run is not None or self.mirror is not None:
                raise RuntimeError("pilot tracking must remain disabled")
            return
        if self.mirror is None:
            if entries:
                raise RuntimeError("authoritative mirror batches require formal tracking")
            return
        registration_error: str | None = None
        try:
            for entry in entries:
                self.mirror.register_authoritative(entry)
        except Exception as error:
            registration_error = f"{type(error).__name__}: {error}"
        registration_errors = _all_gather(registration_error, self.world_size)
        if any(error is not None for error in registration_errors):
            raise RuntimeError(
                "Query-State authoritative mirror registration failed: "
                + repr(registration_errors)
            )
        self._flush_pending()

    def publish(self, entry: QueryStateAuthoritativeEntry) -> None:
        self.restore_authoritative((entry,))


def run_query_state_training(
    config: QueryStateTrainingConfig,
    *,
    repo_root: Path,
    device: torch.device,
    rank: int,
    world_size: int,
) -> QueryStateTrainingRunResult:
    """Execute the approved deterministic owner under an initialized process group."""

    if world_size != int(config.resources["world_size"]) or not 0 <= rank < world_size:
        raise ValueError("Query-State backend rank/world-size differs from config")
    assembly = construct_query_state_training_backend(config, repo_root=repo_root, device=device)
    updates = build_query_state_training_updates(
        assembly.training_ordinals,
        epochs=int(config.schedule["epochs"]),
        seed=int(config.schedule["seed"]),
        rank=rank,
        world_size=world_size,
        rows_per_rank_update=int(config.schedule["rows_per_rank_update"]),
        expected_updates=int(config.schedule["max_updates"]),
    )
    run_root = Path(str(config.output["run_root"]))
    run_identity = query_state_training_run_identity(config)
    controller = QueryStateTrainingController(
        run_root=run_root,
        controller_root=Path(str(config.output["controller_root"])),
        run_identity=run_identity,
        mode=config.mode,
    )
    resume_mode = str(config.initialization["resume_mode"])
    fresh = resume_mode == "fresh"
    crash_replay = resume_mode == "crash_replay"
    claim_error: BaseException | None = None
    if rank == 0:
        try:
            resolved_config = json.loads(
                Path(str(config.output["resolved_config_path"])).read_text(
                    encoding="utf-8"
                )
            )
            command_manifest_text = Path(
                str(config.output["command_manifest_path"])
            ).read_text(encoding="utf-8")
            command_manifest = json.loads(command_manifest_text)
            if not isinstance(resolved_config, dict) or not isinstance(
                command_manifest, dict
            ):
                raise ValueError(
                    "Query-State process config/command publication requires mappings"
                )
            if fresh:
                controller.claim(
                    resolved_config=resolved_config,
                    command_manifest=command_manifest,
                )
            else:
                controller.verify_existing_claim()
            controller.record_process(
                process_identity=current_process_identity(),
                details={
                    "config_identity": config.identity,
                    "command_identity": config.command["identity"],
                    "resume_mode": resume_mode,
                    "approved_pause_update": int(
                        config.schedule["approved_pause_update"]
                    ),
                    "resume_checkpoint": config.initialization[
                        "resume_checkpoint"
                    ],
                    "resolved_config": resolved_config,
                    "command_manifest_text": command_manifest_text,
                },
            )
        except BaseException as error:
            claim_error = error
    _coordinated_rank0_status(
        claim_error,
        rank=rank,
        world_size=world_size,
        operation="run claim/restart ownership",
    )

    store = None
    store_error: BaseException | None = None
    if rank == 0:
        try:
            store = QueryStateSegmentStore(
                run_root / "durable",
                run_identity=run_identity,
                mode=config.mode,
                wandb_run_id=None if config.mode == "pilot" else config.tracking.run_id,
            )
        except BaseException as error:
            store_error = error
    _coordinated_rank0_status(
        store_error,
        rank=rank,
        world_size=world_size,
        operation="durable store initialization",
    )

    identity = _resume_identity(config)
    start_update = 0
    validation_cursor = -1
    calibration_validation_cursor = -1
    holdout_validation_cursor = -1
    early_stopping_cursor = QueryStateEarlyStoppingCursor.initial()
    log_cursor = 0
    tracking_cursor = 0
    restart_mirror_entries: tuple[QueryStateAuthoritativeEntry, ...] = ()
    if resume_mode == "exact_restart":
        if config.mode == "pilot":
            restart_path = run_root / "FORCED_RESTART_REQUIRED.json"
            try:
                restart = json.loads(restart_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError("pilot exact restart lacks its forced-boundary receipt") from error
            if (
                not isinstance(restart, dict)
                or restart.get("run_identity") != run_identity
                or restart.get("checkpoint") != str(config.initialization["resume_checkpoint"])
                or restart.get("first_process_identity") == current_process_identity()
            ):
                raise ValueError("pilot exact restart receipt/process identity mismatch")
        control, scheduler_state = restore_query_state_distributed_checkpoint(
            Path(str(config.initialization["resume_checkpoint"])),
            root=assembly.distributed_worker.root,
            optimizer=assembly.distributed_worker.optimizer,
            expected_identity=identity,
            rank=rank,
        )
        assembly.scheduler.load_state_dict(dict(scheduler_state))
        start_update = control.global_step
        data_cursor = dict(control.data_cursor)
        metric_cursor = dict(control.metric_cursor)
        if data_cursor.get("next_update") != start_update + 1 or data_cursor.get("mode") != config.mode:
            raise ValueError("Query-State restored data cursor is not exact")
        validation_cursor = int(metric_cursor.get("validation", -1))
        calibration_validation_cursor = int(
            metric_cursor.get("calibration_validation", validation_cursor)
        )
        holdout_validation_cursor = int(
            metric_cursor.get("holdout_validation", validation_cursor)
        )
        if config.mode == "formal":
            early_raw = metric_cursor.get("early_stopping")
            if not isinstance(early_raw, Mapping):
                raise ValueError("formal exact restart lacks the early-stop cursor")
            early_stopping_cursor = QueryStateEarlyStoppingCursor.from_mapping(
                early_raw
            )
            _validate_formal_restart_early_stopping_cursor(
                early_stopping_cursor,
                start_update=start_update,
                epoch_updates=int(config.schedule["epoch_updates"]),
            )
        log_cursor = int(metric_cursor.get("log", -1))
        tracking_cursor = int(metric_cursor.get("wandb", -1))
        recovery_error: BaseException | None = None
        if store is not None:
            try:
                if store.recover(
                    checkpoint_root=run_root / "checkpoints"
                ).resume_update != start_update:
                    raise ValueError(
                        "Query-State durable index/checkpoint resume cursor mismatch"
                    )
            except BaseException as error:
                recovery_error = error
        _coordinated_rank0_status(
            recovery_error,
            rank=rank,
            world_size=world_size,
            operation="durable exact-restart recovery",
        )
        mirror_payload: list[Any] = [(), None]
        if config.mode == "formal" and rank == 0:
            try:
                if store is None:
                    raise RuntimeError("rank-zero durable store is absent")
                mirror_payload[0] = _authoritative_entries_for_restart(
                    store,
                    checkpoint_update=start_update,
                    mirror_cursor=tracking_cursor,
                )
            except BaseException as error:
                mirror_payload[1] = f"{type(error).__name__}: {error}"
        if world_size > 1:
            torch.distributed.broadcast_object_list(mirror_payload, src=0)
        if mirror_payload[1] is not None:
            raise RuntimeError(
                "Query-State authoritative mirror restart recovery failed: "
                + str(mirror_payload[1])
            )
        restart_mirror_entries = tuple(mirror_payload[0])
    elif crash_replay:
        replay_recovery: QueryStateRecovery | None = None
        replay_error: BaseException | None = None
        if store is not None:
            try:
                replay_recovery = _recover_first_boundary_crash(
                    store,
                    checkpoint_root=run_root / "checkpoints",
                )
            except BaseException as error:
                replay_error = error
        replay_payload: list[Any] = [replay_recovery, None if replay_error is None else f"{type(replay_error).__name__}: {replay_error}"]
        if world_size > 1:
            torch.distributed.broadcast_object_list(replay_payload, src=0)
        if replay_payload[1] is not None or replay_payload[0] is None:
            raise RuntimeError(
                "Query-State first-boundary crash replay recovery failed: "
                + str(replay_payload[1])
            )
        validation_cursor = 0
        calibration_validation_cursor = 0
        holdout_validation_cursor = 0
        log_cursor = 0
        tracking_cursor = 0
    tracking = _FormalTrackingOwner(config, rank=rank, world_size=world_size)
    tracking.initialize(tracking_cursor)
    tracking.restore_authoritative(restart_mirror_entries)
    if tracking.mirror is not None:
        tracking_cursor = tracking.mirror.cursor

    validation_updates = {int(value) for value in config.schedule["validation_updates"]}
    actor_baseline: Mapping[str, tuple[float, ...]] | None = None
    actor_baseline_identity: str | None = None
    if start_update == 0 and not crash_replay:
        if config.mode == "formal":
            calibration_baseline = _run_detached_validation(
                config,
                assembly,
                update=0,
                split="calibration",
                generation_format_due=False,
                rank=rank,
                world_size=world_size,
                baseline_action_logits=None,
            )
            holdout_baseline = _run_detached_validation(
                config,
                assembly,
                update=0,
                split="holdout",
                generation_format_due=True,
                rank=rank,
                world_size=world_size,
                baseline_action_logits=None,
            )
            overlap = set(calibration_baseline.baseline_action_logits) & set(
                holdout_baseline.baseline_action_logits
            )
            if overlap:
                raise RuntimeError("formal calibration/holdout actor baselines overlap")
            actor_baseline = {
                **dict(calibration_baseline.baseline_action_logits),
                **dict(holdout_baseline.baseline_action_logits),
            }
            if len(actor_baseline) != int(config.data["external_rows"]):
                raise RuntimeError("formal actor baseline does not cover all external rows")
            calibration_validation_cursor = 0
            holdout_validation_cursor = 0
            baseline_publication: Mapping[str, Any] = {
                "update": 0,
                "calibration": calibration_baseline.publication,
                "holdout": holdout_baseline.publication,
                "safety": {
                    "passed": (
                        calibration_baseline.publication["safety"]["passed"] is True
                        and holdout_baseline.publication["safety"]["passed"] is True
                    ),
                    "calibration": calibration_baseline.publication["safety"],
                    "holdout": holdout_baseline.publication["safety"],
                },
                "early_stopping_control": "calibration_only",
                "holdout_controls_early_stop": False,
            }
        else:
            pilot_baseline = _run_detached_validation(
                config,
                assembly,
                update=0,
                split="calibration",
                generation_format_due=(0 in {
                    int(value)
                    for value in config.validation["generation_format_updates"]
                }),
                rank=rank,
                world_size=world_size,
                baseline_action_logits=None,
            )
            actor_baseline = pilot_baseline.baseline_action_logits
            calibration_validation_cursor = 0
            baseline_publication = pilot_baseline.publication
        validation_cursor = 0
        baseline_error: BaseException | None = None
        baseline_identity_payload: list[str | None] = [None]
        if rank == 0:
            try:
                actor_baseline_identity = _publish_actor_baseline(
                    _actor_baseline_path(run_root),
                    config=config,
                    baseline=actor_baseline,
                )
                baseline_identity_payload[0] = actor_baseline_identity
                path = run_root / "validation_update_00000000.json"
                path.write_text(
                    json.dumps(
                        {
                            **dict(baseline_publication),
                            "actor_baseline_identity": actor_baseline_identity,
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except BaseException as error:
                baseline_error = error
        _coordinated_rank0_status(
            baseline_error,
            rank=rank,
            world_size=world_size,
            operation="baseline validation publication",
        )
        if world_size > 1:
            torch.distributed.broadcast_object_list(baseline_identity_payload, src=0)
        actor_baseline_identity = baseline_identity_payload[0]
        if baseline_publication["safety"]["passed"] is not True:
            update_zero_forensic = run_root / "forensics" / "unsafe_update_00000000"
            update_zero_metric_cursor = {
                "validation": 0,
                "calibration_validation": calibration_validation_cursor,
                "holdout_validation": holdout_validation_cursor,
                "log": 0,
                "wandb": tracking_cursor,
                "teacher_memo": _global_teacher_memo_metric(
                    asdict(assembly.dino_teacher.memo_report()),
                    world_size=world_size,
                ),
                "early_stopping": (
                    early_stopping_cursor.to_mapping()
                    if config.mode == "formal"
                    else None
                ),
                "actual_terminal": None,
            }
            update_zero_control = QueryStateDistributedControl(
                identity=identity,
                global_step=0,
                data_cursor={
                    "mode": config.mode,
                    "next_update": 1,
                    "total_updates": len(updates),
                    "schedule_seed": config.schedule["seed"],
                    "train_manifest_identity": config.data["train_manifest_identity"],
                },
                metric_cursor=update_zero_metric_cursor,
                terminal_primary=False,
                forensic_only=True,
            )
            update_zero_save_error: BaseException | None = None
            try:
                save_query_state_distributed_checkpoint(
                    update_zero_forensic,
                    root=assembly.distributed_worker.root,
                    optimizer=assembly.distributed_worker.optimizer,
                    scheduler_state=assembly.scheduler.state_dict(),
                    control=update_zero_control,
                    rank=rank,
                )
            except BaseException as error:
                update_zero_save_error = error
            terminal_error: BaseException | None = None
            if rank == 0:
                assert store is not None
                publication_errors: list[str] = []
                forensic_evidence_published = False
                try:
                    if update_zero_save_error is None:
                        store.record_unsafe_forensic_checkpoint(
                            start_update=0,
                            end_update=0,
                            checkpoint_path=update_zero_forensic,
                            checkpoint_control_hash=_checkpoint_control_hash(
                                update_zero_forensic
                            ),
                            validation=baseline_publication,
                            safety=baseline_publication["safety"],
                        )
                    else:
                        store.record_forensic_save_failure(
                            update=0,
                            validation=baseline_publication,
                            safety=baseline_publication["safety"],
                            error=(
                                f"{type(update_zero_save_error).__name__}: "
                                f"{update_zero_save_error}"
                            ),
                        )
                    forensic_evidence_published = True
                except BaseException as error:
                    publication_errors.append(
                        f"durable failure evidence: {type(error).__name__}: {error}"
                    )
                try:
                    controller.record_terminal(
                        status="validator_failed",
                        details={
                            "update": 0,
                            "validation": baseline_publication,
                            "non_resumable_safety_failure": True,
                            "forensic_checkpoint_preserved": (
                                update_zero_save_error is None
                            ),
                            "forensic_failure_evidence_published": (
                                forensic_evidence_published
                            ),
                            "forensic_checkpoint": (
                                str(update_zero_forensic)
                                if update_zero_save_error is None
                                else None
                            ),
                            "forensic_checkpoint_save_error": (
                                None
                                if update_zero_save_error is None
                                else f"{type(update_zero_save_error).__name__}: "
                                f"{update_zero_save_error}"
                            ),
                            "authoritative_index_advanced": False,
                        },
                    )
                except BaseException as error:
                    publication_errors.append(
                        f"controller terminal evidence: {type(error).__name__}: {error}"
                    )
                if publication_errors:
                    terminal_error = RuntimeError("; ".join(publication_errors))
            _coordinated_rank0_status(
                terminal_error,
                rank=rank,
                world_size=world_size,
                operation="update-zero forensic checkpoint publication",
            )
            if update_zero_save_error is not None:
                raise RuntimeError(
                    "Query-State update-zero forensic checkpoint save failed"
                ) from update_zero_save_error
            raise RuntimeError(
                "Query-State update-zero safety failed; forensic checkpoint "
                "preserved without optimizer update or authoritative index"
            )
    else:
        actor_baseline, actor_baseline_identity = _load_actor_baseline(
            _actor_baseline_path(run_root), config=config
        )
        replay_evidence_error: BaseException | None = None
        if crash_replay and rank == 0:
            try:
                _verify_replayed_update_zero_evidence(
                    run_root,
                    actor_baseline_identity=actor_baseline_identity,
                )
            except BaseException as error:
                replay_evidence_error = error
        if crash_replay:
            _coordinated_rank0_status(
                replay_evidence_error,
                rank=rank,
                world_size=world_size,
                operation="crash-replay update-zero evidence verification",
            )
    if actor_baseline is None or actor_baseline_identity is None:
        raise RuntimeError("Query-State immutable ID176 actor baseline is unavailable")

    epoch_updates = int(config.schedule["epoch_updates"])
    cadence = int(config.schedule["checkpoint_cadence_updates"])
    final_checkpoint = (
        Path(str(config.initialization["resume_checkpoint"]))
        if resume_mode == "exact_restart"
        else Path()
    )
    final_update = start_update
    terminal_reason: str | None = None
    terminal_epoch: int | None = None
    for segment_start in range(start_update, len(updates), cadence):
        segment_end = min(segment_start + cadence, len(updates))
        if segment_end - segment_start != cadence:
            raise ValueError("Query-State schedule ends outside a commit boundary")
        segment = None
        segment_error: BaseException | None = None
        if store is not None:
            try:
                segment = store.begin_segment(
                    start_update=segment_start,
                    end_update=segment_end,
                    process_identity=current_process_identity(),
                )
            except BaseException as error:
                segment_error = error
        _coordinated_rank0_status(
            segment_error,
            rank=rank,
            world_size=world_size,
            operation="pending segment creation",
        )
        mirror_records: list[Mapping[str, Any]] = []
        for update_index in range(segment_start, segment_end):
            data = build_query_state_update_dataproto(
                updates[update_index],
                rows_by_ordinal=assembly.rows_by_ordinal,
                padding_row=assembly.padding_row,
                processor=assembly.processor,
                dino_teacher=assembly.dino_teacher,
                max_length=int(config.runtime["max_sequence_length"]),
                source_manifest_identity=str(config.source["source_manifest_identity"]),
            )
            result = assembly.distributed_worker.core.update(data)
            update = update_index + 1
            record = {
                "update": update,
                "metrics": dict(result.metrics),
                "gradient_norm": result.gradient_norm,
                "micro_batch_count": result.micro_batch_count,
            }
            append_error: BaseException | None = None
            if segment is not None:
                try:
                    segment.append_update(record)
                except BaseException as error:
                    append_error = error
            _coordinated_rank0_status(
                append_error,
                rank=rank,
                world_size=world_size,
                operation="pending update-log publication",
            )
            mirror_records.append(record)
            log_cursor = update

        validation: Mapping[str, Any] = {"due": False, "update": segment_end}
        safety: Mapping[str, Any] = {
            "passed": True,
            "scope": "no_validation_due_at_this_commit_boundary",
            "actor_baseline_identity": actor_baseline_identity,
            "automatic_model_quality_pass": None,
        }
        actual_terminal: Mapping[str, Any] | None = None
        if config.mode == "formal" and segment_end % epoch_updates == 0:
            epoch = segment_end // epoch_updates
            calibration_result = _run_detached_validation(
                config,
                assembly,
                update=segment_end,
                split="calibration",
                generation_format_due=False,
                rank=rank,
                world_size=world_size,
                baseline_action_logits=actor_baseline,
            )
            calibration_metrics = calibration_result.publication["diagnostics"][
                "metrics"
            ]
            decision = advance_query_state_early_stopping(
                early_stopping_cursor,
                epoch=epoch,
                update=segment_end,
                calibration_dino_mse=float(
                    calibration_metrics["direct_state/dino_mse"]
                ),
                calibration_assistant_ce=float(
                    calibration_metrics["lm/archived_assistant_ce"]
                ),
                min_epochs=int(config.early_stopping["min_epochs"]),
                max_epochs=int(config.early_stopping["max_epochs"]),
                patience_epochs=int(config.early_stopping["patience_epochs"]),
                min_relative_improvement=float(
                    config.early_stopping["min_relative_improvement"]
                ),
            )
            _coordinate_early_stopping_decision(
                decision,
                world_size=world_size,
            )
            early_stopping_cursor = decision.cursor
            plan = _validation_boundary_plan(
                config,
                update=segment_end,
                epoch=epoch,
                actual_terminal=decision.should_stop,
            )
            holdout_publication: Mapping[str, Any] = {
                "due": False,
                "update": segment_end,
                "reason": "not_in_registered_holdout_cadence",
            }
            holdout_safety: Mapping[str, Any] = {"passed": True}
            if plan["holdout"]:
                holdout_result = _run_detached_validation(
                    config,
                    assembly,
                    update=segment_end,
                    split="holdout",
                    generation_format_due=plan["generation_format"],
                    rank=rank,
                    world_size=world_size,
                    baseline_action_logits=actor_baseline,
                )
                holdout_publication = holdout_result.publication
                holdout_safety = holdout_publication["safety"]
                holdout_validation_cursor = segment_end
            calibration_validation_cursor = segment_end
            validation_cursor = segment_end
            if decision.should_stop:
                actual_terminal = {
                    "epoch": epoch,
                    "update": segment_end,
                    "reason": decision.reason,
                    "terminal_primary": True,
                }
            validation = {
                "due": True,
                "update": segment_end,
                "calibration": calibration_result.publication,
                "holdout": holdout_publication,
                "early_stopping": asdict(decision),
                "actual_terminal": actual_terminal,
                "early_stopping_control": "calibration_only",
                "holdout_controls_early_stop": False,
            }
            safety = {
                "passed": (
                    calibration_result.publication["safety"]["passed"] is True
                    and holdout_safety.get("passed") is True
                ),
                "scope": "global_id176_actor_generation_safety",
                "calibration": calibration_result.publication["safety"],
                "holdout": holdout_safety,
                "actor_baseline_identity": actor_baseline_identity,
                "automatic_model_quality_pass": None,
            }
            mirror_records[-1] = {
                **dict(mirror_records[-1]),
                "calibration_composite": decision.composite,
                "early_stopping": early_stopping_cursor.to_mapping(),
                "actual_terminal": actual_terminal,
            }
        elif config.mode == "formal":
            validation = {
                "due": False,
                "update": segment_end,
                "scope": "no_validation_due_at_sub_epoch_commit",
                "early_stopping": early_stopping_cursor.to_mapping(),
                "actual_terminal": None,
            }
            safety = {
                "passed": True,
                "scope": "no_validation_due_at_sub_epoch_commit",
                "actor_baseline_identity": actor_baseline_identity,
                "automatic_model_quality_pass": None,
            }
        elif segment_end in validation_updates:
            validation_result = _run_detached_validation(
                config,
                assembly,
                update=segment_end,
                split="calibration",
                generation_format_due=(segment_end in {
                    int(value)
                    for value in config.validation["generation_format_updates"]
                }),
                rank=rank,
                world_size=world_size,
                baseline_action_logits=actor_baseline,
            )
            validation = validation_result.publication
            safety = {
                **dict(validation["safety"]),
                "scope": "global_id176_actor_safety",
                "actor_baseline_identity": actor_baseline_identity,
                "automatic_model_quality_pass": None,
            }
            calibration_validation_cursor = segment_end
            validation_cursor = segment_end
        teacher_memo_metric = _global_teacher_memo_metric(
            asdict(assembly.dino_teacher.memo_report()),
            world_size=world_size,
        )
        metric_cursor = {
            "validation": validation_cursor,
            "calibration_validation": calibration_validation_cursor,
            "holdout_validation": holdout_validation_cursor,
            "log": log_cursor,
            "wandb": tracking_cursor,
            "teacher_memo": teacher_memo_metric,
            "early_stopping": (
                early_stopping_cursor.to_mapping()
                if config.mode == "formal"
                else None
            ),
            "actual_terminal": actual_terminal,
        }
        data_cursor = {
            "mode": config.mode,
            "next_update": segment_end + 1,
            "total_updates": len(updates),
            "schedule_seed": config.schedule["seed"],
            "train_manifest_identity": config.data["train_manifest_identity"],
        }
        if safety.get("passed") is not True:
            forensic_checkpoint = (
                run_root / "forensics" / f"unsafe_update_{segment_end:08d}"
            )
            forensic_metric_cursor = _forensic_metric_cursor(metric_cursor)
            forensic_control = QueryStateDistributedControl(
                identity=identity,
                global_step=segment_end,
                terminal_primary=False,
                forensic_only=True,
                data_cursor=data_cursor,
                metric_cursor=forensic_metric_cursor,
            )
            forensic_save_error: BaseException | None = None
            try:
                save_query_state_distributed_checkpoint(
                    forensic_checkpoint,
                    root=assembly.distributed_worker.root,
                    optimizer=assembly.distributed_worker.optimizer,
                    scheduler_state=assembly.scheduler.state_dict(),
                    control=forensic_control,
                    rank=rank,
                )
            except BaseException as error:
                forensic_save_error = error
            if forensic_save_error is not None:
                save_failure_publication_error: BaseException | None = None
                if rank == 0:
                    assert store is not None
                    publication_errors: list[str] = []
                    forensic_evidence_published = False
                    try:
                        store.record_forensic_save_failure(
                            update=segment_end,
                            validation=validation,
                            safety=safety,
                            error=(
                                f"{type(forensic_save_error).__name__}: "
                                f"{forensic_save_error}"
                            ),
                        )
                        forensic_evidence_published = True
                    except BaseException as error:
                        publication_errors.append(
                            f"durable failure evidence: {type(error).__name__}: {error}"
                        )
                    try:
                        controller.record_terminal(
                            status="validator_failed",
                            details={
                                "update": segment_end,
                                "validation": validation,
                                "safety": safety,
                                "non_resumable_safety_failure": True,
                                "forensic_checkpoint_preserved": False,
                                "forensic_failure_evidence_published": (
                                    forensic_evidence_published
                                ),
                                "forensic_checkpoint_save_error": (
                                    f"{type(forensic_save_error).__name__}: "
                                    f"{forensic_save_error}"
                                ),
                                "authoritative_index_advanced": False,
                            },
                        )
                    except BaseException as error:
                        publication_errors.append(
                            f"controller terminal evidence: {type(error).__name__}: {error}"
                        )
                    if publication_errors:
                        save_failure_publication_error = RuntimeError(
                            "; ".join(publication_errors)
                        )
                _coordinated_rank0_status(
                    save_failure_publication_error,
                    rank=rank,
                    world_size=world_size,
                    operation="forensic checkpoint save failure publication",
                )
                raise RuntimeError(
                    "Query-State unsafe forensic checkpoint save failed"
                ) from forensic_save_error
            failure_error: BaseException | None = None
            if segment is not None:
                publication_errors: list[str] = []
                expected_failure_recorded = False
                try:
                    segment.commit(
                        checkpoint_path=forensic_checkpoint,
                        checkpoint_control_hash=_checkpoint_control_hash(
                            forensic_checkpoint
                        ),
                        data_cursor=data_cursor,
                        metric_cursor=forensic_metric_cursor,
                        validation=validation,
                        safety=safety,
                        mirror_records=tuple(mirror_records),
                    )
                except RuntimeError as error:
                    if "preserved a forensic checkpoint" in str(error):
                        expected_failure_recorded = True
                    else:
                        publication_errors.append(
                            f"durable failure evidence: {type(error).__name__}: {error}"
                        )
                except BaseException as error:
                    publication_errors.append(
                        f"durable failure evidence: {type(error).__name__}: {error}"
                    )
                if not expected_failure_recorded and not publication_errors:
                    publication_errors.append(
                        "durable failure evidence: unsafe manifest was not published"
                    )
                try:
                    controller.record_terminal(
                        status="validator_failed",
                        details={
                            "update": segment_end,
                            "validation": validation,
                            "safety": safety,
                            "non_resumable_safety_failure": True,
                            "forensic_checkpoint_preserved": True,
                            "forensic_failure_evidence_published": (
                                expected_failure_recorded
                            ),
                            "forensic_checkpoint": str(forensic_checkpoint),
                            "checkpoint_control_sha256": (
                                _checkpoint_control_hash(forensic_checkpoint)
                            ),
                            "authoritative_index_advanced": False,
                        },
                    )
                except BaseException as error:
                    publication_errors.append(
                        f"controller terminal evidence: {type(error).__name__}: {error}"
                    )
                if publication_errors:
                    failure_error = RuntimeError("; ".join(publication_errors))
            _coordinated_rank0_status(
                failure_error,
                rank=rank,
                world_size=world_size,
                operation="validator failure forensic checkpoint publication",
            )
            raise RuntimeError(
                "Query-State actor safety failed; forensic checkpoint preserved "
                "without authoritative index"
            )
        checkpoint = run_root / "checkpoints" / f"update_{segment_end:08d}"
        control = QueryStateDistributedControl(
            identity=identity,
            global_step=segment_end,
            terminal_primary=(config.mode == "formal" and actual_terminal is not None),
            data_cursor=data_cursor,
            metric_cursor=metric_cursor,
        )
        save_query_state_distributed_checkpoint(
            checkpoint,
            root=assembly.distributed_worker.root,
            optimizer=assembly.distributed_worker.optimizer,
            scheduler_state=assembly.scheduler.state_dict(),
            control=control,
            rank=rank,
        )
        if world_size > 1:
            torch.distributed.barrier()
        entry = None
        commit_error: BaseException | None = None
        if segment is not None:
            try:
                entry = segment.commit(
                    checkpoint_path=checkpoint,
                    checkpoint_control_hash=_checkpoint_control_hash(checkpoint),
                    data_cursor=control.data_cursor,
                    metric_cursor=metric_cursor,
                    validation=validation,
                    safety=safety,
                    mirror_records=tuple(mirror_records),
                )
            except BaseException as error:
                commit_error = error
        payload = [
            entry,
            None
            if commit_error is None
            else f"{type(commit_error).__name__}: {commit_error}",
        ]
        if world_size > 1:
            torch.distributed.broadcast_object_list(payload, src=0)
        if payload[1] is not None:
            raise RuntimeError(
                "Query-State authoritative segment commit failed: " + str(payload[1])
            )
        entry = payload[0]
        if entry is None:
            raise RuntimeError("Query-State authoritative segment publication failed")
        tracking.publish(entry)
        if tracking.mirror is not None:
            tracking_cursor = tracking.mirror.cursor
        final_checkpoint = checkpoint
        final_update = segment_end
        if actual_terminal is not None:
            terminal_reason = str(actual_terminal["reason"])
            terminal_epoch = int(actual_terminal["epoch"])
            break
        if _approved_pause_due(config, segment_end=segment_end):
            pause_error: BaseException | None = None
            if rank == 0:
                try:
                    controller.record_pause(
                        update=segment_end,
                        details={
                            "checkpoint": str(checkpoint),
                            "checkpoint_control_hash": _checkpoint_control_hash(
                                checkpoint
                            ),
                            "terminal_primary": False,
                            "validation_update": validation_cursor,
                            "calibration_validation_update": (
                                calibration_validation_cursor
                            ),
                            "holdout_validation_update": holdout_validation_cursor,
                            "safety": safety,
                            "early_stopping": early_stopping_cursor.to_mapping(),
                            "next_action": "human approval required before exact restart",
                        },
                    )
                except BaseException as error:
                    pause_error = error
            _coordinated_rank0_status(
                pause_error,
                rank=rank,
                world_size=world_size,
                operation="approved pause receipt publication",
            )
            return QueryStateTrainingRunResult(
                mode=config.mode,
                final_update=segment_end,
                final_checkpoint=str(checkpoint),
                validation_cursor=validation_cursor,
                log_cursor=log_cursor,
                tracking_cursor=tracking_cursor,
                tracking_incomplete=bool(
                    tracking.mirror and tracking.mirror.tracking_incomplete
                ),
                terminal_epoch=None,
                terminal_reason=None,
            )
        if (
            config.mode == "pilot"
            and resume_mode in {"fresh", "crash_replay"}
            and segment_end == int(config.schedule["forced_restart_update"])
        ):
            # This is a planned process boundary, not a terminal run state.  The
            # approved exact-restart command consumes the immutable checkpoint
            # and durable index before the pilot can publish completion.
            restart_error: BaseException | None = None
            if rank == 0:
                try:
                    restart_path = run_root / "FORCED_RESTART_REQUIRED.json"
                    restart_path.write_text(
                        json.dumps(
                            {
                                "run_identity": run_identity,
                                "mode": "pilot",
                                "resume_update": segment_end,
                                "checkpoint": str(checkpoint),
                                "first_process_identity": current_process_identity(),
                            },
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                except BaseException as error:
                    restart_error = error
            _coordinated_rank0_status(
                restart_error,
                rank=rank,
                world_size=world_size,
                operation="forced-restart receipt publication",
            )
            return QueryStateTrainingRunResult(
                mode=config.mode,
                final_update=segment_end,
                final_checkpoint=str(checkpoint),
                validation_cursor=validation_cursor,
                log_cursor=log_cursor,
                tracking_cursor=tracking_cursor,
                tracking_incomplete=False,
                terminal_epoch=None,
                terminal_reason=None,
            )

    if config.mode == "formal" and terminal_reason is None:
        raise RuntimeError("formal training exhausted without an actual terminal verdict")
    if rank == 0:
        controller.record_terminal(
            status="completed",
            details={
                "final_update": final_update,
                "actual_terminal_epoch": terminal_epoch,
                "actual_terminal_reason": terminal_reason,
                "terminal_primary": config.mode == "formal",
                "checkpoint": str(final_checkpoint),
                "checkpoint_control_hash": _checkpoint_control_hash(final_checkpoint),
                "actor_baseline_identity": actor_baseline_identity,
                "terminal_validation_update": validation_cursor,
                "calibration_validation_update": calibration_validation_cursor,
                "holdout_validation_update": holdout_validation_cursor,
                "early_stopping": (
                    early_stopping_cursor.to_mapping()
                    if config.mode == "formal"
                    else None
                ),
                "terminal_safety_passed": True,
            },
        )
    return QueryStateTrainingRunResult(
        mode=config.mode,
        final_update=final_update,
        final_checkpoint=str(final_checkpoint),
        validation_cursor=validation_cursor,
        log_cursor=log_cursor,
        tracking_cursor=tracking_cursor,
        tracking_incomplete=bool(tracking.mirror and tracking.mirror.tracking_incomplete),
        terminal_epoch=terminal_epoch,
        terminal_reason=terminal_reason,
    )


__all__ = [
    "QueryStateTrainingBackendAssembly",
    "QueryStateTrainingRunResult",
    "build_query_state_training_updates",
    "construct_query_state_training_backend",
    "query_state_training_run_identity",
    "run_query_state_training",
]
