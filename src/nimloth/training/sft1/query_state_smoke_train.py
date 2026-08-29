"""Real constructor owner for one Query-State smoke phase.

This module is not a launcher.  A separately approved ``torchrun`` process must
supply a fully resolved config, initialized NCCL process group, exact rank
device, and fresh process identity.  The phase performs one real update and
publishes one immutable Query-State rank checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
import time
from typing import Any, Mapping

import numpy as np
import torch

from nimloth.backbone.dino_grid import (
    DINOV2_LARGE_IDENTITY,
    FrozenDINOGridTargets,
)
from nimloth.backbone.qwen25vl.factory import build_input_builder, load_backbone
from nimloth.backbone.qwen25vl.loading import load_qwen_processor
from nimloth.training.sft1.data import sha256_file
from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.query_state_checkpoint import (
    QueryStateDistributedControl,
    QueryStateResumeIdentity,
)
from nimloth.training.sft1.query_state_data import FreshQueryStateDINOTeacher
from nimloth.training.sft1.query_state_distributed import (
    QueryStateDistributedWorkerAssembly,
    build_query_state_distributed_worker,
)
from nimloth.training.sft1.query_state_driver import (
    QueryStateScheduledRow,
    build_query_state_update_dataproto,
    restore_query_state_distributed_checkpoint,
    save_query_state_distributed_checkpoint,
)
from nimloth.training.sft1.query_state_runtime import (
    QueryStateConstructedRoot,
    QueryStateWorkerAssembly,
    assemble_query_state_training_root,
    construct_query_state_production_root,
)
from nimloth.training.sft1.query_state_smoke_config import QueryStateSmokeConfig
from nimloth.training.sft1.query_state_smoke_runtime import (
    QueryStateSmokeInventoryEvidence,
    QueryStateSmokePhaseContext,
    QueryStateSmokePhaseOutcome,
    build_query_state_inventory_evidence,
    build_query_state_runtime_fingerprint,
    build_query_state_source_manifest_identity,
    collect_query_state_group_gradient_evidence,
    verify_query_state_smoke_rows,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row, index_early4_rows
from nimloth.training.verl.runtime import MixedPrecisionConfig


_EVIDENCE_KIND = "production_path_checkpoint_resume_smoke_not_model_quality_evidence"


@dataclass(frozen=True)
class QueryStateSmokeTrainingAssembly:
    constructed: QueryStateConstructedRoot
    worker: QueryStateWorkerAssembly
    distributed_worker: QueryStateDistributedWorkerAssembly
    scheduler: torch.optim.lr_scheduler.LambdaLR
    input_builder: Any
    processor: Any
    rows_by_ordinal: Mapping[int, SFT1V2Early4Row]
    padding_row: SFT1V2Early4Row
    dino_teacher: FreshQueryStateDINOTeacher
    inventory: QueryStateSmokeInventoryEvidence
    source_manifest_identity: str
    model_class: str
    processor_class: str


def _seed_runtime(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Query-State smoke seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype(name: str) -> torch.dtype:
    values = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported Query-State smoke dtype: {name}")
    return values[name]


def _backbone_args(config: QueryStateSmokeConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model=config.initialization.actor_checkpoint,
        max_pixels=config.runtime.max_pixels,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        attn_implementation=config.runtime.attention_implementation,
        llm_tune=config.state_contract.llm_tune,
        vision_tune=config.state_contract.vision_tune,
        query_tune=config.state_contract.query_tune,
        lora=config.state_contract.lora,
        resume=False,
    )


def _run_identity(config: QueryStateSmokeConfig) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema": "nimloth_sft1_query_state_smoke_run_v1",
                "source_commit": config.source.expected_commit,
                "source_manifest_identity": config.data.source_manifest_identity,
                "config_identity": config.identity,
                "run_root": config.output.run_root,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _resume_identity(config: QueryStateSmokeConfig) -> QueryStateResumeIdentity:
    return QueryStateResumeIdentity(
        source_commit=config.source.expected_commit,
        source_manifest_identity=config.data.source_manifest_identity,
        config_identity=config.identity,
        run_identity=_run_identity(config),
        world_size=int(config.runtime.world_size),
    )


def _all_gather_strings(value: str, *, world_size: int) -> Mapping[str, str]:
    if len(value) != 64:
        raise ValueError("Query-State smoke fingerprint must be SHA256")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[str | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, value)
        if any(not isinstance(item, str) or len(item) != 64 for item in gathered):
            raise RuntimeError("Query-State smoke gathered fingerprint is invalid")
        return {str(rank): str(item) for rank, item in enumerate(gathered)}
    if world_size != 1:
        raise ValueError("multi-rank Query-State smoke requires a process group")
    return {"0": value}


def _all_gather_records(
    value: Mapping[str, Any],
    *,
    world_size: int,
) -> Mapping[str, Mapping[str, Any]]:
    """Gather JSON-safe rank evidence so rank zero never stands in for peers."""

    # Validate serializability/finite numeric payload before entering a
    # collective; callers build this from already coordinated mechanics data.
    json.dumps(value, sort_keys=True, allow_nan=False)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[Mapping[str, Any] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, dict(value))
        if any(not isinstance(item, Mapping) for item in gathered):
            raise RuntimeError("Query-State smoke gathered rank evidence is invalid")
        return {
            str(rank): dict(item)  # type: ignore[arg-type]
            for rank, item in enumerate(gathered)
        }
    if world_size != 1:
        raise ValueError("multi-rank Query-State smoke requires a process group")
    return {"0": dict(value)}


def _parameter_sha256(root: torch.nn.Module, suffix: str) -> str:
    matches = [
        parameter
        for name, parameter in root.named_parameters()
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Query-State smoke requires one parameter ending {suffix}")
    tensor = matches[0].detach().cpu().contiguous().reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(matches[0].shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Query-State smoke {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Query-State smoke {label} hash mismatch")


def preflight_query_state_smoke(
    config: QueryStateSmokeConfig,
    *,
    phase: str,
) -> Mapping[str, Any]:
    """Read-only source/asset/row/output gate; it never loads Qwen/DINO weights."""

    if not config.preflight_locked or phase not in {"fresh", "resume"}:
        raise PermissionError(
            "Query-State smoke preflight requires an operationally locked phase"
        )
    _require_digest(Path(config.data.train_jsonl), config.data.train_sha256, "train source")
    _require_digest(
        Path(config.data.validation_jsonl),
        config.data.validation_sha256,
        "validation source",
    )
    actor = Path(config.initialization.actor_checkpoint)
    if not actor.is_dir():
        raise FileNotFoundError(f"Query-State smoke ID176 checkpoint is missing: {actor}")
    completion = actor.parent / "complete.marker"
    actor_config = actor / "config.json"
    model_index = actor / "model.safetensors.index.json"
    action_head = actor / "action_head_repair.pt"
    for path, digest, label in (
        (completion, config.initialization.actor_completion_sha256, "ID176 completion"),
        (actor_config, config.initialization.actor_config_sha256, "ID176 config"),
        (model_index, config.initialization.actor_model_index_sha256, "ID176 model index"),
        (action_head, config.initialization.actor_action_head_sha256, "ID176 action head"),
    ):
        _require_digest(path, digest, label)
    actor_payload = json.loads(actor_config.read_text(encoding="utf-8"))
    if (
        actor_payload.get("hidden_size") != 2048
        or actor_payload.get("nimloth_latent_token_count") != 16
        or actor_payload.get("nimloth_latent_query_mode") != "inject"
        or tuple(actor_payload.get("nimloth_action_token_ids", ()))
        != config.initialization.action_token_ids
    ):
        raise ValueError("Query-State smoke ID176 K16/action/hidden contract mismatch")
    index_payload = json.loads(model_index.read_text(encoding="utf-8"))
    weight_map = index_payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Query-State smoke ID176 model index is invalid")
    shard_names = sorted(set(str(value) for value in weight_map.values()))
    if any(
        Path(name).name != name or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise ValueError("Query-State smoke ID176 shard filename is unsafe")
    if len(shard_names) != len(config.initialization.actor_model_shards_sha256):
        raise ValueError("Query-State smoke ID176 shard count mismatch")
    for name, digest in zip(
        shard_names,
        config.initialization.actor_model_shards_sha256,
        strict=True,
    ):
        _require_digest(actor / name, digest, "ID176 model shard")
    processor_identity = audit_id176_processor_identity(actor)
    for name in (
        "processor_sha256",
        "tokenizer_sha256",
        "prompt_template_sha256",
        "token_table_sha256",
        "action_token_ids",
    ):
        if getattr(processor_identity, name) != getattr(config.initialization, name):
            raise ValueError(f"Query-State smoke ID176 processor identity mismatch: {name}")

    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not Path(hf_home).is_absolute():
        raise ValueError("Query-State smoke preflight requires explicit absolute HF_HOME")
    dino_snapshot = (
        Path(hf_home)
        / "hub/models--facebook--dinov2-large/snapshots"
        / config.initialization.dino_revision
    )
    if not dino_snapshot.is_dir():
        raise FileNotFoundError("Query-State smoke pinned DINO snapshot is missing")
    if not (dino_snapshot / "config.json").is_file() or not (
        dino_snapshot / "preprocessor_config.json"
    ).is_file():
        raise FileNotFoundError("Query-State smoke pinned DINO metadata is incomplete")
    weights = tuple(dino_snapshot.glob("*.safetensors")) + tuple(
        dino_snapshot.glob("pytorch_model*.bin")
    )
    if not weights or any(not path.is_file() for path in weights):
        raise FileNotFoundError("Query-State smoke pinned DINO weights are missing")

    processor_bundle = load_qwen_processor(
        config.initialization.actor_checkpoint,
        max_pixels=int(config.runtime.max_pixels),
        latent_token_count=16,
    )
    rows, audit = index_early4_rows(config)
    manifest = build_query_state_source_manifest_identity(rows, audit)
    if manifest != config.data.source_manifest_identity:
        raise ValueError("Query-State smoke source manifest identity mismatch")
    selected = tuple(
        row for row in rows
        if row.ordinal in {item.ordinal for item in config.data.smoke_rows}
    )
    verified = verify_query_state_smoke_rows(
        config,
        rows=selected,
        processor=processor_bundle.processor,
    )

    run_root = Path(config.output.run_root)
    resume_root = run_root / config.output.resume_child
    fresh_checkpoint = (
        run_root
        / config.output.fresh_child
        / config.checkpoint.fresh_checkpoint_name
    )
    if phase == "fresh" and run_root.exists():
        raise FileExistsError("Query-State smoke fresh run root already exists")
    if phase == "resume":
        if not (fresh_checkpoint / config.checkpoint.completion_marker).is_file():
            raise FileNotFoundError("Query-State smoke fresh checkpoint is incomplete")
        if resume_root.exists():
            raise FileExistsError("Query-State smoke resume child already exists")
    disk_parent = run_root.parent
    while not disk_parent.exists() and disk_parent != disk_parent.parent:
        disk_parent = disk_parent.parent
    if shutil.disk_usage(disk_parent).free < int(config.output.minimum_free_bytes):
        raise OSError("Query-State smoke output filesystem free-space gate failed")
    return {
        "kind": "query_state_smoke_read_only_preflight",
        "config_identity": config.identity,
        "source_manifest_identity": manifest,
        "phase": phase,
        "verified_rows": [
            {
                "ordinal": item.row.ordinal,
                "record_id": item.row.record_id,
                "step_index": item.row.step_index,
                "image_sha256": item.row.original_image_sha256,
            }
            for item in verified
        ],
        "dino_snapshot": str(dino_snapshot),
        "output_ownership_verified": True,
        "fresh_run_root_absent": phase == "fresh",
        "resume_child_absent": phase == "resume",
    }


def build_query_state_smoke_training_assembly(
    config: QueryStateSmokeConfig,
    *,
    device: torch.device,
    repo_root: Path,
) -> QueryStateSmokeTrainingAssembly:
    """Load Qwen, wrap the complete root, then load frozen online DINO."""

    if not config.launch_locked:
        raise PermissionError("Query-State smoke training requires launch-locked config")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Query-State production-path smoke requires a CUDA rank")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("Query-State production-path smoke requires a process group")
    loaded = load_backbone(
        _backbone_args(config),
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
        resume_dir=None,
        resume_state_path=None,
    )
    constructed = construct_query_state_production_root(loaded)
    inventory = build_query_state_inventory_evidence(constructed.root)
    worker = assemble_query_state_training_root(
        constructed=constructed,
        device=device,
        repo_root=Path(repo_root),
        wrap_policy=config.runtime.fsdp_wrap_policy,
        mixed_precision=MixedPrecisionConfig(
            param_dtype=_dtype(config.runtime.mixed_precision_param_dtype),
            reduce_dtype=_dtype(config.runtime.mixed_precision_reduce_dtype),
            buffer_dtype=_dtype(config.runtime.mixed_precision_buffer_dtype),
        ),
        language_learning_rate=config.optimizer.language_learning_rate,
        direct_state_learning_rate=config.optimizer.direct_state_learning_rate,
        weight_decay=config.optimizer.weight_decay,
        adam_betas=config.optimizer.betas,
        adam_epsilon=config.optimizer.epsilon,
    )
    model = loaded.backbone.model
    if not model.training:
        raise RuntimeError("Query-State Qwen must be in train mode after FSDP assembly")
    checkpoint_modules = [
        module
        for module in model.modules()
        if bool(getattr(module, "gradient_checkpointing", False))
    ]
    if not checkpoint_modules or not all(module.training for module in checkpoint_modules):
        raise RuntimeError("Query-State gradient checkpointing/train-mode gate is inactive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    if config.optimizer.scheduler != "constant_lambda_1":
        raise ValueError("Query-State smoke runner owns only constant LambdaLR")
    input_builder = build_input_builder(
        loaded,
        max_length=int(config.runtime.max_sequence_length),
        latent_token_count=16,
        mask_latent_query_labels=True,
    )

    # Source/row identity is recomputed online.  No student hidden/state or DINO
    # target cache is accepted by this path.
    rows, audit = index_early4_rows(config)
    manifest = build_query_state_source_manifest_identity(rows, audit)
    if manifest != config.data.source_manifest_identity:
        raise ValueError("Query-State smoke source manifest identity mismatch")
    verified = verify_query_state_smoke_rows(
        config,
        rows=tuple(
            row for row in rows
            if row.ordinal in {item.ordinal for item in config.data.smoke_rows}
        ),
        processor=loaded.processor,
    )
    rows_by_ordinal = {item.row.ordinal: item.row for item in verified}
    train_rows = tuple(row for row in rows if row.split == config.data.train_split)
    if not train_rows:
        raise ValueError("Query-State smoke source has no real train padding owner")

    # Load DINO only after Qwen has been FULL_SHARD wrapped.  It remains outside
    # the training root and optimizer.
    dino = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=device,
        dtype=_dtype(config.runtime.model_dtype),
        grid_size=config.initialization.dino_grid_size,
        batch_size=1,
    )
    dino_teacher = FreshQueryStateDINOTeacher(dino)
    distributed_worker = build_query_state_distributed_worker(
        worker=worker,
        input_builder=input_builder,
        device=device,
        max_padded_tokens=int(config.runtime.max_padded_tokens),
        max_rows=config.runtime.max_rows_per_micro_batch,
        max_grad_norm=config.optimizer.max_grad_norm,
        scheduler=scheduler,
    )
    return QueryStateSmokeTrainingAssembly(
        constructed=constructed,
        worker=worker,
        distributed_worker=distributed_worker,
        scheduler=scheduler,
        input_builder=input_builder,
        processor=loaded.processor,
        rows_by_ordinal=rows_by_ordinal,
        padding_row=train_rows[0],
        dino_teacher=dino_teacher,
        inventory=inventory,
        source_manifest_identity=manifest,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        processor_class=(
            f"{type(loaded.processor).__module__}."
            f"{type(loaded.processor).__qualname__}"
        ),
    )


def _cursor(
    phase: str,
    *,
    world_size: int,
    config: QueryStateSmokeConfig,
) -> Mapping[str, Any]:
    consumed = 1 if phase == "fresh" else 2
    completed_phases = {"fresh"} if phase == "fresh" else {"fresh", "resume"}
    completed_rows = {
        item_phase: {
            str(rank): {
                "ordinal": next(
                    row.ordinal
                    for row in config.data.smoke_rows
                    if row.phase == item_phase and row.rank == rank
                ),
                "row_identity": next(
                    row.row_identity
                    for row in config.data.smoke_rows
                    if row.phase == item_phase and row.rank == rank
                ),
            }
            for rank in range(world_size)
        }
        for item_phase in ("fresh", "resume")
        if item_phase in completed_phases
    }
    return {
        "schema": "nimloth_sft1_query_state_smoke_cursor_v1",
        "phase_completed": phase,
        "next_phase": "resume" if phase == "fresh" else "complete",
        "updates_completed": consumed,
        "consumed_rank_rows": {str(rank): consumed for rank in range(world_size)},
        "completed_rows_by_phase_and_rank": completed_rows,
        "world_size": world_size,
        "row_schedule_identity": hashlib.sha256(
            json.dumps(
                [asdict(row) for row in config.data.smoke_rows],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


def execute_query_state_smoke_phase(
    context: QueryStateSmokePhaseContext,
    *,
    repo_root: Path,
    device: torch.device,
) -> QueryStateSmokePhaseOutcome:
    """Execute exactly one update for an already authorized phase context."""

    config = context.config
    if context.phase not in {"fresh", "resume"}:
        raise ValueError("Query-State smoke phase is invalid")
    seed = config.runtime.seed
    if not isinstance(seed, int):
        raise ValueError("Query-State smoke seed remains unresolved")
    _seed_runtime(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    phase_started = time.perf_counter()
    build_started = time.perf_counter()
    assembly = build_query_state_smoke_training_assembly(
        config,
        device=device,
        repo_root=repo_root,
    )
    build_seconds = time.perf_counter() - build_started
    identity = _resume_identity(config)

    restore_seconds = 0.0
    restored_fingerprints: Mapping[str, str] | None = None
    if context.phase == "resume":
        if context.previous_checkpoint_path is None:
            raise ValueError("Query-State resume phase lacks the fresh checkpoint")
        restore_started = time.perf_counter()
        control, scheduler_state = restore_query_state_distributed_checkpoint(
            context.previous_checkpoint_path,
            root=assembly.distributed_worker.root,
            optimizer=assembly.distributed_worker.optimizer,
            expected_identity=identity,
            rank=context.rank,
        )
        assembly.scheduler.load_state_dict(dict(scheduler_state))
        restore_seconds = time.perf_counter() - restore_started
        if (
            control.global_step != 1
            or dict(control.data_cursor)
            != dict(
                _cursor(
                    "fresh",
                    world_size=context.world_size,
                    config=config,
                )
            )
        ):
            raise ValueError("Query-State smoke resume cursor is not step-1 complete")
        if (
            context.previous_process_identity is None
            or control.metric_cursor.get("process_identity")
            != context.previous_process_identity
        ):
            raise ValueError("Query-State smoke fresh-process checkpoint binding mismatch")
        restored = build_query_state_runtime_fingerprint(
            assembly.distributed_worker.root,
            assembly.distributed_worker.optimizer,
            scheduler_state=assembly.scheduler.state_dict(),
        )
        restored_fingerprints = _all_gather_strings(
            restored.identity,
            world_size=context.world_size,
        )
        expected = control.metric_cursor.get("runtime_fingerprints")
        if restored_fingerprints != expected:
            raise ValueError("Query-State smoke restored runtime fingerprint mismatch")

    descriptor = context.row_descriptor
    if descriptor.ordinal not in assembly.rows_by_ordinal:
        raise ValueError("Query-State smoke exact phase row is absent after verification")
    data = build_query_state_update_dataproto(
        (QueryStateScheduledRow(ordinal=descriptor.ordinal, row_valid=True),),
        rows_by_ordinal=assembly.rows_by_ordinal,
        padding_row=assembly.padding_row,
        processor=assembly.processor,
        dino_teacher=assembly.dino_teacher,
        max_length=int(config.runtime.max_sequence_length),
        source_manifest_identity=assembly.source_manifest_identity,
    )
    direct_before = _parameter_sha256(
        assembly.distributed_worker.root,
        "objective.projector.linear.weight",
    )
    direct_before_by_rank = _all_gather_strings(
        direct_before,
        world_size=context.world_size,
    )
    update_started = time.perf_counter()
    update = assembly.distributed_worker.core.update(data)
    update_seconds = time.perf_counter() - update_started
    group_gradients = collect_query_state_group_gradient_evidence(
        assembly.distributed_worker.optimizer,
        device=device,
    )
    direct_after = _parameter_sha256(
        assembly.distributed_worker.root,
        "objective.projector.linear.weight",
    )
    direct_after_by_rank = _all_gather_strings(
        direct_after,
        world_size=context.world_size,
    )
    if direct_before_by_rank == direct_after_by_rank:
        raise RuntimeError("Query-State smoke direct-state owner did not update")

    fingerprint = build_query_state_runtime_fingerprint(
        assembly.distributed_worker.root,
        assembly.distributed_worker.optimizer,
        scheduler_state=assembly.scheduler.state_dict(),
    )
    fingerprints = _all_gather_strings(
        fingerprint.identity,
        world_size=context.world_size,
    )
    expected_update_metrics = {
        "loss/direct_state_mse",
        "count/direct_state_mse",
        "loss/lm_ce",
        "count/lm_ce",
    }
    if set(update.metrics) != expected_update_metrics:
        raise RuntimeError("Query-State smoke active metric set is incomplete or widened")
    phase_descriptors = tuple(
        row for row in config.data.smoke_rows if row.phase == context.phase
    )
    expected_state_count = context.world_size * 16 * 1024
    expected_lm_count = sum(row.valid_lm_token_count for row in phase_descriptors)
    if (
        update.metrics["count/direct_state_mse"] != float(expected_state_count)
        or update.metrics["count/lm_ce"] != float(expected_lm_count)
        or update.micro_batch_count != 1
    ):
        raise RuntimeError(
            "Query-State smoke global denominator/microbatch evidence mismatch"
        )
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in update.metrics.values()
    ):
        raise RuntimeError("Query-State smoke mechanics metrics are non-finite")
    metrics = {
        **update.metrics,
        "loss/weighted_total": (
            config.state_contract.state_weight
            * update.metrics["loss/direct_state_mse"]
            + config.state_contract.lm_weight * update.metrics["loss/lm_ce"]
        ),
        "gradient_norm": update.gradient_norm,
        "micro_batch_count": update.micro_batch_count,
        "optimizer_group_gradient_norms": dict(group_gradients.group_norms),
    }
    control = QueryStateDistributedControl(
        identity=identity,
        global_step=context.expected_global_step,
        data_cursor=_cursor(
            context.phase,
            world_size=context.world_size,
            config=config,
        ),
        metric_cursor={
            "kind": _EVIDENCE_KIND,
            "process_identity": context.process_identity,
            "approved_command_sha256": (
                config.authorization.approved_command_sha256
            ),
            "runtime_identity": {
                "python": config.source.python_version,
                "torch": config.source.torch_version,
                "transformers": config.source.transformers_version,
                "model_class": assembly.model_class,
                "processor_class": assembly.processor_class,
                "processor_sha256": config.initialization.processor_sha256,
                "tokenizer_sha256": config.initialization.tokenizer_sha256,
                "action_token_ids": list(config.initialization.action_token_ids),
                "dino_source": config.initialization.dino_source,
                "dino_revision": config.initialization.dino_revision,
                "dino_processor_fingerprint": (
                    config.initialization.dino_processor_fingerprint
                ),
                "hf_home": os.environ.get("HF_HOME"),
                "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
                "python_pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
                "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "group_rank": os.environ.get("GROUP_RANK"),
                "local_world_size": os.environ.get("LOCAL_WORLD_SIZE"),
            },
            "runtime_fingerprints": dict(fingerprints),
            "restored_runtime_fingerprints": (
                None if restored_fingerprints is None else dict(restored_fingerprints)
            ),
            "inventory": asdict(assembly.inventory),
            "metrics": metrics,
            "row_descriptors_by_rank": {
                str(rank): asdict(
                    next(
                        row
                        for row in config.data.smoke_rows
                        if row.phase == context.phase and row.rank == rank
                    )
                )
                for rank in range(context.world_size)
            },
            "direct_state_sha256_before_by_rank": dict(direct_before_by_rank),
            "direct_state_sha256_after_by_rank": dict(direct_after_by_rank),
        },
    )
    checkpoint_started = time.perf_counter()
    save_query_state_distributed_checkpoint(
        context.checkpoint_path,
        root=assembly.distributed_worker.root,
        optimizer=assembly.distributed_worker.optimizer,
        scheduler_state=assembly.scheduler.state_dict(),
        control=control,
        rank=context.rank,
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    per_rank_mechanics = _all_gather_records(
        {
            "row_descriptor": asdict(descriptor),
            "timing_seconds": {
                "assembly": build_seconds,
                "restore": restore_seconds,
                "update": update_seconds,
                "checkpoint": checkpoint_seconds,
                "phase": time.perf_counter() - phase_started,
            },
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        },
        world_size=context.world_size,
    )
    evidence = {
        "kind": _EVIDENCE_KIND,
        "phase": context.phase,
        "global_step": context.expected_global_step,
        "config_identity": config.identity,
        "source_manifest_identity": assembly.source_manifest_identity,
        "run_identity": identity.run_identity,
        "approved_command_manifest": context.approved_command_manifest,
        "approved_command_sha256": config.authorization.approved_command_sha256,
        "runtime_identity": {
            "python": config.source.python_version,
            "torch": config.source.torch_version,
            "transformers": config.source.transformers_version,
            "model_class": assembly.model_class,
            "processor_class": assembly.processor_class,
            "processor_sha256": config.initialization.processor_sha256,
            "tokenizer_sha256": config.initialization.tokenizer_sha256,
            "action_token_ids": list(config.initialization.action_token_ids),
            "dino_source": config.initialization.dino_source,
            "dino_revision": config.initialization.dino_revision,
            "dino_processor_fingerprint": (
                config.initialization.dino_processor_fingerprint
            ),
            "hf_home": os.environ.get("HF_HOME"),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "python_pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "cuda_device_name": torch.cuda.get_device_name(device),
            "group_rank": os.environ.get("GROUP_RANK"),
            "local_world_size": os.environ.get("LOCAL_WORLD_SIZE"),
        },
        "row_descriptor": asdict(descriptor),
        "metrics": metrics,
        "runtime_fingerprints": dict(fingerprints),
        "restored_runtime_fingerprints": (
            None if restored_fingerprints is None else dict(restored_fingerprints)
        ),
        "direct_state_sha256_before_by_rank": dict(direct_before_by_rank),
        "direct_state_sha256_after_by_rank": dict(direct_after_by_rank),
        "per_rank_mechanics": dict(per_rank_mechanics),
        "inventory": asdict(assembly.inventory),
    }
    return QueryStateSmokePhaseOutcome(
        global_step=context.expected_global_step,
        checkpoint_path=context.checkpoint_path,
        evidence=evidence,
    )


__all__ = [
    "QueryStateSmokeTrainingAssembly",
    "build_query_state_smoke_training_assembly",
    "execute_query_state_smoke_phase",
    "preflight_query_state_smoke",
]
