"""Deterministic production driver interfaces for SFT1-v2 smoke/train/resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn

from nimloth.backbone.qwen25vl.factory import build_input_builder, load_backbone
from nimloth.training.sft1.checkpoint import (
    SFT1V2ControlState,
    capture_sft1_v2_rank_state,
    finalize_sft1_v2_checkpoint,
    load_sft1_v2_rank_state,
    restore_sft1_v2_rank_state,
    save_sft1_v2_rank_state,
)
from nimloth.training.sft1.experiment_config import SFT1V2Config
from nimloth.training.sft1.data import prepare_sft1_v2_row
from nimloth.training.sft1.manifest import (
    SFT1V2Manifest,
    SFT1_V2_MANIFEST_SCHEMA,
    SFT1_V2_SUPERVISION_SCHEMA,
)
from nimloth.training.sft1.objective import SFT1V2Objective, SFT1V2TrainingRoot
from nimloth.training.sft1.real_rows import (
    SFT1V2Early4Row,
    render_early4_row,
)
from nimloth.training.sft1.teacher_cache import (
    SFT1V2CacheSummary,
    SFT1V2TeacherCacheReader,
)
from nimloth.training.sft1.verl_adapter import build_sft1_v2_dataproto
from nimloth.training.sft1.verl_worker import (
    SFT1V2ParameterGroups,
    SFT1V2WorkerAssembly,
    build_sft1_v2_fsdp_worker,
    capture_sft1_v2_parameter_groups,
)
from nimloth.training.verl.runtime import MixedPrecisionConfig
from nimloth.wm.grid import SharedSlotProjector


@dataclass(frozen=True)
class SFT1V2ScheduledRow:
    ordinal: int | None
    row_valid: bool


@dataclass(frozen=True)
class SFT1V2DataCursor:
    epoch: int
    update_index: int
    consumed_rank_rows: int
    schedule_identity: str
    world_size: int
    rank: int
    epoch_loss_sums: Mapping[str, float] = field(default_factory=dict)
    epoch_loss_counts: Mapping[str, float] = field(default_factory=dict)
    gradient_norm_sum: float = 0.0
    gradient_norm_count: int = 0
    token_counts: tuple[int, ...] = ()
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.epoch < 0
            or self.update_index < 0
            or self.consumed_rank_rows < 0
            or self.world_size < 1
            or not 0 <= self.rank < self.world_size
            or len(self.schedule_identity) != 64
            or any(char not in "0123456789abcdef" for char in self.schedule_identity)
            or self.gradient_norm_count < 0
            or self.gradient_norm_sum < 0
            or not torch.isfinite(torch.tensor(float(self.gradient_norm_sum)))
            or self.elapsed_seconds < 0
            or not torch.isfinite(torch.tensor(float(self.elapsed_seconds)))
            or any(int(value) < 1 for value in self.token_counts)
        ):
            raise ValueError("SFT1-v2 data cursor is invalid")
        if any(
            not torch.isfinite(torch.tensor(float(value)))
            for value in self.epoch_loss_sums.values()
        ) or any(
            float(value) < 0
            or not torch.isfinite(torch.tensor(float(value)))
            for value in self.epoch_loss_counts.values()
        ):
            raise ValueError("SFT1-v2 data cursor metric state is invalid")


@dataclass(frozen=True)
class SFT1V2ProductionAssembly:
    worker: SFT1V2WorkerAssembly
    input_builder: Any
    loaded_backbone: Any
    parameter_groups: SFT1V2ParameterGroups
    model_state_keys: tuple[str, ...]


@dataclass(frozen=True)
class SFT1V2RunResult:
    final_epoch: int
    global_step: int
    stopped_for_actor_safety: bool
    checkpoint_paths: tuple[str, ...]
    validation_reports: tuple[str, ...]


def _identity_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_training_manifest(
    config: SFT1V2Config,
    cache: SFT1V2CacheSummary,
) -> SFT1V2Manifest:
    """Bind the complete fresh-cache and source identities used by training."""

    dino_identity = {
        "source": config.teacher.dino_source,
        "revision": config.teacher.dino_revision,
        "processor_fingerprint": config.teacher.dino_processor_fingerprint,
        "shape": [16, 1024],
    }
    return SFT1V2Manifest(
        schema=SFT1_V2_MANIFEST_SCHEMA,
        objective_version=config.state.objective_version,
        supervision_schema=SFT1_V2_SUPERVISION_SCHEMA,
        source_commit=config.source.expected_commit,
        vagen_commit=config.source.vagen_commit,
        verl_commit=config.source.verl_commit,
        actor_checkpoint_sha256=config.teacher.actor_completion_sha256,
        actor_config_sha256=config.teacher.actor_config_sha256,
        actor_model_index_sha256=config.teacher.actor_model_index_sha256,
        actor_model_shards_sha256=config.teacher.actor_model_shards_sha256,
        processor_sha256=config.teacher.processor_sha256,
        tokenizer_sha256=config.teacher.tokenizer_sha256,
        prompt_template_sha256=config.teacher.prompt_template_sha256,
        token_table_sha256=config.teacher.token_table_sha256,
        dino_checkpoint_sha256=_identity_digest(dino_identity),
        dino_processor_sha256=_identity_digest({
            "fingerprint": config.teacher.dino_processor_fingerprint
        }),
        train_trajectory_sha256=config.data.train_sha256,
        validation_trajectory_sha256=config.data.validation_sha256,
        teacher_cache_sha256=cache.root_manifest_sha256,
        latent_query_mode=config.state.latent_query_mode,
        query_count=config.state.grid_tokens,
        action_count=config.state.action_dim,
        action_token_ids=config.teacher.action_token_ids,
        train_split=config.data.train_split,
        external_validation_split=config.data.validation_split,
    )


def iter_schedule_updates(
    schedule: Sequence[SFT1V2ScheduledRow],
    *,
    rows_per_rank_update: int,
) -> Iterator[tuple[SFT1V2ScheduledRow, ...]]:
    if rows_per_rank_update < 1:
        raise ValueError("rows_per_rank_update must be positive")
    for start in range(0, len(schedule), rows_per_rank_update):
        yield tuple(schedule[start : start + rows_per_rank_update])


def build_update_dataproto(
    scheduled: Sequence[SFT1V2ScheduledRow],
    *,
    rows_by_ordinal: Mapping[int, SFT1V2Early4Row],
    padding_row: SFT1V2Early4Row,
    cache_reader: SFT1V2TeacherCacheReader,
    manifest: SFT1V2Manifest,
    processor: Any,
    config: SFT1V2Config,
    repo_root: Path,
) -> Any:
    """Re-render raw rows and attach detached targets for one rank update."""

    if not scheduled:
        raise ValueError("rank update schedule must not be empty")
    prepared = []
    valid: list[bool] = []
    for item in scheduled:
        row = rows_by_ordinal[int(item.ordinal)] if item.row_valid else padding_row
        rendered = render_early4_row(
            row,
            processor=processor,
            max_length=config.runtime.max_sequence_length,
        )
        teacher = cache_reader.load(row.ordinal)
        prepared.append(prepare_sft1_v2_row(
            dict(row.record),
            step_index=row.step_index,
            encoded_tensors=rendered.encoded_tensors,
            teacher=teacher,
            manifest=manifest,
        ))
        valid.append(item.row_valid)
    data = build_sft1_v2_dataproto(
        prepared,
        manifest=manifest,
        repo_root=repo_root,
    )
    data.batch["row_valid"] = torch.tensor(valid, dtype=torch.bool)
    data.batch["feasibility_label_valid"] &= data.batch["row_valid"]
    return data


def deterministic_epoch_schedule(
    ordinals: Sequence[int],
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[tuple[SFT1V2ScheduledRow, ...], str]:
    """Partition whole row identities and pad every rank to equal forward count."""

    if epoch < 0 or seed < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("deterministic schedule arguments are invalid")
    values = tuple(int(value) for value in ordinals)
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("schedule ordinals must be unique non-negative identities")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{epoch}:{value}".encode()).digest(), value
        ),
    )
    schedules = [ordered[item_rank::world_size] for item_rank in range(world_size)]
    length = max((len(schedule) for schedule in schedules), default=0)
    local = schedules[rank]
    padded = tuple(
        SFT1V2ScheduledRow(
            ordinal=local[index] if index < len(local) else None,
            row_valid=index < len(local),
        )
        for index in range(length)
    )
    identity_payload = f"v1:{seed}:{epoch}:{world_size}:" + ",".join(map(str, ordered))
    return padded, hashlib.sha256(identity_payload.encode()).hexdigest()


def deterministic_update_schedule(
    ordinals: Sequence[int],
    *,
    movement_ordinals: frozenset[int],
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
    rows_per_rank_update: int,
) -> tuple[tuple[SFT1V2ScheduledRow, ...], str]:
    """Plan equal rank updates and guarantee one global movement label each."""

    if (
        rows_per_rank_update < 1
        or world_size < 1
        or not 0 <= rank < world_size
        or epoch < 0
        or seed < 0
    ):
        raise ValueError("update schedule arguments are invalid")
    values = tuple(int(value) for value in ordinals)
    if len(set(values)) != len(values) or any(value < 0 for value in values):
        raise ValueError("update schedule ordinals must be unique non-negative values")
    if not movement_ordinals <= set(values):
        raise ValueError("movement ordinal set is outside schedule rows")
    ordered = sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}:{epoch}:{value}".encode()).digest(), value
        ),
    )
    width = world_size * rows_per_rank_update
    chunks: list[list[int | None]] = [
        list(ordered[start : start + width])
        for start in range(0, len(ordered), width)
    ]
    if chunks:
        chunks[-1].extend([None] * (width - len(chunks[-1])))
    for index, chunk in enumerate(chunks):
        if any(value in movement_ordinals for value in chunk if value is not None):
            continue
        donor = next((
            other for other in range(len(chunks))
            if sum(value in movement_ordinals for value in chunks[other] if value is not None) >= 2
        ), None)
        if donor is None:
            raise ValueError("cannot give every global update a movement label")
        source_position = next(
            position for position, value in enumerate(chunks[donor])
            if value in movement_ordinals
        )
        target_position = next(
            position for position, value in enumerate(chunk) if value is not None
        )
        chunks[donor][source_position], chunk[target_position] = (
            chunk[target_position], chunks[donor][source_position]
        )
    local: list[SFT1V2ScheduledRow] = []
    start = rank * rows_per_rank_update
    for chunk in chunks:
        for value in chunk[start : start + rows_per_rank_update]:
            local.append(SFT1V2ScheduledRow(
                ordinal=value,
                row_valid=value is not None,
            ))
    identity = _identity_digest({
        "schema": "nimloth_sft1_v2_update_schedule_v1",
        "seed": seed,
        "epoch": epoch,
        "world_size": world_size,
        "rows_per_rank_update": rows_per_rank_update,
        "global_chunks": chunks,
    })
    return tuple(local), identity


def resume_schedule(
    schedule: Sequence[SFT1V2ScheduledRow],
    cursor: SFT1V2DataCursor,
    *,
    expected_identity: str,
    rank: int,
    world_size: int,
) -> tuple[SFT1V2ScheduledRow, ...]:
    if cursor.schedule_identity != expected_identity:
        raise ValueError("resume data schedule identity mismatch")
    if cursor.rank != rank or cursor.world_size != world_size:
        raise ValueError("resume data cursor rank/world-size mismatch")
    if cursor.consumed_rank_rows < 0 or cursor.consumed_rank_rows > len(schedule):
        raise ValueError("resume data cursor is outside deterministic schedule")
    return tuple(schedule[cursor.consumed_rank_rows:])


def assert_gradient_checkpointing_train_mode(root: nn.Module) -> None:
    """Fail closed unless the actual Qwen checkpointing gate is active."""

    candidates = [
        module for module in root.modules()
        if hasattr(module, "gradient_checkpointing")
    ]
    if not candidates:
        raise RuntimeError("training Qwen exposes no gradient_checkpointing runtime gate")
    active = [
        module for module in candidates
        if bool(getattr(module, "gradient_checkpointing")) and module.training
    ]
    if not active:
        raise RuntimeError("gradient checkpointing requires the training Qwen in train mode")


def construct_sft1_v2_production(
    *,
    config: SFT1V2Config,
    backbone_args: Any,
    device: torch.device,
    repo_root: Path,
    wrap_policy: Mapping[str, Any] | None,
    mixed_precision: MixedPrecisionConfig,
    load_backbone_fn: Callable[..., Any] = load_backbone,
) -> SFT1V2ProductionAssembly:
    """Load ID176 and build the same complete root used by smoke and formal train.

    Tests may inject a tiny loader, but the default path is the existing real Qwen
    loader; there is no proxy model in the production entry point.
    """

    if str(getattr(backbone_args, "model", "")) != config.teacher.actor_checkpoint:
        raise ValueError("production Qwen source differs from the bound ID176 checkpoint")
    if getattr(backbone_args, "llm_tune", "freeze") != "freeze" or getattr(backbone_args, "vision_tune", "freeze") != "freeze":
        raise ValueError("production SFT1-v2 must freeze Qwen language and vision")
    if not bool(getattr(backbone_args, "gradient_checkpointing", False)):
        raise ValueError("production SFT1-v2 requires gradient checkpointing")
    if getattr(backbone_args, "query_tune", None) != "adapter":
        raise ValueError("production SFT1-v2 requires the additive query adapter")

    loaded = load_backbone_fn(
        backbone_args, device=device, latent_token_count=16,
        model_parallel_size=1, resume_dir=None, resume_state_path=None,
    )
    if loaded.query_adapter is None:
        raise RuntimeError("production loader did not install the query adapter")
    model_state_keys = tuple(sorted(loaded.backbone.model.state_dict()))
    projector = SharedSlotProjector(
        config.state.qwen_hidden_dim, config.state.state_dim,
        hidden_dim=config.state.projector_hidden_dim,
        grid_tokens=config.state.grid_tokens,
    )
    objective = SFT1V2Objective(
        projector=projector, state_dim=config.state.state_dim,
        instruction_teacher_dim=config.state.instruction_teacher_dim,
        grid_tokens=config.state.grid_tokens,
        movement_action_indices=config.state.movement_action_indices,
        policy_temperature=config.objective.policy_temperature,
        contrastive_temperature=config.objective.contrastive_temperature,
        weights=config.objective.weights, action_dim=config.state.action_dim,
    )
    root = SFT1V2TrainingRoot(loaded.backbone, objective)
    root.train()
    assert_gradient_checkpointing_train_mode(root)
    groups = capture_sft1_v2_parameter_groups(root)
    input_builder = build_input_builder(
        loaded, max_length=config.runtime.max_sequence_length,
        latent_token_count=16, mask_latent_query_labels=True,
    )
    effective_wrap_policy = dict(wrap_policy or {})
    if not effective_wrap_policy:
        qwen_layers = getattr(loaded.backbone.model, "_no_split_modules", None)
        if not qwen_layers:
            raise RuntimeError("Qwen model does not declare FSDP transformer layers")
        effective_wrap_policy = {
            "transformer_layer_cls_to_wrap": tuple(str(name) for name in qwen_layers)
        }
    worker = build_sft1_v2_fsdp_worker(
        objective_root=root, input_builder=input_builder, device=device,
        repo_root=repo_root, wrap_policy=effective_wrap_policy,
        mixed_precision=mixed_precision,
        query_learning_rate=config.optimizer.query_learning_rate,
        projector_readout_learning_rate=config.optimizer.projector_readout_learning_rate,
        weight_decay=config.optimizer.weight_decay, adam_betas=config.optimizer.betas,
        adam_epsilon=config.optimizer.epsilon,
        max_padded_tokens=config.runtime.max_padded_tokens,
        max_rows=config.runtime.max_rows_per_micro_batch,
        max_grad_norm=config.runtime.max_grad_norm,
        scheduler_factory=None, parameter_groups=groups,
    )
    return SFT1V2ProductionAssembly(
        worker=worker, input_builder=input_builder,
        loaded_backbone=loaded, parameter_groups=groups,
        model_state_keys=model_state_keys,
    )


def save_training_checkpoint(
    assembly: SFT1V2ProductionAssembly,
    path: Path,
    *,
    cursor: SFT1V2DataCursor,
    manifest: SFT1V2Manifest,
    config: SFT1V2Config,
    run_identity: str,
    rank: int,
    world_size: int,
) -> Path:
    state = capture_sft1_v2_rank_state(
        assembly.worker.root,
        assembly.worker.optimizer,
    )
    save_sft1_v2_rank_state(
        path, rank=rank, world_size=world_size, state=state
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        finalize_sft1_v2_checkpoint(
            path,
            control=SFT1V2ControlState(
                global_step=cursor.update_index,
                data_cursor=asdict(cursor),
                manifest_identity=manifest.identity,
                config_identity=config.identity,
                objective_version=config.state.objective_version,
                world_size=world_size,
                run_identity=run_identity,
                source_commit=config.source.expected_commit,
            ),
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return path


def restore_training_checkpoint(
    assembly: SFT1V2ProductionAssembly,
    path: Path,
    *,
    manifest: SFT1V2Manifest,
    config: SFT1V2Config,
    run_identity: str,
    rank: int,
    world_size: int,
) -> SFT1V2DataCursor:
    state, control = load_sft1_v2_rank_state(
        path,
        rank=rank,
        expected_world_size=world_size,
        expected_manifest_identity=manifest.identity,
        expected_config_identity=config.identity,
        expected_run_identity=run_identity,
        expected_source_commit=config.source.expected_commit,
    )
    restore_sft1_v2_rank_state(
        assembly.worker.root,
        assembly.worker.optimizer,
        state,
    )
    return SFT1V2DataCursor(**control.data_cursor)


def run_sft1_v2_epochs(
    *,
    assembly: SFT1V2ProductionAssembly,
    config: SFT1V2Config,
    rows: Sequence[SFT1V2Early4Row],
    cache_reader: SFT1V2TeacherCacheReader,
    manifest: SFT1V2Manifest,
    repo_root: Path,
    rank: int,
    world_size: int,
    seed: int,
    checkpoint_callback: Callable[[int, int, SFT1V2DataCursor], Path],
    validation_callback: Callable[[int, int, Mapping[str, float]], tuple[Path, bool]],
    update_callback: Callable[[int, int, Mapping[str, float]], None] | None = None,
    resume_cursor: SFT1V2DataCursor | None = None,
) -> SFT1V2RunResult:
    """Run the exact epoch/update graph; callbacks own durable artifacts."""

    if world_size != config.runtime.world_size or not 0 <= rank < world_size:
        raise ValueError("runtime rank/world size differs from resolved config")
    train_rows = tuple(row for row in rows if row.split == config.data.train_split)
    if len(train_rows) != config.selection.train_rows:
        raise ValueError("training row count differs from the approved early-4 contract")
    rows_by_ordinal = {row.ordinal: row for row in rows}
    if len(rows_by_ordinal) != len(rows):
        raise ValueError("early-4 row ordinals are duplicated")
    padding_row = train_rows[0]
    global_step = int(resume_cursor.update_index) if resume_cursor is not None else 0
    start_epoch = int(resume_cursor.epoch) if resume_cursor is not None else 0
    checkpoints: list[str] = []
    reports: list[str] = []

    # Epoch-0 validation is mandatory only for a fresh run.
    if resume_cursor is None:
        report, safety = validation_callback(0, global_step, {})
        reports.append(str(report))
        if safety:
            return SFT1V2RunResult(0, global_step, True, (), tuple(reports))

    stopped = False
    final_epoch = start_epoch
    if (
        resume_cursor is not None
        and resume_cursor.consumed_rank_rows == 0
        and resume_cursor.epoch > 0
    ):
        report, stopped = validation_callback(
            resume_cursor.epoch,
            global_step,
            {},
        )
        reports.append(str(report))
        if stopped:
            return SFT1V2RunResult(
                resume_cursor.epoch,
                global_step,
                True,
                (),
                tuple(reports),
            )
    for epoch in range(start_epoch, config.runtime.epochs):
        schedule, identity = deterministic_update_schedule(
            tuple(row.ordinal for row in train_rows),
            movement_ordinals=frozenset(
                row.ordinal for row in train_rows
                if row.movement_success is not None
            ),
            epoch=epoch,
            seed=seed,
            rank=rank,
            world_size=world_size,
            rows_per_rank_update=config.runtime.rows_per_rank_update,
        )
        if resume_cursor is not None and epoch == resume_cursor.epoch:
            schedule = resume_schedule(
                schedule,
                resume_cursor,
                expected_identity=identity,
                rank=rank,
                world_size=world_size,
            )
            consumed = resume_cursor.consumed_rank_rows
            epoch_sums = dict(resume_cursor.epoch_loss_sums)
            epoch_counts = dict(resume_cursor.epoch_loss_counts)
            gradient_norm_sum = float(resume_cursor.gradient_norm_sum)
            gradient_norm_count = int(resume_cursor.gradient_norm_count)
            token_counts = list(resume_cursor.token_counts)
            previous_elapsed = float(resume_cursor.elapsed_seconds)
        else:
            consumed = 0
            epoch_sums: dict[str, float] = {}
            epoch_counts: dict[str, float] = {}
            gradient_norm_sum = 0.0
            gradient_norm_count = 0
            token_counts: list[int] = []
            previous_elapsed = 0.0
        epoch_started = perf_counter()
        for scheduled in iter_schedule_updates(
            schedule,
            rows_per_rank_update=config.runtime.rows_per_rank_update,
        ):
            data = build_update_dataproto(
                scheduled,
                rows_by_ordinal=rows_by_ordinal,
                padding_row=padding_row,
                cache_reader=cache_reader,
                manifest=manifest,
                processor=assembly.loaded_backbone.processor,
                config=config,
                repo_root=repo_root,
            )
            token_counts.extend(int(value) for value in data.batch["token_counts"].tolist())
            result = assembly.worker.core.update(data)
            global_step += 1
            consumed += len(scheduled)
            if update_callback is not None:
                update_callback(
                    epoch,
                    global_step,
                    {
                        **result.metrics,
                        "gradient_norm": result.gradient_norm,
                        "micro_batch_count": float(result.micro_batch_count),
                    },
                )
            gradient_norm_sum += result.gradient_norm
            gradient_norm_count += 1
            for name, value in result.metrics.items():
                if name.startswith("count/"):
                    epoch_counts[name[6:]] = epoch_counts.get(name[6:], 0.0) + value
                elif name.startswith("loss/"):
                    count = result.metrics.get(f"count/{name[5:]}", 0.0)
                    epoch_sums[name[5:]] = epoch_sums.get(name[5:], 0.0) + value * count
            cursor = SFT1V2DataCursor(
                epoch=epoch,
                update_index=global_step,
                consumed_rank_rows=consumed,
                schedule_identity=identity,
                world_size=world_size,
                rank=rank,
                epoch_loss_sums=epoch_sums,
                epoch_loss_counts=epoch_counts,
                gradient_norm_sum=gradient_norm_sum,
                gradient_norm_count=gradient_norm_count,
                token_counts=tuple(token_counts),
                elapsed_seconds=previous_elapsed + perf_counter() - epoch_started,
            )
            if global_step % config.checkpoint.cadence_steps == 0:
                checkpoints.append(str(checkpoint_callback(epoch, global_step, cursor)))
        elapsed = previous_elapsed + perf_counter() - epoch_started
        elapsed_tensor = torch.tensor(
            elapsed,
            device=assembly.worker.core.device,
            dtype=torch.float64,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                elapsed_tensor, op=torch.distributed.ReduceOp.MAX
            )
            gathered_tokens: list[Any] = [None] * world_size
            torch.distributed.all_gather_object(gathered_tokens, token_counts)
            global_tokens = [
                int(value) for values in gathered_tokens for value in values
            ]
        else:
            global_tokens = token_counts
        runtime: dict[str, float] = {}
        for name, count in epoch_counts.items():
            runtime[f"count/{name}"] = count
            runtime[f"loss/{name}"] = epoch_sums[name] / count if count else 0.0
        runtime.update({
            "gradient_norm": gradient_norm_sum / max(gradient_norm_count, 1),
            "throughput_rows_per_second": (
                consumed * world_size / max(float(elapsed_tensor.item()), 1e-9)
            ),
            "token_count_mean": sum(global_tokens) / max(len(global_tokens), 1),
            "token_count_p95": float(
                torch.tensor(global_tokens, dtype=torch.float32).quantile(0.95).item()
            ),
            "peak_memory_bytes": float(
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else 0
            ),
        })
        _, next_identity = deterministic_update_schedule(
            tuple(row.ordinal for row in train_rows),
            movement_ordinals=frozenset(
                row.ordinal for row in train_rows
                if row.movement_success is not None
            ),
            epoch=epoch + 1,
            seed=seed,
            rank=rank,
            world_size=world_size,
            rows_per_rank_update=config.runtime.rows_per_rank_update,
        )
        epoch_cursor = SFT1V2DataCursor(
            epoch=epoch + 1,
            update_index=global_step,
            consumed_rank_rows=0,
            schedule_identity=next_identity,
            world_size=world_size,
            rank=rank,
        )
        checkpoints.append(str(checkpoint_callback(epoch + 1, global_step, epoch_cursor)))
        report, stopped = validation_callback(epoch + 1, global_step, runtime)
        reports.append(str(report))
        final_epoch = epoch + 1
        resume_cursor = None
        if stopped:
            break
    return SFT1V2RunResult(
        final_epoch=final_epoch,
        global_step=global_step,
        stopped_for_actor_safety=stopped,
        checkpoint_paths=tuple(checkpoints),
        validation_reports=tuple(reports),
    )


def run_one_update_smoke(
    assembly: SFT1V2ProductionAssembly,
    data: Any,
    *,
    checkpoint_callback: Callable[[SFT1V2WorkerAssembly], Path],
    resume_callback: Callable[[Path], SFT1V2ProductionAssembly],
) -> Mapping[str, Any]:
    """Exercise the production update/checkpoint/resume constructor boundary."""

    result = assembly.worker.core.update(data)
    checkpoint = checkpoint_callback(assembly.worker)
    if not Path(checkpoint).is_dir():
        raise RuntimeError("smoke checkpoint callback did not publish a checkpoint")
    resumed = resume_callback(Path(checkpoint))
    if type(resumed) is not SFT1V2ProductionAssembly:
        raise TypeError("resume smoke must return the production assembly type")
    return {
        "kind": "production_path_smoke_not_model_evidence",
        "micro_batch_count": result.micro_batch_count,
        "gradient_norm": result.gradient_norm,
        "checkpoint": str(checkpoint),
        "resume_constructor_verified": True,
    }


__all__ = [
    "SFT1V2DataCursor", "SFT1V2ProductionAssembly", "SFT1V2RunResult",
    "SFT1V2ScheduledRow", "build_training_manifest", "build_update_dataproto",
    "deterministic_update_schedule", "iter_schedule_updates", "restore_training_checkpoint",
    "run_sft1_v2_epochs", "save_training_checkpoint",
    "assert_gradient_checkpointing_train_mode", "construct_sft1_v2_production",
    "deterministic_epoch_schedule", "resume_schedule", "run_one_update_smoke",
]
