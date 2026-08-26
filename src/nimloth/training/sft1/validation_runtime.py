"""Distributed real-row validation and report publication for SFT1-v2."""

from __future__ import annotations

from collections import defaultdict
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from nimloth.training.sft1.experiment_config import SFT1V2Config
from nimloth.training.sft1.driver import (
    SFT1V2ProductionAssembly,
    build_update_dataproto,
    deterministic_update_schedule,
    iter_schedule_updates,
)
from nimloth.training.sft1.manifest import SFT1V2Manifest
from nimloth.training.sft1.objective import SFT1V2Normalization
from nimloth.training.sft1.real_rows import SFT1V2Early4Row
from nimloth.training.sft1.teacher_cache import SFT1V2TeacherCacheReader
from nimloth.training.sft1.validation import (
    SFT1V2ValidationInputs,
    SFT1V2ValidationReport,
    load_validation_report,
    publish_validation_report,
    validate_sft1_v2_components,
)
from nimloth.training.sft1.verl_adapter import sft1_v2_update_inputs


_TARGET_RE = re.compile(r"^navigate to the (.+?) in the room and be as close as possible to it$")


def _all_reduce_tensor(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def _all_reduce_count(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return int(tensor.item())


def _target_object(instruction: str) -> str:
    match = _TARGET_RE.fullmatch(instruction)
    if match is None or not match.group(1):
        raise ValueError("target-object probe requires the exact archived instruction template")
    return match.group(1)


def _centroids(
    rows: Sequence[SFT1V2Early4Row],
    reader: SFT1V2TeacherCacheReader,
    *,
    label: str,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    buckets: dict[str, list[torch.Tensor]] = defaultdict(list)
    seen_records: set[str] = set()
    for row in rows:
        if row.split != "train" or row.record_id in seen_records:
            continue
        seen_records.add(row.record_id)
        key = (
            row.instruction_equivalence_group
            if label == "instruction"
            else _target_object(row.instruction)
        )
        buckets[key].append(reader.load(row.ordinal).instruction_teacher.float())
    keys = tuple(sorted(buckets))
    values = torch.stack([
        torch.stack(buckets[key]).mean(dim=0) for key in keys
    ])
    return F.normalize(values, dim=-1), keys


def _probe_correct(
    prediction: torch.Tensor,
    true_labels: Sequence[str],
    centroids: torch.Tensor,
    centroid_labels: Sequence[str],
) -> torch.Tensor:
    scores = F.normalize(prediction.float(), dim=-1) @ centroids.transpose(0, 1)
    predicted = [centroid_labels[index] for index in scores.argmax(dim=-1).tolist()]
    return torch.tensor(
        [left == right for left, right in zip(predicted, true_labels, strict=True)],
        dtype=torch.float32,
    )


def _concat_payloads(payloads: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    tensor_names = (
        "visual_prediction", "dino_regions", "instruction_prediction",
        "instruction_teacher", "feasibility_logits", "executed_action_indices",
        "movement_success", "feasibility_label_valid", "actor_student_logits",
        "actor_teacher_log_probs", "state_policy_logits", "external_eligible",
    )
    result: dict[str, Any] = {}
    for name in tensor_names:
        result[name] = torch.cat([payload[name] for payload in payloads], dim=0)
    for name in (
        "image_content_groups", "instruction_equivalence_groups", "instructions",
        "token_counts",
    ):
        result[name] = tuple(
            value for payload in payloads for value in payload[name]
        )
    return result


def run_validation_epoch(
    *,
    assembly: SFT1V2ProductionAssembly,
    config: SFT1V2Config,
    rows: Sequence[SFT1V2Early4Row],
    cache_reader: SFT1V2TeacherCacheReader,
    manifest: SFT1V2Manifest,
    repo_root: Path,
    rank: int,
    world_size: int,
    epoch: int,
    checkpoint_step: int,
    checkpoint_identity: str,
    report_path: Path,
    runtime_metrics: Mapping[str, float],
    epoch0_report_path: Path | None,
) -> tuple[Path, bool]:
    """Run equal-order FSDP validation and publish the rank-zero report."""

    validation_rows = tuple(row for row in rows if row.split == config.data.validation_split)
    if len(validation_rows) != config.selection.raw_validation_rows:
        raise ValueError("validation row count differs from approved early-4 contract")
    rows_by_ordinal = {row.ordinal: row for row in rows}
    schedule, _ = deterministic_update_schedule(
        tuple(row.ordinal for row in validation_rows),
        movement_ordinals=frozenset(
            row.ordinal for row in validation_rows if row.movement_success is not None
        ),
        epoch=0,
        seed=config.validation.bootstrap_seed,
        rank=rank,
        world_size=world_size,
        rows_per_rank_update=config.runtime.rows_per_rank_update,
    )
    local: dict[str, list[Any]] = defaultdict(list)
    loss_sums: dict[str, torch.Tensor] = {}
    loss_counts: dict[str, torch.Tensor] = {}
    token_counts: list[int] = []
    started = perf_counter()
    root = assembly.worker.root
    root.eval()
    try:
        with torch.no_grad():
            for scheduled in iter_schedule_updates(
                schedule,
                rows_per_rank_update=config.runtime.rows_per_rank_update,
            ):
                data = build_update_dataproto(
                    scheduled,
                    rows_by_ordinal=rows_by_ordinal,
                    padding_row=validation_rows[0],
                    cache_reader=cache_reader,
                    manifest=manifest,
                    processor=assembly.loaded_backbone.processor,
                    config=config,
                    repo_root=repo_root,
                )
                update = sft1_v2_update_inputs(
                    data,
                    input_builder=assembly.input_builder,
                )
                valid = data.batch["row_valid"].bool()
                actions = data.batch["executed_action_indices"].long()
                movement_valid = (
                    valid
                    & data.batch["feasibility_label_valid"].bool()
                    & ((actions == 0) | (actions == 2) | (actions == 3))
                )
                sample_count = _all_reduce_count(int(valid.sum()), assembly.worker.core.device)
                movement_count = _all_reduce_count(
                    int(movement_valid.sum()), assembly.worker.core.device
                )
                if sample_count < 1 or movement_count < 1:
                    raise ValueError("validation update lacks global sample/movement rows")
                output = root(
                    update.student_batch,
                    update.targets,
                    SFT1V2Normalization(
                        global_sample_valid_count=sample_count,
                        global_feasibility_valid_count=movement_count,
                        gradient_average_world_size=world_size,
                    ),
                )
                token_counts.extend(
                    int(value) for value in data.batch["token_counts"].tolist()
                )
                for name, value in output.loss_sums.items():
                    loss_sums[name] = loss_sums.get(
                        name, torch.zeros_like(value)
                    ) + value.detach()
                for name, value in output.local_valid_counts.items():
                    loss_counts[name] = loss_counts.get(
                        name, torch.zeros_like(value)
                    ) + value.detach()
                indices = torch.nonzero(valid, as_tuple=False).flatten()
                for name, value in (
                    ("visual_prediction", output.visual_prediction),
                    ("instruction_prediction", output.instruction_prediction),
                    ("feasibility_logits", output.feasibility_logits),
                    ("actor_student_logits", output.actor_student_logits),
                    ("state_policy_logits", output.state_policy_logits),
                    ("dino_regions", update.targets.dino_regions),
                    ("instruction_teacher", update.targets.instruction_teacher),
                    ("executed_action_indices", update.targets.executed_action_indices),
                    ("movement_success", update.targets.movement_success),
                    ("feasibility_label_valid", update.targets.feasibility_label_valid),
                    ("actor_teacher_log_probs", update.targets.actor_teacher_log_probs),
                ):
                    local[name].append(value.index_select(0, indices.to(value.device)).detach().cpu())
                selected_rows = [
                    rows_by_ordinal[int(item.ordinal)] for item in scheduled if item.row_valid
                ]
                local["image_content_groups"].extend(
                    row.image_content_group for row in selected_rows
                )
                local["instruction_equivalence_groups"].extend(
                    row.instruction_equivalence_group for row in selected_rows
                )
                local["instructions"].extend(row.instruction for row in selected_rows)
                local["external_eligible"].append(torch.tensor(
                    [row.external_eligible for row in selected_rows], dtype=torch.bool
                ))
    finally:
        root.train()
    elapsed = perf_counter() - started
    measured_runtime: dict[str, float] = {}
    for name in sorted(loss_sums):
        total_sum = _all_reduce_tensor(loss_sums[name].to(assembly.worker.core.device))
        total_count = _all_reduce_tensor(
            loss_counts[name].to(assembly.worker.core.device)
        )
        count = float(total_count.item())
        measured_runtime[f"loss/{name}"] = (
            float(total_sum.item()) / count if count else 0.0
        )
        measured_runtime[f"count/{name}"] = count
    measured_runtime.update({
        "gradient_norm": 0.0,
        "throughput_rows_per_second": len(validation_rows) / max(elapsed, 1e-9),
        "token_count_mean": sum(token_counts) / max(len(token_counts), 1),
        "token_count_p95": float(
            torch.tensor(token_counts, dtype=torch.float32).quantile(0.95).item()
        ),
        "peak_memory_bytes": float(
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        ),
    })
    effective_runtime = dict(measured_runtime if not runtime_metrics else runtime_metrics)
    payload: dict[str, Any] = {}
    for name, values in local.items():
        payload[name] = torch.cat(values) if values and isinstance(values[0], torch.Tensor) else tuple(values)
    payload["token_counts"] = tuple(token_counts)
    gathered: list[Any] = [None] * world_size
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    safety = False
    if rank == 0:
        merged = _concat_payloads(gathered)
        if not runtime_metrics:
            all_tokens = merged["token_counts"]
            effective_runtime["token_count_mean"] = sum(all_tokens) / len(all_tokens)
            effective_runtime["token_count_p95"] = float(
                torch.tensor(all_tokens, dtype=torch.float32).quantile(0.95).item()
            )
        instruction_centroids, instruction_labels = _centroids(
            rows, cache_reader, label="instruction"
        )
        target_centroids, target_labels = _centroids(rows, cache_reader, label="target")
        exact_correct = _probe_correct(
            merged["instruction_prediction"],
            merged["instruction_equivalence_groups"],
            instruction_centroids,
            instruction_labels,
        )
        true_targets = tuple(_target_object(value) for value in merged["instructions"])
        target_correct = _probe_correct(
            merged["instruction_prediction"],
            true_targets,
            target_centroids,
            target_labels,
        )
        inputs = SFT1V2ValidationInputs(
            visual_prediction=merged["visual_prediction"],
            dino_regions=merged["dino_regions"],
            instruction_prediction=merged["instruction_prediction"],
            instruction_teacher=merged["instruction_teacher"],
            feasibility_logits=merged["feasibility_logits"],
            executed_action_indices=merged["executed_action_indices"],
            movement_success=merged["movement_success"],
            feasibility_label_valid=merged["feasibility_label_valid"],
            actor_student_logits=merged["actor_student_logits"],
            actor_teacher_log_probs=merged["actor_teacher_log_probs"],
            state_policy_logits=merged["state_policy_logits"],
            image_content_groups=merged["image_content_groups"],
            instruction_equivalence_groups=merged["instruction_equivalence_groups"],
            external_eligible=merged["external_eligible"],
            exact_instruction_probe_correct=exact_correct,
            target_object_probe_correct=target_correct,
        )
        feasibility_train_rates = {
            action: (
                sum(
                    int(bool(row.movement_success))
                    for row in rows
                    if row.split == config.data.train_split
                    and row.executed_action_index == action
                    and row.movement_success is not None
                )
                / sum(
                    1
                    for row in rows
                    if row.split == config.data.train_split
                    and row.executed_action_index == action
                    and row.movement_success is not None
                )
            )
            for action in (0, 2, 3)
        }
        epoch0: SFT1V2ValidationReport | None = (
            load_validation_report(epoch0_report_path)
            if epoch0_report_path is not None
            else None
        )
        report = validate_sft1_v2_components(
            inputs,
            objective_version=config.state.objective_version,
            config_identity=config.identity,
            manifest_identity=manifest.identity,
            cache_manifest_sha256=cache_reader.summary.root_manifest_sha256,
            checkpoint_identity=checkpoint_identity,
            checkpoint_step=checkpoint_step,
            epoch=epoch,
            bootstrap_seed=config.validation.bootstrap_seed,
            bootstrap_resamples=config.validation.bootstrap_resamples,
            contrastive_temperature=config.objective.contrastive_temperature,
            runtime_metrics=effective_runtime,
            feasibility_train_rates=feasibility_train_rates,
            expected_external_rows=config.selection.external_validation_rows,
            expected_same_image_groups=config.selection.same_image_multi_instruction_groups,
            expected_same_instruction_groups=config.selection.same_instruction_multi_image_groups,
            epoch0_report=epoch0,
        )
        publish_validation_report(report_path, report)
        safety = report.safety_stop
    safety_tensor = torch.tensor(int(safety), device=assembly.worker.core.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(safety_tensor, src=0)
        torch.distributed.barrier()
    return report_path, bool(safety_tensor.item())


__all__ = ["run_validation_epoch"]
