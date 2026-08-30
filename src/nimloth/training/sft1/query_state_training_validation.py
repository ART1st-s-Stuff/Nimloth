"""Detached, globally attributable diagnostics for Query-State pilot/formal runs."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import re
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F

from nimloth.backbone.qwen25vl.state_training import QwenStateTrainingOutput
from nimloth.training.sft1.real_rows import SFT1V2Early4Row
from nimloth.wm.grid import DirectSlotProjector


QUERY_STATE_TRAINING_DIAGNOSTIC_SCHEMA = "nimloth_sft1_query_state_training_diagnostic_v1"
_EFFECTIVE_RANK_FORMULA = "entropy_rank_rows_slots_centered_float64_eps1e-12"
_EPSILON = 1e-12


@dataclass(frozen=True)
class QueryStateValidationMetadata:
    row_identity: str
    record_id: str
    step_index: int
    original_image_sha256: str
    image_content_group: str
    instruction: str
    instruction_equivalence_group: str
    executed_action_index: int
    movement_success: bool | None
    external_eligible: bool

    @classmethod
    def from_row(cls, row: SFT1V2Early4Row) -> "QueryStateValidationMetadata":
        if not isinstance(row, SFT1V2Early4Row):
            raise TypeError("validation metadata requires an audited early-4 row")
        return cls(
            row_identity=row.identity,
            record_id=row.record_id,
            step_index=row.step_index,
            original_image_sha256=row.original_image_sha256,
            image_content_group=row.image_content_group,
            instruction=row.instruction,
            instruction_equivalence_group=row.instruction_equivalence_group,
            executed_action_index=row.executed_action_index,
            movement_success=row.movement_success,
            external_eligible=row.external_eligible,
        )


def build_validation_metadata(
    rows: Sequence[SFT1V2Early4Row],
    *,
    expected_row_identities: Sequence[str],
) -> tuple[QueryStateValidationMetadata, ...]:
    values = tuple(rows)
    expected = tuple(expected_row_identities)
    actual = tuple(row.identity for row in values)
    if actual != expected:
        raise ValueError("validation metadata strict identity join changed row order")
    metadata = tuple(QueryStateValidationMetadata.from_row(row) for row in values)
    if len({item.row_identity for item in metadata}) != len(metadata):
        raise ValueError("validation metadata contains duplicate row identities")
    return metadata


def controlled_gather_query_state_diagnostics(
    tensors: Mapping[str, torch.Tensor],
    metadata: Sequence[QueryStateValidationMetadata],
    *,
    max_global_rows: int,
) -> tuple[dict[str, torch.Tensor], tuple[QueryStateValidationMetadata, ...]]:
    """All-rank controlled CPU gather with an explicit global row budget."""

    required = {
        "raw_query_hidden",
        "canonical_state",
        "dino_regions",
        "action_logits",
        "baseline_action_logits",
        "fused_image_features",
        "instruction_features",
    }
    if set(tensors) != required or max_global_rows < 1:
        raise ValueError("controlled diagnostic gather contract is incomplete")
    local_metadata = tuple(metadata)
    local_rows = len(local_metadata)
    local_tensors: dict[str, torch.Tensor] = {}
    for name, value in tensors.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape[0] != local_rows
            or value.requires_grad
            or not torch.isfinite(value).all()
        ):
            raise ValueError("controlled diagnostic gather tensors are invalid")
        local_tensors[name] = value.detach().cpu()
    payload = (local_tensors, local_metadata)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered: list[object] = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
        rank_payloads = gathered
    else:
        rank_payloads = [payload]
    tensor_parts: dict[str, list[torch.Tensor]] = {name: [] for name in required}
    global_metadata: list[QueryStateValidationMetadata] = []
    for item in rank_payloads:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RuntimeError("distributed diagnostic gather payload is invalid")
        rank_tensors, rank_metadata = item
        if not isinstance(rank_tensors, dict) or set(rank_tensors) != required:
            raise RuntimeError("distributed diagnostic gather tensor keys differ by rank")
        rank_metadata = tuple(rank_metadata)
        for name in required:
            value = rank_tensors[name]
            if not isinstance(value, torch.Tensor) or value.shape[0] != len(rank_metadata):
                raise RuntimeError("distributed diagnostic gather rows differ by rank")
            tensor_parts[name].append(value)
        global_metadata.extend(rank_metadata)
    if len(global_metadata) > max_global_rows:
        raise ValueError("controlled diagnostic gather exceeds the preregistered row budget")
    identities = [item.row_identity for item in global_metadata]
    if len(set(identities)) != len(identities):
        raise ValueError("controlled diagnostic gather duplicated row identities")
    return (
        {name: torch.cat(parts, dim=0) for name, parts in tensor_parts.items()},
        tuple(global_metadata),
    )


@dataclass(frozen=True)
class QueryStateSameForwardDiagnostic:
    raw_query_hidden: torch.Tensor
    canonical_state: torch.Tensor
    action_logits: torch.Tensor
    fused_image_features: torch.Tensor
    instruction_features: torch.Tensor


def build_same_forward_diagnostic(
    output: QwenStateTrainingOutput,
    *,
    projector: DirectSlotProjector,
) -> QueryStateSameForwardDiagnostic:
    """Detach all diagnostic views from one already-completed Qwen call."""

    if not isinstance(projector, DirectSlotProjector):
        raise TypeError("same-forward diagnostic requires the canonical direct head")
    if output.fused_image_features is None or output.instruction_features is None:
        raise ValueError("validation Qwen output omitted diagnostic-only same-forward features")
    if output.query_hidden.ndim != 3 or output.query_hidden.shape[1:] != (16, 2048):
        raise ValueError("same-forward diagnostic raw Query shape is invalid")
    with torch.no_grad():
        state = projector(output.query_hidden).detach()
    return QueryStateSameForwardDiagnostic(
        raw_query_hidden=output.query_hidden.detach(),
        canonical_state=state,
        action_logits=output.action_logits.detach(),
        fused_image_features=output.fused_image_features.detach(),
        instruction_features=output.instruction_features.detach(),
    )


def _offdiagonal_slot_cosine(value: torch.Tensor) -> float:
    normalized = F.normalize(value.float(), dim=-1, eps=_EPSILON)
    similarity = normalized @ normalized.transpose(1, 2)
    slots = int(value.shape[1])
    mask = ~torch.eye(slots, dtype=torch.bool, device=value.device)
    return float(similarity[:, mask].mean().item())


def _effective_rank(value: torch.Tensor) -> float:
    matrix = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1, value.shape[-1])
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(matrix)
    energy = singular.square()
    total = energy.sum()
    if not torch.isfinite(total) or float(total.item()) <= _EPSILON:
        return 0.0
    probability = energy / total
    entropy = -(probability * torch.log(probability.clamp_min(_EPSILON))).sum()
    return float(torch.exp(entropy).item())


def _row_relation(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape[0] != right.shape[0]:
        raise ValueError("diagnostic representation row counts disagree")
    left_rows = left.float().reshape(left.shape[0], -1)
    right_rows = right.float().reshape(right.shape[0], -1)
    if left_rows.shape[0] < 2:
        return 0.0
    left_similarity = F.normalize(left_rows, dim=-1, eps=_EPSILON) @ F.normalize(left_rows, dim=-1, eps=_EPSILON).T
    right_similarity = F.normalize(right_rows, dim=-1, eps=_EPSILON) @ F.normalize(right_rows, dim=-1, eps=_EPSILON).T
    indices = torch.triu_indices(left_rows.shape[0], left_rows.shape[0], offset=1)
    left_values = left_similarity[indices[0], indices[1]]
    right_values = right_similarity[indices[0], indices[1]]
    left_values = left_values - left_values.mean()
    right_values = right_values - right_values.mean()
    denominator = left_values.norm() * right_values.norm()
    if float(denominator.item()) <= _EPSILON:
        return 0.0
    return float((left_values @ right_values / denominator).item())


def _content_relation(state: torch.Tensor, dino: torch.Tensor) -> float:
    state_norm = F.normalize(state.float(), dim=-1, eps=_EPSILON)
    dino_norm = F.normalize(dino.float(), dim=-1, eps=_EPSILON)
    state_relation = state_norm @ state_norm.transpose(1, 2)
    dino_relation = dino_norm @ dino_norm.transpose(1, 2)
    slots = state.shape[1]
    mask = ~torch.eye(slots, dtype=torch.bool, device=state.device)
    left = state_relation[:, mask].flatten()
    right = dino_relation[:, mask].flatten()
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    return 0.0 if float(denominator.item()) <= _EPSILON else float((left @ right / denominator).item())


def _paired_distance(
    state_rows: torch.Tensor,
    metadata: Sequence[QueryStateValidationMetadata],
    *,
    group_field: str,
    varying_field: str,
) -> tuple[float, int]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(metadata):
        groups.setdefault(str(getattr(item, group_field)), []).append(index)
    distances: list[torch.Tensor] = []
    row_state = state_rows.float().mean(dim=1)
    for indices in groups.values():
        for left_index, left in enumerate(indices):
            for right in indices[left_index + 1 :]:
                if getattr(metadata[left], varying_field) == getattr(metadata[right], varying_field):
                    continue
                distances.append(1.0 - F.cosine_similarity(row_state[left], row_state[right], dim=0))
    if not distances:
        return 0.0, 0
    return float(torch.stack(distances).mean().item()), len(distances)


@dataclass(frozen=True)
class QueryStateCompleteDiagnosticReport:
    schema: str
    sample_count: int
    record_ids: tuple[str, ...]
    metrics: Mapping[str, float]
    effective_rank_formula: str
    effective_rank_collapse_threshold: float
    global_aggregation: bool
    detached_only: bool
    automatic_checkpoint_selection: bool = False
    automatic_sft2_authorization: bool = False


@dataclass(frozen=True)
class QueryStateActorSafetyVerdict:
    passed: bool
    checks: Mapping[str, bool]
    tolerances: Mapping[str, float]


def evaluate_actor_safety(
    report: QueryStateCompleteDiagnosticReport,
    *,
    tolerances: Mapping[str, float],
) -> QueryStateActorSafetyVerdict:
    expected = {
        "kl_max",
        "top1_min",
        "logit_rms_ratio_min",
        "logit_rms_ratio_max",
    }
    if set(tolerances) != expected or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in tolerances.values()
    ):
        raise ValueError("actor safety requires all pre-registered tolerances")
    values = {name: float(value) for name, value in tolerances.items()}
    if (
        values["kl_max"] < 0.0
        or not 0.0 <= values["top1_min"] <= 1.0
        or values["logit_rms_ratio_min"] <= 0.0
        or values["logit_rms_ratio_max"] < values["logit_rms_ratio_min"]
    ):
        raise ValueError("actor safety tolerances are inconsistent")
    metrics = report.metrics
    checks = {
        "kl": metrics["actor/kl_baseline_to_current"] <= values["kl_max"],
        "top1": metrics["actor/top1_agreement"] >= values["top1_min"],
        "logit_rms_min": metrics["actor/logit_rms_ratio"]
        >= values["logit_rms_ratio_min"],
        "logit_rms_max": metrics["actor/logit_rms_ratio"]
        <= values["logit_rms_ratio_max"],
    }
    return QueryStateActorSafetyVerdict(
        passed=all(checks.values()),
        checks=checks,
        tolerances=values,
    )


def compute_query_state_diagnostics(
    *,
    raw_query_hidden: torch.Tensor,
    canonical_state: torch.Tensor,
    dino_regions: torch.Tensor,
    action_logits: torch.Tensor,
    baseline_action_logits: torch.Tensor,
    fused_image_features: torch.Tensor,
    instruction_features: torch.Tensor,
    archived_assistant_ce: float,
    archived_action_ce: float,
    metadata: Sequence[QueryStateValidationMetadata],
    effective_rank_collapse_threshold: float,
    globally_aggregated: bool = False,
) -> QueryStateCompleteDiagnosticReport:
    """Compute the preregistered formulas on controlled globally joined rows."""

    if not globally_aggregated:
        raise ValueError("Query-State report cannot claim global aggregation from rank0-only data")
    if (
        not math.isfinite(archived_assistant_ce)
        or not math.isfinite(archived_action_ce)
        or archived_assistant_ce < 0.0
        or archived_action_ce < 0.0
    ):
        raise ValueError("archived assistant/action CE must be finite global means")
    tensors = (
        raw_query_hidden,
        canonical_state,
        dino_regions,
        action_logits,
        baseline_action_logits,
        fused_image_features,
        instruction_features,
    )
    if any(not isinstance(value, torch.Tensor) or value.requires_grad or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("diagnostic tensors must be detached and finite")
    size = int(raw_query_hidden.shape[0])
    if (
        raw_query_hidden.ndim != 3
        or raw_query_hidden.shape[1] != 16
        or canonical_state.ndim != 3
        or canonical_state.shape[0:2] != (size, 16)
        or dino_regions.shape != canonical_state.shape
        or action_logits.shape != (size, 8)
        or baseline_action_logits.shape != (size, 8)
        or fused_image_features.ndim != 2
        or fused_image_features.shape[0] != size
        or instruction_features.ndim != 2
        or instruction_features.shape[0] != size
        or len(metadata) != size
    ):
        raise ValueError("diagnostic tensor/metadata shapes do not align")
    if len({item.row_identity for item in metadata}) != size:
        raise ValueError("diagnostic metadata identities are duplicated")
    if not math.isfinite(effective_rank_collapse_threshold) or effective_rank_collapse_threshold <= 0:
        raise ValueError("effective-rank collapse threshold must be pre-registered and positive")

    raw_rank = _effective_rank(raw_query_hidden)
    state_rank = _effective_rank(canonical_state)
    baseline_probability = torch.softmax(baseline_action_logits.float(), dim=-1)
    current_log_probability = torch.log_softmax(action_logits.float(), dim=-1)
    baseline_log_probability = torch.log_softmax(baseline_action_logits.float(), dim=-1)
    actor_kl = (baseline_probability * (baseline_log_probability - current_log_probability)).sum(dim=-1).mean()
    baseline_top1 = baseline_action_logits.argmax(dim=-1)
    current_top1 = action_logits.argmax(dim=-1)
    baseline_rms = baseline_action_logits.float().square().mean().sqrt()
    current_rms = action_logits.float().square().mean().sqrt()
    same_image, same_image_count = _paired_distance(
        canonical_state,
        metadata,
        group_field="image_content_group",
        varying_field="instruction_equivalence_group",
    )
    same_instruction, same_instruction_count = _paired_distance(
        canonical_state,
        metadata,
        group_field="instruction_equivalence_group",
        varying_field="image_content_group",
    )
    state_row = canonical_state.float().mean(dim=1)
    raw_row = raw_query_hidden.float().mean(dim=1)
    observed_movement = [
        item.movement_success
        for item in metadata
        if item.movement_success is not None
    ]
    metrics = {
        "raw_query/norm_mean": float(raw_query_hidden.float().norm(dim=-1).mean().item()),
        "raw_query/slot_variance": float(raw_query_hidden.float().var(dim=1, unbiased=False).mean().item()),
        "raw_query/offdiag_pairwise_cosine": _offdiagonal_slot_cosine(raw_query_hidden),
        "raw_query/effective_rank": raw_rank,
        "canonical_state/norm_mean": float(canonical_state.float().norm(dim=-1).mean().item()),
        "canonical_state/slot_variance": float(canonical_state.float().var(dim=1, unbiased=False).mean().item()),
        "canonical_state/offdiag_pairwise_cosine": _offdiagonal_slot_cosine(canonical_state),
        "canonical_state/effective_rank": state_rank,
        "canonical_state/collapse": float(state_rank < effective_rank_collapse_threshold),
        "direct_state/dino_mse": float((canonical_state.float() - dino_regions.float()).square().mean().item()),
        "direct_state/dino_cosine": float(F.cosine_similarity(canonical_state.float(), dino_regions.float(), dim=-1).mean().item()),
        "direct_state/content_relation": _content_relation(canonical_state, dino_regions),
        "lm/archived_assistant_ce": float(archived_assistant_ce),
        "lm/archived_action_ce": float(archived_action_ce),
        "actor/kl_baseline_to_current": float(actor_kl.item()),
        "actor/top1_agreement": float((baseline_top1 == current_top1).float().mean().item()),
        "actor/logit_rms_ratio": float((current_rms / baseline_rms.clamp_min(_EPSILON)).item()),
        "upstream/fused_to_raw_relation": _row_relation(fused_image_features, raw_row),
        "upstream/instruction_to_state_relation": _row_relation(instruction_features, state_row),
        "pairs/same_image_multi_instruction_state_distance": same_image,
        "pairs/same_instruction_multi_image_state_distance": same_instruction,
        "pairs/same_image_pair_count": float(same_image_count),
        "pairs/same_instruction_pair_count": float(same_instruction_count),
        "executed_outcome/authoritative_movement_rows": float(len(observed_movement)),
        "executed_outcome/movement_success_rows": float(sum(observed_movement)),
        "executed_outcome/movement_failure_rows": float(
            len(observed_movement) - sum(observed_movement)
        ),
    }
    if any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("complete Query-State diagnostic metric is non-finite")
    return QueryStateCompleteDiagnosticReport(
        schema=QUERY_STATE_TRAINING_DIAGNOSTIC_SCHEMA,
        sample_count=size,
        record_ids=tuple(item.record_id for item in metadata),
        metrics=metrics,
        effective_rank_formula=_EFFECTIVE_RANK_FORMULA,
        effective_rank_collapse_threshold=float(effective_rank_collapse_threshold),
        global_aggregation=True,
        detached_only=True,
    )


def _training_qwen(root: torch.nn.Module) -> Any:
    backbone = getattr(root, "backbone", None)
    return getattr(backbone, "model", None)


def _gradient_checkpointing_active(qwen: Any) -> bool:
    modules = getattr(qwen, "modules", None)
    if callable(modules):
        checkpoint_modules = [
            module
            for module in modules()
            if bool(getattr(module, "gradient_checkpointing", False))
        ]
        return bool(checkpoint_modules) and all(
            bool(getattr(module, "training", False))
            for module in checkpoint_modules
        )
    return (
        getattr(qwen, "gradient_checkpointing", None) is True
        and getattr(qwen, "training", None) is True
    )


@contextmanager
def validation_mode(root: torch.nn.Module) -> Iterator[None]:
    """Enter eval and always restore train plus gradient-checkpointing contract."""

    if not root.training:
        raise RuntimeError("Query-State validation requires an active training root")
    root.eval()
    caught: BaseException | None = None
    try:
        yield
    except BaseException as error:
        caught = error
        raise
    finally:
        root.train(True)
        qwen = _training_qwen(root)
        restore_error: RuntimeError | None = None
        if not root.training or qwen is None or getattr(qwen, "training", None) is not True:
            restore_error = RuntimeError(
                "Query-State validation failed to restore training mode"
            )
        elif not _gradient_checkpointing_active(qwen):
            restore_error = RuntimeError(
                "Query-State validation failed to restore gradient checkpointing"
            )
        if restore_error is not None:
            if caught is not None:
                raise restore_error from caught
            raise restore_error


def _parameter_report_group(name: str) -> str | None:
    if "objective.projector.linear.weight" in name:
        return "direct_state_head"
    if "lm_head" in name:
        return "lm_head"
    if "embed_tokens" in name:
        return "language_embedding"
    match = re.search(r"(?:language_model\.)?layers\.(\d+)\.", name)
    if match is not None:
        return f"language_layer_{match.group(1)}"
    if re.search(r"(?:language_model\.)?norm\.", name):
        return "language_norm"
    return None


def _reduce_group_squares(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    groups: dict[str, torch.Tensor] = {}
    for name, tensor in values.items():
        group = _parameter_report_group(name)
        if group is None:
            continue
        value = tensor.detach().float().square().sum()
        groups[group] = groups.get(group, torch.zeros_like(value)) + value
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for value in groups.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return {name: float(value.cpu().item()) for name, value in sorted(groups.items())}


def local_shard_group_squared_norms(
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, float]:
    """Aggregate local parameter shards without copying the full model."""

    return _reduce_group_squares(named_parameters)


def local_shard_group_gradient_squared_norms(
    named_parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, float]:
    """All-reduce local-shard gradient squared norms by production owner."""

    gradients = {
        name: parameter.grad
        for name, parameter in named_parameters.items()
        if parameter.grad is not None
    }
    return _reduce_group_squares(gradients)


def local_shard_group_update_squared_norms(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Compare immutable same-layout checkpoint shards without full-state copies."""

    if set(before) != set(after):
        raise ValueError("update-delta checkpoint shard keys differ")
    deltas: dict[str, torch.Tensor] = {}
    for name in before:
        left, right = before[name], after[name]
        if left.shape != right.shape:
            raise ValueError("update-delta checkpoint shard shapes differ")
        deltas[name] = right.detach().float() - left.detach().float()
    return _reduce_group_squares(deltas)


__all__ = [
    "QUERY_STATE_TRAINING_DIAGNOSTIC_SCHEMA",
    "QueryStateActorSafetyVerdict",
    "QueryStateCompleteDiagnosticReport",
    "QueryStateSameForwardDiagnostic",
    "QueryStateValidationMetadata",
    "build_same_forward_diagnostic",
    "build_validation_metadata",
    "compute_query_state_diagnostics",
    "controlled_gather_query_state_diagnostics",
    "evaluate_actor_safety",
    "local_shard_group_gradient_squared_norms",
    "local_shard_group_squared_norms",
    "local_shard_group_update_squared_norms",
    "validation_mode",
]
