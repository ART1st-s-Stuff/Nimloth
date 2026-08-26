"""Deterministic report-first validation for the early-4 SFT1-v2 canary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


VALIDATION_REPORT_SCHEMA = "nimloth_sft1_state_v2_report_v1"


@dataclass(frozen=True)
class SFT1V2Metric:
    name: str
    value: float
    interval_low: float
    interval_high: float
    sample_count: int
    statistical_unit: str
    split: str
    checkpoint_step: int
    aggregation: str
    provenance: str
    epoch0_value: float | None = None
    delta_from_epoch0: float | None = None
    delta_interval_low: float | None = None
    delta_interval_high: float | None = None


@dataclass(frozen=True)
class SFT1V2ValidationInputs:
    visual_prediction: torch.Tensor
    dino_regions: torch.Tensor
    instruction_prediction: torch.Tensor
    instruction_teacher: torch.Tensor
    feasibility_logits: torch.Tensor
    executed_action_indices: torch.Tensor
    movement_success: torch.Tensor
    feasibility_label_valid: torch.Tensor
    actor_student_logits: torch.Tensor
    actor_teacher_log_probs: torch.Tensor
    state_policy_logits: torch.Tensor
    image_content_groups: tuple[str, ...]
    instruction_equivalence_groups: tuple[str, ...]
    external_eligible: torch.Tensor
    exact_instruction_probe_correct: torch.Tensor
    target_object_probe_correct: torch.Tensor


@dataclass(frozen=True)
class SFT1V2ValidationReport:
    schema: str
    objective_version: str
    config_identity: str
    manifest_identity: str
    cache_manifest_sha256: str
    checkpoint_identity: str
    checkpoint_step: int
    epoch: int
    metrics: tuple[SFT1V2Metric, ...]
    runtime_metrics: Mapping[str, float]
    validity_reasons: tuple[str, ...]
    safety_stop: bool
    safety_stop_reasons: tuple[str, ...]
    report_complete: bool
    interpretation_owner: str = "human"
    automatic_model_quality_pass: None = None


@dataclass(frozen=True)
class _Rows:
    values: np.ndarray
    unit: str = "row"


def _finite_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"validation tensor {name} is missing or non-finite")


def _bootstrap_interval(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    resamples: int,
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    results = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sample = values[rng.integers(0, values.shape[0], size=values.shape[0])]
        results[index] = statistic(sample)
    return float(np.quantile(results, 0.025)), float(np.quantile(results, 0.975))


def _metric(
    name: str,
    rows: np.ndarray,
    *,
    checkpoint_step: int,
    seed: int,
    resamples: int,
    provenance: str,
    statistic: Callable[[np.ndarray], float] = lambda values: float(values.mean()),
    unit: str = "row",
    aggregation: str = "mean with fixed-seed percentile bootstrap 95% CI",
) -> SFT1V2Metric:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 1 or rows.size < 1 or not np.isfinite(rows).all():
        raise ValueError(f"metric {name} requires finite one-dimensional samples")
    value = statistic(rows)
    low, high = _bootstrap_interval(rows, statistic, seed=seed, resamples=resamples)
    return SFT1V2Metric(
        name=name, value=value, interval_low=low, interval_high=high,
        sample_count=int(rows.size), statistical_unit=unit,
        split="validation_external_exact_image_decontaminated",
        checkpoint_step=checkpoint_step, aggregation=aggregation,
        provenance=provenance,
    )


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).mean()
    ties = (positive[:, None] == negative[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    tp = np.cumsum(ordered)
    fp = np.cumsum(1 - ordered)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.sum((recall - np.concatenate(([0.0], recall[:-1]))) * precision))


def _classification_metric(
    name: str,
    scores: np.ndarray,
    labels: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    checkpoint_step: int,
    seed: int,
    resamples: int,
) -> SFT1V2Metric:
    pairs = np.stack((scores, labels), axis=1)
    def paired_stat(rows: np.ndarray) -> float:
        return statistic(rows[:, 0], rows[:, 1].astype(np.int64))
    value = paired_stat(pairs)
    if not math.isfinite(value):
        raise ValueError(f"metric {name} requires both success and failure samples")
    rng = np.random.default_rng(seed)
    boot: list[float] = []
    for _ in range(resamples):
        sampled = pairs[rng.integers(0, len(pairs), size=len(pairs))]
        result = paired_stat(sampled)
        if math.isfinite(result):
            boot.append(result)
    if not boot:
        raise ValueError(f"metric {name} has no valid bootstrap resample")
    return SFT1V2Metric(
        name=name, value=value,
        interval_low=float(np.quantile(boot, 0.025)),
        interval_high=float(np.quantile(boot, 0.975)),
        sample_count=len(pairs), statistical_unit="executed movement row",
        split="validation_external_exact_image_decontaminated",
        checkpoint_step=checkpoint_step,
        aggregation="per-action classification with fixed-seed row bootstrap 95% CI",
        provenance="authoritative executed-action feedback only",
    )


def _pair_group_rows(
    features_a: np.ndarray,
    features_b: np.ndarray,
    primary_groups: Sequence[str],
    differing_groups: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one mean distance per natural group to avoid pair pseudoreplication."""

    a_values: list[float] = []
    b_values: list[float] = []
    for group in sorted(set(primary_groups)):
        indices = [index for index, value in enumerate(primary_groups) if value == group]
        pairs = [
            (left, right)
            for offset, left in enumerate(indices)
            for right in indices[offset + 1:]
            if differing_groups[left] != differing_groups[right]
        ]
        if not pairs:
            continue
        a_values.append(float(np.mean([
            np.linalg.norm(features_a[left] - features_a[right])
            for left, right in pairs
        ])))
        b_values.append(float(np.mean([
            np.linalg.norm(features_b[left] - features_b[right])
            for left, right in pairs
        ])))
    return np.asarray(a_values), np.asarray(b_values), len(a_values)


def validate_sft1_v2_components(
    inputs: SFT1V2ValidationInputs,
    *,
    objective_version: str,
    config_identity: str,
    manifest_identity: str,
    cache_manifest_sha256: str,
    checkpoint_identity: str,
    checkpoint_step: int,
    epoch: int,
    bootstrap_seed: int,
    bootstrap_resamples: int,
    contrastive_temperature: float,
    runtime_metrics: Mapping[str, float],
    feasibility_train_rates: Mapping[int, float],
    expected_external_rows: int = 1413,
    expected_same_image_groups: int = 42,
    expected_same_instruction_groups: int = 101,
    epoch0_report: SFT1V2ValidationReport | None = None,
) -> SFT1V2ValidationReport:
    """Build a complete component report; never emit a quality pass/best bit."""

    for name in SFT1V2ValidationInputs.__dataclass_fields__:
        value = getattr(inputs, name)
        if isinstance(value, torch.Tensor):
            _finite_tensor(name, value)
    size = int(inputs.visual_prediction.shape[0])
    shapes = {
        "visual_prediction": (size, 16, 1024), "dino_regions": (size, 16, 1024),
        "instruction_prediction": (size, 2048), "instruction_teacher": (size, 2048),
        "feasibility_logits": (size, 3), "actor_student_logits": (size, 8),
        "actor_teacher_log_probs": (size, 8), "state_policy_logits": (size, 8),
    }
    for name, expected in shapes.items():
        if tuple(getattr(inputs, name).shape) != expected:
            raise ValueError(f"validation {name} must have shape {expected}")
    for name in (
        "executed_action_indices", "movement_success", "feasibility_label_valid",
        "external_eligible", "exact_instruction_probe_correct", "target_object_probe_correct",
    ):
        if tuple(getattr(inputs, name).shape) != (size,):
            raise ValueError(f"validation {name} must have shape ({size},)")
    if len(inputs.image_content_groups) != size or len(inputs.instruction_equivalence_groups) != size:
        raise ValueError("validation natural-pair group identities do not align")
    mask = inputs.external_eligible.bool().cpu().numpy()
    if not mask.any():
        raise ValueError("external validation mask is empty")
    if int(mask.sum()) != expected_external_rows:
        raise ValueError(
            f"external validation row count mismatch: {int(mask.sum())} != {expected_external_rows}"
        )

    visual = inputs.visual_prediction.float()
    dino = inputs.dino_regions.float()
    instruction = inputs.instruction_prediction.float()
    instruction_teacher = inputs.instruction_teacher.float()
    visual_content = F.cosine_similarity(visual, dino, dim=-1).mean(dim=-1).cpu().numpy()[mask]
    visual_rel = (F.normalize(visual, dim=-1) @ F.normalize(visual, dim=-1).transpose(1, 2) - F.normalize(dino, dim=-1) @ F.normalize(dino, dim=-1).transpose(1, 2)).square().mean(dim=(1, 2)).cpu().numpy()[mask]
    instruction_cos = F.cosine_similarity(instruction, instruction_teacher, dim=-1).cpu().numpy()[mask]
    logits = F.normalize(instruction, dim=-1) @ F.normalize(instruction_teacher, dim=-1).transpose(0, 1)
    logits = logits / contrastive_temperature
    groups = inputs.instruction_equivalence_groups
    contrastive_rows: list[float] = []
    external_indices = np.nonzero(mask)[0].tolist()
    for index in external_indices:
        denominator = torch.logsumexp(logits[index, external_indices], dim=0)
        positives = [other for other in external_indices if groups[other] == groups[index]]
        contrastive_rows.append(float(denominator - torch.logsumexp(logits[index, positives], dim=0)))

    metrics = [
        _metric("visual/content_cosine", visual_content, checkpoint_step=checkpoint_step, seed=bootstrap_seed, resamples=bootstrap_resamples, provenance="fresh original-image DINO target"),
        _metric("visual/slot_relation_error", visual_rel, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 1, resamples=bootstrap_resamples, provenance="fresh original-image DINO slot relations"),
        _metric("instruction/cosine", instruction_cos, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 2, resamples=bootstrap_resamples, provenance="fresh exact-context instruction embedding"),
        _metric("instruction/contrastive_loss", np.asarray(contrastive_rows), checkpoint_step=checkpoint_step, seed=bootstrap_seed + 3, resamples=bootstrap_resamples, provenance="rank-local exact instruction groups; world size checkpoint invariant"),
        _metric("instruction/exact_probe_accuracy", inputs.exact_instruction_probe_correct.float().cpu().numpy()[mask], checkpoint_step=checkpoint_step, seed=bootstrap_seed + 4, resamples=bootstrap_resamples, provenance="pre-registered exact-instruction probe"),
        _metric("instruction/target_object_probe_accuracy", inputs.target_object_probe_correct.float().cpu().numpy()[mask], checkpoint_step=checkpoint_step, seed=bootstrap_seed + 5, resamples=bootstrap_resamples, provenance="pre-registered target-object probe"),
    ]

    action_to_column = {0: 0, 2: 1, 3: 2}
    if set(feasibility_train_rates) != set(action_to_column) or any(
        not 0.0 < float(value) < 1.0
        for value in feasibility_train_rates.values()
    ):
        raise ValueError("validation requires all three authoritative train success rates")
    actions = inputs.executed_action_indices.cpu().numpy()
    labels = inputs.movement_success.cpu().numpy().astype(np.int64)
    feasible = inputs.feasibility_label_valid.bool().cpu().numpy() & mask
    probabilities = torch.sigmoid(inputs.feasibility_logits.float()).cpu().numpy()
    for offset, (action, column) in enumerate(action_to_column.items()):
        selected = feasible & (actions == action)
        scores = probabilities[selected, column]
        actual = labels[selected]
        for suffix, function in (("roc_auc", _roc_auc), ("pr_auc", _pr_auc)):
            metrics.append(_classification_metric(
                f"feasibility/action_{action}_{suffix}", scores, actual, function,
                checkpoint_step=checkpoint_step, seed=bootstrap_seed + 10 + offset,
                resamples=bootstrap_resamples,
            ))
        metrics.append(_metric(
            f"feasibility/action_{action}_brier", (scores - actual) ** 2,
            checkpoint_step=checkpoint_step, seed=bootstrap_seed + 20 + offset,
            resamples=bootstrap_resamples, provenance="authoritative executed-action feedback only",
            unit="executed movement row",
        ))
        train_rate = float(feasibility_train_rates[action])
        metrics.append(_metric(
            f"feasibility/action_{action}_constant_train_rate_brier",
            (train_rate - actual) ** 2,
            checkpoint_step=checkpoint_step,
            seed=bootstrap_seed + 25 + offset,
            resamples=bootstrap_resamples,
            provenance="authoritative train executed-action success rate",
            unit="executed movement row",
        ))
        clipped = np.clip(scores, 1e-7, 1 - 1e-7)
        metrics.append(_metric(
            f"feasibility/action_{action}_nll", -(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)),
            checkpoint_step=checkpoint_step, seed=bootstrap_seed + 30 + offset,
            resamples=bootstrap_resamples, provenance="authoritative executed-action feedback only",
            unit="executed movement row",
        ))
        baseline_clipped = np.clip(train_rate, 1e-7, 1 - 1e-7)
        metrics.append(_metric(
            f"feasibility/action_{action}_constant_train_rate_nll",
            -(actual * np.log(baseline_clipped) + (1 - actual) * np.log(1 - baseline_clipped)),
            checkpoint_step=checkpoint_step,
            seed=bootstrap_seed + 35 + offset,
            resamples=bootstrap_resamples,
            provenance="authoritative train executed-action success rate",
            unit="executed movement row",
        ))
        metrics.append(_metric(
            f"feasibility/action_{action}_success_count", np.ones(int(actual.sum())),
            checkpoint_step=checkpoint_step, seed=bootstrap_seed + 40 + offset,
            resamples=bootstrap_resamples, provenance="authoritative executed-action feedback only",
            statistic=lambda values: float(values.sum()), unit="successful executed movement row", aggregation="count",
        ))
        failures = int((1 - actual).sum())
        metrics.append(_metric(
            f"feasibility/action_{action}_failure_count", np.ones(failures),
            checkpoint_step=checkpoint_step, seed=bootstrap_seed + 50 + offset,
            resamples=bootstrap_resamples, provenance="authoritative executed-action feedback only",
            statistic=lambda values: float(values.sum()), unit="failed executed movement row", aggregation="count",
        ))

    teacher = inputs.actor_teacher_log_probs.float()
    actor_log = F.log_softmax(inputs.actor_student_logits.float(), dim=-1)
    actor_kl = (teacher.exp() * (teacher - actor_log)).sum(dim=-1).cpu().numpy()[mask]
    actor_agree = (teacher.argmax(dim=-1) == actor_log.argmax(dim=-1)).float().cpu().numpy()[mask]
    state_log = F.log_softmax(inputs.state_policy_logits.float(), dim=-1)
    state_kl = (teacher.exp() * (teacher - state_log)).sum(dim=-1).cpu().numpy()[mask]
    state_agree = (teacher.argmax(dim=-1) == state_log.argmax(dim=-1)).float().cpu().numpy()[mask]
    for prefix, kl, agreement in (("actor", actor_kl, actor_agree), ("state_policy", state_kl, state_agree)):
        metrics.extend([
            _metric(f"{prefix}/kl_mean", kl, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 60, resamples=bootstrap_resamples, provenance="fresh ID176 normalized eight-action distribution"),
            _metric(f"{prefix}/kl_p50", kl, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 61, resamples=bootstrap_resamples, provenance="fresh ID176 normalized eight-action distribution", statistic=lambda values: float(np.quantile(values, 0.5)), aggregation="median with fixed-seed percentile bootstrap 95% CI"),
            _metric(f"{prefix}/kl_p95", kl, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 62, resamples=bootstrap_resamples, provenance="fresh ID176 normalized eight-action distribution", statistic=lambda values: float(np.quantile(values, 0.95)), aggregation="p95 with fixed-seed percentile bootstrap 95% CI"),
            _metric(f"{prefix}/top1_agreement", agreement, checkpoint_step=checkpoint_step, seed=bootstrap_seed + 63, resamples=bootstrap_resamples, provenance="fresh ID176 normalized eight-action distribution"),
        ])

    # Unit-normalized complete grids and semantic vectors make modality-distance
    # margins comparable; fixed mean visual pooling would hide slot changes.
    ext_visual = F.normalize(visual.flatten(1), dim=-1).cpu().numpy()[mask]
    ext_instruction = F.normalize(instruction, dim=-1).cpu().numpy()[mask]
    ext_images = tuple(value for value, keep in zip(inputs.image_content_groups, mask, strict=True) if keep)
    ext_instructions = tuple(value for value, keep in zip(inputs.instruction_equivalence_groups, mask, strict=True) if keep)
    semantic_distance, visual_distance, same_image_groups = _pair_group_rows(
        ext_instruction, ext_visual, ext_images, ext_instructions
    )
    visual_distance_2, semantic_distance_2, same_instruction_groups = _pair_group_rows(
        ext_visual, ext_instruction, ext_instructions, ext_images
    )
    if same_image_groups != expected_same_image_groups or same_instruction_groups != expected_same_instruction_groups:
        raise ValueError(
            f"natural-pair group count mismatch: same_image={same_image_groups}, same_instruction={same_instruction_groups}"
        )
    for name, values, seed_offset in (
        ("paired/same_image_semantic_distance", semantic_distance, 70),
        ("paired/same_image_visual_distance", visual_distance, 71),
        ("paired/same_image_semantic_minus_visual", semantic_distance - visual_distance, 72),
        ("paired/same_instruction_visual_distance", visual_distance_2, 73),
        ("paired/same_instruction_semantic_distance", semantic_distance_2, 74),
        ("paired/same_instruction_visual_minus_semantic", visual_distance_2 - semantic_distance_2, 75),
    ):
        metrics.append(_metric(
            name, values, checkpoint_step=checkpoint_step,
            seed=bootstrap_seed + seed_offset, resamples=bootstrap_resamples,
            provenance="natural archived row pairs with real archived CoT; no synthesized counterfactual",
            unit="natural archived identity group",
        ))

    required_runtime = {
        *{f"loss/{name}" for name in (
            "visual_content", "visual_relation", "instruction_cosine",
            "instruction_contrastive", "observed_feasibility", "actor_kl",
            "state_policy_kl",
        )},
        *{f"count/{name}" for name in (
            "visual_content", "visual_relation", "instruction_cosine",
            "instruction_contrastive", "observed_feasibility", "actor_kl",
            "state_policy_kl",
        )},
        "gradient_norm", "throughput_rows_per_second", "token_count_mean",
        "token_count_p95", "peak_memory_bytes",
    }
    missing_runtime = sorted(required_runtime - set(runtime_metrics))
    invalid_runtime = [name for name, value in runtime_metrics.items() if not math.isfinite(float(value))]
    validity_reasons = tuple(
        [f"missing runtime metric: {name}" for name in missing_runtime]
        + [f"non-finite runtime metric: {name}" for name in invalid_runtime]
    )
    if epoch == 0:
        metrics = [
            replace(
                metric, epoch0_value=metric.value, delta_from_epoch0=0.0,
                delta_interval_low=0.0, delta_interval_high=0.0,
            )
            for metric in metrics
        ]
    else:
        if epoch0_report is None or epoch0_report.epoch != 0:
            raise ValueError("post-epoch validation requires the complete epoch-0 report")
        if (
            epoch0_report.config_identity != config_identity
            or epoch0_report.manifest_identity != manifest_identity
        ):
            raise ValueError("epoch-0 report identity mismatch")
        baseline = {metric.name: metric for metric in epoch0_report.metrics}
        if set(baseline) != {metric.name for metric in metrics}:
            raise ValueError("epoch-0 report component set mismatch")
        metrics = [
            replace(
                metric,
                epoch0_value=baseline[metric.name].value,
                delta_from_epoch0=metric.value - baseline[metric.name].value,
                delta_interval_low=metric.interval_low - baseline[metric.name].interval_high,
                delta_interval_high=metric.interval_high - baseline[metric.name].interval_low,
            )
            for metric in metrics
        ]

    actor_mean = next(metric.value for metric in metrics if metric.name == "actor/kl_mean")
    actor_top1 = next(metric.value for metric in metrics if metric.name == "actor/top1_agreement")
    safety = []
    if actor_mean > 0.1:
        safety.append(f"external actor mean KL {actor_mean:.8g} > 0.1")
    if actor_top1 < 0.90:
        safety.append(f"external actor top1 agreement {actor_top1:.8g} < 0.90")
    return SFT1V2ValidationReport(
        schema=VALIDATION_REPORT_SCHEMA, objective_version=objective_version,
        config_identity=config_identity, manifest_identity=manifest_identity,
        cache_manifest_sha256=cache_manifest_sha256,
        checkpoint_identity=checkpoint_identity, checkpoint_step=checkpoint_step,
        epoch=epoch, metrics=tuple(metrics), runtime_metrics=dict(runtime_metrics),
        validity_reasons=validity_reasons, safety_stop=bool(safety),
        safety_stop_reasons=tuple(safety), report_complete=not validity_reasons,
    )


def load_validation_report(path: Path) -> SFT1V2ValidationReport:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != VALIDATION_REPORT_SCHEMA:
        raise ValueError("unsupported validation report schema")
    metrics = raw.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError("validation report metrics are invalid")
    return SFT1V2ValidationReport(
        **{
            **{key: value for key, value in raw.items() if key != "metrics"},
            "metrics": tuple(SFT1V2Metric(**value) for value in metrics),
        }
    )


def publish_validation_report(path: Path, report: SFT1V2ValidationReport) -> str:
    """Atomically publish and reopen a complete report; existing paths are immutable."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError("validation report is immutable")
    if not report.report_complete:
        raise ValueError("incomplete validation report cannot be published as complete")
    payload = asdict(report)
    temporary: Path | None = None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
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
    reopened = json.loads(destination.read_text(encoding="utf-8"))
    if reopened.get("schema") != VALIDATION_REPORT_SCHEMA or reopened.get("automatic_model_quality_pass", "missing") is not None:
        raise ValueError("reopened validation report violates report-first schema")
    return hashlib.sha256(destination.read_bytes()).hexdigest()


__all__ = [
    "SFT1V2Metric", "SFT1V2ValidationInputs", "SFT1V2ValidationReport",
    "VALIDATION_REPORT_SCHEMA", "load_validation_report",
    "publish_validation_report",
    "validate_sft1_v2_components",
]
