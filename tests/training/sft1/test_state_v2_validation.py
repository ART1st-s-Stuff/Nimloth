from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.training.sft1.validation import (
    SFT1V2ValidationInputs,
    publish_validation_report,
    validate_sft1_v2_components,
)


def _runtime_metrics() -> dict[str, float]:
    losses = (
        "visual_content", "visual_relation", "instruction_cosine",
        "instruction_contrastive", "observed_feasibility", "actor_kl",
        "state_policy_kl",
    )
    return {
        **{f"loss/{name}": 0.1 for name in losses},
        **{f"count/{name}": 6.0 for name in losses},
        "gradient_norm": 0.5,
        "throughput_rows_per_second": 2.0,
        "token_count_mean": 100.0,
        "token_count_p95": 120.0,
        "peak_memory_bytes": 1024.0,
    }


def _inputs() -> SFT1V2ValidationInputs:
    generator = torch.Generator().manual_seed(7)
    visual = torch.randn(6, 16, 1024, generator=generator)
    instruction = torch.randn(6, 2048, generator=generator)
    teacher_logits = torch.randn(6, 8, generator=generator)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    return SFT1V2ValidationInputs(
        visual_prediction=visual,
        dino_regions=visual + 0.01,
        instruction_prediction=instruction,
        instruction_teacher=instruction + 0.01,
        feasibility_logits=torch.tensor([
            [-2.0, 0.0, 0.0], [2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0], [0.0, 2.0, 0.0],
            [0.0, 0.0, -2.0], [0.0, 0.0, 2.0],
        ]),
        executed_action_indices=torch.tensor([0, 0, 2, 2, 3, 3]),
        movement_success=torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32),
        feasibility_label_valid=torch.ones(6, dtype=torch.bool),
        actor_student_logits=teacher_logits.clone(),
        actor_teacher_log_probs=teacher_log_probs,
        state_policy_logits=teacher_logits.clone(),
        image_content_groups=("image-a", "image-a", "image-b", "image-c", "image-d", "image-e"),
        instruction_equivalence_groups=("x", "y", "z", "z", "u", "v"),
        external_eligible=torch.ones(6, dtype=torch.bool),
        exact_instruction_probe_correct=torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.float32),
        target_object_probe_correct=torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32),
    )


def test_report_first_validation_emits_components_natural_pairs_and_no_quality_pass(
    tmp_path: Path,
) -> None:
    baseline = validate_sft1_v2_components(
        _inputs(), objective_version="nimloth_state_interface_v2_canary",
        config_identity="a" * 64, manifest_identity="b" * 64,
        cache_manifest_sha256="c" * 64, checkpoint_identity="0" * 64,
        checkpoint_step=0, epoch=0, bootstrap_seed=17, bootstrap_resamples=40,
        contrastive_temperature=0.1, runtime_metrics=_runtime_metrics(),
        feasibility_train_rates={0: 0.5, 2: 0.5, 3: 0.5},
        expected_external_rows=6, expected_same_image_groups=1,
        expected_same_instruction_groups=1,
    )
    report = validate_sft1_v2_components(
        _inputs(), objective_version="nimloth_state_interface_v2_canary",
        config_identity="a" * 64, manifest_identity="b" * 64,
        cache_manifest_sha256="c" * 64, checkpoint_identity="d" * 64,
        checkpoint_step=10, epoch=1, bootstrap_seed=17, bootstrap_resamples=40,
        contrastive_temperature=0.1, runtime_metrics=_runtime_metrics(),
        feasibility_train_rates={0: 0.5, 2: 0.5, 3: 0.5},
        expected_external_rows=6, expected_same_image_groups=1,
        expected_same_instruction_groups=1, epoch0_report=baseline,
    )
    names = {metric.name for metric in report.metrics}
    assert {
        "visual/content_cosine", "visual/slot_relation_error",
        "instruction/cosine", "instruction/contrastive_loss",
        "feasibility/action_0_roc_auc",
        "feasibility/action_0_constant_train_rate_brier",
        "feasibility/action_0_constant_train_rate_nll", "actor/kl_mean",
        "state_policy/kl_mean", "paired/same_image_semantic_minus_visual",
        "paired/same_instruction_visual_minus_semantic",
    } <= names
    assert report.report_complete
    assert not report.safety_stop
    assert report.automatic_model_quality_pass is None
    paired = next(
        metric for metric in report.metrics
        if metric.name == "paired/same_image_semantic_minus_visual"
    )
    assert paired.sample_count == 1
    assert paired.statistical_unit == "natural archived identity group"
    output = tmp_path / "epoch1-report.json"
    digest = publish_validation_report(output, report)
    assert len(digest) == 64
    reopened = json.loads(output.read_text())
    assert reopened["interpretation_owner"] == "human"
    assert reopened["automatic_model_quality_pass"] is None
    assert all(metric["statistical_unit"] for metric in reopened["metrics"])
    assert all(metric["delta_from_epoch0"] is not None for metric in reopened["metrics"])


def test_actor_guard_is_a_continuation_stop_not_a_quality_gate() -> None:
    inputs = _inputs()
    bad = SFT1V2ValidationInputs(**{
        **inputs.__dict__,
        "actor_student_logits": -100.0 * inputs.actor_teacher_log_probs,
    })
    baseline = validate_sft1_v2_components(
        _inputs(), objective_version="nimloth_state_interface_v2_canary",
        config_identity="a" * 64, manifest_identity="b" * 64,
        cache_manifest_sha256="c" * 64, checkpoint_identity="0" * 64,
        checkpoint_step=0, epoch=0, bootstrap_seed=17, bootstrap_resamples=20,
        contrastive_temperature=0.1, runtime_metrics=_runtime_metrics(),
        feasibility_train_rates={0: 0.5, 2: 0.5, 3: 0.5},
        expected_external_rows=6, expected_same_image_groups=1,
        expected_same_instruction_groups=1,
    )
    report = validate_sft1_v2_components(
        bad, objective_version="nimloth_state_interface_v2_canary",
        config_identity="a" * 64, manifest_identity="b" * 64,
        cache_manifest_sha256="c" * 64, checkpoint_identity="d" * 64,
        checkpoint_step=10, epoch=1, bootstrap_seed=17, bootstrap_resamples=20,
        contrastive_temperature=0.1, runtime_metrics=_runtime_metrics(),
        feasibility_train_rates={0: 0.5, 2: 0.5, 3: 0.5},
        expected_external_rows=6, expected_same_image_groups=1,
        expected_same_instruction_groups=1, epoch0_report=baseline,
    )
    assert report.safety_stop
    assert report.safety_stop_reasons
    assert report.automatic_model_quality_pass is None
