from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.training.sft1.query_state_training_validation import (
    QueryStateValidationMetadata,
    build_validation_metadata,
    compute_query_state_diagnostics,
    controlled_gather_query_state_diagnostics,
    evaluate_actor_safety,
    local_shard_group_gradient_squared_norms,
    local_shard_group_squared_norms,
    local_shard_group_update_squared_norms,
    validation_mode,
)
from nimloth.training.sft1.real_rows import SFT1V2Early4Row


_SHA = "a" * 64


def _row(index: int, *, action: int = 0, success: bool | None = True) -> SFT1V2Early4Row:
    instruction = "Find chair" if index < 2 else "Find lamp"
    image = "1" * 64 if index % 2 == 0 else f"{index + 1:064x}"
    return SFT1V2Early4Row(
        schema="nimloth_sft1_state_v2_early4_row_v2",
        ordinal=index,
        source_path="/data/source.jsonl",
        source_sha256=_SHA,
        split="val",
        record_id=f"record-{index}",
        step_index=index % 4,
        original_image_path=f"/images/{index}.png",
        original_image_sha256=image,
        image_content_group=image,
        instruction=instruction,
        instruction_char_span=(0, len(instruction)),
        instruction_equivalence_group=instruction.lower(),
        archived_assistant_response="<think>real</think><|action_start|>",
        executed_action_index=action,
        movement_success=success,
        external_eligible=True,
        record={},
    )


def test_validation_metadata_preserves_exact_read_only_identity_and_feedback() -> None:
    rows = (_row(0), _row(1, action=4, success=None))
    metadata = build_validation_metadata(rows, expected_row_identities=tuple(row.identity for row in rows))
    assert metadata[0].movement_success is True
    assert metadata[1].movement_success is None
    assert metadata[0].external_eligible is True
    assert metadata[0].image_content_group == rows[0].image_content_group
    assert metadata[0].instruction_equivalence_group == rows[0].instruction_equivalence_group
    with pytest.raises(ValueError, match="identity join"):
        build_validation_metadata(rows, expected_row_identities=(rows[1].identity, rows[0].identity))


def test_complete_diagnostics_lock_pairwise_effective_rank_content_and_actor_formula() -> None:
    torch.manual_seed(3)
    batch = 4
    raw = torch.randn(batch, 16, 8)
    state = torch.randn(batch, 16, 6)
    dino = state + 0.1 * torch.randn_like(state)
    action = torch.randn(batch, 8)
    baseline = action + 0.05 * torch.randn_like(action)
    fused = torch.randn(batch, 8)
    instruction = torch.randn(batch, 8)
    metadata = tuple(
        QueryStateValidationMetadata.from_row(_row(index, action=index % 4))
        for index in range(batch)
    )

    gathered, gathered_metadata = controlled_gather_query_state_diagnostics(
        {
            "raw_query_hidden": raw,
            "canonical_state": state,
            "dino_regions": dino,
            "action_logits": action,
            "baseline_action_logits": baseline,
            "fused_image_features": fused,
            "instruction_features": instruction,
        },
        metadata,
        max_global_rows=10,
    )
    report = compute_query_state_diagnostics(
        raw_query_hidden=gathered["raw_query_hidden"],
        canonical_state=gathered["canonical_state"],
        dino_regions=gathered["dino_regions"],
        action_logits=gathered["action_logits"],
        baseline_action_logits=gathered["baseline_action_logits"],
        fused_image_features=gathered["fused_image_features"],
        instruction_features=gathered["instruction_features"],
        archived_assistant_ce=1.5,
        archived_action_ce=0.4,
        metadata=gathered_metadata,
        effective_rank_collapse_threshold=1.5,
        globally_aggregated=True,
    )

    assert report.sample_count == batch
    assert report.effective_rank_formula == "entropy_rank_rows_slots_centered_float64_eps1e-12"
    for key in (
        "raw_query/offdiag_pairwise_cosine",
        "raw_query/effective_rank",
        "canonical_state/offdiag_pairwise_cosine",
        "canonical_state/effective_rank",
        "canonical_state/collapse",
        "direct_state/dino_mse",
        "direct_state/dino_cosine",
        "direct_state/content_relation",
        "lm/archived_assistant_ce",
        "lm/archived_action_ce",
        "actor/kl_baseline_to_current",
        "actor/top1_agreement",
        "actor/logit_rms_ratio",
        "upstream/fused_to_raw_relation",
        "upstream/instruction_to_state_relation",
        "pairs/same_image_multi_instruction_state_distance",
        "pairs/same_instruction_multi_image_state_distance",
        "executed_outcome/authoritative_movement_rows",
        "executed_outcome/movement_success_rows",
        "executed_outcome/movement_failure_rows",
    ):
        assert key in report.metrics
        assert torch.isfinite(torch.tensor(report.metrics[key]))
    assert report.global_aggregation is True
    assert report.detached_only is True
    safety = evaluate_actor_safety(
        report,
        tolerances={
            "kl_max": 1.0,
            "top1_min": 0.0,
            "logit_rms_ratio_min": 0.1,
            "logit_rms_ratio_max": 10.0,
        },
    )
    assert safety.passed is True
    assert evaluate_actor_safety(
        report,
        tolerances={
            "kl_max": 0.0,
            "top1_min": 1.0,
            "logit_rms_ratio_min": 1.0,
            "logit_rms_ratio_max": 1.0,
        },
    ).passed is False


def test_diagnostics_reject_rank0_only_claim_and_metadata_misalignment() -> None:
    tensor = torch.randn(2, 16, 4)
    action = torch.randn(2, 8)
    metadata = tuple(QueryStateValidationMetadata.from_row(_row(index)) for index in range(2))
    with pytest.raises(ValueError, match="global aggregation"):
        compute_query_state_diagnostics(
            raw_query_hidden=tensor,
            canonical_state=tensor,
            dino_regions=tensor,
            action_logits=action,
            baseline_action_logits=action,
            fused_image_features=torch.randn(2, 4),
            instruction_features=torch.randn(2, 4),
            archived_assistant_ce=1.0,
            archived_action_ce=0.5,
            metadata=metadata,
            effective_rank_collapse_threshold=1.0,
            globally_aggregated=False,
        )
    with pytest.raises(ValueError, match="metadata"):
        compute_query_state_diagnostics(
            raw_query_hidden=tensor,
            canonical_state=tensor,
            dino_regions=tensor,
            action_logits=action,
            baseline_action_logits=action,
            fused_image_features=torch.randn(2, 4),
            instruction_features=torch.randn(2, 4),
            archived_assistant_ce=1.0,
            archived_action_ce=0.5,
            metadata=metadata[:1],
            effective_rank_collapse_threshold=1.0,
            globally_aggregated=True,
        )


def test_validation_mode_restores_training_and_gradient_checkpointing_even_on_error() -> None:
    class _Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))
            self.backbone = SimpleNamespace(
                model=SimpleNamespace(
                    gradient_checkpointing=True,
                    training=True,
                )
            )

        def train(self, mode: bool = True):
            super().train(mode)
            self.backbone.model.training = mode
            return self

    root = _Root().train()
    with pytest.raises(RuntimeError, match="probe"):
        with validation_mode(root):
            assert root.training is False
            raise RuntimeError("probe failed")
    assert root.training is True
    assert root.backbone.model.training is True
    assert root.backbone.model.gradient_checkpointing is True

    root.backbone.model.gradient_checkpointing = False
    with pytest.raises(RuntimeError, match="gradient checkpointing"):
        with validation_mode(root):
            pass


def test_per_layer_update_norms_use_local_shards_without_full_model_copy() -> None:
    parameters = {
        "backbone.language_model.layers.0.weight": nn.Parameter(torch.tensor([3.0, 4.0])),
        "backbone.language_model.layers.1.weight": nn.Parameter(torch.tensor([1.0, 2.0])),
        "backbone.language_model.lm_head.weight": nn.Parameter(torch.tensor([2.0])),
        "objective.projector.linear.weight": nn.Parameter(torch.tensor([5.0])),
    }
    norms = local_shard_group_squared_norms(parameters)
    assert norms == {
        "language_layer_0": 25.0,
        "language_layer_1": 5.0,
        "lm_head": 4.0,
        "direct_state_head": 25.0,
    }
    for parameter in parameters.values():
        parameter.grad = torch.ones_like(parameter)
    assert local_shard_group_gradient_squared_norms(parameters) == {
        "language_layer_0": 2.0,
        "language_layer_1": 2.0,
        "lm_head": 1.0,
        "direct_state_head": 1.0,
    }
    before = {name: value.detach().clone() for name, value in parameters.items()}
    after = {name: value + 1.0 for name, value in before.items()}
    assert local_shard_group_update_squared_norms(before, after) == {
        "language_layer_0": 2.0,
        "language_layer_1": 2.0,
        "lm_head": 1.0,
        "direct_state_head": 1.0,
    }
