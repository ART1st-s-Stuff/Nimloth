from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from nimloth.training.sft2.action_head_repair import (
    ActionTokenRowDelta,
    apply_action_row_delta_,
    balanced_action_sample_indices,
    fit_action_token_row_delta,
    population_action_spread,
    restricted_action_cross_entropy,
)


@dataclass(frozen=True)
class _Sample:
    record_id: str
    step_index: int
    action_index: int


def _samples(per_action: int = 5) -> list[_Sample]:
    return [
        _Sample(
            record_id=f"trajectory-{action}-{row}",
            step_index=row,
            action_index=action,
        )
        for action in range(3)
        for row in range(per_action)
    ]


def test_balanced_action_sample_indices_are_deterministic_and_complete() -> None:
    samples = _samples()

    first = balanced_action_sample_indices(
        samples,
        action_count=3,
        examples_per_action=4,
        seed=42002,
    )
    second = balanced_action_sample_indices(
        list(reversed(samples)),
        action_count=3,
        examples_per_action=4,
        seed=42002,
    )

    first_keys = {
        (samples[index].record_id, samples[index].step_index)
        for index in first
    }
    reversed_samples = list(reversed(samples))
    second_keys = {
        (reversed_samples[index].record_id, reversed_samples[index].step_index)
        for index in second
    }
    assert first_keys == second_keys
    assert len(first) == 12
    assert {
        action: sum(samples[index].action_index == action for index in first)
        for action in range(3)
    } == {0: 4, 1: 4, 2: 4}


def test_balanced_action_sample_indices_reject_missing_coverage() -> None:
    with pytest.raises(ValueError, match="action 2 has 0 examples"):
        balanced_action_sample_indices(
            [sample for sample in _samples() if sample.action_index != 2],
            action_count=3,
            examples_per_action=1,
            seed=1,
        )


def test_action_token_row_delta_is_fp32_and_trainable_only_through_delta() -> None:
    base_rows = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=torch.bfloat16,
    )
    module = ActionTokenRowDelta(base_rows)
    hidden = torch.tensor([[2.0, 3.0]], dtype=torch.bfloat16, requires_grad=True)

    logits = module(hidden)
    loss = restricted_action_cross_entropy(logits, torch.tensor([2]))
    loss.backward()

    assert logits.dtype == torch.float32
    assert logits.tolist() == [[2.0, 3.0, -2.0]]
    assert module.base_action_rows.dtype == torch.float32
    assert module.base_action_rows.requires_grad is False
    assert module.delta.grad is not None
    assert torch.count_nonzero(module.delta.grad).item() > 0
    assert hidden.grad is None


def test_apply_action_row_delta_changes_only_selected_rows() -> None:
    weight = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    original = weight.clone()
    delta = torch.tensor([[1.0, 2.0, 3.0], [-3.0, -2.0, -1.0]])

    apply_action_row_delta_(
        weight,
        action_token_ids=(2, 7),
        delta=delta,
    )

    assert torch.equal(weight[2], original[2] + delta[0])
    assert torch.equal(weight[7], original[7] + delta[1])
    untouched = [index for index in range(10) if index not in {2, 7}]
    assert torch.equal(weight[untouched], original[untouched])


def test_fit_action_token_row_delta_improves_heldout_restricted_nll() -> None:
    base = torch.zeros(3, 3)
    train_hidden = torch.eye(3).repeat_interleave(4, dim=0)
    train_targets = torch.arange(3).repeat_interleave(4)
    val_hidden = torch.eye(3).repeat_interleave(2, dim=0)
    val_targets = torch.arange(3).repeat_interleave(2)

    result = fit_action_token_row_delta(
        base_action_rows=base,
        train_hidden=train_hidden,
        train_targets=train_targets,
        validation_hidden=val_hidden,
        validation_targets=val_targets,
        learning_rate=0.1,
        weight_decay=0.0,
        max_epochs=100,
        early_stopping_patience=20,
        minimum_validation_improvement=0.1,
        device=torch.device("cpu"),
    )

    assert result.best_epoch >= 1
    assert result.validation_nll_after < result.validation_nll_before - 0.1
    assert result.delta.shape == (3, 3)
    assert torch.isfinite(result.delta).all()
    assert population_action_spread(result.validation_logits_after).median() > 0


def test_fit_action_token_row_delta_requires_balanced_training_targets() -> None:
    with pytest.raises(ValueError, match="equal per-action counts"):
        fit_action_token_row_delta(
            base_action_rows=torch.zeros(2, 2),
            train_hidden=torch.eye(2),
            train_targets=torch.tensor([0, 0]),
            validation_hidden=torch.eye(2),
            validation_targets=torch.tensor([0, 1]),
            learning_rate=0.1,
            weight_decay=0.0,
            max_epochs=2,
            early_stopping_patience=1,
            minimum_validation_improvement=0.0,
            device=torch.device("cpu"),
        )


def test_apply_action_row_delta_rejects_duplicate_tokens_and_nonfinite_delta() -> None:
    weight = torch.zeros(10, 3)
    with pytest.raises(ValueError, match="unique"):
        apply_action_row_delta_(
            weight,
            action_token_ids=(2, 2),
            delta=torch.zeros(2, 3),
        )
    with pytest.raises(ValueError, match="finite"):
        apply_action_row_delta_(
            weight,
            action_token_ids=(2, 7),
            delta=torch.tensor([[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]]),
        )
