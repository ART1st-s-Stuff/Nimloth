from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.outcome_metrics import (
    OutcomeMetrics,
    merge_outcome_metrics,
)


def test_mask_horizon_threshold_and_unique_transition_support():
    batch = SimpleNamespace(
        outcome_targets=torch.tensor([[0., 1.], [1., 0.], [0., 0.]]),
        outcome_mask=torch.tensor([[True, True], [True, False], [True, True]]),
        sample_weights=torch.tensor([1., 1., 0.]),
        next_indices=torch.tensor([[0, 1], [1, 2], [0, 1]]),
        state_keys=(("a", 1), ("a", 2), ("a", 3)))
    accumulator = OutcomeMetrics()
    accumulator.update(batch, torch.tensor([[-1., 0.], [-1., float('nan')], [0., 0.]]))
    result = accumulator.metrics()
    assert result['outcome_h1_failure_tp'] == 1
    assert result['outcome_h1_failure_fp'] == 1
    assert result['outcome_h2_failure_tn'] == 1  # zero predicts success
    assert result['outcome_all_count'] == 3
    assert result['outcome_unique_transition_count'] == 2
    assert result['outcome_all_failure_precision'] == .5
    assert result['outcome_h2_failure_recall_defined'] == 0
    assert 'outcome_h2_failure_recall' not in result
    assert result['outcome_all_always_success_accuracy'] == pytest.approx(2/3)


def test_rank_counts_merge_before_ratios_and_union_keys():
    result = merge_outcome_metrics([
        ({1: [1, 0, 0, 0, 2.]}, {('a', 1)}),
        ({1: [0, 9, 0, 0, 8.], 2: [0, 0, 0, 0, 0]}, {('a', 1), ('b', 1)})])
    assert result['outcome_h1_failure_precision'] == .1
    assert result['outcome_h1_bce'] == 1.
    assert result['outcome_unique_transition_count'] == 2
    assert result['outcome_h2_accuracy_defined'] == 0
    assert 'outcome_h2_accuracy' not in result


def test_single_class_discrimination_is_explicitly_undefined():
    result = merge_outcome_metrics([({1: [0, 2, 0, 3, 2.]}, set())])
    for key in ('failure_precision', 'failure_recall', 'failure_f1'):
        assert result['outcome_h1_' + key + '_defined'] == 0
        assert 'outcome_h1_' + key not in result
    assert result['outcome_h1_accuracy'] == .6


def test_distributed_wrapper_collects_rank_payloads(monkeypatch):
    from nimloth.training.sft.stage3 import outcome_metrics as module
    monkeypatch.setattr(module.dist, 'is_available', lambda: True)
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda: 2)
    def gather(parts, local):
        parts[:] = [local, ({1: [0, 1, 0, 0, 1.]}, {('b', 1)})]
    monkeypatch.setattr(module.dist, 'all_gather_object', gather)
    accumulator = OutcomeMetrics()
    accumulator.rows = {1: [1, 0, 0, 0, 1.]}
    accumulator.transitions = {('a', 1)}
    assert accumulator.metrics()['outcome_all_failure_precision'] == .5


def test_evaluate_publishes_global_outcome_counts():
    from nimloth.training.sft.stage3.evaluate import evaluate
    batch = SimpleNamespace(outcome_targets=torch.tensor([[0., 1.]]),
        outcome_mask=torch.ones(1, 2, dtype=torch.bool), sample_weights=torch.ones(1),
        next_indices=torch.tensor([[0, 1]]), state_keys=(("a", 1), ("a", 2)))
    output = SimpleNamespace(metrics={"outcome_bce": .5, "outcome_count": 2.},
        sample_count=1, diagnostics={"outcome_logits": torch.tensor([[-1., 1.]])})
    runtime = SimpleNamespace(agent=SimpleNamespace(trainable_modules=()))
    runtime.unwrapped = lambda: runtime
    result = evaluate(SimpleNamespace(evaluation_step=lambda *_: output), runtime, [batch],
        batch_builder=SimpleNamespace(prepare=lambda value: value))
    assert result['outcome_count'] == 2
    assert result['outcome_unique_transition_count'] == 2
    assert result['outcome_all_failure_recall'] == 1.
