"""Outcome readout: no extra predictor, masked BCE and gradient boundaries."""
from types import SimpleNamespace

import pytest
import torch

from nimloth.training.sft.stage3.algorithm import SFT2Algorithm
from nimloth.wm.outcome import ActionOutcomeHead


def algorithm(weight=1):
    return SFT2Algorithm(history_size=1, sigreg=None, sigreg_weight=0,
                         value_weight=1, ce_weight=1, outcome_weight=weight)


@pytest.mark.parametrize('horizon', [1, 4])
def test_bce_matches_valid_labels_and_gradients(horizon):
    head = ActionOutcomeHead(8)
    states = torch.randn(2, horizon, 4, 8, requires_grad=True)
    labels = torch.zeros(2, horizon)
    mask = torch.ones_like(labels, dtype=torch.bool)
    mask[0, 0] = False
    labels[0, 0] = float('nan')
    batch = SimpleNamespace(outcome_targets=labels, outcome_mask=mask, sample_weights=torch.ones(2))
    loss = algorithm()._outcome_loss(batch, head(states))
    reference = torch.nn.functional.binary_cross_entropy_with_logits(head(states)[mask], labels[mask])
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert states.grad is not None and torch.isfinite(states.grad).all()
    assert all(p.grad is not None for p in head.parameters())


def test_empty_labels_connected_zero_and_disabled_no_loss():
    head = ActionOutcomeHead(8)
    states = torch.randn(2, 1, 4, 8, requires_grad=True)
    batch = SimpleNamespace(outcome_targets=torch.full((2, 1), float('nan')),
                            outcome_mask=torch.zeros(2, 1, dtype=torch.bool), sample_weights=torch.ones(2))
    loss = algorithm()._outcome_loss(batch, head(states))
    assert loss.item() == 0
    loss.backward()
    assert states.grad is not None and states.grad.count_nonzero() == 0
    assert algorithm(0)._outcome_loss(batch, head(states)) is None
