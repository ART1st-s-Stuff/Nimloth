"""CPU mechanism tests; these do not establish model quality."""

import torch
from torch.nn import functional as F

from experiments.training.sft.diagnosis.format_transfer import (
    fit_precision,
    full_vocab_row_loss,
)


def test_full_vocab_loss_keeps_old_token_competitors():
    hidden = torch.tensor([[1.0, 0.0]])
    rows = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
    ids = torch.tensor([1])
    target = torch.tensor([1])
    frozen = torch.tensor([[8.0, -10.0, 2.0]])
    loss = full_vocab_row_loss(hidden, rows, ids, target, frozen)
    assert torch.allclose(loss, F.cross_entropy(torch.tensor([[8.0, 0.0, 2.0]]), target))
    loss.backward()
    assert rows.grad[0, 0] < -0.9
    assert torch.equal(frozen, torch.tensor([[8.0, -10.0, 2.0]]))


def test_precision_fit_measures_actual_writes_and_preserves_source():
    hidden = torch.tensor([[1.0, 1.0]])
    weights = torch.full((3, 2), 0.02, dtype=torch.bfloat16)
    before = weights.clone()
    report = fit_precision(hidden, weights, torch.tensor([1]), torch.tensor([1]), steps=2, lr=5e-6)
    bf16 = report['torch.bfloat16']['history'][0]
    fp32 = report['torch.float32']['history'][0]
    assert bf16['actual_abs_mean'] == 0
    assert fp32['actual_abs_mean'] > 0
    assert torch.equal(weights, before)
