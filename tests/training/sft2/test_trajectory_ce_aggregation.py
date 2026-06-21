"""Tests for trajectory CE aggregation semantics."""

from __future__ import annotations

import torch

from nimloth.training.sft2.trajectory_once import ce_loss_from_logits, legacy_batch_ce_loss


def test_token_weighted_ce_matches_legacy_batch_mean() -> None:
    torch.manual_seed(0)
    vocab = 16
    labels_a = torch.full((6,), -100)
    labels_a[2:5] = torch.tensor([3, 4, 5])
    labels_b = torch.full((4,), -100)
    labels_b[1:3] = torch.tensor([6, 7])
    logits_a = torch.randn(6, vocab)
    logits_b = torch.randn(4, vocab)

    once_labels = torch.full((10,), -100)
    once_labels[2:5] = torch.tensor([3, 4, 5])
    once_labels[7:9] = torch.tensor([6, 7])
    once_logits = torch.zeros(10, vocab)
    once_logits[:6] = logits_a
    once_logits[6:10] = logits_b

    once_loss = ce_loss_from_logits(once_logits, once_labels)
    legacy_loss = legacy_batch_ce_loss([labels_a, labels_b], [logits_a.unsqueeze(0), logits_b.unsqueeze(0)])
    assert torch.allclose(once_loss, legacy_loss, atol=1e-6)
