from __future__ import annotations

import torch
from torch import nn

from nimloth.wm.lewm import SafeBatchNorm1d


def test_safe_batchnorm_updates_from_existing_running_stats() -> None:
    torch.manual_seed(0)
    safe = SafeBatchNorm1d(8)
    reference = nn.BatchNorm1d(8)
    reference.load_state_dict(safe.state_dict())
    batch = torch.randn(32, 8)

    safe(batch)
    reference(batch)

    torch.testing.assert_close(safe.running_mean, reference.running_mean)
    torch.testing.assert_close(safe.running_var, reference.running_var)
    assert torch.all(safe.running_var >= 0)
    safe.eval()
    assert torch.isfinite(safe(batch)).all()


def test_safe_batchnorm_supports_two_forwards_before_backward() -> None:
    safe = SafeBatchNorm1d(8)
    first = torch.randn(16, 8, requires_grad=True)
    second = torch.randn(16, 8, requires_grad=True)

    (safe(first).square().mean() + safe(second).square().mean()).backward()

    assert first.grad is not None
    assert second.grad is not None
    assert torch.isfinite(safe.running_var).all()
    assert torch.all(safe.running_var >= 0)
