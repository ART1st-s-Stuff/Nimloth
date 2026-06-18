from __future__ import annotations

from nimloth.training.sft2.metrics import batch_step_success_rate


def test_batch_step_success_rate() -> None:
    items = [{"success": True}, {"success": False}]
    assert batch_step_success_rate(items) == 0.5
