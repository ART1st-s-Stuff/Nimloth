from __future__ import annotations

from nimloth.training.sft2.metrics import batch_step_success_rate, val_rollout_success_rate


def test_batch_step_success_rate() -> None:
    items = [{"success": True}, {"success": False}]
    assert batch_step_success_rate(items) == 0.5


def test_val_rollout_success_rate_empty(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert val_rollout_success_rate(path) == 0.0


def test_val_rollout_success_rate_from_records(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text(
        '{"success": true}\n{"success": false}\n{"success": true}\n',
        encoding="utf-8",
    )
    assert val_rollout_success_rate(path) == 2 / 3
