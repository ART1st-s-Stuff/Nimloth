from __future__ import annotations

from nimloth.eval.rollout import val_rollout_success_rate


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
