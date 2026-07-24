"""Tests for model-independent rollout dataset statistics."""

from __future__ import annotations

from nimloth.wm.statistics import dataset_rollout_success_rate


def test_dataset_rollout_success_rate_empty(tmp_path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert dataset_rollout_success_rate(path) == 0.0


def test_dataset_rollout_success_rate_from_records(tmp_path) -> None:
    path = tmp_path / "val.jsonl"
    path.write_text(
        '{"success": true}\n{"success": false}\n{"success": true}\n',
        encoding="utf-8",
    )
    assert dataset_rollout_success_rate(path) == 2 / 3
