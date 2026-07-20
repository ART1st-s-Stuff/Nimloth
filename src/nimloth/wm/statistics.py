"""Descriptive statistics for model-independent rollout datasets."""

from __future__ import annotations

from pathlib import Path

from nimloth.wm.dataset import load_jsonl_records


def dataset_rollout_success_rate(jsonl_path: Path, *, max_records: int = -1) -> float:
    """Return success-label prevalence without running a model."""

    records = load_jsonl_records(jsonl_path, max_records=max_records)
    if not records:
        return 0.0
    successes = sum(1 for record in records if bool(record.get("success", False)))
    return successes / len(records)
