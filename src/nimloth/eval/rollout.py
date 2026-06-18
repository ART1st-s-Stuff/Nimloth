"""Offline evaluation metrics on Nimloth rollout jsonl."""

from __future__ import annotations

from pathlib import Path

from nimloth.wm.dataset import load_jsonl_records


def val_rollout_success_rate(val_jsonl: Path, *, max_records: int = -1) -> float:
    """Trajectory-level success rate on a held-out Nimloth jsonl split."""

    records = load_jsonl_records(val_jsonl, max_records=max_records)
    if not records:
        return 0.0
    successes = sum(1 for record in records if bool(record.get("success", False)))
    return successes / len(records)
