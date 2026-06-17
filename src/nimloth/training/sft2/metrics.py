"""SFT2-specific evaluation metrics."""

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


def batch_step_success_rate(items: list[dict]) -> float:
    """Per-step success flag average within a collated batch."""

    if not items:
        return 0.0
    return sum(1.0 for item in items if item.get("success")) / len(items)
