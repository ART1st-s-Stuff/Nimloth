"""Training-batch metrics for SFT2 (not offline eval)."""

from __future__ import annotations


def batch_step_success_rate(items: list[dict]) -> float:
    """Per-step success flag average within a collated batch."""

    if not items:
        return 0.0
    return sum(1.0 for item in items if item.get("success")) / len(items)
