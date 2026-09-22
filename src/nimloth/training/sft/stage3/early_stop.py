"""Checkpointable adjacent-epoch validation convergence rule."""
from __future__ import annotations

import math


def initialize_early_stop(config, saved: dict | None) -> dict | None:
    metric = getattr(config, "early_stop_metric", None)
    if metric is None:
        if saved is not None:
            raise ValueError("cannot disable checkpoint early stopping on resume")
        return None
    contract = {"metric": metric, "relative_improvement": config.early_stop_relative_improvement,
                "patience": config.early_stop_patience}
    if saved is not None:
        if any(saved.get(key) != value for key, value in contract.items()):
            raise ValueError("early stopping resume contract mismatch")
        return saved
    baseline = config.early_stop_baseline
    if baseline is None or not math.isfinite(baseline) or baseline < 0:
        raise ValueError("new early stopping history requires an explicit finite nonnegative baseline")
    return {**contract, "previous": baseline, "bad_epochs": 0, "history": []}


def update_early_stop(state: dict, metrics: dict, epoch: int) -> bool:
    value = float(metrics[state["metric"]])
    if not math.isfinite(value) or value < 0:
        raise ValueError("early stopping metric must be finite and nonnegative")
    previous = state["previous"]
    improvement = (previous - value) / max(abs(previous), 1e-12)
    state["bad_epochs"] = state["bad_epochs"] + 1 if improvement < state["relative_improvement"] else 0
    state["previous"] = value
    state["history"].append({"epoch": epoch, "value": value, "relative_improvement": improvement})
    return state["bad_epochs"] >= state["patience"]
