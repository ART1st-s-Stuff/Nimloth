"""Validate returns retained after dropping a transition with no next observation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

FINITE_HORIZON_FORMAT = "finite_horizon_tail_drop_v1"


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def validate_action_successes(record: Mapping[str, Any]) -> None:
    values = record.get("action_successes")
    if values is None:
        return
    if not isinstance(values, list) or len(values) != len(record["action_indices"]):
        raise ValueError("action_successes must align one-to-one with actions")
    if any(type(value) is not bool for value in values):
        raise ValueError("action_successes must contain only booleans")


def validated_action_value_targets(
    record: Mapping[str, Any], *, gamma: float | None = None,
) -> list[float] | None:
    """Recompute full-episode returns before slicing; never trust cached targets alone."""
    values = record.get("action_value_targets")
    provenance = record.get("finite_horizon_provenance")
    if values is None and not provenance:
        return None
    if values is None or not isinstance(provenance, Mapping):
        raise ValueError("explicit action_value_targets require finite_horizon_provenance")
    if provenance.get("format") != FINITE_HORIZON_FORMAT:
        raise ValueError("unsupported finite_horizon_provenance format")
    n = len(record["action_indices"])
    original_n = provenance.get("original_action_count")
    horizon = provenance.get("max_action_horizon")
    if type(original_n) is not int or original_n != n + 1:
        raise ValueError("finite-horizon conversion must remove exactly one final action")
    if type(horizon) is not int or not 1 <= original_n <= horizon:
        raise ValueError("original action count exceeds finite task horizon")
    reason = provenance.get("bootstrap_reason")
    if reason not in {"task_horizon", "environment_terminated"}:
        raise ValueError("bootstrap requires task horizon or environment termination evidence")
    dones = provenance.get("original_dones")
    if not isinstance(dones, list) or len(dones) != original_n or any(type(v) is not bool for v in dones):
        raise ValueError("original_dones must contain one boolean per original action")
    if any(dones[:-1]):
        raise ValueError("source actions cannot continue after environment termination")
    if reason == "environment_terminated" and not dones[-1]:
        raise ValueError("environment_terminated requires final source done evidence")
    if reason == "task_horizon" and dones[-1]:
        raise ValueError("source final done must use environment_terminated reason")
    if reason == "task_horizon" and original_n != horizon:
        raise ValueError("task_horizon bootstrap requires reaching the original horizon")
    if _finite(provenance.get("bootstrap"), "bootstrap") != 0.0:
        raise ValueError("finite-horizon original endpoint bootstrap must be zero")
    saved_gamma = _finite(provenance.get("gamma"), "gamma")
    if not 0 <= saved_gamma <= 1:
        raise ValueError("finite-horizon gamma must be in [0, 1]")
    if gamma is not None and saved_gamma != _finite(gamma, "requested gamma"):
        raise ValueError("precomputed action-value gamma does not match requested gamma")
    source_hash = provenance.get("source_record_sha256")
    if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
        raise ValueError("finite-horizon provenance requires source_record_sha256")
    original = provenance.get("original_rewards")
    if not isinstance(original, list) or len(original) != original_n:
        raise ValueError("original_rewards must cover the entire source action sequence")
    rewards = [_finite(value, "original reward") for value in original]
    if record.get("reward_provenance") != "step_rewards":
        raise ValueError("finite-horizon targets require explicit step_rewards provenance")
    retained = record.get("rewards")
    if not isinstance(retained, list):
        raise ValueError("retained rewards must be an explicit list")
    retained = [_finite(value, "retained reward") for value in retained]
    if retained != rewards[:-1]:
        raise ValueError("retained rewards must equal original_rewards without final action")
    if _finite(provenance.get("removed_reward"), "removed_reward") != rewards[-1]:
        raise ValueError("removed_reward must match the original final reward")
    removed_action = provenance.get("removed_action_index")
    from nimloth.environment import get_action_space

    action_count = len(get_action_space(str(record["action_space_id"]), int(record["action_space_version"])))
    if type(removed_action) is not int or not 0 <= removed_action < action_count:
        raise ValueError("removed_action_index must belong to the record action space")
    if not isinstance(values, list) or len(values) != n:
        raise ValueError("action_value_targets must align one-to-one with actions")
    targets = [_finite(value, "action_value_target") for value in values]
    running = 0.0
    full_returns = [0.0] * original_n
    for index in range(original_n - 1, -1, -1):
        running = rewards[index] + saved_gamma * running
        full_returns[index] = running
    if any(not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8)
           for actual, expected in zip(targets, full_returns[:-1], strict=True)):
        raise ValueError("action_value_targets disagree with full original reward returns")
    return targets
