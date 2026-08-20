"""Versioned, policy-capability-aware rollout browser records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ROLLOUT_AUDIT_SCHEMA = "nimloth_rollout_audit_v2"
EVALUATION_MANIFEST_SCHEMA = "nimloth_evaluation_rollout_browser_manifest_v1"
CAPABILITY_KEYS = frozenset(
    {
        "task",
        "observations",
        "terminal_observation",
        "cot",
        "token_trace",
        "action_distribution",
        "direct_q",
        "state_value",
        "planner",
        "mcts",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_or_null_vector(value: Any, name: str) -> list[float | None]:
    vector = list(_sequence(value, name))
    if not vector:
        raise ValueError(f"{name} must not be empty")
    for index, item in enumerate(vector):
        if item is not None:
            _finite(item, f"{name}[{index}]")
    return vector


def validate_capabilities(raw: Any) -> dict[str, bool]:
    capabilities = dict(_mapping(raw, "capabilities"))
    if set(capabilities) != CAPABILITY_KEYS:
        raise ValueError(
            "capability keys mismatch: "
            f"missing={sorted(CAPABILITY_KEYS - set(capabilities))}, "
            f"unknown={sorted(set(capabilities) - CAPABILITY_KEYS)}"
        )
    if any(not isinstance(value, bool) for value in capabilities.values()):
        raise ValueError("capabilities must be bool values")
    if capabilities["mcts"] and not capabilities["planner"]:
        raise ValueError("mcts capability requires planner capability")
    if capabilities["state_value"] and not capabilities["direct_q"]:
        raise ValueError("state_value capability requires direct_q capability")
    return capabilities


def _validate_observation(raw: Any, name: str) -> None:
    observation = _mapping(raw, name)
    for key in ("text", "image", "sha256"):
        if not isinstance(observation.get(key), str):
            raise ValueError(f"{name}.{key} must be a string")
    if not observation["image"] or not observation["sha256"].startswith("sha256:"):
        raise ValueError(f"{name} image provenance is incomplete")


def _validate_planner(raw: Any, turn_name: str, *, require_mcts: bool) -> None:
    planner = _mapping(raw, f"{turn_name}.planner")
    mode = planner.get("search_mode")
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{turn_name}.planner.search_mode must be non-empty")
    horizon = planner.get("horizon")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"{turn_name}.planner.horizon must be positive")
    root_scores = _finite_or_null_vector(
        planner.get("root_scores"), f"{turn_name}.planner.root_scores"
    )
    candidates = list(
        _sequence(planner.get("candidates"), f"{turn_name}.planner.candidates")
    )
    if not candidates:
        raise ValueError(f"{turn_name}.planner candidates must not be empty")
    visits_sum = 0
    for index, raw_candidate in enumerate(candidates):
        candidate = _mapping(
            raw_candidate, f"{turn_name}.planner.candidates[{index}]"
        )
        action_ids = list(
            _sequence(candidate.get("action_ids"), "candidate.action_ids")
        )
        actions = list(_sequence(candidate.get("actions"), "candidate.actions"))
        if len(action_ids) != horizon or len(actions) != horizon:
            raise ValueError("planner candidate horizon mismatch")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in action_ids):
            raise ValueError("planner candidate action ids must be ints")
        if any(not isinstance(value, str) or not value for value in actions):
            raise ValueError("planner candidate action names must be non-empty")
        _finite(candidate.get("score"), "planner candidate score")
        visits = candidate.get("visits")
        if visits is not None:
            if isinstance(visits, bool) or not isinstance(visits, int) or visits < 1:
                raise ValueError("planner candidate visits must be positive")
            visits_sum += visits
    if require_mcts:
        if mode != "mcts":
            raise ValueError("mcts capability requires mcts search mode")
        simulations = planner.get("num_simulations")
        if (
            isinstance(simulations, bool)
            or not isinstance(simulations, int)
            or simulations < 1
        ):
            raise ValueError("mcts num_simulations must be positive")
        _finite(planner.get("exploration_constant"), "mcts exploration constant")
        root_visits = list(
            _sequence(planner.get("root_visits"), "mcts root_visits")
        )
        if len(root_visits) != len(root_scores) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in root_visits
        ):
            raise ValueError("mcts root visits must align with root scores")
        if sum(root_visits) != simulations or visits_sum != simulations:
            raise ValueError("mcts candidate visits must sum to num_simulations")


def validate_rollout_audit(raw: Mapping[str, Any]) -> None:
    """Fail closed if a rollout browser audit is incomplete or contradictory."""

    audit = _mapping(raw, "rollout audit")
    if audit.get("schema") != ROLLOUT_AUDIT_SCHEMA:
        raise ValueError("unsupported rollout audit schema")
    identity = _mapping(audit.get("identity"), "identity")
    if not any(
        isinstance(identity.get(key), str) and identity.get(key)
        for key in ("rollout_sample_id", "record_id")
    ):
        raise ValueError("rollout identity requires sample or record id")
    repeat = identity.get("rollout_repeat_index")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError("rollout repeat index must be non-negative")
    policy_family = audit.get("policy_family")
    if not isinstance(policy_family, str) or not policy_family:
        raise ValueError("policy_family must be non-empty")
    capabilities = validate_capabilities(audit.get("capabilities"))
    task = audit.get("task")
    if capabilities["task"] and (not isinstance(task, str) or not task.strip()):
        raise ValueError("task capability requires a non-empty task")
    if not isinstance(audit.get("success"), bool):
        raise ValueError("rollout success must be bool")
    _finite(audit.get("reward"), "rollout reward")
    for key in ("terminated", "truncated"):
        if not isinstance(audit.get(key), bool):
            raise ValueError(f"rollout {key} must be bool")
    turns = list(_sequence(audit.get("turns"), "turns"))
    if audit.get("turn_count") != len(turns) or not turns:
        raise ValueError("rollout turn count mismatch")
    for expected_index, raw_turn in enumerate(turns):
        turn_name = f"turns[{expected_index}]"
        turn = _mapping(raw_turn, turn_name)
        if turn.get("turn_index") != expected_index:
            raise ValueError("rollout turn indices must be contiguous")
        if capabilities["observations"]:
            _validate_observation(turn.get("observation"), f"{turn_name}.observation")
        if capabilities["cot"]:
            cot = turn.get("cot")
            raw_response = turn.get("raw_response")
            if (
                not isinstance(cot, str)
                or not cot.startswith("<think>")
                or not cot.endswith("</think>")
                or not isinstance(raw_response, str)
                or not raw_response
            ):
                raise ValueError("cot capability requires actual response and CoT")
        raw_action = turn.get("executed_action")
        if raw_action is not None:
            action = _mapping(raw_action, f"{turn_name}.executed_action")
            if (
                isinstance(action.get("id"), bool)
                or not isinstance(action.get("id"), int)
                or not isinstance(action.get("name"), str)
                or not action.get("name")
            ):
                raise ValueError("executed action is invalid")
        environment = _mapping(turn.get("environment"), f"{turn_name}.environment")
        _finite(environment.get("reward"), "turn environment reward")
        if capabilities["action_distribution"]:
            distribution = _mapping(
                turn.get("action_distribution"), f"{turn_name}.action_distribution"
            )
            _finite_or_null_vector(
                distribution.get("log_probabilities"),
                f"{turn_name}.action_distribution.log_probabilities",
            )
        elif "action_distribution" in turn:
            raise ValueError("action distribution exists without capability")
        if capabilities["direct_q"]:
            direct_q = _mapping(turn.get("direct_q"), f"{turn_name}.direct_q")
            _finite_or_null_vector(direct_q.get("values"), "direct_q.values")
            if capabilities["state_value"]:
                _finite(direct_q.get("state_value"), "direct_q.state_value")
        elif "direct_q" in turn:
            raise ValueError("direct Q exists without capability")
        if capabilities["planner"]:
            if "planner" not in turn:
                raise ValueError("planner capability requires planner evidence")
            _validate_planner(turn["planner"], turn_name, require_mcts=capabilities["mcts"])
        elif "planner" in turn:
            raise ValueError("planner evidence exists without capability")
        if capabilities["token_trace"] and "token_trace" not in turn:
            raise ValueError("token_trace capability requires token evidence")
        if "terminal" in turn:
            if expected_index != len(turns) - 1:
                raise ValueError("terminal evidence must be on the final turn")
            terminal = _mapping(turn["terminal"], "terminal")
            if terminal.get("action_executed") is not False:
                raise ValueError("terminal CoT must not execute an action")
            if capabilities["terminal_observation"]:
                _validate_observation(terminal.get("observation"), "terminal.observation")


__all__ = [
    "CAPABILITY_KEYS",
    "EVALUATION_MANIFEST_SCHEMA",
    "ROLLOUT_AUDIT_SCHEMA",
    "validate_capabilities",
    "validate_rollout_audit",
]
