"""Versioned, policy-capability-aware rollout browser records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

ROLLOUT_AUDIT_SCHEMA = "nimloth_rollout_audit_v3"
_LEGACY_ROLLOUT_AUDIT_SCHEMA = "nimloth_rollout_audit_v2"
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
        "model_state",
        "mcts_process",
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


def validate_capabilities(
    raw: Any,
    *,
    allow_legacy_state_absence: bool = False,
) -> dict[str, bool]:
    capabilities = dict(_mapping(raw, "capabilities"))
    legacy_keys = CAPABILITY_KEYS - {"model_state", "mcts_process"}
    if allow_legacy_state_absence and set(capabilities) == legacy_keys:
        capabilities.update({"model_state": False, "mcts_process": False})
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
    if capabilities["mcts_process"] and not (
        capabilities["mcts"] and capabilities["model_state"]
    ):
        raise ValueError("mcts_process capability requires mcts and model_state")
    return capabilities


def _validate_observation(raw: Any, name: str) -> None:
    observation = _mapping(raw, name)
    for key in ("text", "image", "sha256"):
        if not isinstance(observation.get(key), str):
            raise ValueError(f"{name}.{key} must be a string")
    if not observation["image"] or not observation["sha256"].startswith("sha256:"):
        raise ValueError(f"{name} image provenance is incomplete")


def _validate_model_state(raw: Any, turn_name: str) -> None:
    state = _mapping(raw, f"{turn_name}.model_state")
    if state.get("schema") != "nimloth_k4_model_state_archive_v1":
        raise ValueError("unsupported model state archive schema")
    archive = state.get("archive")
    digest = state.get("sha256")
    if (
        not isinstance(archive, str)
        or not archive.endswith(".npz")
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
    ):
        raise ValueError("model state archive provenance is incomplete")
    arrays = _mapping(state.get("arrays"), "model_state.arrays")
    expected = {
        "latent_hidden": [16, 2048],
        "current_state": [16, 1024],
        "mcts_node_states": None,
    }
    if set(arrays) != set(expected):
        raise ValueError("model state archive arrays mismatch")
    for key, fixed_shape in expected.items():
        spec = _mapping(arrays[key], f"model_state.arrays.{key}")
        shape = list(_sequence(spec.get("shape"), f"model_state.{key}.shape"))
        if spec.get("key") != key or spec.get("dtype") != "float32":
            raise ValueError("model state array key/dtype mismatch")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape):
            raise ValueError("model state array shape is invalid")
        if fixed_shape is not None and shape != fixed_shape:
            raise ValueError(f"model state {key} shape mismatch")
        if key == "mcts_node_states" and (
            len(shape) != 3 or shape[0] < 1 or shape[1:] != [16, 1024]
        ):
            raise ValueError("MCTS node-state tensor shape mismatch")


def _validate_mcts_process(raw: Any, turn_name: str, *, simulations: int, horizon: int) -> None:
    process = _mapping(raw, f"{turn_name}.planner.mcts_process")
    if process.get("schema") != "nimloth_k4_mcts_process_v1":
        raise ValueError("unsupported MCTS process schema")
    nodes = list(_sequence(process.get("tree_nodes"), "mcts_process.tree_nodes"))
    traces = list(_sequence(process.get("simulations"), "mcts_process.simulations"))
    if len(traces) != simulations or not nodes:
        raise ValueError("MCTS process does not contain every simulation")
    sequences: set[tuple[int, ...]] = set()
    state_indices: set[int] = set()
    for raw_node in nodes:
        node = _mapping(raw_node, "mcts_process.tree_node")
        sequence = tuple(_sequence(node.get("sequence"), "MCTS node sequence"))
        if sequence in sequences or node.get("depth") != len(sequence):
            raise ValueError("MCTS tree node identity mismatch")
        sequences.add(sequence)
        _finite(node.get("value_sum"), "MCTS node value_sum")
        _finite(node.get("mean_value"), "MCTS node mean_value")
        if not sequence:
            if node.get("state_index") is not None:
                raise ValueError("MCTS root must reference current_state")
        else:
            state_index = node.get("state_index")
            if isinstance(state_index, bool) or not isinstance(state_index, int) or state_index < 0:
                raise ValueError("predicted MCTS node requires state index")
            state_indices.add(state_index)
    if () not in sequences or state_indices != set(range(len(nodes) - 1)):
        raise ValueError("MCTS predicted-state indices are incomplete")
    for index, raw_trace in enumerate(traces):
        trace = _mapping(raw_trace, "mcts_process.simulation")
        if trace.get("simulation_index") != index:
            raise ValueError("MCTS simulation indices must be contiguous")
        steps = list(_sequence(trace.get("selection_steps"), "MCTS selection steps"))
        backups = list(_sequence(trace.get("backups"), "MCTS backups"))
        if len(steps) != horizon or len(backups) != horizon + 1:
            raise ValueError("MCTS simulation path/backup length mismatch")
        for depth, raw_step in enumerate(steps):
            step = _mapping(raw_step, "MCTS selection step")
            if step.get("depth") != depth or step.get("operation") not in {"expand", "select"}:
                raise ValueError("MCTS selection step identity mismatch")
            parent_sequence = tuple(_sequence(step.get("parent_sequence"), "MCTS parent sequence"))
            child_sequence = tuple(_sequence(step.get("child_sequence"), "MCTS child sequence"))
            if (
                len(parent_sequence) != depth
                or len(child_sequence) != depth + 1
                or child_sequence[:-1] != parent_sequence
                or child_sequence not in sequences
            ):
                raise ValueError("MCTS selection edge is not in the persisted tree")
            candidates = list(_sequence(step.get("uct_candidates"), "MCTS UCT candidates"))
            if step["operation"] == "select" and not candidates:
                raise ValueError("MCTS select step requires UCT candidates")
            for raw_candidate in candidates:
                candidate = _mapping(raw_candidate, "MCTS UCT candidate")
                for field in ("mean_value", "exploration_bonus", "uct_score"):
                    _finite(candidate.get(field), f"MCTS UCT {field}")
        leaf = _mapping(trace.get("leaf"), "MCTS leaf")
        action_values = list(_sequence(leaf.get("action_values"), "MCTS leaf action values"))
        if not action_values or any(not math.isfinite(float(value)) for value in action_values):
            raise ValueError("MCTS leaf action values are incomplete")
        leaf_value = _finite(leaf.get("value"), "MCTS leaf value")
        for raw_backup in backups:
            backup = _mapping(raw_backup, "MCTS backup")
            before = backup.get("visit_count_before")
            after = backup.get("visit_count_after")
            if (
                isinstance(before, bool)
                or not isinstance(before, int)
                or before < 0
                or after != before + 1
            ):
                raise ValueError("MCTS backup visit increment mismatch")
            before_sum = _finite(backup.get("value_sum_before"), "MCTS backup value_sum_before")
            after_sum = _finite(backup.get("value_sum_after"), "MCTS backup value_sum_after")
            if not math.isclose(after_sum, before_sum + leaf_value, rel_tol=0.0, abs_tol=1e-5):
                raise ValueError("MCTS backup value increment mismatch")
            _finite(backup.get("mean_value_after"), "MCTS backup mean_value_after")


def _validate_planner(raw: Any, turn_name: str, *, require_mcts: bool, require_process: bool) -> None:
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
        if require_process:
            _validate_mcts_process(
                planner.get("mcts_process"),
                turn_name,
                simulations=simulations,
                horizon=horizon,
            )
        elif "mcts_process" in planner:
            raise ValueError("MCTS process exists without capability")


def validate_rollout_audit(raw: Mapping[str, Any]) -> None:
    """Fail closed if a rollout browser audit is incomplete or contradictory."""

    audit = _mapping(raw, "rollout audit")
    schema = audit.get("schema")
    if schema not in {ROLLOUT_AUDIT_SCHEMA, _LEGACY_ROLLOUT_AUDIT_SCHEMA}:
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
    capabilities = validate_capabilities(
        audit.get("capabilities"),
        allow_legacy_state_absence=schema == _LEGACY_ROLLOUT_AUDIT_SCHEMA,
    )
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
            _validate_planner(
                turn["planner"],
                turn_name,
                require_mcts=capabilities["mcts"],
                require_process=capabilities["mcts_process"],
            )
        elif "planner" in turn:
            raise ValueError("planner evidence exists without capability")
        if capabilities["model_state"]:
            _validate_model_state(turn.get("model_state"), turn_name)
        elif "model_state" in turn:
            raise ValueError("model state exists without capability")
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
