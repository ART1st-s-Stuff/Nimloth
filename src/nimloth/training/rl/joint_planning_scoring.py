"""Identity-safe record for TP-rank-zero K4 planner scoring."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import torch

K4_PLANNING_SCORING_SCHEMA = "nimloth_frozen_k4_planning_scoring_v1"
_K4_POLICY_STATE_SCHEMA = "nimloth_policy_state_k4_mcts_v1"
_SCORE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


@dataclass(frozen=True)
class FrozenK4PlanningScoringRecord:
    """Direct Q, MCTS root means, and complete search evidence for one turn."""

    schema: str
    request_id: str
    generation_id: str
    contract_id: str
    snapshot_id: str
    snapshot_source_step: int
    activation_version: int
    latent_token_ids: tuple[int, ...]
    action_start_token_id: int
    action_token_ids: tuple[int, ...]
    score_dtype: str
    prior_logits: tuple[float, ...]
    direct_all_action_q: tuple[float, ...]
    planner_root_mean_values: tuple[float, ...]
    planner_root_visit_counts: tuple[int, ...]
    candidate_sequences: tuple[tuple[int, ...], ...]
    candidate_mean_values: tuple[float, ...]
    candidate_visit_counts: tuple[int, ...]
    planning_horizon: int
    mcts_num_simulations: int
    mcts_exploration_constant: float
    planner_latency_seconds: float

    def __post_init__(self) -> None:
        if self.schema != K4_PLANNING_SCORING_SCHEMA:
            raise ValueError(
                f"unsupported K4 planning scoring schema: {self.schema!r}"
            )
        request = _nonempty_string(self.request_id, "request_id")
        generation = _nonempty_string(self.generation_id, "generation_id")
        if request == generation:
            raise ValueError("K4 scoring generation_id must differ from request_id")
        contract = _nonempty_string(self.contract_id, "contract_id")
        snapshot = _nonempty_string(self.snapshot_id, "snapshot_id")
        source = _nonnegative_int(self.snapshot_source_step, "snapshot_source_step")
        activation = _nonnegative_int(self.activation_version, "activation_version")
        latent_ids = tuple(_token_ids(self.latent_token_ids, "latent_token_ids"))
        action_start = _nonnegative_int(
            self.action_start_token_id,
            "action_start_token_id",
        )
        if action_start in latent_ids:
            raise ValueError("K4 scoring action_start token must differ from latent tokens")
        action_ids = tuple(_token_ids(self.action_token_ids, "action_token_ids"))
        if self.score_dtype not in _SCORE_DTYPES:
            raise ValueError("K4 scoring score_dtype is unsupported")
        action_count = len(action_ids)
        prior = tuple(
            _quantized_vector(self.prior_logits, "prior_logits", self.score_dtype)
        )
        direct_q = tuple(
            _quantized_vector(
                self.direct_all_action_q,
                "direct_all_action_q",
                self.score_dtype,
            )
        )
        planner = tuple(
            _quantized_vector(
                self.planner_root_mean_values,
                "planner_root_mean_values",
                self.score_dtype,
            )
        )
        if len(prior) != action_count or len(direct_q) != action_count or len(planner) != action_count:
            raise ValueError("K4 scoring action vectors must align with token table")
        horizon = _positive_int(self.planning_horizon, "planning_horizon")
        if horizon != 4:
            raise ValueError("K4 scoring planning_horizon must be exactly 4")
        simulations = _positive_int(
            self.mcts_num_simulations,
            "mcts_num_simulations",
        )
        root_visits = tuple(
            _visit_counts(
                self.planner_root_visit_counts,
                expected_length=action_count,
                expected_total=simulations,
                field="planner_root_visit_counts",
            )
        )
        exploration = _finite_float(
            self.mcts_exploration_constant,
            "mcts_exploration_constant",
        )
        if exploration < 0.0:
            raise ValueError("K4 scoring exploration constant must be non-negative")
        candidates = tuple(
            _candidate_sequence(row, horizon=horizon, action_count=action_count)
            for row in self.candidate_sequences
        )
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("K4 scoring candidate sequences must be non-empty and unique")
        candidate_values = tuple(
            _quantized_vector(
                self.candidate_mean_values,
                "candidate_mean_values",
                self.score_dtype,
            )
        )
        candidate_visits = tuple(
            _visit_counts(
                self.candidate_visit_counts,
                expected_length=len(candidates),
                expected_total=simulations,
                field="candidate_visit_counts",
            )
        )
        if len(candidate_values) != len(candidates):
            raise ValueError("K4 scoring candidate values must align with sequences")
        latency = _finite_float(
            self.planner_latency_seconds,
            "planner_latency_seconds",
        )
        if latency < 0.0:
            raise ValueError("K4 scoring planner latency must be non-negative")
        object.__setattr__(self, "request_id", request)
        object.__setattr__(self, "generation_id", generation)
        object.__setattr__(self, "contract_id", contract)
        object.__setattr__(self, "snapshot_id", snapshot)
        object.__setattr__(self, "snapshot_source_step", source)
        object.__setattr__(self, "activation_version", activation)
        object.__setattr__(self, "latent_token_ids", latent_ids)
        object.__setattr__(self, "action_start_token_id", action_start)
        object.__setattr__(self, "action_token_ids", action_ids)
        object.__setattr__(self, "prior_logits", prior)
        object.__setattr__(self, "direct_all_action_q", direct_q)
        object.__setattr__(self, "planner_root_mean_values", planner)
        object.__setattr__(self, "planner_root_visit_counts", root_visits)
        object.__setattr__(self, "candidate_sequences", candidates)
        object.__setattr__(self, "candidate_mean_values", candidate_values)
        object.__setattr__(self, "candidate_visit_counts", candidate_visits)
        object.__setattr__(self, "mcts_exploration_constant", exploration)
        object.__setattr__(self, "planner_latency_seconds", latency)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "FrozenK4PlanningScoringRecord":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "K4 planning scoring record",
        )
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in (
            "latent_token_ids",
            "action_token_ids",
            "prior_logits",
            "direct_all_action_q",
            "planner_root_mean_values",
            "planner_root_visit_counts",
            "candidate_mean_values",
            "candidate_visit_counts",
        ):
            raw[field] = list(raw[field])
        raw["candidate_sequences"] = [
            list(sequence) for sequence in self.candidate_sequences
        ]
        return raw

    def record_id(self) -> str:
        payload = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def k4_scoring_record_from_policy_state(
    policy_state: Mapping[str, Any],
    *,
    expected_request_id: str,
    expected_generation_id: str,
    expected_latent_token_ids: Sequence[int],
    expected_action_start_token_id: int,
    expected_action_token_ids: Sequence[int],
    expected_snapshot_id: str,
    expected_snapshot_source_step: int,
    expected_contract_id: str,
    expected_activation_version: int,
    expected_score_dtype: str,
    expected_planning_horizon: int,
    expected_mcts_num_simulations: int,
    expected_mcts_exploration_constant: float,
) -> FrozenK4PlanningScoringRecord:
    """Validate the same-generation capture and embedded rank-zero planner result."""

    required = {
        "schema",
        "request_id",
        "generation_id",
        "latent_token_ids",
        "action_start_token_id",
        "action_token_ids",
        "latent_hidden",
        "action_logits",
        "frozen_k4_planning",
    }
    values = _exact_mapping(policy_state, required, "K4 policy state")
    if values["schema"] != _K4_POLICY_STATE_SCHEMA:
        raise ValueError(f"unsupported K4 policy state schema: {values['schema']!r}")
    for actual, expected, field in (
        (values["request_id"], expected_request_id, "request_id"),
        (values["generation_id"], expected_generation_id, "generation_id"),
        (values["latent_token_ids"], list(expected_latent_token_ids), "latent_token_ids"),
        (values["action_start_token_id"], expected_action_start_token_id, "action_start_token_id"),
        (values["action_token_ids"], list(expected_action_token_ids), "action_token_ids"),
    ):
        if actual != expected:
            raise ValueError(f"K4 policy state {field} mismatch")
    planning = values["frozen_k4_planning"]
    if not isinstance(planning, Mapping):
        raise ValueError("K4 policy state planning result must be a mapping")
    planning_fields = {
        "snapshot_id",
        "source_step",
        "contract_id",
        "activation_version",
        "tensor_parallel_rank",
        "scored",
        "score_dtype",
        "planning_config",
        "direct_all_action_q",
        "planner_root_mean_values",
        "planner_root_visit_counts",
        "candidate_sequences",
        "candidate_mean_values",
        "candidate_visit_counts",
        "planner_latency_seconds",
    }
    missing_planning = planning_fields - set(planning)
    unexpected_planning = set(planning) - planning_fields - {"current_state", "mcts_trace"}
    if missing_planning or unexpected_planning:
        raise ValueError(
            "K4 policy planning result fields mismatch: "
            f"missing={sorted(missing_planning)}, unexpected={sorted(unexpected_planning)}"
        )
    plan = dict(planning)
    if plan["scored"] is not True or plan["tensor_parallel_rank"] != 0:
        raise ValueError("K4 policy planning result must come from TP rank zero")
    current_state = plan.get("current_state")
    mcts_trace = plan.get("mcts_trace")
    if (current_state is None) != (mcts_trace is None):
        raise ValueError("K4 current state and MCTS trace must be captured together")
    if current_state is not None:
        projected = _finite_matrix(current_state, "current_state")
        if len(projected) != 16 or len(projected[0]) != 1024:
            raise ValueError("K4 captured current state must have shape (16, 1024)")
        if not isinstance(mcts_trace, Mapping):
            raise ValueError("K4 captured MCTS trace must be a mapping")
    config = plan["planning_config"]
    if not isinstance(config, Mapping) or set(config) != {
        "horizon",
        "num_simulations",
        "exploration_constant",
    }:
        raise ValueError("K4 policy planning config is invalid")
    for actual, expected, field in (
        (plan["snapshot_id"], expected_snapshot_id, "snapshot_id"),
        (plan["source_step"], expected_snapshot_source_step, "source_step"),
        (plan["contract_id"], expected_contract_id, "contract_id"),
        (plan["activation_version"], expected_activation_version, "activation_version"),
        (plan["score_dtype"], expected_score_dtype, "score_dtype"),
        (config["horizon"], expected_planning_horizon, "planning_horizon"),
        (config["num_simulations"], expected_mcts_num_simulations, "mcts_num_simulations"),
        (config["exploration_constant"], expected_mcts_exploration_constant, "mcts_exploration_constant"),
    ):
        if actual != expected:
            raise ValueError(f"K4 policy planning {field} mismatch")
    latent = _finite_matrix(values["latent_hidden"], "latent_hidden")
    if len(latent) != len(expected_latent_token_ids):
        raise ValueError("K4 policy latent row count mismatch")
    prior = _finite_vector(values["action_logits"], "action_logits")
    if len(prior) != len(expected_action_token_ids):
        raise ValueError("K4 policy action logit count mismatch")
    return FrozenK4PlanningScoringRecord(
        schema=K4_PLANNING_SCORING_SCHEMA,
        request_id=expected_request_id,
        generation_id=expected_generation_id,
        contract_id=expected_contract_id,
        snapshot_id=expected_snapshot_id,
        snapshot_source_step=expected_snapshot_source_step,
        activation_version=expected_activation_version,
        latent_token_ids=tuple(expected_latent_token_ids),
        action_start_token_id=expected_action_start_token_id,
        action_token_ids=tuple(expected_action_token_ids),
        score_dtype=expected_score_dtype,
        prior_logits=tuple(prior),
        direct_all_action_q=tuple(plan["direct_all_action_q"]),
        planner_root_mean_values=tuple(plan["planner_root_mean_values"]),
        planner_root_visit_counts=tuple(plan["planner_root_visit_counts"]),
        candidate_sequences=tuple(
            tuple(row) for row in plan["candidate_sequences"]
        ),
        candidate_mean_values=tuple(plan["candidate_mean_values"]),
        candidate_visit_counts=tuple(plan["candidate_visit_counts"]),
        planning_horizon=expected_planning_horizon,
        mcts_num_simulations=expected_mcts_num_simulations,
        mcts_exploration_constant=expected_mcts_exploration_constant,
        planner_latency_seconds=plan["planner_latency_seconds"],
    )


def _quantized_vector(
    values: Sequence[Real],
    field: str,
    dtype_name: str,
) -> list[float]:
    tensor = torch.tensor(_finite_vector(values, field), dtype=_SCORE_DTYPES[dtype_name])
    if not torch.isfinite(tensor).all():
        raise ValueError(f"K4 scoring {field} is non-finite after dtype conversion")
    return [float(value) for value in tensor.tolist()]


def _visit_counts(
    values: Sequence[int],
    *,
    expected_length: int,
    expected_total: int,
    field: str,
) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 scoring {field} must be a sequence")
    result = list(values)
    if len(result) != expected_length or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError(f"K4 scoring {field} must contain positive ints")
    if sum(result) != expected_total:
        raise ValueError(f"K4 scoring {field} must sum to simulations")
    return result


def _candidate_sequence(
    values: Sequence[int],
    *,
    horizon: int,
    action_count: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("K4 candidate sequence must be a sequence")
    result = tuple(values)
    if len(result) != horizon or any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < action_count
        for value in result
    ):
        raise ValueError("K4 candidate sequence is outside action/horizon contract")
    return result


def _token_ids(values: Sequence[int], field: str) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 scoring {field} must be a sequence")
    result = list(values)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ) or len(set(result)) != len(result):
        raise ValueError(f"K4 scoring {field} must contain unique non-negative ints")
    return result


def _finite_matrix(values: Any, field: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 scoring {field} must be a matrix")
    rows = [_finite_vector(row, field) for row in values]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"K4 scoring {field} must be a non-empty rectangular matrix")
    return rows


def _finite_vector(values: Any, field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 scoring {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"K4 scoring {field} must be non-empty")
    return result


def _finite_float(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"K4 scoring {field} must be finite")
    return float(value)


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"K4 scoring {field} must be non-empty")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"K4 scoring {field} must be a non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result < 1:
        raise ValueError(f"K4 scoring {field} must be positive")
    return result


def _exact_mapping(
    raw: Mapping[str, Any],
    fields: set[str] | frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(fields) - set(raw)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    unexpected = set(raw) - set(fields)
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")
    return {field: raw[field] for field in fields}


__all__ = [
    "K4_PLANNING_SCORING_SCHEMA",
    "FrozenK4PlanningScoringRecord",
    "k4_scoring_record_from_policy_state",
]
