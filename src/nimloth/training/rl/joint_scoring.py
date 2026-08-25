"""Strict capture-to-frozen-Q scoring contract for VAGEN joint policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

import torch

from nimloth.training.rl.joint_critic import FrozenJointCriticSnapshot

_POLICY_STATE_SCHEMA = "nimloth_policy_state_v2"
_SCORING_SCHEMA = "nimloth_frozen_q_scoring_v1"
_SCORE_DTYPES = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
}


@dataclass(frozen=True)
class FrozenQScoringRecord:
    """Immutable identity-bound result of scoring one captured policy state."""

    schema: str
    request_id: str
    generation_id: str
    contract_id: str
    snapshot_id: str
    snapshot_source_step: int
    latent_token_ids: tuple[int, ...]
    action_start_token_id: int
    action_token_ids: tuple[int, ...]
    score_dtype: str
    prior_logits: tuple[float, ...]
    frozen_all_action_q: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.schema != _SCORING_SCHEMA:
            raise ValueError(f"unsupported frozen Q scoring schema: {self.schema!r}")
        request_id = _nonempty_string(self.request_id, "request_id")
        generation_id = _nonempty_string(self.generation_id, "generation_id")
        if request_id == generation_id:
            raise ValueError(
                "frozen Q generation_id must differ from sticky request_id"
            )
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "contract_id", _nonempty_string(self.contract_id, "contract_id"))
        object.__setattr__(self, "snapshot_id", _nonempty_string(self.snapshot_id, "snapshot_id"))
        if (
            isinstance(self.snapshot_source_step, bool)
            or not isinstance(self.snapshot_source_step, int)
            or self.snapshot_source_step < 0
        ):
            raise ValueError("frozen Q snapshot_source_step must be a non-negative int")
        latent_ids, action_start, action_ids = _token_table(
            latent_token_ids=self.latent_token_ids,
            action_start_token_id=self.action_start_token_id,
            action_token_ids=self.action_token_ids,
        )
        object.__setattr__(self, "latent_token_ids", latent_ids)
        object.__setattr__(self, "action_start_token_id", action_start)
        object.__setattr__(self, "action_token_ids", action_ids)
        if self.score_dtype not in set(_SCORE_DTYPES.values()):
            raise ValueError(
                "frozen Q score_dtype must be float32, bfloat16, or float64"
            )
        logits = _quantized_vector(
            self.prior_logits,
            "prior logits",
            self.score_dtype,
        )
        q_values = _quantized_vector(
            self.frozen_all_action_q,
            "frozen Q",
            self.score_dtype,
        )
        if len(logits) != len(action_ids) or len(q_values) != len(action_ids):
            raise ValueError(
                "frozen Q prior logits, action values, and action token table must align"
            )
        object.__setattr__(self, "prior_logits", tuple(logits))
        object.__setattr__(self, "frozen_all_action_q", tuple(q_values))

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        generation_id: str,
        contract_id: str,
        snapshot_id: str,
        snapshot_source_step: int,
        latent_token_ids: Sequence[int],
        action_start_token_id: int,
        action_token_ids: Sequence[int],
        score_dtype: str,
        prior_logits: Sequence[Real],
        frozen_all_action_q: Sequence[Real],
    ) -> "FrozenQScoringRecord":
        request = _nonempty_string(request_id, "request_id")
        generation = _nonempty_string(generation_id, "generation_id")
        contract = _nonempty_string(contract_id, "contract_id")
        snapshot = _nonempty_string(snapshot_id, "snapshot_id")
        if (
            isinstance(snapshot_source_step, bool)
            or not isinstance(snapshot_source_step, int)
            or snapshot_source_step < 0
        ):
            raise ValueError("frozen Q snapshot_source_step must be a non-negative int")
        latent_ids, action_start, action_ids = _token_table(
            latent_token_ids=latent_token_ids,
            action_start_token_id=action_start_token_id,
            action_token_ids=action_token_ids,
        )
        if score_dtype not in set(_SCORE_DTYPES.values()):
            raise ValueError(
                "frozen Q score_dtype must be float32, bfloat16, or float64"
            )
        logits = _finite_vector(prior_logits, "prior logits")
        q_values = _finite_vector(frozen_all_action_q, "frozen Q")
        if len(logits) != len(action_ids) or len(q_values) != len(action_ids):
            raise ValueError(
                "frozen Q prior logits, action values, and action token table must align"
            )
        return cls(
            schema=_SCORING_SCHEMA,
            request_id=request,
            generation_id=generation,
            contract_id=contract,
            snapshot_id=snapshot,
            snapshot_source_step=snapshot_source_step,
            latent_token_ids=latent_ids,
            action_start_token_id=action_start,
            action_token_ids=action_ids,
            score_dtype=score_dtype,
            prior_logits=tuple(logits),
            frozen_all_action_q=tuple(q_values),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenQScoringRecord":
        if not isinstance(raw, Mapping):
            raise ValueError("frozen Q scoring record must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"frozen Q scoring record is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"frozen Q scoring record has unexpected fields: {sorted(unexpected)}"
            )
        if raw["schema"] != _SCORING_SCHEMA:
            raise ValueError(f"unsupported frozen Q scoring schema: {raw['schema']!r}")
        return cls.build(
            request_id=raw["request_id"],
            generation_id=raw["generation_id"],
            contract_id=raw["contract_id"],
            snapshot_id=raw["snapshot_id"],
            snapshot_source_step=raw["snapshot_source_step"],
            latent_token_ids=raw["latent_token_ids"],
            action_start_token_id=raw["action_start_token_id"],
            action_token_ids=raw["action_token_ids"],
            score_dtype=raw["score_dtype"],
            prior_logits=raw["prior_logits"],
            frozen_all_action_q=raw["frozen_all_action_q"],
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in (
            "latent_token_ids",
            "action_token_ids",
            "prior_logits",
            "frozen_all_action_q",
        ):
            raw[field] = list(raw[field])
        return raw

    def record_id(self) -> str:
        payload = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def score_captured_policy_state(
    policy_state: Mapping[str, Any],
    *,
    snapshot: FrozenJointCriticSnapshot,
    expected_request_id: str,
    expected_generation_id: str,
    expected_latent_token_ids: Sequence[int],
    expected_action_start_token_id: int,
    expected_action_token_ids: Sequence[int],
    expected_contract_id: str,
) -> FrozenQScoringRecord:
    """Validate one same-generation capture and score it with a frozen critic."""

    if not isinstance(policy_state, Mapping):
        raise ValueError("captured policy state must be a mapping")
    required_fields = {
        "schema",
        "request_id",
        "generation_id",
        "latent_token_ids",
        "action_start_token_id",
        "action_token_ids",
        "latent_hidden",
        "action_logits",
    }
    missing = required_fields - set(policy_state)
    if missing:
        raise ValueError(f"captured policy state is missing fields: {sorted(missing)}")
    unexpected = set(policy_state) - required_fields
    if unexpected:
        raise ValueError(f"captured policy state has unexpected fields: {sorted(unexpected)}")
    if policy_state["schema"] != _POLICY_STATE_SCHEMA:
        raise ValueError(
            f"unsupported captured policy state schema: {policy_state['schema']!r}"
        )
    if not isinstance(snapshot, FrozenJointCriticSnapshot):
        raise ValueError("frozen Q scorer requires FrozenJointCriticSnapshot")
    expected_request = _nonempty_string(expected_request_id, "expected_request_id")
    if policy_state["request_id"] != expected_request:
        raise ValueError(
            "captured policy state request identity mismatch: "
            f"actual={policy_state['request_id']!r}, expected={expected_request!r}"
        )
    expected_generation = _nonempty_string(
        expected_generation_id,
        "expected_generation_id",
    )
    if expected_generation == expected_request:
        raise ValueError(
            "expected_generation_id must differ from sticky expected_request_id"
        )
    if policy_state["generation_id"] != expected_generation:
        raise ValueError(
            "captured policy state generation identity mismatch: "
            f"actual={policy_state['generation_id']!r}, expected={expected_generation!r}"
        )
    expected_latent_ids, expected_action_start, expected_action_ids = _token_table(
        latent_token_ids=expected_latent_token_ids,
        action_start_token_id=expected_action_start_token_id,
        action_token_ids=expected_action_token_ids,
    )
    actual_latent_ids, actual_action_start, actual_action_ids = _token_table(
        latent_token_ids=policy_state["latent_token_ids"],
        action_start_token_id=policy_state["action_start_token_id"],
        action_token_ids=policy_state["action_token_ids"],
    )
    if actual_latent_ids != expected_latent_ids:
        raise ValueError("captured policy state latent token identity mismatch")
    if actual_action_start != expected_action_start:
        raise ValueError("captured policy state action-start token identity mismatch")
    if actual_action_ids != expected_action_ids:
        raise ValueError("captured policy state action token identity mismatch")
    contract = _nonempty_string(expected_contract_id, "expected_contract_id")
    if snapshot.contract_id != contract:
        raise ValueError(
            "frozen Q snapshot contract mismatch: "
            f"snapshot={snapshot.contract_id}, expected={contract}"
        )
    score_dtypes = {name: dtype for dtype, name in _SCORE_DTYPES.items()}
    if snapshot.score_dtype not in score_dtypes:
        raise ValueError(
            "frozen Q snapshot has unsupported contract-bound score_dtype: "
            f"{snapshot.score_dtype!r}"
        )
    score_dtype = score_dtypes[snapshot.score_dtype]
    if len(expected_latent_ids) != snapshot.spec.grid_tokens:
        raise ValueError(
            "captured policy state latent hidden shape does not match snapshot grid tokens"
        )
    if len(expected_action_ids) != snapshot.spec.action_count:
        raise ValueError(
            "captured policy state action count does not match frozen Q snapshot"
        )

    latent_rows = _finite_matrix(policy_state["latent_hidden"], "latent hidden")
    expected_hidden_shape = (
        snapshot.spec.grid_tokens,
        snapshot.spec.qwen_hidden_dim,
    )
    actual_hidden_shape = (
        len(latent_rows),
        len(latent_rows[0]) if latent_rows else 0,
    )
    if actual_hidden_shape != expected_hidden_shape:
        raise ValueError(
            "captured policy state latent hidden shape mismatch: "
            f"actual={actual_hidden_shape}, expected={expected_hidden_shape}"
        )
    raw_prior_logits = _finite_vector(
        policy_state["action_logits"],
        "action logits",
    )
    if len(raw_prior_logits) != snapshot.spec.action_count:
        raise ValueError(
            "captured policy state action logits shape does not match action count"
        )

    parameter = next(snapshot.parameters(), None)
    if parameter is None:
        raise ValueError("frozen Q snapshot must contain critic parameters")
    latent_tensor = torch.tensor(
        latent_rows,
        dtype=parameter.dtype,
        device=parameter.device,
    ).unsqueeze(0)
    frozen_q = snapshot(latent_tensor)
    if tuple(frozen_q.shape) != (1, snapshot.spec.action_count):
        raise RuntimeError(
            "frozen Q snapshot returned invalid action-value shape: "
            f"{tuple(frozen_q.shape)}"
        )
    frozen_q = frozen_q[0].to(dtype=score_dtype, device="cpu")
    prior_logits = torch.tensor(
        raw_prior_logits,
        dtype=score_dtype,
        device="cpu",
    )
    if not torch.isfinite(frozen_q).all():
        raise ValueError("frozen Q snapshot returned non-finite action values")
    if not torch.isfinite(prior_logits).all():
        raise ValueError(
            "action logits are non-finite after contract-bound dtype conversion"
        )

    return FrozenQScoringRecord.build(
        request_id=expected_request,
        generation_id=expected_generation,
        contract_id=contract,
        snapshot_id=snapshot.snapshot_id,
        snapshot_source_step=snapshot.source_step,
        latent_token_ids=expected_latent_ids,
        action_start_token_id=expected_action_start,
        action_token_ids=expected_action_ids,
        score_dtype=snapshot.score_dtype,
        prior_logits=prior_logits.tolist(),
        frozen_all_action_q=frozen_q.tolist(),
    )


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"frozen Q {field} must be non-empty")
    return value


def _token_ids(values: Sequence[int], field: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"frozen Q {field} must be a sequence")
    result = tuple(values)
    if not result:
        raise ValueError(f"frozen Q {field} must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result):
        raise ValueError(f"frozen Q {field} must contain non-negative ints")
    if len(set(result)) != len(result):
        raise ValueError(f"frozen Q {field} must be unique")
    return result


def _token_table(
    *,
    latent_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    latent_ids = _token_ids(latent_token_ids, "latent token ids")
    action_ids = _token_ids(action_token_ids, "action token ids")
    if (
        isinstance(action_start_token_id, bool)
        or not isinstance(action_start_token_id, int)
        or action_start_token_id < 0
    ):
        raise ValueError("frozen Q action-start token id must be a non-negative int")
    if action_start_token_id in latent_ids or action_start_token_id in action_ids:
        raise ValueError("frozen Q action-start token id must be distinct")
    if set(latent_ids) & set(action_ids):
        raise ValueError("frozen Q latent and action token ids must be disjoint")
    return latent_ids, action_start_token_id, action_ids


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"frozen Q {field} values must be finite real numbers")
    return float(value)


def _finite_vector(values: Sequence[Real], field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"frozen Q {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"frozen Q {field} must be non-empty")
    return result


def _quantized_vector(
    values: Sequence[Real],
    field: str,
    score_dtype: str,
) -> list[float]:
    score_dtypes = {name: dtype for dtype, name in _SCORE_DTYPES.items()}
    if score_dtype not in score_dtypes:
        raise ValueError(
            "frozen Q score_dtype must be float32, bfloat16, or float64"
        )
    tensor = torch.tensor(
        _finite_vector(values, field),
        dtype=score_dtypes[score_dtype],
        device="cpu",
    )
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"frozen Q {field} values must remain finite after "
            f"{score_dtype} conversion"
        )
    return tensor.tolist()


def _finite_matrix(values: Sequence[Sequence[Real]], field: str) -> list[list[float]]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"frozen Q {field} must be a matrix")
    rows = [_finite_vector(row, field) for row in values]
    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"frozen Q {field} must be a non-empty rectangular matrix")
    return rows


__all__ = ["FrozenQScoringRecord", "score_captured_policy_state"]
