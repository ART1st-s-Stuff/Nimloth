"""Batch-pinned lifecycle owner for one active frozen-Q rollout snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nimloth.training.rl.joint_critic import (
    FrozenJointCriticSnapshot,
    FrozenJointCriticSnapshotState,
    export_frozen_critic_snapshot,
    restore_frozen_critic_snapshot,
)
from nimloth.training.rl.joint_scoring import (
    FrozenQScoringRecord,
    score_captured_policy_state,
)

FROZEN_Q_BATCH_PIN_SCHEMA = "nimloth_frozen_q_batch_pin_v1"
FROZEN_Q_OWNER_SCORE_REQUEST_SCHEMA = "nimloth_frozen_q_owner_score_request_v1"
FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA = "nimloth_frozen_q_owner_score_result_v1"
FROZEN_Q_OWNER_CHECKPOINT_SCHEMA = "nimloth_frozen_q_owner_checkpoint_v1"
FROZEN_Q_OWNER_STATUS_SCHEMA = "nimloth_frozen_q_owner_status_v1"


@dataclass(frozen=True)
class FrozenQBatchPin:
    """One rollout batch's authoritative active-snapshot binding."""

    schema: str
    batch_id: str
    policy_step: int
    snapshot_id: str
    snapshot_source_step: int
    contract_id: str
    activation_version: int

    def __post_init__(self) -> None:
        if self.schema != FROZEN_Q_BATCH_PIN_SCHEMA:
            raise ValueError(f"unsupported frozen Q batch pin schema: {self.schema!r}")
        object.__setattr__(self, "batch_id", _nonempty_string(self.batch_id, "batch_id"))
        object.__setattr__(self, "policy_step", _nonnegative_int(self.policy_step, "policy_step"))
        object.__setattr__(self, "snapshot_id", _nonempty_string(self.snapshot_id, "snapshot_id"))
        object.__setattr__(
            self,
            "snapshot_source_step",
            _nonnegative_int(self.snapshot_source_step, "snapshot_source_step"),
        )
        object.__setattr__(self, "contract_id", _nonempty_string(self.contract_id, "contract_id"))
        object.__setattr__(
            self,
            "activation_version",
            _nonnegative_int(self.activation_version, "activation_version"),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenQBatchPin":
        values = _exact_mapping(raw, frozenset(cls.__dataclass_fields__), "frozen Q batch pin")
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class FrozenQOwnerScoringResult:
    """Serializable score result retaining its exact batch pin."""

    schema: str
    batch_pin: FrozenQBatchPin
    scoring_record: FrozenQScoringRecord

    def __post_init__(self) -> None:
        if self.schema != FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA:
            raise ValueError(
                f"unsupported frozen Q owner score result schema: {self.schema!r}"
            )
        pin = _canonical_pin(self.batch_pin)
        record = _canonical_scoring_record(self.scoring_record)
        if record.snapshot_id != pin.snapshot_id:
            raise ValueError("frozen Q owner score result snapshot does not match batch pin")
        if record.snapshot_source_step != pin.snapshot_source_step:
            raise ValueError("frozen Q owner score result source step does not match batch pin")
        if record.contract_id != pin.contract_id:
            raise ValueError("frozen Q owner score result contract does not match batch pin")
        object.__setattr__(self, "batch_pin", pin)
        object.__setattr__(self, "scoring_record", record)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenQOwnerScoringResult":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "frozen Q owner score result",
        )
        pin = values["batch_pin"]
        record = values["scoring_record"]
        if not isinstance(pin, Mapping) or not isinstance(record, Mapping):
            raise ValueError("frozen Q owner score result nested values must be mappings")
        return cls(
            schema=values["schema"],
            batch_pin=FrozenQBatchPin.from_mapping(pin),
            scoring_record=FrozenQScoringRecord.from_mapping(record),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "batch_pin": self.batch_pin.to_mapping(),
            "scoring_record": self.scoring_record.to_mapping(),
        }


class FrozenQSnapshotOwner:
    """Single-threaded state machine used inside the dedicated CPU Ray actor."""

    def __init__(
        self,
        *,
        initial_snapshot_state: FrozenJointCriticSnapshotState | Mapping[str, Any],
        activation_version: int = 0,
    ) -> None:
        self._active = restore_frozen_critic_snapshot(initial_snapshot_state)
        _validate_cpu_snapshot(self._active)
        self._activation_version = _nonnegative_int(
            activation_version,
            "activation_version",
        )
        self._staged: FrozenJointCriticSnapshot | None = None
        self._open_pins: dict[str, FrozenQBatchPin] = {}

    @classmethod
    def from_checkpoint_state(
        cls,
        raw: Mapping[str, Any],
    ) -> "FrozenQSnapshotOwner":
        values = _exact_mapping(
            raw,
            {"schema", "activation_version", "active_snapshot_state"},
            "frozen Q owner checkpoint",
        )
        if values["schema"] != FROZEN_Q_OWNER_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"unsupported frozen Q owner checkpoint schema: {values['schema']!r}"
            )
        state = values["active_snapshot_state"]
        if not isinstance(state, Mapping):
            raise ValueError("frozen Q owner active_snapshot_state must be a mapping")
        return cls(
            initial_snapshot_state=state,
            activation_version=values["activation_version"],
        )

    def status(self) -> dict[str, Any]:
        return {
            "schema": FROZEN_Q_OWNER_STATUS_SCHEMA,
            "active_snapshot_id": self._active.snapshot_id,
            "active_source_step": self._active.source_step,
            "contract_id": self._active.contract_id,
            "score_dtype": self._active.score_dtype,
            "parameter_dtype": str(_snapshot_parameter_dtype(self._active)).removeprefix("torch."),
            "activation_version": self._activation_version,
            "staged_snapshot_id": (
                None if self._staged is None else self._staged.snapshot_id
            ),
            "open_batch_count": len(self._open_pins),
        }

    def pin_batch(
        self,
        *,
        batch_id: str,
        policy_step: int,
        expected_snapshot_id: str,
        expected_activation_version: int,
    ) -> dict[str, Any]:
        batch = _nonempty_string(batch_id, "batch_id")
        step = _nonnegative_int(policy_step, "policy_step")
        expected_snapshot = _nonempty_string(
            expected_snapshot_id,
            "expected_snapshot_id",
        )
        expected_version = _nonnegative_int(
            expected_activation_version,
            "expected_activation_version",
        )
        self._validate_active_cas(expected_snapshot, expected_version)
        pin = FrozenQBatchPin(
            schema=FROZEN_Q_BATCH_PIN_SCHEMA,
            batch_id=batch,
            policy_step=step,
            snapshot_id=self._active.snapshot_id,
            snapshot_source_step=self._active.source_step,
            contract_id=self._active.contract_id,
            activation_version=self._activation_version,
        )
        existing = self._open_pins.get(batch)
        if existing is not None:
            if existing != pin:
                raise ValueError("frozen Q batch id is already pinned with another identity")
            return existing.to_mapping()
        self._open_pins[batch] = pin
        return pin.to_mapping()

    def unpin_batch(self, value: FrozenQBatchPin | Mapping[str, Any]) -> dict[str, Any]:
        pin = _canonical_pin(value)
        existing = self._open_pins.get(pin.batch_id)
        if existing is None:
            raise ValueError(f"frozen Q batch is not pinned: {pin.batch_id!r}")
        if existing != pin:
            raise ValueError("frozen Q unpin request does not match open batch pin")
        del self._open_pins[pin.batch_id]
        return self.status()

    def score(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(
            raw,
            {
                "schema",
                "batch_pin",
                "policy_state",
                "expected_request_id",
                "expected_generation_id",
                "expected_latent_token_ids",
                "expected_action_start_token_id",
                "expected_action_token_ids",
                "expected_contract_id",
            },
            "frozen Q owner score request",
        )
        if values["schema"] != FROZEN_Q_OWNER_SCORE_REQUEST_SCHEMA:
            raise ValueError(
                f"unsupported frozen Q owner score request schema: {values['schema']!r}"
            )
        pin_raw = values["batch_pin"]
        policy_state = values["policy_state"]
        if not isinstance(pin_raw, Mapping):
            raise ValueError("frozen Q owner score request batch_pin must be a mapping")
        if not isinstance(policy_state, Mapping):
            raise ValueError("frozen Q owner score request policy_state must be a mapping")
        pin = FrozenQBatchPin.from_mapping(pin_raw)
        active_pin = self._open_pins.get(pin.batch_id)
        if active_pin is None:
            raise ValueError(f"frozen Q batch is not pinned: {pin.batch_id!r}")
        if active_pin != pin:
            raise ValueError("frozen Q score request batch pin does not match open pin")
        if pin.snapshot_id != self._active.snapshot_id:
            raise RuntimeError("frozen Q open batch no longer matches active snapshot")
        record = score_captured_policy_state(
            policy_state,
            snapshot=self._active,
            expected_request_id=values["expected_request_id"],
            expected_generation_id=values["expected_generation_id"],
            expected_latent_token_ids=values["expected_latent_token_ids"],
            expected_action_start_token_id=values["expected_action_start_token_id"],
            expected_action_token_ids=values["expected_action_token_ids"],
            expected_contract_id=values["expected_contract_id"],
        )
        return FrozenQOwnerScoringResult(
            schema=FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA,
            batch_pin=pin,
            scoring_record=record,
        ).to_mapping()

    def stage_snapshot(
        self,
        *,
        new_snapshot_state: FrozenJointCriticSnapshotState | Mapping[str, Any],
        expected_active_snapshot_id: str,
        expected_activation_version: int,
    ) -> dict[str, Any]:
        self._validate_active_cas(
            _nonempty_string(
                expected_active_snapshot_id,
                "expected_active_snapshot_id",
            ),
            _nonnegative_int(
                expected_activation_version,
                "expected_activation_version",
            ),
        )
        candidate = restore_frozen_critic_snapshot(new_snapshot_state)
        _validate_cpu_snapshot(candidate)
        if candidate.source_step <= self._active.source_step:
            raise ValueError("staged frozen Q snapshot source step must be strictly newer")
        if candidate.contract_id != self._active.contract_id:
            raise ValueError("staged frozen Q snapshot contract does not match active snapshot")
        if candidate.score_dtype != self._active.score_dtype:
            raise ValueError("staged frozen Q snapshot score dtype does not match active snapshot")
        if _snapshot_parameter_dtype(candidate) != _snapshot_parameter_dtype(self._active):
            raise ValueError(
                "staged frozen Q snapshot parameter dtype does not match active snapshot"
            )
        if candidate.spec != self._active.spec:
            raise ValueError("staged frozen Q snapshot architecture does not match active snapshot")
        if self._staged is not None:
            if self._staged.snapshot_id != candidate.snapshot_id:
                raise ValueError("a different frozen Q snapshot is already staged")
            return self.status()
        self._staged = candidate
        return self.status()

    def activate_staged(
        self,
        *,
        staged_snapshot_id: str,
        expected_active_snapshot_id: str,
        expected_activation_version: int,
    ) -> dict[str, Any]:
        self._validate_active_cas(
            _nonempty_string(
                expected_active_snapshot_id,
                "expected_active_snapshot_id",
            ),
            _nonnegative_int(
                expected_activation_version,
                "expected_activation_version",
            ),
        )
        if self._open_pins:
            raise ValueError("cannot activate staged frozen Q snapshot with open batch pins")
        if self._staged is None:
            raise ValueError("no frozen Q snapshot is staged")
        expected_staged = _nonempty_string(staged_snapshot_id, "staged_snapshot_id")
        if self._staged.snapshot_id != expected_staged:
            raise ValueError("staged frozen Q snapshot identity mismatch")
        self._active = self._staged
        self._staged = None
        self._activation_version += 1
        return self.status()

    def checkpoint_state(self) -> dict[str, Any]:
        if self._open_pins:
            raise ValueError("cannot checkpoint frozen Q owner with open batch pins")
        if self._staged is not None:
            raise ValueError("cannot checkpoint frozen Q owner with a staged snapshot")
        return {
            "schema": FROZEN_Q_OWNER_CHECKPOINT_SCHEMA,
            "activation_version": self._activation_version,
            "active_snapshot_state": export_frozen_critic_snapshot(
                self._active
            ).to_mapping(),
        }

    def _validate_active_cas(
        self,
        expected_snapshot_id: str,
        expected_activation_version: int,
    ) -> None:
        if self._active.snapshot_id != expected_snapshot_id:
            raise ValueError("frozen Q active snapshot compare-and-swap mismatch")
        if self._activation_version != expected_activation_version:
            raise ValueError("frozen Q activation version compare-and-swap mismatch")


def _validate_cpu_snapshot(snapshot: FrozenJointCriticSnapshot) -> None:
    if not isinstance(snapshot, FrozenJointCriticSnapshot):
        raise ValueError("frozen Q owner requires FrozenJointCriticSnapshot")
    snapshot._validate_unchanged()
    if any(parameter.device.type != "cpu" for parameter in snapshot.parameters()):
        raise ValueError("frozen Q owner snapshot parameters must be on CPU")
    if any(parameter.requires_grad for parameter in snapshot.parameters()):
        raise ValueError("frozen Q owner snapshot parameters must not require gradients")
    if any(module.training for module in snapshot.modules()):
        raise ValueError("frozen Q owner snapshot modules must be in eval mode")


def _snapshot_parameter_dtype(snapshot: FrozenJointCriticSnapshot) -> object:
    dtypes = {parameter.dtype for parameter in snapshot.parameters()}
    if len(dtypes) != 1:
        raise ValueError("frozen Q owner snapshot parameters must use one dtype")
    return next(iter(dtypes))


def _canonical_pin(value: FrozenQBatchPin | Mapping[str, Any]) -> FrozenQBatchPin:
    raw = value.to_mapping() if isinstance(value, FrozenQBatchPin) else value
    if not isinstance(raw, Mapping):
        raise ValueError("frozen Q batch pin must be a mapping or pin")
    return FrozenQBatchPin.from_mapping(raw)


def _canonical_scoring_record(
    value: FrozenQScoringRecord | Mapping[str, Any],
) -> FrozenQScoringRecord:
    raw = value.to_mapping() if isinstance(value, FrozenQScoringRecord) else value
    if not isinstance(raw, Mapping):
        raise ValueError("frozen Q scoring record must be a mapping or record")
    return FrozenQScoringRecord.from_mapping(raw)


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


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"frozen Q owner {field} must be a non-negative int")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"frozen Q owner {field} must be non-empty")
    return value


__all__ = [
    "FROZEN_Q_BATCH_PIN_SCHEMA",
    "FROZEN_Q_OWNER_CHECKPOINT_SCHEMA",
    "FROZEN_Q_OWNER_SCORE_REQUEST_SCHEMA",
    "FROZEN_Q_OWNER_SCORE_RESULT_SCHEMA",
    "FROZEN_Q_OWNER_STATUS_SCHEMA",
    "FrozenQBatchPin",
    "FrozenQOwnerScoringResult",
    "FrozenQSnapshotOwner",
]
