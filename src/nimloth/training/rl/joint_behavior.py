"""Pure scoring-to-behavior assembly for the VAGEN joint-policy boundary."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
from nimloth.training.rl.joint_scoring import FrozenQScoringRecord
from vagen.joint_policy import (
    GuidedActionExecutionRequest,
    GuidedPolicyActionDrawRecord,
    GuidedPolicyBehaviorRecord,
)

_RESPONSE_TRACE_SCHEMA = "nimloth_policy_response_trace_v1"


@dataclass(frozen=True)
class NimlothPolicyResponseTrace:
    """Immutable identity-bearing token evidence from one Nimloth generation."""

    schema: str
    request_id: str
    generation_id: str
    generation_spec_id: str
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    raw_response: str

    def __post_init__(self) -> None:
        if self.schema != _RESPONSE_TRACE_SCHEMA:
            raise ValueError(
                f"unsupported Nimloth policy response trace schema: {self.schema!r}"
            )
        request_id = _nonempty_string(self.request_id, "request_id")
        generation_id = _nonempty_string(self.generation_id, "generation_id")
        if request_id == generation_id:
            raise ValueError(
                "Nimloth policy response generation_id must differ from request_id"
            )
        if not _is_sha256_id(self.generation_spec_id):
            raise ValueError(
                "Nimloth policy response generation_spec_id must be canonical sha256"
            )
        response_ids = tuple(_token_ids(self.response_ids, "response_ids"))
        response_mask = tuple(_response_mask(self.response_mask))
        response_logprobs = tuple(_response_logprobs(self.response_logprobs))
        if len({len(response_ids), len(response_mask), len(response_logprobs)}) != 1:
            raise ValueError(
                "Nimloth policy response ids, mask, and log-probs must have the same length"
            )
        raw_response = _nonempty_string(self.raw_response, "raw_response")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "response_ids", response_ids)
        object.__setattr__(self, "response_mask", response_mask)
        object.__setattr__(self, "response_logprobs", response_logprobs)
        object.__setattr__(self, "raw_response", raw_response)

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        generation_id: str,
        response_ids: Sequence[int],
        response_mask: Sequence[int | bool],
        response_logprobs: Sequence[Real],
        raw_response: str,
        generation_spec: TurnGenerationSpec,
        tokenizer: Any,
    ) -> "NimlothPolicyResponseTrace":
        spec = _generation_spec(generation_spec)
        trace = cls(
            schema=_RESPONSE_TRACE_SCHEMA,
            request_id=request_id,
            generation_id=generation_id,
            generation_spec_id=_generation_spec_id(spec),
            response_ids=tuple(response_ids),
            response_mask=tuple(response_mask),
            response_logprobs=tuple(response_logprobs),
            raw_response=raw_response,
        )
        trace.validate_protocol(generation_spec=spec, tokenizer=tokenizer)
        return trace

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NimlothPolicyResponseTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("Nimloth policy response trace must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(
                f"Nimloth policy response trace is missing fields: {sorted(missing)}"
            )
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                "Nimloth policy response trace has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        return cls(
            schema=raw["schema"],
            request_id=raw["request_id"],
            generation_id=raw["generation_id"],
            generation_spec_id=raw["generation_spec_id"],
            response_ids=tuple(raw["response_ids"]),
            response_mask=tuple(raw["response_mask"]),
            response_logprobs=tuple(raw["response_logprobs"]),
            raw_response=raw["raw_response"],
        )

    def validate_protocol(
        self,
        *,
        generation_spec: TurnGenerationSpec,
        tokenizer: Any,
    ) -> None:
        spec = _generation_spec(generation_spec)
        if self.generation_spec_id != _generation_spec_id(spec):
            raise ValueError(
                "Nimloth policy response generation spec identity mismatch"
            )
        latent_ids = spec.injected_token_ids[:-1]
        action_start_id = spec.injected_token_ids[-1]
        action_positions = [
            index
            for index, token_id in enumerate(self.response_ids)
            if token_id in spec.action_token_ids
        ]
        if len(action_positions) != 1:
            raise ValueError(
                "Nimloth policy response must contain exactly one action token"
            )
        prior_index = action_positions[0]
        prior_token_id = self.response_ids[prior_index]
        expected_suffix = (
            *latent_ids,
            action_start_id,
            prior_token_id,
            spec.action_end_token_id,
        )
        suffix_start = len(self.response_ids) - len(expected_suffix)
        if suffix_start < 0 or self.response_ids[suffix_start:] != expected_suffix:
            raise ValueError(
                "Nimloth policy response does not contain the exact protocol suffix"
            )
        for token_id in {
            *latent_ids,
            action_start_id,
            spec.action_end_token_id,
        }:
            if self.response_ids.count(token_id) != 1:
                raise ValueError(
                    "Nimloth policy response protocol suffix token identity is ambiguous"
                )
        sampled_prefix = min(suffix_start, spec.max_reasoning_tokens)
        expected_mask = (
            (1,) * sampled_prefix
            + (0,) * (suffix_start - sampled_prefix)
            + (0,) * (len(latent_ids) + 1)
            + (1, 0)
        )
        if self.response_mask != expected_mask:
            raise ValueError(
                "Nimloth policy response mask does not match canonical policy ownership"
            )
        decoded = tokenizer.decode(
            list(self.response_ids),
            skip_special_tokens=False,
        )
        if not isinstance(decoded, str):
            raise ValueError("Nimloth policy response tokenizer decode must return str")
        expected_raw_response = f"<think>{decoded}"
        if self.raw_response != expected_raw_response:
            raise ValueError(
                "Nimloth policy response raw text does not match response_ids"
            )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in ("response_ids", "response_mask", "response_logprobs"):
            raw[field] = list(raw[field])
        return raw

    def trace_id(self) -> str:
        return _sha256_id(self.to_mapping())


def build_guided_execution_from_scoring(
    *,
    scoring_record: FrozenQScoringRecord | Mapping[str, Any],
    action_draw: GuidedPolicyActionDrawRecord | Mapping[str, Any],
    response_trace: NimlothPolicyResponseTrace | Mapping[str, Any],
    generation_spec: TurnGenerationSpec,
    tokenizer: Any,
    expected_request_id: str,
    expected_generation_id: str,
    expected_snapshot_id: str,
    expected_contract_id: str,
    expected_generation_spec_id: str,
) -> GuidedActionExecutionRequest:
    """Assemble behavior for an externally selected action without RNG/current Q."""

    score = _canonical_scoring_record(scoring_record)
    draw = _canonical_action_draw(action_draw)
    trace = _canonical_response_trace(response_trace)
    spec = _generation_spec(generation_spec)
    trace.validate_protocol(generation_spec=spec, tokenizer=tokenizer)
    expected_request = _nonempty_string(expected_request_id, "expected_request_id")
    expected_generation = _nonempty_string(
        expected_generation_id,
        "expected_generation_id",
    )
    expected_snapshot = _nonempty_string(
        expected_snapshot_id,
        "expected_snapshot_id",
    )
    expected_contract = _nonempty_string(
        expected_contract_id,
        "expected_contract_id",
    )
    expected_spec_id = _nonempty_string(
        expected_generation_spec_id,
        "expected_generation_spec_id",
    )
    actual_spec_id = _generation_spec_id(spec)
    if not _is_sha256_id(expected_spec_id) or actual_spec_id != expected_spec_id:
        raise ValueError(
            "guided behavior generation spec identity mismatch: "
            f"actual={actual_spec_id!r}, expected={expected_spec_id!r}"
        )
    if score.score_dtype != draw.policy_config.score_dtype:
        raise ValueError(
            "guided behavior score_dtype mismatch: "
            f"scoring={score.score_dtype}, draw={draw.policy_config.score_dtype}"
        )
    for actual, expected, field in (
        (score.request_id, expected_request, "scoring request"),
        (trace.request_id, expected_request, "response request"),
        (score.generation_id, expected_generation, "scoring generation"),
        (trace.generation_id, expected_generation, "response generation"),
        (score.snapshot_id, expected_snapshot, "snapshot"),
        (score.contract_id, expected_contract, "contract"),
    ):
        if actual != expected:
            raise ValueError(
                f"guided behavior {field} identity mismatch: "
                f"actual={actual!r}, expected={expected!r}"
            )
    if (
        score.latent_token_ids != spec.injected_token_ids[:-1]
        or score.action_start_token_id != spec.injected_token_ids[-1]
        or score.action_token_ids != spec.action_token_ids
    ):
        raise ValueError(
            "guided behavior scoring token table does not match generation spec"
        )

    if (
        draw.contract_id != score.contract_id
        or draw.contract_id != expected_contract
        or draw.action_token_ids != score.action_token_ids
        or draw.prior_logits != score.prior_logits
        or draw.frozen_all_action_q != score.frozen_all_action_q
    ):
        raise ValueError(
            "guided behavior action draw does not match scoring contract, tokens, prior logits, or frozen Q"
        )

    prior_response_idx = len(trace.response_ids) - 2
    prior_token_id = trace.response_ids[prior_response_idx]
    prior_action_id = score.action_token_ids.index(prior_token_id)
    behavior = GuidedPolicyBehaviorRecord.build(
        action_space=draw.action_space,
        action_space_names=draw.action_space_names,
        action_token_ids=score.action_token_ids,
        snapshot_id=score.snapshot_id,
        prior_token_id=prior_token_id,
        prior_action_id=prior_action_id,
        prior_response_idx=prior_response_idx,
        behavior_llm_prior_logprob=trace.response_logprobs[prior_response_idx],
        prior_logits=score.prior_logits,
        frozen_all_action_q=score.frozen_all_action_q,
        guided_action_id=draw.guided_action_id,
        behavior_guided_logprob=draw.behavior_guided_logprob,
        config=draw.policy_config,
    )
    return GuidedActionExecutionRequest.from_behavior(
        behavior,
        raw_response=trace.raw_response,
        response_trace_id=trace.trace_id(),
        action_draw_record_id=draw.record_id(),
    )


def _canonical_scoring_record(
    value: FrozenQScoringRecord | Mapping[str, Any],
) -> FrozenQScoringRecord:
    raw = value.to_mapping() if isinstance(value, FrozenQScoringRecord) else value
    if not isinstance(raw, Mapping):
        raise ValueError("guided behavior scoring_record must be a mapping or record")
    return FrozenQScoringRecord.from_mapping(raw)


def _canonical_action_draw(
    value: GuidedPolicyActionDrawRecord | Mapping[str, Any],
) -> GuidedPolicyActionDrawRecord:
    raw = value.to_mapping() if isinstance(value, GuidedPolicyActionDrawRecord) else value
    if not isinstance(raw, Mapping):
        raise ValueError("guided behavior action_draw must be a mapping or record")
    return GuidedPolicyActionDrawRecord.from_mapping(raw)


def _canonical_response_trace(
    value: NimlothPolicyResponseTrace | Mapping[str, Any],
) -> NimlothPolicyResponseTrace:
    raw = value.to_mapping() if isinstance(value, NimlothPolicyResponseTrace) else value
    if not isinstance(raw, Mapping):
        raise ValueError("guided behavior response_trace must be a mapping or record")
    return NimlothPolicyResponseTrace.from_mapping(raw)


def _generation_spec(value: TurnGenerationSpec) -> TurnGenerationSpec:
    if not isinstance(value, TurnGenerationSpec):
        raise ValueError("guided behavior generation_spec must be TurnGenerationSpec")
    return TurnGenerationSpec.from_extra_args(value.to_extra_args())  # type: ignore[return-value]


def _generation_spec_id(spec: TurnGenerationSpec) -> str:
    return _sha256_id(spec.to_extra_args())


def _sha256_id(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _is_sha256_id(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"guided behavior {field} must be non-empty")
    return value


def _token_ids(values: Sequence[int], field: str) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"guided behavior {field} must be a sequence")
    result = list(values)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in result
    ):
        raise ValueError(f"guided behavior {field} must contain non-negative ints")
    return result


def _response_mask(values: Sequence[int | bool]) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("guided behavior response_mask must be a sequence")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            result.append(int(value))
        elif isinstance(value, int) and value in (0, 1):
            result.append(value)
        else:
            raise ValueError("guided behavior response_mask must contain only 0/1")
    return result


def _response_logprobs(values: Sequence[Real]) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("guided behavior response_logprobs must be a sequence")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("guided behavior response log-probs must be real numbers")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("guided behavior response log-probs must be finite")
        result.append(normalized)
    return result


__all__ = [
    "NimlothPolicyResponseTrace",
    "build_guided_execution_from_scoring",
]
