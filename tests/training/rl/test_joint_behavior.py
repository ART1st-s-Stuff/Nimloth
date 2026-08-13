from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import pytest

from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
from nimloth.training.rl.joint_behavior import NimlothPolicyResponseTrace
from nimloth.training.rl.joint_scoring import FrozenQScoringRecord
from vagen.joint_policy import (
    FrozenQGuidedPolicyConfig,
    GuidedActionExecutionRequest,
    GuidedPolicyActionDrawRecord,
    sample_frozen_q_guided_action,
)


_ACTION_NAMES = ("move_forward", "turn_right")
_ACTION_TOKEN_IDS = (100, 101)
_RAW_RESPONSE = (
    "<think>real thought</think>"
    "<|latent_state|><|latent_state_1|>"
    "<|action_start|><|action_(0)|><|action_end|>"
)
_RESPONSE_IDS = (1, 2, 7, 8, 90, 91, 92, 100, 93)
_RESPONSE_MASK = (1, 1, 1, 1, 0, 0, 0, 1, 0)
_TOKEN_TEXT = {
    1: "real",
    2: " thought",
    7: "</",
    8: "think>",
    90: "<|latent_state|>",
    91: "<|latent_state_1|>",
    92: "<|action_start|>",
    93: "<|action_end|>",
    94: "<|wrong_end|>",
    100: "<|action_(0)|>",
    101: "<|action_(1)|>",
}


class _Tokenizer:
    def decode(self, token_ids, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        return "".join(_TOKEN_TEXT.get(token_id, f"<{token_id}>") for token_id in token_ids)


def _spec(*, action_end_token_id: int = 93) -> TurnGenerationSpec:
    return TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=(7, 8),
        injected_token_ids=(90, 91, 92),
        action_token_ids=_ACTION_TOKEN_IDS,
        action_end_token_id=action_end_token_id,
        forbidden_reasoning_token_ids=(),
        max_reasoning_tokens=4,
    )


def _config(*, score_dtype: str = "float64") -> FrozenQGuidedPolicyConfig:
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": 1.0,
            "prior_temperature": 1.0,
            "backprop_to_llm": True,
            "score_dtype": score_dtype,
        }
    )


def _scoring_record(
    *,
    config: FrozenQGuidedPolicyConfig | None = None,
) -> FrozenQScoringRecord:
    config = config or _config()
    return FrozenQScoringRecord.build(
        request_id="session-17",
        generation_id="generation-23",
        contract_id=config.contract_id(
            "navigation_v1", _ACTION_NAMES, _ACTION_TOKEN_IDS
        ),
        snapshot_id="sha256:frozen-step-7",
        snapshot_source_step=7,
        latent_token_ids=(90, 91),
        action_start_token_id=92,
        action_token_ids=_ACTION_TOKEN_IDS,
        score_dtype=config.score_dtype,
        prior_logits=(0.0, 0.0),
        frozen_all_action_q=(0.0, math.log(3.0)),
    )


def _draw(
    *,
    config: FrozenQGuidedPolicyConfig | None = None,
    scoring_record: FrozenQScoringRecord | None = None,
    uniform_draw: float = 0.5,
    action_space_names=_ACTION_NAMES,
) -> GuidedPolicyActionDrawRecord:
    config = config or _config()
    score = scoring_record or _scoring_record(config=config)
    return sample_frozen_q_guided_action(
        action_space="navigation_v1",
        action_space_names=action_space_names,
        action_token_ids=score.action_token_ids,
        prior_logits=score.prior_logits,
        frozen_all_action_q=score.frozen_all_action_q,
        uniform_draw=uniform_draw,
        config=config,
    )


def _response_logprobs(*, action_logprob: float = math.log(0.5)) -> tuple[float, ...]:
    return (-0.1, -0.2, -0.3, -0.4, 0.0, 0.0, 0.0, action_logprob, 0.0)


def _trace(
    *,
    response_ids=_RESPONSE_IDS,
    response_mask=_RESPONSE_MASK,
    response_logprobs=None,
    raw_response: str = _RAW_RESPONSE,
    request_id: str = "session-17",
    generation_id: str = "generation-23",
    spec: TurnGenerationSpec | None = None,
) -> NimlothPolicyResponseTrace:
    return NimlothPolicyResponseTrace.build(
        request_id=request_id,
        generation_id=generation_id,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=(
            _response_logprobs()
            if response_logprobs is None
            else response_logprobs
        ),
        raw_response=raw_response,
        generation_spec=spec or _spec(),
        tokenizer=_Tokenizer(),
    )


def _build(**overrides):
    from nimloth.training.rl.joint_behavior import build_guided_execution_from_scoring

    scoring_record = overrides.pop("scoring_record", _scoring_record())
    action_draw = overrides.pop("action_draw", None)
    if action_draw is None:
        action_draw = _draw(scoring_record=scoring_record)
    kwargs = {
        "scoring_record": scoring_record,
        "action_draw": action_draw,
        "response_trace": _trace(),
        "generation_spec": _spec(),
        "tokenizer": _Tokenizer(),
        "expected_request_id": "session-17",
        "expected_generation_id": "generation-23",
        "expected_snapshot_id": "sha256:frozen-step-7",
        "expected_contract_id": (
            scoring_record.contract_id
            if isinstance(scoring_record, FrozenQScoringRecord)
            else scoring_record["contract_id"]
        ),
        "expected_generation_spec_id": _trace().generation_spec_id,
    }
    kwargs.update(overrides)
    return build_guided_execution_from_scoring(**kwargs)


def test_builds_identity_bound_behavior_without_selecting_action() -> None:
    trace = _trace()
    request = _build(response_trace=trace)
    assert isinstance(request, GuidedActionExecutionRequest)
    assert request.response_trace_id == trace.trace_id()
    behavior = request.behavior_record
    assert behavior.prior_action_id == 0
    assert behavior.prior_token_id == 100
    assert behavior.prior_response_idx == 7
    assert behavior.behavior_llm_prior_logprob == math.log(0.5)
    assert behavior.guided_action_id == 1
    assert behavior.behavior_guided_logprob == pytest.approx(math.log(0.75))
    assert behavior.snapshot_id == "sha256:frozen-step-7"
    assert behavior.prior_logits == (0.0, 0.0)
    assert behavior.frozen_all_action_q == (0.0, math.log(3.0))
    request.validate_raw_response(_RAW_RESPONSE)


def test_external_action_choice_is_deterministic_and_can_equal_prior() -> None:
    draw = _draw(uniform_draw=0.0)
    first = _build(action_draw=draw)
    second = _build(action_draw=draw)
    assert first == second
    assert first.guided_action_id == first.prior_action_id == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_request_id", "other-session", "request"),
        ("expected_generation_id", "other-generation", "generation"),
        ("expected_snapshot_id", "other-snapshot", "snapshot"),
        ("expected_contract_id", "other-contract", "contract"),
        ("expected_generation_spec_id", "sha256:" + "0" * 64, "generation spec"),
    ],
)
def test_rejects_scoring_identity_mismatch(field: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**{field: value})


def test_rejects_response_trace_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="response request"):
        _build(response_trace=_trace(request_id="other-session"))
    with pytest.raises(ValueError, match="response generation"):
        _build(response_trace=_trace(generation_id="other-generation"))


def test_rejects_draw_score_dtype_or_action_table_mismatch() -> None:
    score = _scoring_record()
    other_config = _config(score_dtype="float32")
    other_score = _scoring_record(config=other_config)
    with pytest.raises(ValueError, match="score_dtype|contract"):
        _build(
            scoring_record=score,
            action_draw=_draw(config=other_config, scoring_record=other_score),
        )
    with pytest.raises(ValueError, match="action table|contract"):
        _build(
            scoring_record=score,
            action_draw=_draw(
                scoring_record=score,
                action_space_names=("turn_right", "move_forward"),
            ),
        )


@pytest.mark.parametrize(
    ("trace_kwargs", "message"),
    [
        ({"response_ids": _RESPONSE_IDS[:-1]}, "same length|protocol suffix"),
        ({"response_mask": _RESPONSE_MASK[:-1]}, "same length"),
        ({"response_logprobs": _response_logprobs()[:-1]}, "same length"),
        (
            {"response_ids": (*_RESPONSE_IDS[:4], 91, 90, *_RESPONSE_IDS[6:])},
            "protocol suffix",
        ),
        ({"response_ids": (*_RESPONSE_IDS[:-1], 94)}, "protocol suffix"),
        (
            {
                "response_ids": (101, *_RESPONSE_IDS),
                "response_mask": (1, *_RESPONSE_MASK),
                "response_logprobs": (-0.5, *_response_logprobs()),
                "raw_response": "<think><|action_(1)|>real thought</think>"
                "<|latent_state|><|latent_state_1|>"
                "<|action_start|><|action_(0)|><|action_end|>",
            },
            "exactly one action token",
        ),
        ({"response_mask": (*_RESPONSE_MASK[:7], 0, 0)}, "canonical policy ownership"),
        (
            {"response_mask": (1, 0, 1, 1, 0, 0, 0, 1, 0)},
            "canonical policy ownership",
        ),
        (
            {"response_mask": (*_RESPONSE_MASK[:4], 1, 0, 0, 1, 0)},
            "canonical policy ownership",
        ),
        (
            {"response_logprobs": (*_response_logprobs()[:4], float("nan"), 0, 0, math.log(0.5), 0)},
            "finite",
        ),
    ],
)
def test_rejects_noncanonical_response_trace(trace_kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _trace(**trace_kwargs)


def test_accepts_forced_close_tokens_after_reasoning_budget() -> None:
    short_spec = replace(_spec(), max_reasoning_tokens=2)
    trace = _trace(
        response_mask=(1, 1, 0, 0, 0, 0, 0, 1, 0),
        spec=short_spec,
    )
    request = _build(
        response_trace=trace,
        generation_spec=short_spec,
        expected_generation_spec_id=trace.generation_spec_id,
    )
    assert request.behavior_record.prior_response_idx == 7


def test_rejects_raw_response_that_does_not_decode_from_response_ids() -> None:
    with pytest.raises(ValueError, match="raw text"):
        _trace(raw_response=_RAW_RESPONSE.replace("real thought", "other thought"))


def test_rejects_changed_action_end_even_when_trace_and_spec_change_together() -> None:
    altered_spec = _spec(action_end_token_id=94)
    altered_ids = (*_RESPONSE_IDS[:-1], 94)
    altered_raw = _RAW_RESPONSE.replace("<|action_end|>", "<|wrong_end|>")
    altered_trace = _trace(
        response_ids=altered_ids,
        raw_response=altered_raw,
        spec=altered_spec,
    )
    with pytest.raises(ValueError, match="generation spec identity"):
        _build(
            response_trace=altered_trace,
            generation_spec=altered_spec,
        )


@pytest.mark.parametrize("score_dtype", ["float64", "float32", "bfloat16"])
def test_sampled_prior_logprob_accepts_dtype_tolerance(score_dtype: str) -> None:
    epsilon = {"float64": 2.0**-52, "float32": 2.0**-23, "bfloat16": 2.0**-7}[score_dtype]
    config = _config(score_dtype=score_dtype)
    trace = _trace(response_logprobs=_response_logprobs(action_logprob=math.log(0.5) + 4 * epsilon))
    score = _scoring_record(config=config)
    request = _build(
        scoring_record=score,
        action_draw=_draw(config=config, scoring_record=score),
        response_trace=trace,
    )
    assert request.behavior_record.guided_action_id == 1


@pytest.mark.parametrize("score_dtype", ["float64", "float32", "bfloat16"])
def test_sampled_prior_logprob_rejects_outside_dtype_tolerance(score_dtype: str) -> None:
    epsilon = {"float64": 2.0**-52, "float32": 2.0**-23, "bfloat16": 2.0**-7}[score_dtype]
    config = _config(score_dtype=score_dtype)
    trace = _trace(response_logprobs=_response_logprobs(action_logprob=math.log(0.5) + 16 * epsilon))
    with pytest.raises(ValueError, match="LLM prior log-prob"):
        score = _scoring_record(config=config)
        _build(
            scoring_record=score,
            action_draw=_draw(config=config, scoring_record=score),
            response_trace=trace,
        )


def test_rejects_draw_forgery_and_trace_mapping_forgery() -> None:
    forged_draw = _draw().to_mapping()
    forged_draw["guided_action_id"] = 0
    with pytest.raises(ValueError, match="guided_action_id"):
        _build(action_draw=forged_draw)
    forged = _trace().to_mapping()
    forged["response_logprobs"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _build(response_trace=forged)


def test_mapping_scoring_is_revalidated_and_no_current_q_is_accepted() -> None:
    record = _scoring_record().to_mapping()
    forged = deepcopy(record)
    forged["frozen_all_action_q"][0] = float("nan")
    with pytest.raises(ValueError, match="frozen Q"):
        _build(scoring_record=forged, action_draw=_draw())
    with pytest.raises(TypeError, match="unexpected keyword"):
        _build(current_q=(1.0, 2.0))


def test_direct_trace_replacement_is_revalidated_at_helper_boundary() -> None:
    forged = replace(_trace(), raw_response="forged")
    with pytest.raises(ValueError, match="raw text"):
        _build(response_trace=forged)
