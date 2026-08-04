from __future__ import annotations

import torch

from nimloth.backbone.qwen25vl.turn_generation import (
    TurnGenerationSpec,
    allowed_turn_token_ids,
    apply_turn_response_logits,
)


def _spec() -> TurnGenerationSpec:
    return TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=(10, 11, 12),
        injected_token_ids=(20, 21),
        action_token_ids=(30, 31, 32),
        action_end_token_id=40,
        forbidden_reasoning_token_ids=(20, 21, 30, 31, 32, 40, 50, 51),
        max_reasoning_tokens=4,
    )


def test_turn_generation_transitions_in_one_continuation() -> None:
    spec = _spec()

    assert allowed_turn_token_ids((1, 2), spec) is None
    assert allowed_turn_token_ids((1, 10, 11, 12), spec) == (20,)
    assert allowed_turn_token_ids((1, 10, 11, 12, 20), spec) == (21,)
    assert allowed_turn_token_ids((1, 10, 11, 12, 20, 21), spec) == (
        30,
        31,
        32,
    )
    assert allowed_turn_token_ids((1, 10, 11, 12, 20, 21, 31), spec) == (40,)


def test_turn_generation_uses_decoded_close_boundary_for_merged_bpe() -> None:
    spec = _spec()

    # Token 13 represents a piece such as ".</" and therefore cannot contain
    # the canonical close-token subsequence. The decoded matcher has already
    # established that token index 3 ends in literal text ``</think>``.
    assert allowed_turn_token_ids(
        (1, 13, 14),
        spec,
        decoded_close_end=3,
    ) == (20,)
    assert allowed_turn_token_ids(
        (1, 13, 14, 20),
        spec,
        decoded_close_end=3,
    ) == (21,)


def test_turn_generation_forces_close_after_reasoning_limit() -> None:
    spec = _spec()

    assert allowed_turn_token_ids((1, 2, 3, 4), spec) == (10,)
    assert allowed_turn_token_ids((1, 2, 3, 10), spec) == (11,)
    assert allowed_turn_token_ids((1, 2, 3, 10, 11), spec) == (12,)


def test_turn_logits_mask_protocol_during_reasoning_and_force_boundaries() -> None:
    spec = _spec()
    logits = torch.arange(64, dtype=torch.float32)

    reasoning = apply_turn_response_logits((1,), logits, spec=spec)
    assert torch.isneginf(
        reasoning[list(spec.forbidden_reasoning_token_ids)]
    ).all()
    assert reasoning[5] == logits[5]

    action = apply_turn_response_logits(
        (1, 10, 11, 12, 20, 21),
        logits,
        spec=spec,
    )
    finite = torch.isfinite(action).nonzero().flatten().tolist()
    assert finite == list(spec.action_token_ids)


def test_turn_generation_spec_extra_args_roundtrip() -> None:
    spec = _spec()

    assert TurnGenerationSpec.from_extra_args(spec.to_extra_args()) == spec
