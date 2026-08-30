from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nimloth.backbone.qwen25vl.turn_generation import (
    TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY,
    TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY,
    build_turn_generation_spec,
    build_turn_response_policy_prompt,
    parse_turn_continuation,
    response_policy_prompt_identity,
    run_fsdp_greedy_turn_probe,
    turn_generation_spec_identity,
)
from nimloth.backbone.qwen25vl.vllm_policy import QwenVLLMAgentPolicy
from nimloth.latent import LatentActionTokens, all_special_tokens_for_latent_count


class _Tokenizer:
    unk_token_id = None
    all_special_ids = ()
    added_tokens_decoder = {}

    def __init__(self) -> None:
        names = all_special_tokens_for_latent_count(
            LatentActionTokens(), latent_token_count=16
        )
        self.ids = {name: index + 20 for index, name in enumerate(names)}
        self.reverse = {value: name for name, value in self.ids.items()}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids[token]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "</think>":
            return [10, 11, 12]
        if text in self.ids:
            return [self.ids[text]]
        raise AssertionError(text)

    def decode(self, token_ids, **kwargs) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
            "spaces_between_special_tokens": False,
        }
        pieces = []
        values = list(token_ids)
        index = 0
        while index < len(values):
            if values[index : index + 3] == [10, 11, 12]:
                pieces.append("</think>")
                index += 3
            elif values[index] == 13:
                pieces.append("</think> trailing")
                index += 1
            elif values[index] == 60:
                pieces.append("reason ")
                index += 1
            else:
                pieces.append(self.reverse.get(values[index], f"token-{values[index]}"))
                index += 1
        return "".join(pieces)


def _contract(max_response_tokens: int = 24):
    tokenizer = _Tokenizer()
    token_map = dict(tokenizer.ids)
    actions = tuple(
        token_map[token] for token in LatentActionTokens().action_tokens
    )
    spec = build_turn_generation_spec(
        tokenizer=tokenizer,
        token_id_map=token_map,
        action_token_ids=actions,
        latent_token_count=16,
        max_response_tokens=max_response_tokens,
    )
    return tokenizer, token_map, actions, spec


def _valid_continuation(spec, *, action_index: int = 3) -> tuple[int, ...]:
    return (
        60,
        *spec.close_token_ids,
        *spec.injected_token_ids,
        spec.action_token_ids[action_index],
        spec.action_end_token_id,
    )


def test_fsdp_probe_delegates_prompt_to_production_response_policy_owner() -> None:
    expected = SimpleNamespace(
        messages=({"role": "user", "content": "observation"}, {"role": "assistant", "content": "<think>"}),
        images=("image",),
    )

    class Template:
        def build_response_policy_prompt(self, transcript):
            assert transcript == "real-unacted-transcript"
            return expected

    prompt = build_turn_response_policy_prompt(
        Template(), "real-unacted-transcript"
    )
    assert prompt is expected
    expected.template = SimpleNamespace(to_record=lambda: {"identifier": "production"})
    identity = response_policy_prompt_identity(expected)
    assert len(identity) == 64
    assert len(TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY) == 64
    assert len(TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY) == 64

    bad = SimpleNamespace(messages=({"role": "assistant", "content": "fixed thought"},), images=())
    with pytest.raises(ValueError, match="<think>|response-policy"):
        build_turn_response_policy_prompt(
            SimpleNamespace(build_response_policy_prompt=lambda _transcript: bad),
            "real-unacted-transcript",
        )


def test_shared_builder_is_exactly_the_production_vllm_builder() -> None:
    tokenizer, token_map, actions, shared = _contract()
    policy = QwenVLLMAgentPolicy(
        engine=SimpleNamespace(),
        processor=SimpleNamespace(tokenizer=tokenizer),
        temperature=0.0,
        top_p=1.0,
        latent_token_count=16,
        credit_assignment="turn",
        max_response_tokens=24,
    )
    assert policy.token_id_map == token_map
    assert policy.action_token_ids == actions
    assert policy._response_generation_spec() == shared
    assert turn_generation_spec_identity(policy._response_generation_spec()) == turn_generation_spec_identity(shared)


def test_pure_parser_rejects_malformed_control_and_roundtrips_exact_response() -> None:
    tokenizer, _token_map, _actions, spec = _contract()
    valid = _valid_continuation(spec)
    parsed = parse_turn_continuation(valid, tokenizer=tokenizer, spec=spec)
    assert parsed.thought == "reason "
    assert parsed.action_index == 3
    assert parsed.response.startswith("<think>reason </think>")
    assert parsed.response[len("<think>") :] == tokenizer.decode(
        valid,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
        spaces_between_special_tokens=False,
    )

    # A merged BPE token containing an interior close remains real reasoning;
    # only the later clean terminal close owns injection, matching production.
    assert parse_turn_continuation((13, *valid), tokenizer=tokenizer, spec=spec).thought.startswith("</think> trailing")

    malformed = {
        "interior_without_clean_close": (13, *spec.injected_token_ids, spec.action_token_ids[0], spec.action_end_token_id),
        "forbidden_reasoning": (spec.injected_token_ids[0], *valid),
        "missing_query": valid[: 1 + len(spec.close_token_ids)] + valid[-2:],
        "partial_query": valid[:-5],
        "duplicate_query": valid[:-2] + spec.injected_token_ids + valid[-2:],
        "wrong_action": valid[:-2] + (99, valid[-1]),
        "missing_action_end": valid[:-1],
        "extra_suffix": valid + (60,),
    }
    for name, continuation in malformed.items():
        with pytest.raises((ValueError, RuntimeError), match="close|forbidden|injected|action|suffix|round-trip"):
            parse_turn_continuation(continuation, tokenizer=tokenizer, spec=spec)


class _GreedyFSDP(nn.Module):
    def __init__(self, vocab: int, preferred_action: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.vocab = vocab
        self.preferred_action = preferred_action
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **_kwargs):
        self.calls += 1
        logits = torch.full((1, 1, self.vocab), -10.0)
        logits[..., 60] = 10.0
        logits[..., self.preferred_action] = 20.0
        return SimpleNamespace(logits=logits)


def test_actual_model_logits_drive_fsdp_greedy_probe_without_side_effects() -> None:
    tokenizer, _token_map, actions, spec = _contract()
    model = _GreedyFSDP(vocab=128, preferred_action=actions[-1])
    prompt_inputs = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    with pytest.raises(TypeError, match="FSDP"):
        run_fsdp_greedy_turn_probe(
            model,
            prompt_inputs=prompt_inputs,
            tokenizer=tokenizer,
            spec=spec,
            checkpoint_identity="a" * 64,
            prompt_identity="b" * 64,
        )
    result = run_fsdp_greedy_turn_probe(
        model,
        prompt_inputs=prompt_inputs,
        tokenizer=tokenizer,
        spec=spec,
        checkpoint_identity="a" * 64,
        prompt_identity="b" * 64,
        require_fsdp=False,
    )
    assert model.calls == len(result.continuation_token_ids)
    assert result.parsed.action_index == 7
    assert result.parsed.reasoning_truncated is True
    assert result.action_executed is False
    assert result.rollout_persisted is False
    assert result.deployable_materialized is False
    assert result.used_current_model_logits is True
