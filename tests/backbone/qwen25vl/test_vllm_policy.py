from __future__ import annotations

import sys
from types import SimpleNamespace

from nimloth.backbone.qwen25vl import vllm_policy as module
from nimloth.backbone.qwen25vl.vllm_policy import QwenVLLMAgentPolicy
from nimloth.latent import LatentActionTokens


class _Tokenizer:
    unk_token_id = None

    def __init__(self) -> None:
        tokens = LatentActionTokens()
        names = [
            tokens.latent_state,
            tokens.action_start,
            *tokens.action_tokens,
            tokens.action_end,
        ]
        self._ids = {name: index + 10 for index, name in enumerate(names)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [self._ids[token]]


class _Engine:
    def __init__(self, action_ids: tuple[int, ...]) -> None:
        self.action_ids = action_ids
        self.requests = []

    def generate(self, prompts, sampling_params, *, use_tqdm=False):
        self.requests.append((prompts, sampling_params, use_tqdm))
        logprobs = {
            token_id: SimpleNamespace(logprob=float(index - 7))
            for index, token_id in enumerate(self.action_ids)
        }
        completion = SimpleNamespace(logprobs=[logprobs])
        return [SimpleNamespace(outputs=[completion])]


def test_vllm_policy_reuses_agent_sampling_contract(monkeypatch) -> None:
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    action_ids = tuple(
        processor.tokenizer.convert_tokens_to_ids(token)
        for token in LatentActionTokens().action_tokens
    )
    engine = _Engine(action_ids)
    monkeypatch.setattr(module, "render_policy_messages", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(module, "collect_policy_images", lambda messages: [])
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    policy = QwenVLLMAgentPolicy(
        engine=engine,
        processor=processor,
        temperature=0.0,
        top_p=1.0,
    )
    prompt = SimpleNamespace(bound_messages=lambda: [])
    decision = policy.select_action(prompt)  # type: ignore[arg-type]

    assert decision.action_index == 7
    request, params, use_tqdm = engine.requests[0]
    assert request == [{"prompt": "prompt"}]
    assert params.allowed_token_ids == list(action_ids)
    assert params.logprobs == 8
    assert use_tqdm is False
