from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import pytest

from nimloth.backbone.qwen25vl import vllm_policy as module
from nimloth.backbone.qwen25vl.vllm_policy import QwenVLLMAgentPolicy
from nimloth.latent import LatentActionTokens, all_special_tokens_for_latent_count


class _Tokenizer:
    unk_token_id = None
    all_special_ids = (900, 901)
    added_tokens_decoder = {
        902: SimpleNamespace(special=True),
        903: SimpleNamespace(special=False),
    }

    def __init__(self) -> None:
        tokens = LatentActionTokens()
        names = list(all_special_tokens_for_latent_count(tokens, latent_token_count=16))
        self._ids = {name: index + 10 for index, name in enumerate(names)}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._ids[token]

    def encode(self, token: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if token == "</think>":
            return [501, 502, 503]
        return [self._ids[token]]

    def decode(self, token_ids, **kwargs) -> str:
        assert kwargs["skip_special_tokens"] is False
        values = list(token_ids)
        pieces: list[str] = []
        reverse = {value: token for token, value in self._ids.items()}
        index = 0
        while index < len(values):
            if values[index : index + 3] == [501, 502, 503]:
                pieces.append("</think>")
                index += 3
            elif values[index : index + 2] == [700, 701]:
                pieces.append("move left")
                index += 2
            else:
                pieces.append(reverse[values[index]])
                index += 1
        return "".join(pieces)


class _Engine:
    def __init__(self, action_ids: tuple[int, ...]) -> None:
        self.action_ids = action_ids
        self.requests = []

    def generate(self, prompts, sampling_params, *, use_tqdm=False):
        self.requests.append((prompts, sampling_params, use_tqdm))
        logprobs = {
            token_id: SimpleNamespace(logprob=-math.log(len(self.action_ids)))
            for token_id in self.action_ids
        }
        completion = SimpleNamespace(
            token_ids=[self.action_ids[-1]],
            logprobs=[logprobs],
        )
        return [SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2, 3])]


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
    assert params.temperature == 0.0
    assert use_tqdm is False
    assert decision.token_trace is not None
    assert decision.token_trace.token_roles == ("action", "injected")
    assert decision.token_trace.action_token_ids == action_ids


def test_from_model_forwards_ray_backend(monkeypatch) -> None:
    captured = {}
    engine = _Engine(())

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return engine

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fake_llm))
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    policy = QwenVLLMAgentPolicy.from_model(
        "/model",
        processor=processor,
        tensor_parallel_size=8,
        temperature=0.7,
        top_p=0.95,
        max_model_len=32768,
        max_images=6,
        gpu_memory_utilization=0.85,
        latent_token_count=16,
        distributed_executor_backend="ray",
        enable_prefix_caching=True,
        mm_processor_cache_gb=2.5,
    )

    assert policy.engine is engine
    assert captured["tensor_parallel_size"] == 8
    assert captured["distributed_executor_backend"] == "ray"
    assert captured["logprobs_mode"] == "processed_logprobs"
    assert captured["enable_prefix_caching"] is True
    assert captured["mm_processor_cache_gb"] == 2.5
    assert policy.latent_token_count == 16


def test_from_model_registers_turn_logits_adapter_and_eager_mode(monkeypatch) -> None:
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return _Engine(())

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fake_llm))
    QwenVLLMAgentPolicy.from_model(
        "/model",
        processor=SimpleNamespace(tokenizer=_Tokenizer()),
        tensor_parallel_size=4,
        temperature=0.7,
        top_p=0.95,
        max_model_len=32768,
        max_images=6,
        gpu_memory_utilization=0.85,
        latent_token_count=16,
        credit_assignment="turn",
        distributed_executor_backend="ray",
        enforce_eager=True,
    )

    assert captured["logits_processors"] == [
        "nimloth.backbone.qwen25vl.vllm_logits:TurnResponseLogitsProcessor"
    ]
    assert captured["enforce_eager"] is True
    assert captured["enable_prefix_caching"] is False
    assert captured["mm_processor_cache_gb"] == 0


def test_from_model_registers_policy_state_worker_extension(monkeypatch) -> None:
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return _Engine(())

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=fake_llm))
    QwenVLLMAgentPolicy.from_model(
        "/model",
        processor=SimpleNamespace(tokenizer=_Tokenizer()),
        tensor_parallel_size=4,
        temperature=0.7,
        top_p=0.95,
        max_model_len=32768,
        max_images=6,
        gpu_memory_utilization=0.85,
        latent_token_count=16,
        credit_assignment="token",
        enforce_eager=True,
        capture_policy_state=True,
    )

    assert captured["worker_extension_cls"] == (
        "nimloth.backbone.qwen25vl.vllm_hidden."
        "PolicyStateCaptureWorkerExtension"
    )


def test_turn_credit_generates_reasoning_then_constrained_action(monkeypatch) -> None:
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    action_ids = tuple(
        processor.tokenizer.convert_tokens_to_ids(token)
        for token in LatentActionTokens().action_tokens
    )

    class Engine:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, prompts, sampling_params, *, use_tqdm=False):
            self.requests.append((prompts, sampling_params, use_tqdm))
            tokens = LatentActionTokens()
            ids = [
                700,
                701,
                501,
                502,
                503,
                processor.tokenizer.convert_tokens_to_ids(tokens.latent_state),
                processor.tokenizer.convert_tokens_to_ids(tokens.action_start),
                action_ids[3],
                processor.tokenizer.convert_tokens_to_ids(tokens.action_end),
            ]
            action_logprobs = {
                token_id: SimpleNamespace(logprob=-math.log(len(action_ids)))
                for token_id in action_ids
            }
            output = SimpleNamespace(
                token_ids=ids,
                logprobs=[
                    {token_id: SimpleNamespace(logprob=-0.2)}
                    if index < 5
                    else action_logprobs
                    if index == 7
                    else {token_id: SimpleNamespace(logprob=0.0)}
                    for index, token_id in enumerate(ids)
                ],
            )
            return [SimpleNamespace(outputs=[output], prompt_token_ids=[1, 2])]

    monkeypatch.setattr(
        module,
        "render_policy_messages",
        lambda *args, **kwargs: "prompt<|image_pad|>",
    )
    monkeypatch.setattr(module, "collect_policy_images", lambda messages: ["image"])
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    engine = Engine()
    policy = QwenVLLMAgentPolicy(
        engine=engine,
        processor=processor,
        temperature=0.7,
        top_p=0.95,
        latent_token_count=1,
        credit_assignment="turn",
        max_response_tokens=15,
    )
    prompt = SimpleNamespace(
        messages=({"role": "assistant", "content": "<think>"},),
        bound_messages=lambda: [{"role": "user", "content": "image"}],
    )

    decision = policy.select_action(prompt)  # type: ignore[arg-type]

    assert decision.action_index == 3
    assert decision.response == (
        "<think>move left</think><|latent_state|><|action_start|>"
        "<|action_(3)|><|action_end|>"
    )
    assert decision.token_trace is not None
    assert decision.token_trace.token_roles[:3] == ("reasoning",) * 3
    assert decision.token_trace.token_roles[-2:] == ("action", "injected")
    assert decision.token_trace.action_token_ids == action_ids
    assert decision.token_trace.reasoning_text == "move left"
    assert decision.token_trace.finish_reason == "stop"
    assert decision.token_trace.reasoning_truncated is False
    assert len(engine.requests) == 1
    request = engine.requests[0][0][0]
    assert request == {
        "prompt": "prompt<|image_pad|>",
        "multi_modal_data": {"image": ["image"]},
    }
    params = engine.requests[0][1]
    assert params.max_tokens == 15
    assert params.extra_args["nimloth_turn_response"]["max_reasoning_tokens"] == 8
    assert params.stop_token_ids == [
        processor.tokenizer.convert_tokens_to_ids(LatentActionTokens().action_end)
    ]
    assert params.extra_args["nimloth_turn_response"]["action_token_ids"] == list(
        action_ids
    )
    forbidden_ids = params.extra_args["nimloth_turn_response"][
        "forbidden_reasoning_token_ids"
    ]
    assert 900 in forbidden_ids
    assert 901 in forbidden_ids
    assert 902 in forbidden_ids
    assert 903 not in forbidden_ids
    assert set(action_ids).issubset(forbidden_ids)


def test_turn_credit_rejects_forbidden_control_token_in_reasoning(
    monkeypatch,
) -> None:
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    tokens = LatentActionTokens()
    action_ids = tuple(
        processor.tokenizer.convert_tokens_to_ids(token)
        for token in tokens.action_tokens
    )

    class Engine:
        def generate(self, prompts, sampling_params, *, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            ids = [
                900,
                501,
                502,
                503,
                processor.tokenizer.convert_tokens_to_ids(tokens.latent_state),
                processor.tokenizer.convert_tokens_to_ids(tokens.action_start),
                action_ids[0],
                processor.tokenizer.convert_tokens_to_ids(tokens.action_end),
            ]
            action_logprobs = {
                token_id: SimpleNamespace(logprob=-math.log(len(action_ids)))
                for token_id in action_ids
            }
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=ids,
                            logprobs=[
                                {token_id: SimpleNamespace(logprob=-0.2)}
                                if index == 0
                                else action_logprobs
                                if index == 6
                                else {token_id: SimpleNamespace(logprob=0.0)}
                                for index, token_id in enumerate(ids)
                            ],
                        )
                    ]
                )
            ]

    monkeypatch.setattr(
        module,
        "render_policy_messages",
        lambda *args, **kwargs: "prompt",
    )
    monkeypatch.setattr(module, "collect_policy_images", lambda messages: [])
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    policy = QwenVLLMAgentPolicy(
        engine=Engine(),
        processor=processor,
        temperature=0.7,
        top_p=0.95,
        credit_assignment="turn",
        max_response_tokens=16,
    )

    with pytest.raises(RuntimeError, match="forbidden control token ids"):
        policy.select_action(
            SimpleNamespace(
                messages=({"role": "assistant", "content": "<think>"},),
                bound_messages=lambda: [],
            )
        )


def test_request_rejects_unbound_image_placeholder(monkeypatch) -> None:
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    policy = QwenVLLMAgentPolicy(
        engine=SimpleNamespace(),
        processor=processor,
        temperature=0.7,
        top_p=0.95,
    )
    monkeypatch.setattr(
        module,
        "render_policy_messages",
        lambda *args, **kwargs: "<|image_pad|><|image_pad|>",
    )
    monkeypatch.setattr(module, "collect_policy_images", lambda messages: ["image"])

    with pytest.raises(ValueError, match="placeholders do not match bound images"):
        policy._request(SimpleNamespace(bound_messages=lambda: []))


def test_turn_credit_records_forced_reasoning_close_as_truncation(monkeypatch) -> None:
    processor = SimpleNamespace(tokenizer=_Tokenizer())
    tokens = LatentActionTokens()
    action_ids = tuple(
        processor.tokenizer.convert_tokens_to_ids(token)
        for token in tokens.action_tokens
    )

    class Engine:
        def generate(self, prompts, sampling_params, *, use_tqdm=False):
            del prompts, sampling_params, use_tqdm
            ids = [
                700,
                701,
                501,
                502,
                503,
                processor.tokenizer.convert_tokens_to_ids(tokens.latent_state),
                processor.tokenizer.convert_tokens_to_ids(tokens.action_start),
                action_ids[1],
                processor.tokenizer.convert_tokens_to_ids(tokens.action_end),
            ]
            action_logprobs = {
                token_id: SimpleNamespace(logprob=-math.log(len(action_ids)))
                for token_id in action_ids
            }
            return [SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=ids,
                logprobs=[
                    {token_id: SimpleNamespace(logprob=-0.2)}
                    if index < 2
                    else action_logprobs
                    if index == 7
                    else {token_id: SimpleNamespace(logprob=0.0)}
                    for index, token_id in enumerate(ids)
                ],
            )])]

    monkeypatch.setattr(module, "render_policy_messages", lambda *args, **kwargs: "prompt")
    monkeypatch.setattr(module, "collect_policy_images", lambda messages: [])
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    policy = QwenVLLMAgentPolicy(
        engine=Engine(),
        processor=processor,
        temperature=0.7,
        top_p=0.95,
        credit_assignment="turn",
        max_response_tokens=9,
    )
    decision = policy.select_action(SimpleNamespace(
        messages=({"role": "assistant", "content": "<think>"},),
        bound_messages=lambda: [],
    ))

    assert decision.token_trace is not None
    assert decision.token_trace.finish_reason == "length"
    assert decision.token_trace.reasoning_truncated is True
    assert decision.token_trace.token_roles[:2] == ("reasoning", "reasoning")
    assert decision.token_trace.token_roles[2:7] == ("injected",) * 5
    assert decision.token_trace.old_log_probs[2:7] == (None,) * 5
