from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
from nimloth.backbone.qwen25vl.vllm_logits import TurnResponseLogitsProcessor


def test_request_processor_has_vllm_two_argument_signature() -> None:
    spec = TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=(2,),
        injected_token_ids=(3,),
        action_token_ids=(4, 5),
        action_end_token_id=6,
        forbidden_reasoning_token_ids=(3, 4, 5, 6, 7),
        max_reasoning_tokens=1,
    )
    adapter = object.__new__(TurnResponseLogitsProcessor)
    adapter._tokenizer = SimpleNamespace(
        decode=lambda token_ids, **_kwargs: (
            "done</think>" if token_ids == [1, 2] else "done"
        )
    )
    processor = adapter.new_req_logits_processor(
        SimpleNamespace(extra_args=spec.to_extra_args())
    )

    assert processor is not None
    assert len(inspect.signature(processor).parameters) == 2
    logits = torch.zeros(8)
    processed = processor([], logits)
    assert processed.shape == logits.shape


def test_request_processor_matches_decoded_close_across_token_boundaries() -> None:
    spec = TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=(10, 11, 12),
        injected_token_ids=(3,),
        action_token_ids=(4, 5),
        action_end_token_id=6,
        forbidden_reasoning_token_ids=(3, 4, 5, 6, 7),
        max_reasoning_tokens=8,
    )
    adapter = object.__new__(TurnResponseLogitsProcessor)
    adapter._tokenizer = SimpleNamespace(
        decode=lambda token_ids, **_kwargs: (
            "Move left.</think>" if token_ids == [20, 21, 22, 11, 12] else ""
        )
    )
    processor = adapter.new_req_logits_processor(
        SimpleNamespace(extra_args=spec.to_extra_args())
    )
    assert processor is not None

    logits = torch.arange(16, dtype=torch.float32)
    processed = processor([20, 21, 22, 11, 12], logits)

    assert torch.isfinite(processed).nonzero().flatten().tolist() == [3]
