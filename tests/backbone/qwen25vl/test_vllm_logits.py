from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
from nimloth.backbone.qwen25vl.vllm_logits import TurnResponseLogitsProcessor


def test_request_processor_has_vllm_two_argument_signature() -> None:
    spec = TurnGenerationSpec(
        close_token_ids=(2,),
        injected_token_ids=(3,),
        action_token_ids=(4, 5),
        action_end_token_id=6,
        protocol_token_ids=(3, 4, 5, 6),
        max_reasoning_tokens=1,
    )
    adapter = object.__new__(TurnResponseLogitsProcessor)
    processor = adapter.new_req_logits_processor(
        SimpleNamespace(extra_args=spec.to_extra_args())
    )

    assert processor is not None
    assert len(inspect.signature(processor).parameters) == 2
    logits = torch.zeros(8)
    processed = processor([], logits)
    assert processed.shape == logits.shape
