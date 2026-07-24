"""vLLM V1 adapter for Nimloth single-request turn generation."""

from __future__ import annotations

from functools import partial
from typing import Any

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from nimloth.backbone.qwen25vl.turn_generation import (
    TurnGenerationSpec,
    apply_turn_response_logits,
)


class TurnResponseLogitsProcessor(AdapterLogitsProcessor):
    """Create the per-request turn state machine from ``SamplingParams``."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: Any):
        spec = TurnGenerationSpec.from_extra_args(params.extra_args or {})
        if spec is None:
            return None
        return partial(apply_turn_response_logits, spec=spec)


__all__ = ["TurnResponseLogitsProcessor"]
