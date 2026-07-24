"""vLLM V1 adapter for Nimloth single-request turn generation."""

from __future__ import annotations

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

        # vLLM 0.11 decides whether to prepend prompt token IDs by inspecting
        # the request processor's parameter count. A functools.partial that
        # binds ``spec`` still exposes that keyword-only parameter, so vLLM
        # mistakes the processor for its three-argument form. Keep the runtime
        # signature explicitly two-argument: output IDs followed by logits.
        def processor(output_token_ids: list[int], logits: Any):
            return apply_turn_response_logits(
                output_token_ids,
                logits,
                spec=spec,
            )

        return processor


__all__ = ["TurnResponseLogitsProcessor"]
