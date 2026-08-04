"""vLLM V1 adapter for Nimloth single-request turn generation."""

from __future__ import annotations

from typing import Any

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.transformers_utils.tokenizer import get_tokenizer

from nimloth.backbone.qwen25vl.turn_generation import (
    TurnGenerationSpec,
    apply_turn_response_logits,
)


class TurnResponseLogitsProcessor(AdapterLogitsProcessor):
    """Create the per-request turn state machine from ``SamplingParams``."""

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        model_config = vllm_config.model_config
        self._tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            revision=model_config.tokenizer_revision,
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: Any):
        spec = TurnGenerationSpec.from_extra_args(params.extra_args or {})
        if spec is None:
            return None
        decoded_close_end: int | None = None

        # vLLM 0.11 decides whether to prepend prompt token IDs by inspecting
        # the request processor's parameter count. A functools.partial that
        # binds ``spec`` still exposes that keyword-only parameter, so vLLM
        # mistakes the processor for its three-argument form. Keep the runtime
        # signature explicitly two-argument: output IDs followed by logits.
        def processor(output_token_ids: list[int], logits: Any):
            nonlocal decoded_close_end
            if decoded_close_end is None and spec.close_text in self._tokenizer.decode(
                output_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
                spaces_between_special_tokens=False,
            ):
                # This callback runs before every next-token sample. Therefore
                # the first decoded match ends at the current token boundary;
                # the latent queries are forced immediately after it.
                decoded_close_end = len(output_token_ids)
            return apply_turn_response_logits(
                output_token_ids,
                logits,
                spec=spec,
                decoded_close_end=decoded_close_end,
            )

        return processor


__all__ = ["TurnResponseLogitsProcessor"]
