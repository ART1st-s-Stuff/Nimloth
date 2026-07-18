"""Runtime compatibility for tied Qwen checkpoints loaded into vLLM."""

from __future__ import annotations


def _materialize_tied_lm_head(weights, *, tie_word_embeddings: bool):
    materialized = list(weights)
    names = {name for name, _value in materialized}
    has_lm_head = any(name.endswith("lm_head.weight") for name in names)
    if tie_word_embeddings and not has_lm_head:
        embeddings = [
            value
            for name, value in materialized
            if name.endswith("embed_tokens.weight")
        ]
        if len(embeddings) != 1:
            raise ValueError(
                "tied Qwen vLLM load expected exactly one embedding weight, "
                f"found {len(embeddings)}"
            )
        materialized.append(("lm_head.weight", embeddings[0]))
    return materialized


def install_vllm_qwen25vl_tied_head_patch() -> None:
    """Duplicate tied embeddings into vLLM's separate LM-head parameter.

    The SFT merged checkpoint intentionally stores only ``model.embed_tokens``.
    Transformers honors ``tie_word_embeddings=true`` after our override, while
    vLLM's Qwen2.5-VL implementation keeps a distinct load target named
    ``language_model.lm_head``. Feed that target the exact embedding tensor.
    """
    from vllm.model_executor.models.qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
    )

    current = Qwen2_5_VLForConditionalGeneration.load_weights
    if getattr(current, "_nimloth_tied_head_fixed", False):
        return

    def load_weights(self, weights):
        materialized = _materialize_tied_lm_head(
            weights,
            tie_word_embeddings=bool(
                getattr(self.config, "tie_word_embeddings", False)
            ),
        )
        return current(self, materialized)

    load_weights._nimloth_tied_head_fixed = True
    Qwen2_5_VLForConditionalGeneration.load_weights = load_weights
