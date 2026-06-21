"""Experimental monkey patches for HF Qwen2.5-VL forward semantics."""

from __future__ import annotations

from types import MethodType

import torch


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _explicit_causal_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    attn_implementation: str,
) -> torch.Tensor:
    batch_size, seq_len, _ = inputs_embeds.shape
    device = inputs_embeds.device
    causal = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool).tril()
    causal = causal.view(1, 1, seq_len, seq_len).expand(batch_size, 1, seq_len, seq_len)

    if attention_mask is not None:
        if attention_mask.ndim != 2:
            raise ValueError("explicit multimodal causal mask expects 2D attention_mask")
        padding = attention_mask.to(device=device, dtype=torch.bool)
        causal = causal & padding[:, None, None, :] & padding[:, None, :, None]
        empty_rows = ~causal.any(dim=-1, keepdim=True)
        if bool(empty_rows.any()):
            eye = torch.eye(seq_len, device=device, dtype=torch.bool).view(1, 1, seq_len, seq_len)
            causal = causal | (empty_rows & eye)

    if attn_implementation == "eager":
        min_dtype = torch.finfo(inputs_embeds.dtype).min
        zeros = torch.zeros((), device=device, dtype=inputs_embeds.dtype)
        return torch.where(causal, zeros, min_dtype)

    if attn_implementation != "sdpa":
        raise ValueError(
            "explicit multimodal causal mask patch currently supports only sdpa/eager, "
            f"got {attn_implementation!r}"
        )
    return causal


def apply_qwen25vl_force_explicit_causal_mask_patch(model) -> bool:
    """Force an explicit 4D causal mask for multimodal no-cache decoder forwards.

    This is a narrow research patch for the observed Qwen2.5-VL prefix/full
    mismatch. It only changes the text-decoder forward when all of the following
    hold:

    - multimodal 3D position ids are provided (shape `[3, batch, seq]`)
    - a standard 2D attention mask is provided
    - `past_key_values is None`
    - attention backend is `sdpa` or `eager`
    - the text config has no sliding-attention layers

    In that case we bypass the Hugging Face mask builder's skip path and pass a
    materialized lower-triangular 4D mask directly to every decoder layer.
    """

    root = _unwrap_model(model)
    text_model = getattr(getattr(root, "model", None), "language_model", None)
    if text_model is None:
        raise RuntimeError("model does not look like Qwen2.5-VL language_model")
    if getattr(text_model, "_nimloth_force_explicit_causal_mask_patch", False):
        return False

    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    original_forward = text_model.forward

    def patched_forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        force_explicit_mask = (
            position_ids.ndim == 3
            and position_ids.shape[0] == 3
            and attention_mask is not None
            and attention_mask.ndim == 2
            and past_key_values is None
            and not self.has_sliding_layers
            and self.config._attn_implementation in {"sdpa", "eager"}
        )

        if force_explicit_mask:
            causal_mask_mapping = {
                "full_attention": _explicit_causal_mask(
                    inputs_embeds,
                    attention_mask,
                    attn_implementation=self.config._attn_implementation,
                )
            }
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            layer_out = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    text_model.forward = MethodType(patched_forward, text_model)
    text_model._nimloth_force_explicit_causal_mask_patch = True
    text_model._nimloth_original_forward = original_forward
    return True
