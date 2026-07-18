"""Independent Qwen2.5-VL token critic for VAGEN masked GAE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig


def load_independent_qwen_critic(
    model_path: str | Path,
    *,
    dtype: torch.dtype,
    attn_implementation: str,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    """Load VAGEN's independent full Qwen token-classification critic."""

    from verl.models.transformers.modeling_qwen_2_5_vl_patch import (
        Qwen2_5_VLForTokenClassification,
    )

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_labels = 1
    config.classifier_dropout = 0.0
    config.hidden_dropout = 0.0
    critic = Qwen2_5_VLForTokenClassification.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    critic.to(dtype=dtype)
    if gradient_checkpointing:
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return critic


def compute_token_values_for_batch(
    ppo_items: list[dict[str, Any]],
    critic: torch.nn.Module,
    processor: Any,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    history_window: int,
    latent_token_count: int,
) -> tuple[torch.Tensor, list[int]]:
    """Evaluate values immediately before every stochastic response token."""

    from nimloth.latent.extraction import LatentActionTokens, latent_state_tokens
    from nimloth.training.rl.rollout import (
        _append_input_token,
        build_nimloth_policy_messages,
        policy_inputs_from_messages,
    )

    tokens = LatentActionTokens()
    values: list[torch.Tensor] = []
    counts: list[int] = []
    for item in ppo_items:
        messages, images = build_nimloth_policy_messages(
            item["image_history_paths"],
            item["system_prompt"],
            item["observation_texts"],
            item["assistant_responses"],
            history_window=history_window,
        )
        model_inputs, _ = policy_inputs_from_messages(
            critic, processor, messages, images
        )
        prefix_length = int(model_inputs["input_ids"].shape[1])
        thought_ids = [int(token) for token in item["thought_token_ids"]]
        for token_id in thought_ids:
            model_inputs = _append_input_token(model_inputs, token_id)
        for query_name in latent_state_tokens(latent_token_count, tokens):
            model_inputs = _append_input_token(
                model_inputs, token_id_map[query_name]
            )
        model_inputs = _append_input_token(
            model_inputs, token_id_map[tokens.action_start]
        )
        outputs = critic(
            **model_inputs,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )
        positions = [
            *(prefix_length - 1 + offset for offset in range(len(thought_ids))),
            int(model_inputs["input_ids"].shape[1]) - 1,
        ]
        row_values = outputs.logits[0, positions, 0].float()
        values.extend(row_values.unbind())
        counts.append(len(positions))
    if not values:
        raise ValueError("token critic batch contains no stochastic tokens")
    return torch.stack(values), counts
