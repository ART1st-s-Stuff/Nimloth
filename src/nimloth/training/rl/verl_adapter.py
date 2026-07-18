"""Strict Nimloth trajectory tensors to VAGEN/VERL ``DataProto`` conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class VerlReplayRow:
    """One fully tokenized prompt/response row before batch padding.

    ``loss_mask`` and rewards use full-sequence coordinates. Framework-injected
    latent queries and delimiters remain in ``input_ids`` with loss mask zero;
    sampled thought/action tokens have loss mask one.
    """

    trajectory_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prompt_length: int
    loss_mask: torch.Tensor
    token_level_rewards: torch.Tensor
    end_of_response_position_mask: torch.Tensor
    multi_modal_inputs: dict[str, Any] | None = None


def _validate_row(row: VerlReplayRow) -> None:
    if row.input_ids.ndim != 1:
        raise ValueError("VERL replay input_ids must be one-dimensional")
    sequence_length = int(row.input_ids.numel())
    if not 0 < int(row.prompt_length) < sequence_length:
        raise ValueError(
            "VERL replay prompt_length must leave a nonempty response: "
            f"prompt={row.prompt_length}, sequence={sequence_length}"
        )
    for name, tensor in (
        ("attention_mask", row.attention_mask),
        ("loss_mask", row.loss_mask),
        ("token_level_rewards", row.token_level_rewards),
        ("end_of_response_position_mask", row.end_of_response_position_mask),
    ):
        if tensor.ndim != 1 or int(tensor.numel()) != sequence_length:
            raise ValueError(
                f"VERL replay {name} must have shape [{sequence_length}]"
            )
    if row.position_ids.ndim not in {1, 2}:
        raise ValueError("VERL replay position_ids rank must be one or two")
    if int(row.position_ids.shape[-1]) != sequence_length:
        raise ValueError(
            "VERL replay position_ids final dimension must match input_ids"
        )
    if not torch.all((row.attention_mask == 0) | (row.attention_mask == 1)):
        raise ValueError("VERL replay attention_mask must be binary")
    if not torch.all((row.loss_mask == 0) | (row.loss_mask == 1)):
        raise ValueError("VERL replay loss_mask must be binary")
    if not torch.all(
        (row.end_of_response_position_mask == 0)
        | (row.end_of_response_position_mask == 1)
    ):
        raise ValueError("VERL replay end-of-response mask must be binary")

    prompt = slice(0, int(row.prompt_length))
    response = slice(int(row.prompt_length), sequence_length)
    if bool(row.loss_mask[prompt].any()):
        raise ValueError("VERL replay prompt loss mask must be zero")
    if bool(row.token_level_rewards[prompt].ne(0).any()):
        raise ValueError("VERL replay prompt reward must be zero")
    if bool(row.end_of_response_position_mask[prompt].any()):
        raise ValueError("VERL replay prompt end-of-response mask must be zero")
    if not bool(row.loss_mask[response].any()):
        raise ValueError("VERL replay contains no sampled policy tokens")
    if bool((row.loss_mask.bool() & ~row.attention_mask.bool()).any()):
        raise ValueError("VERL replay loss mask selects an unattended token")
    reward_positions = row.token_level_rewards.ne(0)
    if bool(
        (reward_positions & ~row.end_of_response_position_mask.bool()).any()
    ):
        raise ValueError(
            "VERL replay nonzero rewards must use end-of-response positions"
        )
    if row.multi_modal_inputs is not None and not isinstance(
        row.multi_modal_inputs, dict
    ):
        raise ValueError("VERL replay multi_modal_inputs must be a dict or None")


def build_verl_replay_dataproto(
    rows: Sequence[VerlReplayRow],
    *,
    pad_token_id: int,
    temperature: float = 1.0,
    micro_batch_size: int = 1,
):
    """Pad exact replay rows using VERL's prompt-left/response-right convention."""

    if not rows:
        raise ValueError("cannot build an empty VERL replay batch")
    if micro_batch_size <= 0 or len(rows) % micro_batch_size != 0:
        raise ValueError(
            "VERL replay batch size must be divisible by micro_batch_size"
        )
    if temperature <= 0:
        raise ValueError("VERL replay temperature must be positive")
    for row in rows:
        _validate_row(row)

    position_rank = rows[0].position_ids.ndim
    if any(row.position_ids.ndim != position_rank for row in rows):
        raise ValueError("VERL replay position_ids rank must be uniform")
    position_channels = (
        int(rows[0].position_ids.shape[0]) if position_rank == 2 else None
    )
    if position_rank == 2 and any(
        int(row.position_ids.shape[0]) != position_channels for row in rows
    ):
        raise ValueError("VERL replay position_ids channels must be uniform")
    has_multimodal = [row.multi_modal_inputs is not None for row in rows]
    if any(has_multimodal) and not all(has_multimodal):
        raise ValueError(
            "VERL replay batch cannot mix multimodal and text-only rows"
        )

    prompt_width = max(int(row.prompt_length) for row in rows)
    response_width = max(
        int(row.input_ids.numel()) - int(row.prompt_length) for row in rows
    )
    sequence_width = prompt_width + response_width
    batch_size = len(rows)

    input_ids = torch.full(
        (batch_size, sequence_width), int(pad_token_id), dtype=torch.long
    )
    prompts = torch.full(
        (batch_size, prompt_width), int(pad_token_id), dtype=torch.long
    )
    responses = torch.full(
        (batch_size, response_width), int(pad_token_id), dtype=torch.long
    )
    attention_mask = torch.zeros((batch_size, sequence_width), dtype=torch.long)
    loss_mask = torch.zeros_like(attention_mask)
    end_mask = torch.zeros_like(attention_mask)
    full_rewards = torch.zeros((batch_size, sequence_width), dtype=torch.float32)
    response_rewards = torch.zeros(
        (batch_size, response_width), dtype=torch.float32
    )
    if position_rank == 1:
        position_ids = torch.zeros(
            (batch_size, sequence_width), dtype=rows[0].position_ids.dtype
        )
    else:
        assert position_channels is not None
        position_ids = torch.zeros(
            (batch_size, position_channels, sequence_width),
            dtype=rows[0].position_ids.dtype,
        )

    raw_prompt_ids: list[list[int]] = []
    for batch_index, row in enumerate(rows):
        prompt_length = int(row.prompt_length)
        response_length = int(row.input_ids.numel()) - prompt_length
        prompt_start = prompt_width - prompt_length
        response_start = prompt_width
        response_end = response_start + response_length
        source_response = slice(prompt_length, prompt_length + response_length)

        prompt_tokens = row.input_ids[:prompt_length].to(dtype=torch.long)
        response_tokens = row.input_ids[source_response].to(dtype=torch.long)
        prompts[batch_index, prompt_start:] = prompt_tokens
        responses[batch_index, :response_length] = response_tokens
        input_ids[batch_index, prompt_start:prompt_width] = prompt_tokens
        input_ids[batch_index, response_start:response_end] = response_tokens
        attention_mask[batch_index, prompt_start:prompt_width] = (
            row.attention_mask[:prompt_length].to(dtype=torch.long)
        )
        attention_mask[batch_index, response_start:response_end] = (
            row.attention_mask[source_response].to(dtype=torch.long)
        )
        loss_mask[batch_index, response_start:response_end] = row.loss_mask[
            source_response
        ].to(dtype=torch.long)
        end_mask[batch_index, response_start:response_end] = (
            row.end_of_response_position_mask[source_response].to(dtype=torch.long)
        )
        row_rewards = row.token_level_rewards.to(dtype=torch.float32)
        full_rewards[batch_index, response_start:response_end] = row_rewards[
            source_response
        ]
        response_rewards[batch_index, :response_length] = row_rewards[
            source_response
        ]
        if position_rank == 1:
            position_ids[batch_index, prompt_start:prompt_width] = row.position_ids[
                :prompt_length
            ]
            position_ids[batch_index, response_start:response_end] = (
                row.position_ids[source_response]
            )
        else:
            position_ids[batch_index, :, prompt_start:prompt_width] = (
                row.position_ids[:, :prompt_length]
            )
            position_ids[batch_index, :, response_start:response_end] = (
                row.position_ids[:, source_response]
            )
        raw_prompt_ids.append([int(token) for token in prompt_tokens.tolist()])

    from verl.protocol import DataProto

    tensors = {
        "prompts": prompts,
        "responses": responses,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "loss_mask": loss_mask,
        "gae_mask": loss_mask.clone(),
        "end_of_response_position_mask": end_mask,
        "multi_turn_token_level_rewards": full_rewards,
        "token_level_scores": response_rewards,
        "index": torch.arange(batch_size, dtype=torch.long),
        "step_reward_sum": response_rewards.sum(dim=-1),
    }
    non_tensors: dict[str, Any] = {
        "trajectory_id": [row.trajectory_id for row in rows],
        "raw_prompt_ids": raw_prompt_ids,
    }
    if all(has_multimodal):
        non_tensors["multi_modal_inputs"] = [
            row.multi_modal_inputs for row in rows
        ]
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={
            "temperature": float(temperature),
            "micro_batch_size": int(micro_batch_size),
            "use_dynamic_bsz": False,
        },
    )
