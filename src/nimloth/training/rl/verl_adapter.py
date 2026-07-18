"""Strict Nimloth trajectory tensors to VAGEN/VERL ``DataProto`` conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    policy_transcript: str | None = None
    task_instruction: str | None = None
    observation_texts: tuple[str, ...] = ()
    assistant_responses: tuple[str, ...] = ()
    image_paths: tuple[str, ...] = ()


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


def _find_ordered_subsequence(
    values: list[int], subsequence: list[int], *, start: int
) -> int:
    if not subsequence:
        raise ValueError("Nimloth VERL response subsequence must be nonempty")
    stop = len(values) - len(subsequence) + 1
    for index in range(start, stop):
        if values[index:index + len(subsequence)] == subsequence:
            return index
    raise ValueError(
        "Nimloth VERL transcript does not contain the expected ordered "
        f"assistant response after token {start}"
    )


def build_nimloth_verl_trajectory_replay_row(
    *,
    trajectory_id: str,
    transcript_input_ids: torch.Tensor,
    transcript_attention_mask: torch.Tensor,
    transcript_position_ids: torch.Tensor,
    thought_token_ids_by_turn: Sequence[Sequence[int]],
    latent_query_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_ids: Sequence[int],
    action_end_token_id: int,
    turn_rewards: Sequence[float],
    pad_token_id: int,
    multi_modal_inputs: dict[str, Any] | None,
) -> VerlReplayRow:
    """Serialize one complete episode so masked GAE crosses turn boundaries."""

    if transcript_input_ids.ndim != 1 or transcript_input_ids.numel() == 0:
        raise ValueError("Nimloth VERL transcript input_ids must be nonempty and 1D")
    if transcript_attention_mask.shape != transcript_input_ids.shape:
        raise ValueError("Nimloth VERL transcript attention shape mismatch")
    if not bool(transcript_attention_mask.bool().all()):
        raise ValueError("Nimloth VERL unpadded transcript attention must be all ones")
    if transcript_position_ids.ndim not in {1, 2} or int(
        transcript_position_ids.shape[-1]
    ) != int(transcript_input_ids.numel()):
        raise ValueError("Nimloth VERL transcript position_ids shape mismatch")

    thoughts = [[int(token) for token in values] for values in thought_token_ids_by_turn]
    query_ids = [int(token) for token in latent_query_token_ids]
    actions = [int(token) for token in action_token_ids]
    rewards_by_turn = [float(value) for value in turn_rewards]
    turn_count = len(thoughts)
    if turn_count == 0:
        raise ValueError("Nimloth VERL trajectory requires at least one turn")
    if not query_ids:
        raise ValueError("Nimloth VERL trajectory requires latent query tokens")
    if len(actions) != turn_count or len(rewards_by_turn) != turn_count:
        raise ValueError(
            "Nimloth VERL trajectory turn metadata mismatch: "
            f"thoughts={turn_count}, actions={len(actions)}, rewards={len(rewards_by_turn)}"
        )
    if any(not values for values in thoughts):
        raise ValueError("Nimloth VERL trajectory requires sampled thought tokens per turn")

    transcript_ids = [int(token) for token in transcript_input_ids.tolist()]
    transcript_loss_mask = torch.zeros(len(transcript_ids), dtype=torch.long)
    transcript_rewards = torch.zeros(len(transcript_ids), dtype=torch.float32)
    transcript_end_mask = torch.zeros(len(transcript_ids), dtype=torch.long)
    search_start = 0
    for turn, (thought_ids, action_id, reward) in enumerate(
        zip(thoughts, actions, rewards_by_turn, strict=True)
    ):
        expected_response = [
            *thought_ids,
            *query_ids,
            int(action_start_token_id),
            action_id,
            int(action_end_token_id),
        ]
        response_start = _find_ordered_subsequence(
            transcript_ids, expected_response, start=search_start
        )
        action_position = response_start + len(expected_response) - 2
        transcript_loss_mask[
            response_start:response_start + len(thought_ids)
        ] = 1
        transcript_loss_mask[action_position] = 1
        transcript_rewards[action_position] = reward
        transcript_end_mask[action_position] = 1
        search_start = response_start + len(expected_response)

    input_ids = torch.cat(
        (torch.tensor([int(pad_token_id)], dtype=torch.long), transcript_input_ids.cpu())
    )
    attention_mask = torch.cat(
        (torch.zeros(1, dtype=torch.long), transcript_attention_mask.long().cpu())
    )
    loss_mask = torch.cat((torch.zeros(1, dtype=torch.long), transcript_loss_mask))
    token_level_rewards = torch.cat(
        (torch.zeros(1, dtype=torch.float32), transcript_rewards)
    )
    end_mask = torch.cat((torch.zeros(1, dtype=torch.long), transcript_end_mask))
    transcript_position_ids = transcript_position_ids.detach().cpu()
    if transcript_position_ids.ndim == 1:
        position_ids = torch.cat(
            (torch.zeros(1, dtype=transcript_position_ids.dtype), transcript_position_ids)
        )
    else:
        position_ids = torch.cat(
            (
                torch.zeros(
                    (transcript_position_ids.shape[0], 1),
                    dtype=transcript_position_ids.dtype,
                ),
                transcript_position_ids,
            ),
            dim=-1,
        )
    row = VerlReplayRow(
        trajectory_id=str(trajectory_id),
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        prompt_length=1,
        loss_mask=loss_mask,
        token_level_rewards=token_level_rewards,
        end_of_response_position_mask=end_mask,
        multi_modal_inputs=multi_modal_inputs,
    )
    _validate_row(row)
    return row


def build_nimloth_verl_replay_row(
    *,
    trajectory_id: str,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    thought_token_ids: Sequence[int],
    latent_query_token_ids: Sequence[int],
    action_start_token_id: int,
    action_token_id: int,
    action_end_token_id: int,
    reward: float,
    multi_modal_inputs: dict[str, Any] | None,
) -> VerlReplayRow:
    """Serialize one Nimloth turn using VERL response and loss-mask semantics."""

    if prompt_input_ids.ndim != 1 or prompt_input_ids.numel() == 0:
        raise ValueError("Nimloth VERL prompt_input_ids must be nonempty and 1D")
    prompt_length = int(prompt_input_ids.numel())
    if prompt_attention_mask.shape != prompt_input_ids.shape:
        raise ValueError("Nimloth VERL prompt attention shape mismatch")
    if prompt_position_ids.ndim not in {1, 2} or int(
        prompt_position_ids.shape[-1]
    ) != prompt_length:
        raise ValueError("Nimloth VERL prompt position_ids shape mismatch")
    prompt_input_ids = prompt_input_ids.detach().cpu()
    prompt_attention_mask = prompt_attention_mask.detach().cpu()
    prompt_position_ids = prompt_position_ids.detach().cpu()
    thought_ids = [int(token) for token in thought_token_ids]
    query_ids = [int(token) for token in latent_query_token_ids]
    if not thought_ids:
        raise ValueError("Nimloth VERL response requires sampled thought tokens")
    if not query_ids:
        raise ValueError("Nimloth VERL response requires latent query tokens")
    response_ids = [
        *thought_ids,
        *query_ids,
        int(action_start_token_id),
        int(action_token_id),
        int(action_end_token_id),
    ]
    response_length = len(response_ids)
    input_ids = torch.cat(
        (
            prompt_input_ids.to(dtype=torch.long),
            torch.tensor(response_ids, dtype=torch.long),
        )
    )
    attention_mask = torch.cat(
        (
            prompt_attention_mask.to(dtype=torch.long),
            torch.ones(response_length, dtype=torch.long),
        )
    )
    response_loss_mask = torch.tensor(
        [
            *([1] * len(thought_ids)),
            *([0] * len(query_ids)),
            0,
            1,
            0,
        ],
        dtype=torch.long,
    )
    loss_mask = torch.cat(
        (torch.zeros(prompt_length, dtype=torch.long), response_loss_mask)
    )
    action_position = prompt_length + response_length - 2
    rewards = torch.zeros(input_ids.numel(), dtype=torch.float32)
    rewards[action_position] = float(reward)
    end_mask = torch.zeros(input_ids.numel(), dtype=torch.long)
    end_mask[action_position] = 1

    position_delta = torch.arange(
        1,
        response_length + 1,
        dtype=prompt_position_ids.dtype,
        device=prompt_position_ids.device,
    )
    if prompt_position_ids.ndim == 1:
        response_position_ids = prompt_position_ids[-1:] + position_delta
        position_ids = torch.cat((prompt_position_ids, response_position_ids))
    else:
        response_position_ids = prompt_position_ids[:, -1:] + position_delta
        position_ids = torch.cat(
            (prompt_position_ids, response_position_ids), dim=-1
        )
    row = VerlReplayRow(
        trajectory_id=str(trajectory_id),
        input_ids=input_ids.cpu(),
        attention_mask=attention_mask.cpu(),
        position_ids=position_ids.cpu(),
        prompt_length=prompt_length,
        loss_mask=loss_mask,
        token_level_rewards=rewards,
        end_of_response_position_mask=end_mask,
        multi_modal_inputs=multi_modal_inputs,
    )
    _validate_row(row)
    return row


def build_verl_replay_row_from_trajectory(
    processor,
    trajectory,
) -> VerlReplayRow:
    """Tokenize one complete stored trajectory using the live Qwen processor."""

    from nimloth.latent.extraction import (
        LatentActionTokens,
        latent_state_tokens,
        special_token_ids,
    )
    from nimloth.training.rl.rollout import (
        build_nimloth_policy_messages,
        multimodal_policy_messages,
        validate_rollout_trajectory,
    )
    from verl.models.transformers.qwen2_vl import get_rope_index

    validate_rollout_trajectory(trajectory)
    messages, images = build_nimloth_policy_messages(
        list(trajectory.image_paths),
        trajectory.system_prompt,
        list(trajectory.observation_texts),
        list(trajectory.assistant_responses),
        history_window=trajectory.num_steps,
    )
    messages, materialized_images = multimodal_policy_messages(messages, images)
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    model_inputs = processor(
        text=[rendered],
        images=materialized_images,
        return_tensors="pt",
        padding=False,
    )
    if "input_ids" not in model_inputs or "attention_mask" not in model_inputs:
        raise ValueError("Qwen processor omitted transcript ids or attention mask")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            "Nimloth VERL trajectory processor requires one unpadded transcript"
        )
    transcript_input_ids = input_ids[0].detach().cpu()
    transcript_attention_mask = attention_mask[0].detach().cpu()
    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    second_per_grid_ts = model_inputs.get("second_per_grid_ts")
    position_ids = get_rope_index(
        processor,
        input_ids=transcript_input_ids,
        image_grid_thw=(
            image_grid_thw.detach().cpu() if image_grid_thw is not None else None
        ),
        video_grid_thw=(
            video_grid_thw.detach().cpu() if video_grid_thw is not None else None
        ),
        second_per_grid_ts=(
            second_per_grid_ts.detach().cpu()
            if isinstance(second_per_grid_ts, torch.Tensor)
            else second_per_grid_ts
        ),
        attention_mask=transcript_attention_mask,
    )

    tokens = LatentActionTokens()
    token_id_map = special_token_ids(
        processor.tokenizer,
        tokens,
        latent_token_count=int(trajectory.latent_token_count),
    )
    query_ids = [
        token_id_map[name]
        for name in latent_state_tokens(int(trajectory.latent_token_count), tokens)
    ]
    action_token_ids = [
        token_id_map[tokens.action_tokens[int(action_index)]]
        for action_index in trajectory.action_indices
    ]
    for turn, assistant_response in enumerate(trajectory.assistant_responses):
        expected = [
            *[int(token) for token in trajectory.thought_token_ids[turn]],
            *query_ids,
            token_id_map[tokens.action_start],
            action_token_ids[turn],
            token_id_map[tokens.action_end],
        ]
        encoded = processor.tokenizer.encode(
            assistant_response, add_special_tokens=False
        )
        if [int(token) for token in encoded] != expected:
            raise ValueError(
                "stored assistant response tokens disagree with sampled thought and "
                f"deterministic scaffold at turn {turn}"
            )

    turn_rewards = [float(value) for value in trajectory.step_rewards]
    turn_rewards[-1] += float(trajectory.final_reward)
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Qwen tokenizer requires pad_token_id for VERL replay")
    multimodal_inputs: dict[str, Any] = {}
    for key, value in model_inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                f"Qwen multimodal processor output {key} must be a tensor"
            )
        multimodal_inputs[key] = value.detach().cpu()

    row = build_nimloth_verl_trajectory_replay_row(
        trajectory_id=str(trajectory.record_id),
        transcript_input_ids=transcript_input_ids,
        transcript_attention_mask=transcript_attention_mask,
        transcript_position_ids=position_ids,
        thought_token_ids_by_turn=trajectory.thought_token_ids,
        latent_query_token_ids=query_ids,
        action_start_token_id=token_id_map[tokens.action_start],
        action_token_ids=action_token_ids,
        action_end_token_id=token_id_map[tokens.action_end],
        turn_rewards=turn_rewards,
        pad_token_id=int(pad_token_id),
        multi_modal_inputs=multimodal_inputs,
    )
    return replace(
        row,
        policy_transcript=rendered,
        task_instruction=str(trajectory.task_instruction),
        observation_texts=tuple(str(value) for value in trajectory.observation_texts),
        assistant_responses=tuple(str(value) for value in trajectory.assistant_responses),
        image_paths=tuple(str(value) for value in trajectory.image_paths),
    )


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
        "policy_transcript": [row.policy_transcript for row in rows],
        "task_instruction": [row.task_instruction for row in rows],
        "observation_texts": [row.observation_texts for row in rows],
        "assistant_responses": [row.assistant_responses for row in rows],
        "image_paths": [row.image_paths for row in rows],
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


def finalize_verl_exact_replay_batch(
    replay,
    *,
    old_log_prob_output,
    reference_log_prob_output,
    values_output,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    """Attach immutable worker outputs and compute strict masked-GAE in place."""

    from vagen.trainer.ppo.ray_trainer import AdvantageEstimator, compute_advantage

    replay = replay.union(old_log_prob_output)
    replay = replay.union(reference_log_prob_output)
    replay = replay.union(values_output)
    response_shape = replay.batch["responses"].shape
    if len(response_shape) != 2:
        raise ValueError(f"VERL exact replay responses must be 2D, got {response_shape}")
    response_length = int(response_shape[1])
    response_mask = replay.batch["loss_mask"][:, -response_length:].bool()
    response_attention = replay.batch["attention_mask"][:, -response_length:].bool()
    expected_shape = tuple(response_shape)
    for key in ("old_log_probs", "ref_log_prob", "values"):
        if key not in replay.batch:
            raise ValueError(f"VERL exact replay worker output omitted {key}")
        tensor = replay.batch[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"VERL exact replay {key} shape mismatch: "
                f"expected={expected_shape}, got={tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor.masked_select(response_attention)).all()):
            raise ValueError(f"VERL exact replay {key} contains non-finite attended values")
    scores = replay.batch["multi_turn_token_level_rewards"][:, -response_length:]
    if bool((scores.ne(0) & ~response_mask).any()):
        raise ValueError("VERL exact replay rewards must be placed on policy tokens")
    replay.batch["token_level_scores"] = scores.clone()
    replay.batch["token_level_rewards"] = scores.clone()
    replay.meta_info["global_token_num"] = torch.sum(
        replay.batch["attention_mask"], dim=-1
    ).tolist()
    compute_advantage(
        replay,
        adv_estimator=AdvantageEstimator.MASKED_GAE,
        gamma=float(gamma),
        lam=float(lam),
    )
    if not bool(torch.isfinite(replay.batch["advantages"].masked_select(response_mask)).all()):
        raise ValueError("VERL exact replay produced non-finite policy advantages")
    if not bool(torch.isfinite(replay.batch["returns"].masked_select(response_mask)).all()):
        raise ValueError("VERL exact replay produced non-finite policy returns")
    if bool(replay.batch["returns"].masked_select(~response_mask).ne(0).any()):
        raise ValueError("VERL exact replay masked-GAE wrote returns outside loss mask")

    policy_tokens = int(response_mask.sum().item())
    old_policy = replay.batch["old_log_probs"].masked_select(response_mask)
    ref_policy = replay.batch["ref_log_prob"].masked_select(response_mask)
    value_policy = replay.batch["values"].masked_select(response_mask)
    audit = {
        "batch_size": int(response_shape[0]),
        "response_tokens": int(response_attention.sum().item()),
        "policy_tokens": policy_tokens,
        "reward_positions": int(scores.ne(0).sum().item()),
        "finite_old_policy_tokens": int(torch.isfinite(old_policy).sum().item()),
        "finite_ref_policy_tokens": int(torch.isfinite(ref_policy).sum().item()),
        "finite_value_policy_tokens": int(torch.isfinite(value_policy).sum().item()),
        "max_abs_old_ref_delta": float(
            (old_policy - ref_policy).abs().max().item()
        ),
    }
    if policy_tokens <= 0:
        raise ValueError("VERL exact replay contains no policy tokens")
    return replay, audit
