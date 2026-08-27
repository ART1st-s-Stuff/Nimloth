"""Explicit one-forward Qwen boundary for state-interface-v2 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from nimloth.backbone.base import BackboneBatch
from nimloth.backbone.qwen25vl.latent import _final_norm_module
from nimloth.latent import (
    ExtractionPositions,
    LatentActionTokens,
    find_all_latent_state_blocks,
)


@dataclass(frozen=True)
class QwenStateTrainingBatch:
    """Tensor batch plus the archived responses that define its real CoT state."""

    backbone_batch: BackboneBatch
    archived_assistant_responses: tuple[str, ...]
    response_sources: tuple[str, ...]


@dataclass(frozen=True)
class QwenStateTrainingOutput:
    """Student tensors captured from one and the same causal Qwen forward."""

    query_hidden: torch.Tensor
    action_logits: torch.Tensor


def require_archived_assistant_response(
    response: str | None,
    *,
    source: str,
) -> str:
    """Require real persisted response/CoT provenance; never synthesize a fallback."""

    if source != "archived" or not isinstance(response, str) or not response.strip():
        raise ValueError(
            "state training requires a non-empty archived assistant response; "
            "fixed or missing CoT is forbidden"
        )
    think_start = response.find("<think>")
    think_stop = response.find("</think>", think_start + len("<think>"))
    if think_start < 0 or think_stop < 0 or not response[
        think_start + len("<think>") : think_stop
    ].strip():
        raise ValueError("archived assistant response must contain its real non-empty CoT")
    if "<|action_start|>" not in response:
        raise ValueError(
            "archived assistant response must preserve its exact action boundary"
        )
    return response


def _find_last_subsequence(
    sequence: Sequence[int], query: Sequence[int]
) -> tuple[int, int]:
    values = [int(value) for value in sequence]
    needle = [int(value) for value in query]
    if not needle:
        raise ValueError("empty token query")
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start, start + len(needle)
    raise ValueError("token query is absent")


def exact_instruction_token_span(
    input_ids: Sequence[int] | torch.Tensor,
    *,
    tokenizer: Any,
    instruction: str,
) -> tuple[int, int]:
    """Locate an instruction using tokenization with its real field boundaries.

    A token overlapping the instruction/suffix boundary remains part of the
    span. A missing full bounded sequence fails closed rather than dropping it.
    """

    if not isinstance(instruction, str) or not instruction:
        raise ValueError("instruction must be a non-empty archived field")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim != 1:
            raise ValueError("input_ids must contain exactly one rendered row")
        rendered_ids = input_ids.detach().cpu().tolist()
    else:
        rendered_ids = list(input_ids)
    prefix = "Human Instruction: "
    suffixes = ("\nDecide your next action", "\nDecide your next action(s).")
    for suffix in suffixes:
        bounded = prefix + instruction + suffix
        encoded = tokenizer(
            bounded,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        bounded_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [tuple(value) for value in encoded["offset_mapping"]]
        if len(offsets) != len(bounded_ids):
            raise ValueError("instruction boundary tokenization is invalid")
        char_start = len(prefix)
        char_stop = char_start + len(instruction)
        selected = [
            index
            for index, (start, stop) in enumerate(offsets)
            if int(stop) > char_start and int(start) < char_stop
        ]
        if not selected or selected != list(range(selected[0], selected[-1] + 1)):
            raise ValueError("instruction boundary tokenization is invalid")
        try:
            bounded_start, _ = _find_last_subsequence(rendered_ids, bounded_ids)
        except ValueError:
            continue
        return bounded_start + selected[0], bounded_start + selected[-1] + 1
    raise ValueError(
        f"exact bounded instruction span is absent: {instruction!r}"
    )


def find_current_state_extraction_positions(
    input_ids: Sequence[int] | torch.Tensor,
    token_id_map: dict[str, int],
    *,
    expected_pair_count: int | None = None,
    latent_token_count: int = 16,
) -> ExtractionPositions:
    """Validate every structural pair and select the final current boundary."""

    if (
        expected_pair_count is not None
        and (
            isinstance(expected_pair_count, bool)
            or not isinstance(expected_pair_count, int)
            or expected_pair_count < 1
        )
    ):
        raise ValueError("expected structural pair count must be a positive integer")
    tokens = LatentActionTokens()
    latent_blocks = find_all_latent_state_blocks(
        input_ids,
        token_id_map,
        tokens,
        latent_token_count=latent_token_count,
    )
    ids = torch.as_tensor(input_ids, dtype=torch.long).flatten()
    latent_start_id = token_id_map.get(tokens.latent_state)
    if latent_start_id is None:
        raise ValueError("state training latent-start token is unavailable")
    latent_starts = torch.nonzero(
        ids == int(latent_start_id),
        as_tuple=False,
    ).flatten().tolist()
    if len(latent_starts) != len(latent_blocks):
        raise ValueError(
            "state training input contains a malformed latent block: "
            f"starts={len(latent_starts)}, complete={len(latent_blocks)}"
        )
    action_start_id = token_id_map.get(tokens.action_start)
    if action_start_id is None:
        raise ValueError("state training action boundary token is unavailable")
    action_boundaries = torch.nonzero(
        ids == int(action_start_id),
        as_tuple=False,
    ).flatten().tolist()
    if not latent_blocks or len(latent_blocks) != len(action_boundaries):
        raise ValueError(
            "state training structural pair count mismatch: "
            f"latent={len(latent_blocks)}, action={len(action_boundaries)}"
        )
    if expected_pair_count is not None and len(latent_blocks) != expected_pair_count:
        raise ValueError(
            "state training structural pair count mismatch: "
            f"expected={expected_pair_count}, actual={len(latent_blocks)}"
        )
    for turn_index, (latent_indices, boundary) in enumerate(
        zip(latent_blocks, action_boundaries, strict=True)
    ):
        if latent_indices[-1] + 1 != boundary:
            raise ValueError(
                f"state training turn {turn_index} K16 block is not adjacent "
                "to its action boundary"
            )
    current = tuple(latent_blocks[-1])
    return ExtractionPositions(
        latent_state_index=current[0],
        latent_state_indices=current,
        action_start_index=int(action_boundaries[-1]),
    )


def forward_qwen_state_training(
    model: torch.nn.Module,
    batch: QwenStateTrainingBatch,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    latent_token_count: int = 16,
) -> QwenStateTrainingOutput:
    """Return ordered query hidden and exact action logits from one Qwen call."""

    if int(latent_token_count) != 16:
        raise ValueError("state-interface-v2 requires exactly 16 latent queries")
    enc = dict(batch.backbone_batch.tensors)
    if "labels" in enc:
        raise ValueError("state training forward does not accept LM labels")
    input_ids = enc.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("state training input_ids must have shape (B,S)")
    batch_size = int(input_ids.shape[0])
    if len(batch.archived_assistant_responses) != batch_size:
        raise ValueError("state training requires one archived response per tensor row")
    if len(batch.response_sources) != batch_size:
        raise ValueError("state training requires one response source per tensor row")
    for response, source in zip(
        batch.archived_assistant_responses,
        batch.response_sources,
        strict=True,
    ):
        require_archived_assistant_response(response, source=source)
    tokens = LatentActionTokens()
    action_ids: list[int] = []
    for token in tokens.action_tokens:
        if token not in token_id_map:
            raise ValueError(f"missing action token id: {token}")
        action_ids.append(int(token_id_map[token]))
    if len(set(action_ids)) != 8:
        raise ValueError("eight action tokens must map to distinct token ids")

    query_positions: list[tuple[int, ...]] = []
    boundary_positions: list[int] = []
    for row in input_ids.detach().cpu():
        positions = find_current_state_extraction_positions(
            row,
            token_id_map,
            latent_token_count=latent_token_count,
        )
        if positions.latent_state_indices is None or len(positions.latent_state_indices) != 16:
            raise ValueError("state training row has no complete K16 query block")
        if positions.action_start_index is None:
            raise ValueError("state training row has no exact action boundary")
        if positions.action_start_index != positions.latent_state_indices[-1] + 1:
            raise ValueError("K16 query block must immediately precede action_start")
        query_positions.append(tuple(positions.latent_state_indices))
        boundary_positions.append(int(positions.action_start_index))

    # Qwen accepts selected sequence positions before the LM head. A Python list
    # remains valid with Accelerate device maps (unlike a CPU index tensor).
    kept_positions = sorted(set(boundary_positions))
    model_inputs = {
        key: value.to(device, non_blocking=True) for key, value in enc.items()
    }
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = _final_norm_module(model).register_forward_hook(hook)
    try:
        output = model(
            **model_inputs,
            logits_to_keep=kept_positions,
            output_hidden_states=False,
            return_dict=True,
        )
    finally:
        handle.remove()
    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("Qwen final norm hook did not capture state-training hidden")
    logits = getattr(output, "logits", None)
    if logits is None or logits.ndim != 3 or logits.shape[1] != len(kept_positions):
        raise RuntimeError("Qwen did not return logits for exact action boundaries")

    hidden_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    kept_index = {position: index for index, position in enumerate(kept_positions)}
    action_index = torch.tensor(action_ids, device=logits.device, dtype=torch.long)
    for row, (queries, boundary) in enumerate(
        zip(query_positions, boundary_positions, strict=True)
    ):
        query_index = torch.tensor(queries, device=hidden.device, dtype=torch.long)
        hidden_rows.append(hidden[row].index_select(0, query_index))
        action_rows.append(
            logits[row, kept_index[boundary]].index_select(0, action_index)
        )
    query_hidden = torch.stack(hidden_rows, dim=0)
    action_logits = torch.stack(action_rows, dim=0).float()
    if query_hidden.shape[:2] != (input_ids.shape[0], 16):
        raise RuntimeError("Qwen state-training query hidden has an invalid shape")
    if action_logits.shape != (input_ids.shape[0], 8):
        raise RuntimeError("Qwen state-training action logits have an invalid shape")
    if not torch.isfinite(query_hidden).all() or not torch.isfinite(action_logits).all():
        raise RuntimeError("Qwen state-training output contains non-finite values")
    return QwenStateTrainingOutput(
        query_hidden=query_hidden,
        action_logits=action_logits,
    )


__all__ = [
    "QwenStateTrainingBatch",
    "QwenStateTrainingOutput",
    "exact_instruction_token_span",
    "find_current_state_extraction_positions",
    "forward_qwen_state_training",
    "require_archived_assistant_response",
]
