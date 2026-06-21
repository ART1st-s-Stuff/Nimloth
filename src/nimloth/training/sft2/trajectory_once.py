"""Full-trajectory single forward for SFT2 packed mode (strict legacy equivalence)."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from nimloth.latent import extract_latent_state, find_all_latent_state_indices, find_last_latent_state_index
from nimloth.latent.extraction import LatentActionTokens
from nimloth.training.common.qwen_batch import (
    _message_cache_key,
    _offset_cache,
    _template_cache,
    assistant_char_spans,
    encode_qwen_item,
)
from nimloth.training.sft2.qwen_latent import _capture_last_hidden, reset_model_rope_state
from nimloth.training.sft2.trajectory_forward import _batch_enc
from nimloth.wm.collate import prefix_messages_with_images
from nimloth.wm.dataset import TransitionSample


@dataclass(frozen=True)
class TrajectoryOnceForward:
    current_latents: torch.Tensor
    next_latents: torch.Tensor | None
    lm_loss: torch.Tensor | None
    latent_indices: list[int]
    num_steps: int


def supervised_token_count(labels: torch.Tensor) -> int:
    return int((labels != -100).sum().item())


def labels_for_trajectory_steps(
    full_input_ids: torch.Tensor,
    full_text: str,
    steps: list[TransitionSample],
    processor: AutoProcessor,
    max_length: int,
) -> torch.Tensor:
    """Build full-sequence labels supervising each step's last-assistant span only."""

    labels = full_input_ids.clone()
    labels[:] = -100
    offset_rows = _offset_cache(processor).offsets(full_text, max_length)
    usable = min(labels.shape[0], len(offset_rows))
    for sample in steps:
        spans = assistant_char_spans(prefix_messages_with_images(sample), processor)
        for tok_idx in range(usable):
            start, end = offset_rows[tok_idx]
            if end <= start:
                continue
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                labels[tok_idx] = full_input_ids[tok_idx]
    return labels


def ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Token-level CE mean over valid (non -100) label positions."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid = shift_labels != -100
    if not bool(valid.any()):
        return logits.sum() * 0.0
    flat_logits = shift_logits[valid]
    flat_labels = shift_labels[valid]
    return F.cross_entropy(flat_logits, flat_labels)


def legacy_batch_ce_loss(
    per_step_labels: list[torch.Tensor],
    per_step_logits: list[torch.Tensor],
) -> torch.Tensor:
    """Match ``build_qwen_batch`` global mean: sum of token CE / total supervised tokens."""

    total = torch.zeros((), device=per_step_logits[0].device)
    token_count = 0
    for labels, logits in zip(per_step_labels, per_step_logits, strict=True):
        if logits.ndim == 3:
            logits = logits[0]
        if labels.ndim == 2:
            labels = labels[0]
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = labels[1:].contiguous()
        valid = shift_labels != -100
        n = int(valid.sum().item())
        if n == 0:
            continue
        flat_logits = shift_logits[valid]
        flat_labels = shift_labels[valid]
        total = total + F.cross_entropy(flat_logits, flat_labels, reduction="sum")
        token_count += n
    if token_count == 0:
        return per_step_logits[0].sum() * 0.0
    return total / token_count


def assert_packed_steps(steps: list[TransitionSample]) -> None:
    record_id = steps[0].record_id
    for index, sample in enumerate(steps):
        if sample.record_id != record_id:
            raise ValueError("trajectory steps must share record_id")
        if sample.step_index != index:
            raise ValueError(
                f"trajectory steps must be contiguous from 0; expected {index}, got {sample.step_index}"
            )


def _render_messages(processor: AutoProcessor, messages: list[dict]) -> str:
    cache = _template_cache(processor)
    cache_key = _message_cache_key(messages)
    return cache.render(cache_key, False)


def _input_ids_list(input_ids: torch.Tensor | list[int]) -> list[int]:
    """Normalize processor ``input_ids`` to a flat 1D token list."""

    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            raise ValueError("input_ids list must be 1D")
        return [int(x) for x in input_ids]
    flat = input_ids.reshape(-1)
    return flat.tolist()


def find_latent_index_in_last_assistant_span(
    input_ids: torch.Tensor | list[int],
    prefix_messages: list[dict],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    max_length: int,
) -> int:
    """Locate the single ``<|latent_state|>`` token inside the last assistant span."""

    tokens = LatentActionTokens()
    latent_id = token_id_map[tokens.latent_state]
    spans = assistant_char_spans(prefix_messages, processor)
    if not spans:
        raise ValueError("no assistant span for latent lookup")
    span_start, span_end = spans[-1]
    text = _render_messages(processor, prefix_messages)
    offset_rows = _offset_cache(processor).offsets(text, max_length)
    ids = _input_ids_list(input_ids)
    matches: list[int] = []
    for tok_idx, (start, end) in enumerate(offset_rows):
        if tok_idx >= len(ids):
            break
        if end <= start:
            continue
        if start < span_end and end > span_start and ids[tok_idx] == latent_id:
            matches.append(tok_idx)
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        # Qwen-VL vision tokens may break char-offset alignment; match legacy find_last.
        return find_last_latent_state_index(input_ids, token_id_map, tokens)
    raise ValueError(
        "expected exactly one <|latent_state|> in last assistant span, "
        f"found {len(matches)} at {matches}"
    )


def verify_prefix_tokenization(
    steps: list[TransitionSample],
    full_enc: dict[str, torch.Tensor],
    processor: AutoProcessor,
    max_length: int,
    *,
    full_text: str | None = None,
    token_id_map: dict[str, int] | None = None,
) -> None:
    """Ensure each step's legacy prefix encoding matches the full trajectory prefix."""

    full_ids = _input_ids_list(full_enc["input_ids"])
    rendered_full = full_text if full_text is not None else _render_messages(
        processor, prefix_messages_with_images(steps[-1])
    )
    for sample in steps:
        prefix_messages = prefix_messages_with_images(sample)
        prefix_enc = encode_qwen_item(prefix_messages, processor, max_length, include_labels=False)
        prefix_ids = _input_ids_list(prefix_enc["input_ids"])
        rendered_prefix = _render_messages(processor, prefix_messages)
        if not rendered_full.startswith(rendered_prefix):
            raise ValueError(
                f"record {sample.record_id!r} step {sample.step_index}: chat template text is not a "
                "prefix of full trajectory text"
            )
        if full_ids[: len(prefix_ids)] != prefix_ids:
            mismatch = next(
                (idx for idx, (left, right) in enumerate(zip(full_ids, prefix_ids, strict=False)) if left != right),
                len(prefix_ids),
            )
            raise ValueError(
                f"record {sample.record_id!r} step {sample.step_index}: prefix tokenization is not a "
                f"prefix of full trajectory encoding (first mismatch at token {mismatch})"
            )
        if token_id_map is None:
            continue
        find_latent_index_in_last_assistant_span(
            prefix_enc["input_ids"],
            prefix_messages,
            processor,
            token_id_map,
            max_length,
        )


def find_step_latent_indices(
    steps: list[TransitionSample],
    full_enc: dict[str, torch.Tensor],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    max_length: int,
) -> list[int]:
    """Return each step's latent index in the full trajectory encoding.

    Requires ``verify_prefix_tokenization`` to have passed: each legacy prefix
    encoding must be an exact token prefix of the full trajectory encoding.
    """

    full_ids = _input_ids_list(full_enc["input_ids"])
    indices: list[int] = []
    for sample in steps:
        prefix_messages = prefix_messages_with_images(sample)
        prefix_enc = encode_qwen_item(prefix_messages, processor, max_length, include_labels=False)
        prefix_ids = _input_ids_list(prefix_enc["input_ids"])
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"record {sample.record_id!r} step {sample.step_index}: prefix tokenization is not a "
                "prefix of full trajectory encoding"
            )
        indices.append(
            find_latent_index_in_last_assistant_span(
                prefix_enc["input_ids"],
                prefix_messages,
                processor,
                token_id_map,
                max_length,
            )
        )
    return indices


def find_step_latent_indices_in_full(
    full_input_ids: torch.Tensor,
    full_text: str,
    steps: list[TransitionSample],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    max_length: int,
) -> list[int]:
    """Locate each step's <|latent_state|> token via assistant char spans (CPU/fake tests)."""

    tokens = LatentActionTokens()
    latent_id = token_id_map[tokens.latent_state]
    offset_rows = _offset_cache(processor).offsets(full_text, max_length)
    ids = full_input_ids.tolist()
    indices: list[int] = []
    for sample in steps:
        spans = assistant_char_spans(prefix_messages_with_images(sample), processor)
        if not spans:
            raise ValueError(
                f"record {sample.record_id!r} step {sample.step_index}: no assistant span for latent lookup"
            )
        span_start, span_end = spans[-1]
        found: int | None = None
        for tok_idx, (start, end) in enumerate(offset_rows):
            if tok_idx >= len(ids):
                break
            if end <= start:
                continue
            if start < span_end and end > span_start and ids[tok_idx] == latent_id:
                found = tok_idx
        if found is None:
            raise ValueError(
                f"record {sample.record_id!r} step {sample.step_index}: no <|latent_state|> in full trajectory"
            )
        indices.append(found)
    return indices


def encode_full_trajectory(
    steps: list[TransitionSample],
    processor: AutoProcessor,
    max_length: int,
    *,
    token_id_map: dict[str, int] | None = None,
) -> tuple[dict[str, torch.Tensor], str]:
    if not steps:
        raise ValueError("encode_full_trajectory requires at least one step")
    assert_packed_steps(steps)
    full_messages = prefix_messages_with_images(steps[-1])
    cache = _template_cache(processor)
    cache_key = _message_cache_key(full_messages)
    full_text = cache.render(cache_key, False)
    enc = encode_qwen_item(full_messages, processor, max_length, include_labels=False)
    verify_prefix_tokenization(
        steps, enc, processor, max_length, full_text=full_text, token_id_map=token_id_map
    )
    enc["labels"] = labels_for_trajectory_steps(enc["input_ids"], full_text, steps, processor, max_length)
    return enc, full_text


def _extract_latents_at_indices(
    hidden_row: torch.Tensor,
    indices: list[int],
) -> torch.Tensor:
    return torch.stack([extract_latent_state(hidden_row.unsqueeze(0), pos) for pos in indices], dim=0)


def forward_trajectory_once(
    model,
    steps: list[TransitionSample],
    processor: AutoProcessor,
    token_id_map: dict[str, int],
    device: torch.device,
    *,
    max_length: int,
    vision_ema=None,
    full_enc: dict[str, torch.Tensor] | None = None,
) -> TrajectoryOnceForward:
    """One Qwen forward over the full trajectory; extract per-step latents + CE."""

    if full_enc is None:
        enc, full_text = encode_full_trajectory(
            steps, processor, max_length, token_id_map=token_id_map
        )
    else:
        enc = full_enc
        full_text = _render_messages(processor, prefix_messages_with_images(steps[-1]))
        verify_prefix_tokenization(
            steps, enc, processor, max_length, full_text=full_text, token_id_map=token_id_map
        )

    batch = _batch_enc(enc)
    model_inputs = {k: v.to(device) for k, v in batch.items()}
    reset_model_rope_state(model)
    hidden, output = _capture_last_hidden(model, model_inputs)
    logits = output.logits
    latent_indices = find_step_latent_indices(steps, enc, processor, token_id_map, max_length)
    all_latent_indices = find_all_latent_state_indices(
        enc["input_ids"], token_id_map, LatentActionTokens()
    )
    if len(all_latent_indices) < len(steps):
        raise ValueError(
            f"expected {len(steps)} latent tokens, found {len(all_latent_indices)} in full trajectory"
        )

    current_latents = _extract_latents_at_indices(hidden[0], latent_indices)

    next_latents: torch.Tensor | None = None
    if len(latent_indices) > 1:
        ema_ctx = vision_ema.use_ema_weights(model) if vision_ema is not None else contextlib.nullcontext()
        with torch.no_grad(), ema_ctx:
            if vision_ema is not None:
                target_hidden, _ = _capture_last_hidden(model, model_inputs)
                target_row = target_hidden[0]
            else:
                target_row = hidden[0]
            next_latents = _extract_latents_at_indices(target_row, latent_indices[1:])

    labels = enc["labels"].to(device)
    lm_loss = ce_loss_from_logits(logits[0], labels)

    return TrajectoryOnceForward(
        current_latents=current_latents,
        next_latents=next_latents,
        lm_loss=lm_loss,
        latent_indices=latent_indices,
        num_steps=len(steps),
    )
