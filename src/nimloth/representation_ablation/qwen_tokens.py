"""Qwen token extraction helpers for representation ablations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from nimloth.latent.extraction import LatentActionTokens, find_all_latent_state_indices


def repeated_latent_marker(num_tokens: int, *, marker: str = "<|latent_state|>") -> str:
    """Return ``num_tokens`` adjacent latent markers."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    return marker * num_tokens


def expand_latent_markers_in_text(text: str, num_tokens: int, *, marker: str = "<|latent_state|>") -> str:
    """Replace each latent marker in a text string with an adjacent marker run."""

    if num_tokens == 1:
        return text
    if marker not in text:
        return text
    return text.replace(marker, repeated_latent_marker(num_tokens, marker=marker))


def expand_latent_markers_in_messages(
    messages: Sequence[dict[str, Any]],
    num_tokens: int,
    *,
    marker: str = "<|latent_state|>",
) -> list[dict[str, Any]]:
    """Deep-copy messages and expand string text fields containing latent markers.

    Supports both plain string content and Qwen processor list content with text
    parts. Image parts are copied unchanged.
    """

    out = deepcopy(list(messages))
    if num_tokens == 1:
        return out
    for msg in out:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = expand_latent_markers_in_text(content, num_tokens, marker=marker)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = expand_latent_markers_in_text(part["text"], num_tokens, marker=marker)
    return out


def last_latent_token_run_indices(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    *,
    num_tokens: int,
    tokens: LatentActionTokens = LatentActionTokens(),
) -> list[int]:
    """Return the last contiguous run of ``num_tokens`` latent token indices."""

    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    indices = find_all_latent_state_indices(input_ids, token_ids, tokens)
    if num_tokens == 1:
        return [indices[-1]]
    for end in range(len(indices), num_tokens - 1, -1):
        candidate = indices[end - num_tokens : end]
        if all(candidate[i] + 1 == candidate[i + 1] for i in range(len(candidate) - 1)):
            return candidate
    raise ValueError(f"Expected a contiguous run of {num_tokens} {tokens.latent_state} tokens, found indices {indices}")


def extract_latent_token_set(
    hidden_states: Tensor,
    input_ids: Tensor,
    token_ids: Mapping[str, int],
    *,
    num_tokens: int,
) -> Tensor:
    """Extract final hidden states for the last latent-token run.

    Args:
        hidden_states: ``(B, S, D)`` or ``(S, D)`` hidden states.
        input_ids: ``(B, S)`` or ``(S,)`` token ids.
        token_ids: mapping from special token string to tokenizer id.
        num_tokens: latent token count K.
    Returns:
        ``(B, K, D)`` for batched input, or ``(K, D)`` for unbatched input.
    """

    single = hidden_states.ndim == 2
    if single:
        hidden_batch = hidden_states.unsqueeze(0)
        ids_batch = input_ids.unsqueeze(0) if isinstance(input_ids, Tensor) and input_ids.ndim == 1 else torch.as_tensor(input_ids).unsqueeze(0)
    else:
        hidden_batch = hidden_states
        ids_batch = input_ids
    if hidden_batch.ndim != 3:
        raise ValueError(f"hidden_states must have shape (B, S, D) or (S, D), got {tuple(hidden_states.shape)}")
    if not isinstance(ids_batch, Tensor) or ids_batch.ndim != 2:
        raise ValueError(f"input_ids must have shape (B, S) or (S,), got {tuple(getattr(input_ids, 'shape', ())) }")
    rows: list[Tensor] = []
    for row_hidden, row_ids in zip(hidden_batch, ids_batch, strict=True):
        indices = last_latent_token_run_indices(row_ids, token_ids, num_tokens=num_tokens)
        rows.append(row_hidden[torch.tensor(indices, device=row_hidden.device, dtype=torch.long)])
    stacked = torch.stack(rows, dim=0)
    return stacked[0] if single else stacked
