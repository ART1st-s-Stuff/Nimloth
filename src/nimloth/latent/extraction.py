from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentActionTokens:
    """Special tokens used by Nimloth latent-action prompting."""

    latent_state: str = "<|latent_state|>"
    action_start: str = "<|action_start|>"
    action_end: str = "<|action_end|>"
    action_tokens: tuple[str, ...] = tuple(f"<|action_({idx})|>" for idx in range(8))

    @property
    def all_special_tokens(self) -> tuple[str, ...]:
        return (self.latent_state, self.action_start, self.action_end, *self.action_tokens)


@dataclass(frozen=True)
class ExtractionPositions:
    """Token positions needed to extract state and action-prior features."""

    latent_state_index: int
    action_start_index: int | None = None
    first_action_index: int | None = None
    latent_state_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ActionPrior:
    """Action logits over the configured action token set."""

    token_ids: Tensor
    logits: Tensor
    log_probs: Tensor
    probs: Tensor
    token_to_log_prob: Mapping[str, float]


def _validate_latent_token_count(latent_token_count: int) -> int:
    count = int(latent_token_count)
    if count < 1:
        raise ValueError(f"latent_token_count must be >= 1, got {latent_token_count!r}")
    return count


def latent_state_tokens(
    latent_token_count: int = 1,
    tokens: LatentActionTokens = LatentActionTokens(),
) -> tuple[str, ...]:
    """Return the configured latent query token names.

    Slot 0 keeps the legacy token ``<|latent_state|>`` for backward
    compatibility.  Additional slots use stable, distinct token names so each
    query slot can learn a different role.
    """

    count = _validate_latent_token_count(latent_token_count)
    if count == 1:
        return (tokens.latent_state,)
    return (tokens.latent_state, *(f"<|latent_state_{idx}|>" for idx in range(1, count)))


def latent_state_block(
    latent_token_count: int = 1,
    tokens: LatentActionTokens = LatentActionTokens(),
) -> str:
    """Return the serialized latent query token block for prompts."""

    return "".join(latent_state_tokens(latent_token_count, tokens))


def all_special_tokens_for_latent_count(
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> tuple[str, ...]:
    """Return all Nimloth special tokens for the configured latent slots."""

    return (
        *latent_state_tokens(latent_token_count, tokens),
        tokens.action_start,
        tokens.action_end,
        *tokens.action_tokens,
    )


_LATENT_BLOCK_RE = re.compile(r"<\|latent_state\|>(?:(?:<\|latent_state\|>)|(?:<\|latent_state_\d+\|>))*")


def normalize_latent_state_blocks(
    text: str,
    latent_token_count: int = 1,
    tokens: LatentActionTokens = LatentActionTokens(),
) -> str:
    """Normalize every latent block in rendered text to the configured length.

    Existing SFT data contains one ``<|latent_state|>`` per assistant turn.  For
    multi-query training we expand that block at render time, so JSONL files do
    not need to be rewritten.  The function is idempotent for a fixed count.
    """

    _validate_latent_token_count(latent_token_count)
    if tokens.latent_state not in text:
        return text
    return _LATENT_BLOCK_RE.sub(latent_state_block(latent_token_count, tokens), text)


def _as_1d_input_ids(input_ids: Tensor | Sequence[int]) -> Tensor:
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    if ids.ndim == 2:
        if ids.shape[0] != 1:
            raise ValueError("Only a single sequence is supported; pass one row at a time.")
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D or batch size 1, got shape {tuple(ids.shape)}")
    return ids


def _single_token_id(tokenizer, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_id:
        raise ValueError(f"Token {token!r} is not in the tokenizer vocabulary.")

    encoded = tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1 or encoded[0] != token_id:
        raise ValueError(
            f"Token {token!r} must encode as exactly one token. "
            "Add it as a special token before extraction."
        )
    return int(token_id)


def special_token_ids(
    tokenizer,
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> dict[str, int]:
    """Return ids for every Nimloth special token, failing on missing/split tokens."""

    return {
        token: _single_token_id(tokenizer, token)
        for token in all_special_tokens_for_latent_count(tokens, latent_token_count=latent_token_count)
    }


def add_special_tokens(
    tokenizer,
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> int:
    """Add Nimloth special tokens to a HuggingFace tokenizer.

    Returns the number of newly added tokens. If this is non-zero, the caller must resize
    the model embeddings with `model.resize_token_embeddings(len(tokenizer))`.
    """

    return int(
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(
                    all_special_tokens_for_latent_count(tokens, latent_token_count=latent_token_count)
                )
            }
        )
    )


def initialize_extra_latent_token_embeddings(
    model,
    token_ids: Mapping[str, int],
    *,
    latent_token_count: int = 1,
    tokens: LatentActionTokens = LatentActionTokens(),
) -> None:
    """Initialize extra latent query token embeddings from slot 0.

    This avoids giving ``<|latent_state_1|>`` ... random initial embeddings that
    are unrelated to the legacy ``<|latent_state|>`` token.  It is safe to call
    after every ``resize_token_embeddings``; for ``latent_token_count=1`` it is a
    no-op.
    """

    count = _validate_latent_token_count(latent_token_count)
    if count == 1:
        return

    latent_tokens = latent_state_tokens(count, tokens)
    source_id = int(token_ids[latent_tokens[0]])
    dest_ids = [int(token_ids[token]) for token in latent_tokens[1:]]

    def _copy_rows(weight: torch.Tensor) -> None:
        with torch.no_grad():
            source = weight[source_id].detach().clone()
            for dest_id in dest_ids:
                weight[dest_id].copy_(source.to(dtype=weight.dtype, device=weight.device))

    input_embeddings = model.get_input_embeddings()
    input_weight = getattr(input_embeddings, "weight", None) if input_embeddings is not None else None
    if input_weight is not None:
        _copy_rows(input_weight)

    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
    if output_weight is not None:
        if input_weight is None or output_weight.data_ptr() != input_weight.data_ptr():
            _copy_rows(output_weight)


def find_all_latent_state_blocks(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> list[list[int]]:
    """Return token-index blocks for every configured latent query group."""

    ids = _as_1d_input_ids(input_ids)
    latent_names = latent_state_tokens(latent_token_count, tokens)
    latent_ids = [int(token_ids[name]) for name in latent_names]
    start_id = latent_ids[0]
    starts = torch.nonzero(ids == start_id, as_tuple=False).flatten().tolist()
    if not starts:
        raise ValueError(f"Expected at least one {tokens.latent_state} token, found 0.")

    blocks: list[list[int]] = []
    malformed_starts: list[int] = []
    for start in starts:
        end = int(start) + len(latent_ids)
        if end > ids.numel():
            malformed_starts.append(int(start))
            continue
        if all(int(ids[int(start) + offset].item()) == expected for offset, expected in enumerate(latent_ids)):
            blocks.append(list(range(int(start), end)))
        else:
            malformed_starts.append(int(start))

    if not blocks:
        raise ValueError(
            f"Expected at least one contiguous latent block of length {len(latent_ids)}, "
            f"found malformed starts {malformed_starts}."
        )
    return blocks


def find_last_latent_state_block(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> list[int]:
    """Return indices of the final configured latent query block."""

    return find_all_latent_state_blocks(
        input_ids,
        token_ids,
        tokens,
        latent_token_count=latent_token_count,
    )[-1]


def find_extraction_positions(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    tokens: LatentActionTokens = LatentActionTokens(),
    *,
    latent_token_count: int = 1,
) -> ExtractionPositions:
    """Find the per-step latent-state and action-prior positions in a tokenized sequence.

    The latent state is read at the configured latent query token block. The
    action logits for the token immediately after ``<|action_start|>`` are read
    from the ``<|action_start|>`` position, because causal-LM logits at position
    i predict token i+1.
    """

    ids = _as_1d_input_ids(input_ids)
    action_start_id = token_ids[tokens.action_start]
    action_end_id = token_ids[tokens.action_end]

    latent_blocks = find_all_latent_state_blocks(
        ids,
        token_ids,
        tokens,
        latent_token_count=latent_token_count,
    )
    if len(latent_blocks) != 1:
        raise ValueError(f"Expected exactly one latent-state block, found {len(latent_blocks)}.")
    latent_indices = tuple(latent_blocks[0])
    latent_index = int(latent_indices[0])

    action_start_matches = torch.nonzero(ids == action_start_id, as_tuple=False).flatten()
    if action_start_matches.numel() == 0:
        return ExtractionPositions(latent_state_index=latent_index, latent_state_indices=latent_indices)
    if action_start_matches.numel() != 1:
        raise ValueError(f"Expected at most one {tokens.action_start} token, found {action_start_matches.numel()}.")

    action_start_index = int(action_start_matches.item())
    if action_start_index + 1 >= ids.numel():
        raise ValueError(f"{tokens.action_start} cannot be the final token when extracting an action prior.")
    first_action_index = action_start_index + 1
    if int(ids[first_action_index].item()) == action_end_id:
        raise ValueError(f"No action token appears between {tokens.action_start} and {tokens.action_end}.")

    return ExtractionPositions(
        latent_state_index=latent_index,
        action_start_index=action_start_index,
        first_action_index=first_action_index,
        latent_state_indices=latent_indices,
    )


def find_last_latent_state_index(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    tokens: LatentActionTokens = LatentActionTokens(),
) -> int:
    """Return the index of the last legacy ``<|latent_state|>`` token."""

    indices = find_all_latent_state_indices(input_ids, token_ids, tokens)
    return indices[-1]


def find_all_latent_state_indices(
    input_ids: Tensor | Sequence[int],
    token_ids: Mapping[str, int],
    tokens: LatentActionTokens = LatentActionTokens(),
) -> list[int]:
    """Return indices of every legacy ``<|latent_state|>`` token in order."""

    ids = _as_1d_input_ids(input_ids)
    latent_id = token_ids[tokens.latent_state]
    latent_matches = torch.nonzero(ids == latent_id, as_tuple=False).flatten()
    if latent_matches.numel() == 0:
        raise ValueError(f"Expected at least one {tokens.latent_state} token, found 0.")
    return [int(i.item()) for i in latent_matches]


def last_hidden_state(model_output) -> Tensor:
    """Return final-layer hidden states from a HuggingFace-style model output."""

    hidden_states = getattr(model_output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]

    last = getattr(model_output, "last_hidden_state", None)
    if last is not None:
        return last

    raise ValueError("Model output must include hidden_states or last_hidden_state.")


def _select_sequence_row(tensor: Tensor) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError(f"Expected sequence tensor with at least 2 dims, got {tuple(tensor.shape)}")
    if tensor.ndim == 2:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("Only batch size 1 is supported; extract one sequence at a time.")
    return tensor[0]


def extract_latent_state(hidden_states: Tensor, latent_state_index: int) -> Tensor:
    """Extract the final hidden state at one latent query token."""

    sequence_hidden = _select_sequence_row(hidden_states)
    return sequence_hidden[latent_state_index]


def extract_latent_state_block(hidden_states: Tensor, latent_state_indices: Sequence[int]) -> Tensor:
    """Extract final hidden states at a configured latent query token block."""

    sequence_hidden = _select_sequence_row(hidden_states)
    positions = torch.as_tensor(list(latent_state_indices), dtype=torch.long, device=sequence_hidden.device)
    return sequence_hidden.index_select(0, positions)


def extract_action_prior(
    logits: Tensor,
    hidden_states: Tensor,
    action_start_index: int,
    action_token_ids: Sequence[int] | Tensor,
    action_tokens: Sequence[str] | None = None,
    first_action_index: int | None = None,
) -> ActionPrior:
    """Extract action-token logits for the position after ``<|action_start|>``.

    In a causal language model, logits at `action_start_index` predict the next
    token. Therefore these logits are the action logits for `first_action_index`.
    `hidden_states` and `first_action_index` are accepted for API compatibility
    with older call sites; they are not used to define the action logits.
    """

    del hidden_states, first_action_index
    sequence_logits = _select_sequence_row(logits)
    token_ids = torch.as_tensor(action_token_ids, dtype=torch.long, device=sequence_logits.device)
    prior_logits = sequence_logits[action_start_index, token_ids]
    log_probs = torch.log_softmax(prior_logits, dim=-1)
    probs = torch.softmax(prior_logits, dim=-1)

    names = tuple(action_tokens) if action_tokens is not None else tuple(str(int(i)) for i in token_ids.detach().cpu())
    token_to_log_prob = {
        name: float(value)
        for name, value in zip(names, log_probs.detach().cpu().tolist(), strict=True)
    }

    return ActionPrior(
        token_ids=token_ids.detach().clone(),
        logits=prior_logits,
        log_probs=log_probs,
        probs=probs,
        token_to_log_prob=token_to_log_prob,
    )


class LatentActionExtractor:
    """Small wrapper that runs a causal LM and extracts Nimloth per-step features."""

    def __init__(
        self,
        tokenizer,
        tokens: LatentActionTokens = LatentActionTokens(),
        *,
        latent_token_count: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.latent_token_count = _validate_latent_token_count(latent_token_count)
        self.token_ids = special_token_ids(tokenizer, tokens, latent_token_count=self.latent_token_count)
        self.action_token_ids = tuple(self.token_ids[token] for token in tokens.action_tokens)

    @classmethod
    def with_added_special_tokens(
        cls,
        tokenizer,
        tokens: LatentActionTokens = LatentActionTokens(),
        *,
        latent_token_count: int = 1,
    ) -> "LatentActionExtractor":
        add_special_tokens(tokenizer, tokens, latent_token_count=latent_token_count)
        return cls(tokenizer=tokenizer, tokens=tokens, latent_token_count=latent_token_count)

    def positions(self, input_ids: Tensor | Sequence[int]) -> ExtractionPositions:
        return find_extraction_positions(
            input_ids=input_ids,
            token_ids=self.token_ids,
            tokens=self.tokens,
            latent_token_count=self.latent_token_count,
        )

    @torch.no_grad()
    def extract_from_model(self, model, **model_inputs) -> tuple[Tensor, ActionPrior | None, ExtractionPositions]:
        """Run `model` and extract latent state plus optional action prior.

        `model_inputs` must include `input_ids`. The method forces hidden-state output
        while preserving any other caller-provided model inputs such as attention masks
        or image tensors.
        """

        if "input_ids" not in model_inputs:
            raise ValueError("model_inputs must include input_ids.")

        positions = self.positions(model_inputs["input_ids"])
        output = model(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = last_hidden_state(output)
        if self.latent_token_count == 1:
            latent_state = extract_latent_state(hidden, positions.latent_state_index)
        else:
            if positions.latent_state_indices is None:
                raise ValueError("latent_state_indices missing for multi-token extraction")
            latent_state = extract_latent_state_block(hidden, positions.latent_state_indices)

        action_prior = None
        if positions.action_start_index is not None:
            logits = getattr(output, "logits", None)
            if logits is None:
                raise ValueError("Model output must include logits to extract action prior.")
            action_prior = extract_action_prior(
                logits=logits,
                hidden_states=hidden,
                action_start_index=positions.action_start_index,
                action_token_ids=self.action_token_ids,
                action_tokens=self.tokens.action_tokens,
                first_action_index=positions.first_action_index,
            )

        return latent_state, action_prior, positions
