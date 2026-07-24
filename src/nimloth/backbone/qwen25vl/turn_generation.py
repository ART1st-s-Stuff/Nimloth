"""Single-request turn generation constraints shared by vLLM rollout code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


TURN_RESPONSE_EXTRA_ARG = "nimloth_turn_response"


@dataclass(frozen=True)
class TurnGenerationSpec:
    """Token protocol for one reasoning-plus-action continuation."""

    close_token_ids: tuple[int, ...]
    injected_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_end_token_id: int
    protocol_token_ids: tuple[int, ...]
    max_reasoning_tokens: int

    def __post_init__(self) -> None:
        if not self.close_token_ids:
            raise ValueError("turn generation requires closing token ids")
        if not self.injected_token_ids:
            raise ValueError("turn generation requires injected prefix token ids")
        if not self.action_token_ids or len(set(self.action_token_ids)) != len(
            self.action_token_ids
        ):
            raise ValueError("turn generation requires unique action token ids")
        if self.max_reasoning_tokens < 1:
            raise ValueError("max_reasoning_tokens must be positive")

    def to_extra_args(self) -> dict[str, Any]:
        return {
            TURN_RESPONSE_EXTRA_ARG: {
                "close_token_ids": list(self.close_token_ids),
                "injected_token_ids": list(self.injected_token_ids),
                "action_token_ids": list(self.action_token_ids),
                "action_end_token_id": self.action_end_token_id,
                "protocol_token_ids": list(self.protocol_token_ids),
                "max_reasoning_tokens": self.max_reasoning_tokens,
            }
        }

    @classmethod
    def from_extra_args(cls, extra_args: Mapping[str, Any]) -> "TurnGenerationSpec | None":
        raw = extra_args.get(TURN_RESPONSE_EXTRA_ARG)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("turn response extra args must be a mapping")
        return cls(
            close_token_ids=tuple(int(value) for value in raw["close_token_ids"]),
            injected_token_ids=tuple(
                int(value) for value in raw["injected_token_ids"]
            ),
            action_token_ids=tuple(int(value) for value in raw["action_token_ids"]),
            action_end_token_id=int(raw["action_end_token_id"]),
            protocol_token_ids=tuple(
                int(value) for value in raw["protocol_token_ids"]
            ),
            max_reasoning_tokens=int(raw["max_reasoning_tokens"]),
        )

    @property
    def max_output_tokens(self) -> int:
        return (
            self.max_reasoning_tokens
            + len(self.close_token_ids)
            + len(self.injected_token_ids)
            + 2
        )


def find_token_subsequence(
    values: Sequence[int],
    subsequence: Sequence[int],
) -> int | None:
    """Return the first exact subsequence start, or ``None``."""

    width = len(subsequence)
    if width == 0:
        raise ValueError("subsequence must be non-empty")
    for start in range(len(values) - width + 1):
        if tuple(values[start : start + width]) == tuple(subsequence):
            return start
    return None


def _close_prefix_length(output_ids: Sequence[int], close_ids: Sequence[int]) -> int:
    maximum = min(len(output_ids), len(close_ids) - 1)
    for width in range(maximum, 0, -1):
        if tuple(output_ids[-width:]) == tuple(close_ids[:width]):
            return width
    return 0


def allowed_turn_token_ids(
    output_ids: Sequence[int],
    spec: TurnGenerationSpec,
) -> tuple[int, ...] | None:
    """Return the constrained next-token set; ``None`` means reasoning vocab."""

    close_start = find_token_subsequence(output_ids, spec.close_token_ids)
    if close_start is None:
        if len(output_ids) < spec.max_reasoning_tokens:
            return None
        matched = _close_prefix_length(output_ids, spec.close_token_ids)
        return (spec.close_token_ids[matched],)

    suffix = output_ids[close_start + len(spec.close_token_ids) :]
    if len(suffix) < len(spec.injected_token_ids):
        expected = spec.injected_token_ids[len(suffix)]
        if tuple(suffix) != spec.injected_token_ids[: len(suffix)]:
            raise ValueError("generated turn diverged from injected token prefix")
        return (expected,)
    if tuple(suffix[: len(spec.injected_token_ids)]) != spec.injected_token_ids:
        raise ValueError("generated turn has invalid injected token prefix")
    action_suffix = suffix[len(spec.injected_token_ids) :]
    if not action_suffix:
        return spec.action_token_ids
    if len(action_suffix) == 1:
        if action_suffix[0] not in spec.action_token_ids:
            raise ValueError("generated turn has an invalid action token")
        return (spec.action_end_token_id,)
    raise ValueError("generation continued after the action end boundary")


def apply_turn_response_logits(
    output_ids: Sequence[int],
    logits: torch.Tensor,
    *,
    spec: TurnGenerationSpec,
) -> torch.Tensor:
    """Apply the turn protocol before temperature/top-p sampling."""

    allowed = allowed_turn_token_ids(output_ids, spec)
    masked = logits.clone()
    if allowed is None:
        masked[
            torch.tensor(
                spec.protocol_token_ids,
                dtype=torch.long,
                device=masked.device,
            )
        ] = float("-inf")
        return masked

    selected = torch.tensor(allowed, dtype=torch.long, device=masked.device)
    masked.fill_(float("-inf"))
    masked[selected] = logits[selected]
    return masked


__all__ = [
    "TURN_RESPONSE_EXTRA_ARG",
    "TurnGenerationSpec",
    "allowed_turn_token_ids",
    "apply_turn_response_logits",
    "find_token_subsequence",
]
