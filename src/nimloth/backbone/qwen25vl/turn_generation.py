"""Single-request turn generation constraints shared by vLLM rollout code."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import torch

from nimloth.latent import LatentActionTokens, latent_state_tokens


TURN_RESPONSE_EXTRA_ARG = "nimloth_turn_response"
TURN_RESPONSE_PROMPT_PROTOCOL = (
    "nimloth.agent.templates.nimloth:NimlothPromptTemplate."
    "build_response_policy_prompt:nimloth-agent-v1"
)
TURN_RESPONSE_PARSER_PROTOCOL = (
    "nimloth.backbone.qwen25vl.turn_generation:parse_turn_continuation:v1"
)
TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY = hashlib.sha256(
    TURN_RESPONSE_PROMPT_PROTOCOL.encode()
).hexdigest()
TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY = hashlib.sha256(
    TURN_RESPONSE_PARSER_PROTOCOL.encode()
).hexdigest()


@dataclass(frozen=True)
class TurnGenerationSpec:
    """Token protocol for one reasoning-plus-action continuation."""

    close_text: str
    close_token_ids: tuple[int, ...]
    injected_token_ids: tuple[int, ...]
    action_token_ids: tuple[int, ...]
    action_end_token_id: int
    forbidden_reasoning_token_ids: tuple[int, ...]
    max_reasoning_tokens: int

    def __post_init__(self) -> None:
        if not self.close_text:
            raise ValueError("turn generation requires closing text")
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
                "close_text": self.close_text,
                "close_token_ids": list(self.close_token_ids),
                "injected_token_ids": list(self.injected_token_ids),
                "action_token_ids": list(self.action_token_ids),
                "action_end_token_id": self.action_end_token_id,
                "forbidden_reasoning_token_ids": list(
                    self.forbidden_reasoning_token_ids
                ),
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
            close_text=str(raw["close_text"]),
            close_token_ids=tuple(int(value) for value in raw["close_token_ids"]),
            injected_token_ids=tuple(
                int(value) for value in raw["injected_token_ids"]
            ),
            action_token_ids=tuple(int(value) for value in raw["action_token_ids"]),
            action_end_token_id=int(raw["action_end_token_id"]),
            forbidden_reasoning_token_ids=tuple(
                int(value) for value in raw["forbidden_reasoning_token_ids"]
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


@dataclass(frozen=True)
class ParsedTurnResponse:
    continuation_token_ids: tuple[int, ...]
    response: str
    thought: str
    action_index: int
    action_token_id: int
    close_end: int
    reasoning_truncated: bool


@dataclass(frozen=True)
class FSDPGreedyTurnProbeResult:
    checkpoint_identity: str
    prompt_identity: str
    spec_identity: str
    continuation_token_ids: tuple[int, ...]
    parsed: ParsedTurnResponse
    used_current_model_logits: bool
    action_executed: bool = False
    rollout_persisted: bool = False
    deployable_materialized: bool = False


def response_policy_prompt_identity(prompt: Any) -> str:
    """Hash the exact production prompt messages/images/template record."""

    messages = getattr(prompt, "messages", None)
    images = getattr(prompt, "images", None)
    template = getattr(prompt, "template", None)
    to_record = getattr(template, "to_record", None)
    if (
        not isinstance(messages, (list, tuple))
        or not isinstance(images, (list, tuple))
        or not callable(to_record)
        or any(not isinstance(image, (str, bytes)) for image in images)
    ):
        raise TypeError("response-policy prompt identity requires persistable production inputs")
    payload = {
        "protocol_identity": TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY,
        "messages": list(messages),
        "images": [
            image.decode("utf-8") if isinstance(image, bytes) else image
            for image in images
        ],
        "template": to_record(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def build_turn_response_policy_prompt(
    prompt_template: Any,
    transcript: Any,
) -> Any:
    """Delegate to the production prompt owner and reject fixed-thought prompts."""

    builder = getattr(prompt_template, "build_response_policy_prompt", None)
    if not callable(builder):
        raise TypeError("turn probe requires the production response-policy prompt owner")
    prompt = builder(transcript)
    messages = getattr(prompt, "messages", None)
    if (
        not isinstance(messages, (list, tuple))
        or not messages
        or not isinstance(messages[-1], Mapping)
        or messages[-1].get("role") != "assistant"
        or messages[-1].get("content") != "<think>"
    ):
        raise ValueError(
            "production response-policy prompt must prefill only '<think>'"
        )
    return prompt


def build_turn_generation_spec(
    *,
    tokenizer: Any,
    token_id_map: Mapping[str, int],
    action_token_ids: Sequence[int],
    latent_token_count: int,
    max_response_tokens: int,
) -> TurnGenerationSpec:
    """Build the one production protocol shared by vLLM and FSDP probes."""

    if latent_token_count < 1:
        raise ValueError("production turn generation requires latent queries")
    tokens = LatentActionTokens()
    close_ids = tuple(
        int(value)
        for value in tokenizer.encode("</think>", add_special_tokens=False)
    )
    injected_tokens = (*latent_state_tokens(latent_token_count, tokens), tokens.action_start)
    try:
        injected_ids = tuple(int(token_id_map[token]) for token in injected_tokens)
        action_end = int(token_id_map[tokens.action_end])
    except KeyError as error:
        raise ValueError("production turn token table is incomplete") from error
    actions = tuple(int(value) for value in action_token_ids)
    expected_actions = tuple(int(token_id_map[token]) for token in tokens.action_tokens)
    if actions != expected_actions:
        raise ValueError("production turn action-token order changed")
    # Lazy import avoids making the lower-level policy module import this builder
    # while still sharing its exact replay/rollout forbidden-token function.
    from nimloth.backbone.qwen25vl.policy import reasoning_forbidden_token_ids

    forbidden = reasoning_forbidden_token_ids(
        tokenizer,
        dict(token_id_map),
        close_token_ids=close_ids,
    )
    overhead = len(close_ids) + len(injected_ids) + 2
    max_reasoning_tokens = max_response_tokens - overhead
    if max_reasoning_tokens < 1:
        raise ValueError(
            "max_response_tokens is too small for the turn protocol: "
            f"{max_response_tokens} <= {overhead}"
        )
    return TurnGenerationSpec(
        close_text="</think>",
        close_token_ids=close_ids,
        injected_token_ids=injected_ids,
        action_token_ids=actions,
        action_end_token_id=action_end,
        forbidden_reasoning_token_ids=forbidden,
        max_reasoning_tokens=max_reasoning_tokens,
    )


def turn_generation_spec_identity(spec: TurnGenerationSpec) -> str:
    return hashlib.sha256(
        json.dumps(
            spec.to_extra_args(), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _decode(tokenizer: Any, values: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(values),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
            spaces_between_special_tokens=False,
        )
    )


def parse_turn_continuation(
    continuation_token_ids: Sequence[int],
    *,
    tokenizer: Any,
    spec: TurnGenerationSpec,
) -> ParsedTurnResponse:
    """Parse exactly the production continuation without repairing model text."""

    values = tuple(int(value) for value in continuation_token_ids)
    injection_start = find_token_subsequence(values, spec.injected_token_ids)
    if injection_start is None:
        raise ValueError("turn continuation has missing or partial injected K16 prefix")
    decoded_reasoning_close = _decode(tokenizer, values[:injection_start])
    close_start = decoded_reasoning_close.rfind(spec.close_text)
    if close_start < 0 or decoded_reasoning_close[close_start:] != spec.close_text:
        raise ValueError("turn continuation lacks one clean terminal close boundary")
    forbidden = set(spec.forbidden_reasoning_token_ids)
    invalid = sorted(set(values[:injection_start]) & forbidden)
    if invalid:
        raise ValueError(f"turn reasoning contains forbidden control token ids: {invalid}")
    prefix_end = injection_start + len(spec.injected_token_ids)
    if values[injection_start:prefix_end] != spec.injected_token_ids:
        raise ValueError("turn continuation has an invalid injected K16/action-start prefix")
    if len(values) != prefix_end + 2:
        raise ValueError("turn continuation has an invalid action suffix length")
    action_token_id, action_end = values[prefix_end:]
    if action_end != spec.action_end_token_id:
        raise ValueError("turn continuation has a missing or wrong action-end token")
    try:
        action_index = spec.action_token_ids.index(action_token_id)
    except ValueError as error:
        raise ValueError("turn continuation contains a non-action token") from error
    decoded = _decode(tokenizer, values)
    response = "<think>" + decoded
    if response[len("<think>") :] != decoded:
        raise RuntimeError("turn continuation failed exact decode/response round-trip")
    return ParsedTurnResponse(
        continuation_token_ids=values,
        response=response,
        thought=decoded_reasoning_close[:close_start],
        action_index=action_index,
        action_token_id=action_token_id,
        close_end=injection_start,
        reasoning_truncated=injection_start > spec.max_reasoning_tokens,
    )


def _append_token(inputs: Mapping[str, torch.Tensor], token_id: int) -> dict[str, torch.Tensor]:
    current = dict(inputs)
    input_ids = current.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("FSDP greedy probe requires one exact prompt input_ids row")
    token = torch.tensor([[token_id]], dtype=torch.long, device=input_ids.device)
    current["input_ids"] = torch.cat((input_ids, token), dim=1)
    attention = current.get("attention_mask")
    if attention is not None:
        if not isinstance(attention, torch.Tensor) or attention.shape != input_ids.shape:
            raise ValueError("FSDP greedy probe attention mask is misaligned")
        current["attention_mask"] = torch.cat(
            (attention, torch.ones_like(token, dtype=attention.dtype)), dim=1
        )
    return current


@torch.no_grad()
def run_fsdp_greedy_turn_probe(
    model: torch.nn.Module,
    *,
    prompt_inputs: Mapping[str, torch.Tensor],
    tokenizer: Any,
    spec: TurnGenerationSpec,
    checkpoint_identity: str,
    prompt_identity: str,
    require_fsdp: bool = True,
) -> FSDPGreedyTurnProbeResult:
    """Greedily decode current model/FSDP logits; never execute or persist action."""

    for name, identity in (("checkpoint", checkpoint_identity), ("prompt", prompt_identity)):
        if len(identity) != 64 or any(char not in "0123456789abcdef" for char in identity):
            raise ValueError(f"FSDP greedy probe {name} identity must be SHA256")
    if require_fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if not isinstance(model, FSDP):
            raise TypeError("production greedy format probe requires the complete FSDP root")
    inputs = {name: value for name, value in prompt_inputs.items()}
    output_ids: list[int] = []
    for _index in range(spec.max_output_tokens):
        decoded_close_end = None
        if output_ids and _decode(tokenizer, output_ids).endswith(spec.close_text):
            decoded_close_end = len(output_ids)
        if require_fsdp:
            output = model(
                generation_inputs=inputs,
                generation_logits_to_keep=1,
            )
        else:
            output = model(
                **inputs,
                logits_to_keep=1,
                output_hidden_states=False,
                return_dict=True,
            )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
            raise RuntimeError("FSDP greedy probe model did not return next-token logits")
        selected = apply_turn_response_logits(
            output_ids,
            logits[0, -1].float(),
            spec=spec,
            decoded_close_end=decoded_close_end,
        )
        if not torch.isfinite(selected).any():
            raise RuntimeError("FSDP greedy probe has no finite allowed next token")
        token_id = int(torch.argmax(selected).item())
        output_ids.append(token_id)
        inputs = _append_token(inputs, token_id)
        if token_id == spec.action_end_token_id:
            break
    else:
        raise RuntimeError("FSDP greedy probe exhausted the exact turn token budget")
    parsed = parse_turn_continuation(output_ids, tokenizer=tokenizer, spec=spec)
    spec_identity = turn_generation_spec_identity(spec)
    return FSDPGreedyTurnProbeResult(
        checkpoint_identity=checkpoint_identity,
        prompt_identity=prompt_identity,
        spec_identity=spec_identity,
        continuation_token_ids=tuple(output_ids),
        parsed=parsed,
        used_current_model_logits=True,
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
    *,
    decoded_close_end: int | None = None,
) -> tuple[int, ...] | None:
    """Return the constrained next-token set; ``None`` means reasoning vocab."""

    if decoded_close_end is not None:
        if not 1 <= decoded_close_end <= len(output_ids):
            raise ValueError("decoded close boundary is outside generated output")
        close_end = decoded_close_end
    else:
        close_start = find_token_subsequence(output_ids, spec.close_token_ids)
        close_end = (
            close_start + len(spec.close_token_ids)
            if close_start is not None
            else None
        )
    if close_end is None:
        if len(output_ids) < spec.max_reasoning_tokens:
            return None
        matched = _close_prefix_length(output_ids, spec.close_token_ids)
        return (spec.close_token_ids[matched],)

    suffix = output_ids[close_end:]
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
    decoded_close_end: int | None = None,
) -> torch.Tensor:
    """Apply the turn protocol before temperature/top-p sampling."""

    allowed = allowed_turn_token_ids(
        output_ids,
        spec,
        decoded_close_end=decoded_close_end,
    )
    masked = logits.clone()
    if allowed is None:
        masked[
            torch.tensor(
                spec.forbidden_reasoning_token_ids,
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
    "TURN_RESPONSE_PARSER_PROTOCOL",
    "TURN_RESPONSE_PARSER_PROTOCOL_IDENTITY",
    "TURN_RESPONSE_PROMPT_PROTOCOL",
    "TURN_RESPONSE_PROMPT_PROTOCOL_IDENTITY",
    "FSDPGreedyTurnProbeResult",
    "ParsedTurnResponse",
    "TurnGenerationSpec",
    "allowed_turn_token_ids",
    "build_turn_generation_spec",
    "build_turn_response_policy_prompt",
    "apply_turn_response_logits",
    "find_token_subsequence",
    "parse_turn_continuation",
    "response_policy_prompt_identity",
    "run_fsdp_greedy_turn_probe",
    "turn_generation_spec_identity",
]
