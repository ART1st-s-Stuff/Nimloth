"""Shared Agent transcript and Nimloth action-prompt construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from nimloth.latent import LatentActionTokens, latent_state_tokens

PROMPT_VERSION = "nimloth-agent-v1"

NAVIGATION_ACTION_NAMES: tuple[str, ...] = (
    "moveahead",
    "moveback",
    "moveright",
    "moveleft",
    "rotateright",
    "rotateleft",
    "lookup",
    "lookdown",
)
NAVIGATION_ACTION_TO_INDEX = {
    name: index for index, name in enumerate(NAVIGATION_ACTION_NAMES)
}

_INSTRUCTION_RE = re.compile(r"Human Instruction:\s*(.+?)(?:\n|$)")


def navigation_action_name(action_index: int) -> str:
    if not 0 <= action_index < len(NAVIGATION_ACTION_NAMES):
        raise ValueError(f"action_index must be in [0, 8), got {action_index}")
    return NAVIGATION_ACTION_NAMES[action_index]


def instruction_from_observation(observation_text: str) -> str:
    """Extract navigation instruction metadata from a VAGEN initial observation."""

    match = _INSTRUCTION_RE.search(observation_text)
    return match.group(1).strip() if match else ""


@dataclass(frozen=True)
class AgentTranscript:
    """Structured episode history independent of model-specific rendering."""

    system_prompt: str
    observation_texts: tuple[str, ...]
    observation_images: tuple[Any, ...]
    action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("AgentTranscript requires a non-empty system_prompt")
        if len(self.observation_texts) != len(self.observation_images):
            raise ValueError(
                "observation text/image count mismatch: "
                f"{len(self.observation_texts)} != {len(self.observation_images)}"
            )
        if len(self.action_indices) > len(self.observation_texts):
            raise ValueError("an action cannot exist without its preceding observation")
        for action_index in self.action_indices:
            navigation_action_name(action_index)

    def policy_prefix(self, step_index: int) -> AgentTranscript:
        """Return the state immediately before choosing action ``step_index``."""

        if not 0 <= step_index < len(self.observation_texts):
            raise IndexError(
                f"step_index {step_index} outside {len(self.observation_texts)} observations"
            )
        if step_index > len(self.action_indices):
            raise ValueError(
                f"step {step_index} needs {step_index} prior actions, "
                f"only {len(self.action_indices)} available"
            )
        return AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=self.observation_texts[: step_index + 1],
            observation_images=self.observation_images[: step_index + 1],
            action_indices=self.action_indices[:step_index],
        )


def bind_image_placeholders(
    messages: Sequence[dict[str, Any]],
    images: Sequence[Any],
) -> list[dict[str, Any]]:
    """Replace every textual ``<image>`` placeholder with its ordered image."""

    image_index = 0
    bound: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str) or "<image>" not in content:
            bound.append(dict(message))
            continue

        parts: list[dict[str, Any]] = []
        chunks = content.split("<image>")
        for chunk_index, chunk in enumerate(chunks):
            if chunk:
                parts.append({"type": "text", "text": chunk})
            if chunk_index >= len(chunks) - 1:
                continue
            if image_index >= len(images):
                raise ValueError(
                    "messages contain more image placeholders than the "
                    f"{len(images)} images provided"
                )
            parts.append({"type": "image", "image": images[image_index]})
            image_index += 1
        bound.append({**message, "content": parts})

    if image_index != len(images):
        raise ValueError(
            f"messages consumed {image_index} image placeholders but "
            f"{len(images)} images were provided"
        )
    return bound


class NimlothAgentPrompt:
    """Canonical action prompt shared by online Agent, PPO replay, and SFT2."""

    def __init__(
        self,
        *,
        latent_token_count: int = 1,
        thought: str = "What should I do next?",
    ) -> None:
        if latent_token_count < 1:
            raise ValueError("latent_token_count must be >= 1")
        self.latent_token_count = latent_token_count
        self.thought = thought

    @property
    def version(self) -> str:
        return PROMPT_VERSION

    def assistant_prefix(self, *, thought: str | None = None) -> str:
        tokens = LatentActionTokens()
        latent_block = "".join(
            latent_state_tokens(self.latent_token_count, tokens)
        )
        resolved_thought = self.thought if thought is None else thought
        return (
            f"<think>{resolved_thought}</think>"
            f"{latent_block}{tokens.action_start}"
        )

    def assistant_response(
        self,
        action_index: int,
        *,
        thought: str | None = None,
    ) -> str:
        navigation_action_name(action_index)
        tokens = LatentActionTokens()
        return (
            f"{self.assistant_prefix(thought=thought)}"
            f"{tokens.action_tokens[action_index]}"
            f"{tokens.action_end}"
        )

    def build_policy_messages(
        self,
        transcript: AgentTranscript,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        """Build the exact prefix whose final token predicts the next action."""

        if len(transcript.observation_texts) != len(transcript.action_indices) + 1:
            raise ValueError(
                "policy prompt requires exactly one unacted observation: "
                f"observations={len(transcript.observation_texts)}, "
                f"actions={len(transcript.action_indices)}"
            )
        messages = self._completed_messages(transcript)
        messages.append(
            {
                "role": "user",
                "content": transcript.observation_texts[-1],
            }
        )
        messages.append(
            {"role": "assistant", "content": self.assistant_prefix()}
        )
        if bind_images:
            return bind_image_placeholders(messages, transcript.observation_images)
        return messages

    def build_supervised_messages(
        self,
        transcript: AgentTranscript,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        """Build completed action turns for SFT2 or trajectory serialization."""

        messages = self._completed_messages(transcript)
        acted_images = transcript.observation_images[: len(transcript.action_indices)]
        if bind_images:
            return bind_image_placeholders(messages, acted_images)
        return messages

    def _completed_messages(
        self,
        transcript: AgentTranscript,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": transcript.system_prompt}
        ]
        for step_index, action_index in enumerate(transcript.action_indices):
            messages.append(
                {
                    "role": "user",
                    "content": transcript.observation_texts[step_index],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": self.assistant_response(action_index),
                }
            )
        return messages
