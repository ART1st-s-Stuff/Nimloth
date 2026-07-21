"""Stateful Agent runtime backed by a pluggable action policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from nimloth.agent.prompt import (
    AgentTranscript,
    NimlothAgentPrompt,
    navigation_action_name,
)


def validate_action_log_probs(
    action_index: int,
    action_log_probs: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Validate and normalize the Agent's 8-way behavior distribution."""

    navigation_action_name(action_index)
    values = tuple(float(value) for value in action_log_probs)
    if len(values) != 8:
        raise ValueError(
            f"policy must return 8 action log probabilities, got {len(values)}"
        )
    chosen = values[action_index]
    if not math.isfinite(chosen):
        raise ValueError("chosen action must have a finite behavior log probability")
    if any(math.isnan(value) or value == float("inf") for value in values):
        raise ValueError(
            "action log probabilities may contain -inf, but not NaN or +inf"
        )
    probability_sum = sum(math.exp(value) for value in values if math.isfinite(value))
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(
            f"action log probabilities must normalize to 1, got {probability_sum}"
        )
    return values


@dataclass(frozen=True)
class PolicyDecision:
    """Action and behavior-distribution log probabilities returned by a policy."""

    action_index: int
    action_log_probs: tuple[float, ...]

    def __post_init__(self) -> None:
        validate_action_log_probs(self.action_index, self.action_log_probs)


class AgentPolicy(Protocol):
    def select_action(self, messages: list[dict[str, Any]]) -> PolicyDecision:
        """Choose an action from one fully rendered Agent policy prompt."""
        ...


@dataclass(frozen=True)
class AgentAction:
    action_index: int
    action_name: str
    action_log_probs: tuple[float, ...]
    response: str
    prompt_messages: tuple[dict[str, Any], ...]


class NavigationAgent:
    """Own one navigation episode's observations, actions, and policy calls."""

    def __init__(
        self,
        *,
        policy: AgentPolicy,
        prompt: NimlothAgentPrompt | None = None,
    ) -> None:
        self._policy = policy
        self._prompt = prompt or NimlothAgentPrompt()
        self._system_prompt = ""
        self._observation_texts: list[str] = []
        self._observation_images: list[Any] = []
        self._action_indices: list[int] = []

    @property
    def prompt_version(self) -> str:
        return self._prompt.version

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def observation_texts(self) -> tuple[str, ...]:
        return tuple(self._observation_texts)

    @property
    def action_indices(self) -> tuple[int, ...]:
        return tuple(self._action_indices)

    def reset(self, *, system_prompt: str) -> None:
        if not system_prompt.strip():
            raise ValueError("NavigationAgent requires the environment system prompt")
        self._system_prompt = system_prompt
        self._observation_texts.clear()
        self._observation_images.clear()
        self._action_indices.clear()

    def observe(self, *, text: str, image: Any) -> None:
        if not self._system_prompt:
            raise RuntimeError("NavigationAgent.reset() must be called before observe()")
        if len(self._observation_texts) != len(self._action_indices):
            raise RuntimeError("the previous observation has not been acted on")
        if "<image>" not in text:
            raise ValueError("Agent observation text must contain an <image> placeholder")
        self._observation_texts.append(text)
        self._observation_images.append(image)

    def act(self) -> AgentAction:
        transcript = self.transcript()
        unbound_messages = self._prompt.build_policy_messages(
            transcript,
            bind_images=False,
        )
        bound_messages = self._prompt.build_policy_messages(
            transcript,
            bind_images=True,
        )
        decision = self._policy.select_action(bound_messages)
        self._action_indices.append(decision.action_index)
        return AgentAction(
            action_index=decision.action_index,
            action_name=navigation_action_name(decision.action_index),
            action_log_probs=decision.action_log_probs,
            response=self._prompt.assistant_response(decision.action_index),
            prompt_messages=tuple(unbound_messages),
        )

    def transcript(self) -> AgentTranscript:
        if not self._system_prompt:
            raise RuntimeError("NavigationAgent has not been reset")
        return AgentTranscript(
            system_prompt=self._system_prompt,
            observation_texts=tuple(self._observation_texts),
            observation_images=tuple(self._observation_images),
            action_indices=tuple(self._action_indices),
        )

    def completed_messages(self, *, bind_images: bool = False) -> list[dict[str, Any]]:
        return self._prompt.build_supervised_messages(
            self.transcript(),
            bind_images=bind_images,
        )

    def policy_messages_for_step(
        self,
        step_index: int,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        prefix = self.transcript().policy_prefix(step_index)
        return self._prompt.build_policy_messages(prefix, bind_images=bind_images)
