"""由注入的模板、policy 与动作空间组成的 Agent 状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nimloth.agent.policy import (
    AgentPolicy,
    PolicyDecision,
    PolicyTokenTrace,
    validate_action_log_probs,
)
from nimloth.agent.template import (
    AgentPrompt,
    AgentPromptTemplate,
    PromptTemplateSpec,
)
from nimloth.agent.transcript import AgentTranscript
from nimloth.environment.common.action_space import DiscreteActionSpace


@dataclass(frozen=True)
class AgentAction:
    """一次动作及其完整 policy provenance。"""

    action_index: int
    action_key: str
    action_log_probs: tuple[float, ...]
    response: str
    policy_prompt: AgentPrompt
    token_trace: PolicyTokenTrace | None = None

    @property
    def prompt_messages(self) -> tuple[dict[str, Any], ...]:
        """兼容只需要审计消息的旧调用方。"""

        return self.policy_prompt.messages


class AgentRuntime:
    """维护一个 episode 的 transcript，并执行一次次 policy 调用。"""

    def __init__(
        self,
        *,
        policy: AgentPolicy,
        action_space: DiscreteActionSpace,
        prompt_template: AgentPromptTemplate,
    ) -> None:
        self._policy = policy
        self._action_space = action_space
        self._prompt_template = prompt_template
        if prompt_template.action_count != len(action_space):
            raise ValueError(
                "prompt action count must match environment action space: "
                f"{prompt_template.action_count} != {len(action_space)}"
            )
        self._system_prompt = ""
        self._observation_texts: list[str] = []
        self._observation_images: list[Any] = []
        self._action_indices: list[int] = []
        self._assistant_responses: list[str] = []

    @property
    def prompt_template_spec(self) -> PromptTemplateSpec:
        return self._prompt_template.spec

    @property
    def action_space(self) -> DiscreteActionSpace:
        return self._action_space

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
            raise ValueError("Agent requires the environment system prompt")
        self._policy.reset_episode()
        self._system_prompt = system_prompt
        self._observation_texts.clear()
        self._observation_images.clear()
        self._action_indices.clear()
        self._assistant_responses.clear()

    def observe(self, *, text: str, image: Any) -> None:
        if not self._system_prompt:
            raise RuntimeError("Agent.reset() must be called before observe()")
        if len(self._observation_texts) != len(self._action_indices):
            raise RuntimeError("the previous observation has not been acted on")
        self._observation_texts.append(text)
        self._observation_images.append(image)

    def act(self) -> AgentAction:
        prompt_mode = getattr(self._policy, "prompt_mode", "action")
        if prompt_mode == "action":
            policy_prompt = self._prompt_template.build_policy_prompt(
                self.transcript()
            )
        elif prompt_mode == "response":
            policy_prompt = self._prompt_template.build_response_policy_prompt(
                self.transcript()
            )
        else:
            raise ValueError(f"unknown Agent policy prompt mode: {prompt_mode!r}")
        decision = self._policy.select_action(policy_prompt)
        action_log_probs = validate_action_log_probs(
            decision.action_index,
            decision.action_log_probs,
            action_count=len(self._action_space),
        )
        response = decision.response or self._prompt_template.assistant_response(
            decision.action_index
        )
        self._action_indices.append(decision.action_index)
        self._assistant_responses.append(response)
        return AgentAction(
            action_index=decision.action_index,
            action_key=self._action_space.key_for(decision.action_index),
            action_log_probs=action_log_probs,
            response=response,
            policy_prompt=policy_prompt,
            token_trace=decision.token_trace,
        )

    def transcript(self) -> AgentTranscript:
        if not self._system_prompt:
            raise RuntimeError("Agent has not been reset")
        return AgentTranscript(
            system_prompt=self._system_prompt,
            observation_texts=tuple(self._observation_texts),
            observation_images=tuple(self._observation_images),
            action_indices=tuple(self._action_indices),
            assistant_responses=tuple(self._assistant_responses),
        )

    def completed_prompt(self) -> AgentPrompt:
        """返回所有已完成动作轮次的监督 prompt。"""

        return self._prompt_template.build_supervised_prompt(self.transcript())

    def policy_prompt_for_step(self, step_index: int) -> AgentPrompt:
        """按历史位置重建可审计的 policy prompt。"""

        prefix = self.transcript().policy_prefix(step_index)
        prompt_mode = getattr(self._policy, "prompt_mode", "action")
        if prompt_mode == "response":
            return self._prompt_template.build_response_policy_prompt(prefix)
        return self._prompt_template.build_policy_prompt(prefix)


__all__ = ["AgentAction", "AgentRuntime"]
