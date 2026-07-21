"""Nimloth latent-state/action-token prompt 模板。"""

from __future__ import annotations

from typing import Any

from nimloth.agent.registry import register_prompt_template
from nimloth.agent.template import AgentPrompt, PromptTemplateSpec
from nimloth.agent.transcript import AgentTranscript
from nimloth.latent import LatentActionTokens, latent_state_tokens

NIMLOTH_PROMPT_TEMPLATE_ID = "nimloth-latent-action"
PROMPT_VERSION = "nimloth-agent-v1"
DEFAULT_THOUGHT = "What should I do next?"


class NimlothPromptTemplate:
    """SFT2、RL 和在线 Agent 共用的 latent-action 模板。"""

    def __init__(
        self,
        *,
        latent_token_count: int,
        action_count: int,
        thought: str = DEFAULT_THOUGHT,
    ) -> None:
        if latent_token_count < 1:
            raise ValueError("latent_token_count must be >= 1")
        token_action_count = len(LatentActionTokens().action_tokens)
        if not 1 <= action_count <= token_action_count:
            raise ValueError(
                "action_count must fit the configured action-token vocabulary: "
                f"got {action_count}, capacity={token_action_count}"
            )
        if not thought.strip():
            raise ValueError("prompt thought must be non-empty")
        self.latent_token_count = latent_token_count
        self.action_count = action_count
        self.thought = thought

    @property
    def spec(self) -> PromptTemplateSpec:
        return PromptTemplateSpec(
            identifier=NIMLOTH_PROMPT_TEMPLATE_ID,
            version=PROMPT_VERSION,
            config={
                "latent_token_count": self.latent_token_count,
                "thought": self.thought,
            },
        )

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
        if not 0 <= action_index < self.action_count:
            raise ValueError(
                f"action_index must be in [0, {self.action_count}), "
                f"got {action_index}"
            )
        tokens = LatentActionTokens()
        return (
            f"{self.assistant_prefix(thought=thought)}"
            f"{tokens.action_tokens[action_index]}"
            f"{tokens.action_end}"
        )

    def build_policy_prompt(self, transcript: AgentTranscript) -> AgentPrompt:
        """构造最后一个 observation 的动作查询。"""

        if len(transcript.observation_texts) != len(transcript.action_indices) + 1:
            raise ValueError(
                "policy prompt requires exactly one unacted observation: "
                f"observations={len(transcript.observation_texts)}, "
                f"actions={len(transcript.action_indices)}"
            )
        messages = self._completed_messages(transcript)
        messages.append(
            {"role": "user", "content": transcript.observation_texts[-1]}
        )
        messages.append(
            {"role": "assistant", "content": self.assistant_prefix()}
        )
        return AgentPrompt(
            messages=tuple(messages),
            images=transcript.observation_images,
            template=self.spec,
        )

    def build_supervised_prompt(self, transcript: AgentTranscript) -> AgentPrompt:
        """构造 transcript 中所有已完成动作轮次。"""

        messages = self._completed_messages(transcript)
        return AgentPrompt(
            messages=tuple(messages),
            images=transcript.observation_images[: len(transcript.action_indices)],
            template=self.spec,
        )

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


def _from_spec(
    spec: PromptTemplateSpec,
    action_count: int,
) -> NimlothPromptTemplate:
    if spec.version != PROMPT_VERSION:
        raise ValueError(
            f"unsupported prompt version {spec.version!r} for "
            f"{NIMLOTH_PROMPT_TEMPLATE_ID}; expected {PROMPT_VERSION!r}"
        )
    allowed = {"latent_token_count", "thought"}
    unknown = sorted(set(spec.config) - allowed)
    if unknown:
        raise ValueError(
            f"unknown Nimloth prompt config field: {unknown[0]}"
        )
    return NimlothPromptTemplate(
        latent_token_count=int(spec.config.get("latent_token_count", 1)),
        action_count=action_count,
        thought=str(spec.config.get("thought", DEFAULT_THOUGHT)),
    )


register_prompt_template(NIMLOTH_PROMPT_TEMPLATE_ID, _from_spec)
