"""Nimloth latent-state/action-token prompt 模板。"""

from __future__ import annotations

from typing import Any

from nimloth.agent.registry import register_prompt_template
from nimloth.agent.template import AgentPrompt, PromptTemplateSpec
from nimloth.agent.transcript import AgentTranscript
from nimloth.latent import LatentActionTokens, latent_state_tokens

NIMLOTH_PROMPT_TEMPLATE_ID = "nimloth-latent-action"
PROMPT_VERSION = "nimloth-agent-v1"


class NimlothPromptTemplate:
    """SFT2、RL 和在线 Agent 共用的 latent-action 模板。"""

    def __init__(
        self,
        *,
        latent_token_count: int,
        action_count: int,
    ) -> None:
        if latent_token_count < 1:
            raise ValueError("latent_token_count must be >= 1")
        token_action_count = len(LatentActionTokens().action_tokens)
        if not 1 <= action_count <= token_action_count:
            raise ValueError(
                "action_count must fit the configured action-token vocabulary: "
                f"got {action_count}, capacity={token_action_count}"
            )
        self.latent_token_count = latent_token_count
        self.action_count = action_count

    @property
    def spec(self) -> PromptTemplateSpec:
        return PromptTemplateSpec(
            identifier=NIMLOTH_PROMPT_TEMPLATE_ID,
            version=PROMPT_VERSION,
            config={"latent_token_count": self.latent_token_count},
        )

    def assistant_prefix(self, *, thought: str) -> str:
        if not thought.strip():
            raise ValueError("assistant thought must be non-empty")
        tokens = LatentActionTokens()
        latent_block = "".join(
            latent_state_tokens(self.latent_token_count, tokens)
        )
        return (
            f"<think>{thought}</think>"
            f"{latent_block}{tokens.action_start}"
        )

    def assistant_response(
        self,
        action_index: int,
        *,
        thought: str,
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
        """拒绝缺少真实 CoT 的旧 action-only prompt。"""

        raise RuntimeError(
            "action-only prompt would require a fixed CoT; generate and persist "
            "the real assistant response instead"
        )

    def build_state_prompt(
        self,
        transcript: AgentTranscript,
    ) -> AgentPrompt:
        """拒绝无法从未执行 observation 推导真实 CoT 的旧 state prompt。"""

        raise RuntimeError(
            "state prompt requires a persisted real CoT for this observation; "
            "fixed template thoughts are forbidden"
        )

    def build_response_policy_prompt(
        self,
        transcript: AgentTranscript,
    ) -> AgentPrompt:
        """构造仅预填 ``<think>`` 的 prompt，供 behavior policy 采样 CoT。"""

        if len(transcript.observation_texts) != len(transcript.action_indices) + 1:
            raise ValueError(
                "response policy prompt requires exactly one unacted observation: "
                f"observations={len(transcript.observation_texts)}, "
                f"actions={len(transcript.action_indices)}"
            )
        messages = self._completed_messages(transcript)
        messages.append(
            {"role": "user", "content": transcript.observation_texts[-1]}
        )
        messages.append({"role": "assistant", "content": "<think>"})
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
        if transcript.action_indices and not transcript.assistant_responses:
            raise ValueError(
                "completed transcript requires the real assistant response for "
                "every action"
            )
        for step_index, _action_index in enumerate(transcript.action_indices):
            messages.append(
                {
                    "role": "user",
                    "content": transcript.observation_texts[step_index],
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": transcript.assistant_responses[step_index],
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
    allowed = {"latent_token_count"}
    unknown = sorted(set(spec.config) - allowed)
    if unknown:
        raise ValueError(
            f"unknown Nimloth prompt config field: {unknown[0]}"
        )
    return NimlothPromptTemplate(
        latent_token_count=int(spec.config.get("latent_token_count", 1)),
        action_count=action_count,
    )


register_prompt_template(NIMLOTH_PROMPT_TEMPLATE_ID, _from_spec)
