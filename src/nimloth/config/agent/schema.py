"""训练、评估和独立 rollout 共用的 Agent 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nimloth.agent import (
    NIMLOTH_PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    PromptTemplateSpec,
)


@dataclass(frozen=True)
class AgentConfig:
    """选择 prompt 模板及其与环境无关的文本参数。"""

    prompt_template: str = NIMLOTH_PROMPT_TEMPLATE_ID
    thought: str = "What should I do next?"

    def prompt_spec(self, *, latent_token_count: int) -> PromptTemplateSpec:
        """结合模型能力生成可持久化模板 spec。"""

        if self.prompt_template != NIMLOTH_PROMPT_TEMPLATE_ID:
            raise ValueError(
                f"unsupported configured prompt template "
                f"{self.prompt_template!r}"
            )
        return PromptTemplateSpec(
            identifier=self.prompt_template,
            version=PROMPT_VERSION,
            config={
                "latent_token_count": latent_token_count,
                "thought": self.thought,
            },
        )


def parse_agent_config(raw: Mapping[str, Any] | None) -> AgentConfig:
    """严格解析 Agent 配置，未知字段直接报错。"""

    values = {} if raw is None else raw
    if not isinstance(values, Mapping):
        raise ValueError("agent config must be a mapping")
    allowed = {"prompt_template", "thought"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown agent config field: {unknown[0]}")
    prompt_template = str(
        values.get("prompt_template", NIMLOTH_PROMPT_TEMPLATE_ID)
    )
    thought = str(values.get("thought", "What should I do next?"))
    if not prompt_template.strip():
        raise ValueError("agent.prompt_template must be non-empty")
    if not thought.strip():
        raise ValueError("agent.thought must be non-empty")
    return AgentConfig(prompt_template=prompt_template, thought=thought)
