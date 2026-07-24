"""训练、评估和独立 rollout 共用的 Agent 配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from nimloth.agent import (
    NIMLOTH_PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    PromptTemplateSpec,
)


@dataclass(frozen=True)
class AgentPlanningConfig:
    """控制一次真实动作之前的快速 World Model 搜索。"""

    enabled: bool = False
    horizon: int = 4
    beam_width: int = 8


@dataclass(frozen=True)
class AgentConfig:
    """选择 prompt 模板、决策方式及其与环境无关的参数。"""

    prompt_template: str = NIMLOTH_PROMPT_TEMPLATE_ID
    thought: str = "What should I do next?"
    planning: AgentPlanningConfig = field(default_factory=AgentPlanningConfig)

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
    allowed = {"prompt_template", "thought", "planning"}
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
    planning_raw = values.get("planning", {})
    if not isinstance(planning_raw, Mapping):
        raise ValueError("agent.planning must be a mapping")
    unknown_planning = sorted(
        set(planning_raw) - {"enabled", "horizon", "beam_width"}
    )
    if unknown_planning:
        raise ValueError(
            f"unknown agent config field: planning.{unknown_planning[0]}"
        )
    enabled = planning_raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("agent.planning.enabled must be a boolean")
    horizon = int(planning_raw.get("horizon", 4))
    beam_width = int(planning_raw.get("beam_width", 8))
    if horizon < 1:
        raise ValueError("agent.planning.horizon must be >= 1")
    if beam_width < 1:
        raise ValueError("agent.planning.beam_width must be >= 1")
    return AgentConfig(
        prompt_template=prompt_template,
        thought=thought,
        planning=AgentPlanningConfig(
            enabled=enabled,
            horizon=horizon,
            beam_width=beam_width,
        ),
    )
