"""Prompt template spec 的持久化兼容层。"""

from __future__ import annotations

from typing import Any, Mapping

from nimloth.agent.template import PromptTemplateSpec
from nimloth.agent.templates.nimloth import (
    DEFAULT_THOUGHT,
    NIMLOTH_PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
)


def prompt_template_spec_from_record(
    record: Mapping[str, Any],
) -> PromptTemplateSpec:
    """读取新 spec；旧记录按历史 Nimloth 模板显式迁移。"""

    stored = record.get("prompt_template")
    if stored is not None:
        if not isinstance(stored, Mapping):
            raise ValueError("prompt_template must be a mapping")
        return PromptTemplateSpec.from_record(stored)

    return PromptTemplateSpec(
        identifier=NIMLOTH_PROMPT_TEMPLATE_ID,
        version=str(record.get("prompt_version", PROMPT_VERSION)),
        config={
            "latent_token_count": int(record.get("latent_token_count", 1)),
            "thought": str(record.get("prompt_thought", DEFAULT_THOUGHT)),
        },
    )
