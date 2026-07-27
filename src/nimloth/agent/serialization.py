"""Prompt template spec 的持久化读取。"""

from __future__ import annotations

from typing import Any, Mapping

from nimloth.agent.template import PromptTemplateSpec
def prompt_template_spec_from_record(
    record: Mapping[str, Any],
) -> PromptTemplateSpec:
    """读取记录明确保存的 prompt spec。"""

    stored = record.get("prompt_template")
    if not isinstance(stored, Mapping):
        raise ValueError("trajectory record must contain a prompt_template mapping")
    return PromptTemplateSpec.from_record(stored)
