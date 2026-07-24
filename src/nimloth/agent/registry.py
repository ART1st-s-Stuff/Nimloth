"""Prompt template 的显式注册与重建入口。"""

from __future__ import annotations

from collections.abc import Callable

from nimloth.agent.template import AgentPromptTemplate, PromptTemplateSpec

PromptTemplateFactory = Callable[[PromptTemplateSpec, int], AgentPromptTemplate]

_PROMPT_TEMPLATE_FACTORIES: dict[str, PromptTemplateFactory] = {}


def register_prompt_template(
    identifier: str,
    factory: PromptTemplateFactory,
) -> None:
    """注册一种持久化模板；重复注册不同 factory 时直接报错。"""

    existing = _PROMPT_TEMPLATE_FACTORIES.get(identifier)
    if existing is not None and existing is not factory:
        raise ValueError(f"prompt template {identifier!r} is already registered")
    _PROMPT_TEMPLATE_FACTORIES[identifier] = factory


def create_prompt_template(
    spec: PromptTemplateSpec,
    *,
    action_count: int,
) -> AgentPromptTemplate:
    """按持久化 spec 重建模板，未知实现不会静默回退。"""

    # 延迟导入保证直接使用 registry 时内置模板也完成注册。
    from nimloth.agent.templates import nimloth as _nimloth  # noqa: F401

    try:
        factory = _PROMPT_TEMPLATE_FACTORIES[spec.identifier]
    except KeyError as error:
        raise ValueError(
            f"unknown prompt template {spec.identifier!r}"
        ) from error
    return factory(spec, action_count)
