"""Agent prompt 的公共输入对象与模板协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from nimloth.agent.transcript import AgentTranscript


@dataclass(frozen=True)
class PromptTemplateSpec:
    """可持久化、可重建的 prompt 模板身份与参数。"""

    identifier: str
    version: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier.strip():
            raise ValueError("prompt template identifier must be non-empty")
        if not self.version.strip():
            raise ValueError("prompt template version must be non-empty")
        object.__setattr__(self, "config", dict(self.config))

    def to_record(self) -> dict[str, Any]:
        """转换为 JSON 可序列化记录。"""

        return {
            "identifier": self.identifier,
            "version": self.version,
            "config": dict(self.config),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PromptTemplateSpec:
        """从持久化记录恢复模板描述。"""

        config = record.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("prompt template config must be a mapping")
        return cls(
            identifier=str(record.get("identifier", "")),
            version=str(record.get("version", "")),
            config=config,
        )


def bind_image_placeholders(
    messages: Sequence[dict[str, Any]],
    images: Sequence[Any],
) -> list[dict[str, Any]]:
    """按照顺序把消息中的 ``<image>`` 替换为真实图片。"""

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


@dataclass(frozen=True)
class AgentPrompt:
    """模板生成的未绑定消息、图片和模板来源。"""

    messages: tuple[dict[str, Any], ...]
    images: tuple[Any, ...]
    template: PromptTemplateSpec

    def unbound_messages(self) -> list[dict[str, Any]]:
        """返回适合审计和持久化的消息副本。"""

        return [dict(message) for message in self.messages]

    def bound_messages(self) -> list[dict[str, Any]]:
        """仅在 policy 调用前绑定真实图片。"""

        return bind_image_placeholders(self.messages, self.images)


class AgentPromptTemplate(Protocol):
    """Agent runtime 依赖的模型无关 prompt 模板协议。"""

    @property
    def spec(self) -> PromptTemplateSpec:
        ...

    @property
    def action_count(self) -> int:
        ...

    def build_policy_prompt(self, transcript: AgentTranscript) -> AgentPrompt:
        """构造预测下一动作的 prompt。"""
        ...

    def build_supervised_prompt(self, transcript: AgentTranscript) -> AgentPrompt:
        """构造所有已完成动作轮次的监督 prompt。"""
        ...

    def assistant_response(
        self,
        action_index: int,
        *,
        thought: str | None = None,
    ) -> str:
        """把动作编码为环境后端可接收的 assistant 响应。"""
        ...
