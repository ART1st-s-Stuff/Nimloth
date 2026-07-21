"""SFT2、RL 和在线 Agent 共用的 transcript 与动作 prompt。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nimloth.latent import LatentActionTokens, latent_state_tokens

PROMPT_VERSION = "nimloth-agent-v1"

@dataclass(frozen=True)
class AgentTranscript:
    """不依赖具体模型和环境动作语义的 episode 历史。"""

    system_prompt: str
    observation_texts: tuple[str, ...]
    observation_images: tuple[Any, ...]
    action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.system_prompt.strip():
            raise ValueError("AgentTranscript requires a non-empty system_prompt")
        if len(self.observation_texts) != len(self.observation_images):
            raise ValueError(
                "observation text/image count mismatch: "
                f"{len(self.observation_texts)} != {len(self.observation_images)}"
            )
        if len(self.action_indices) > len(self.observation_texts):
            raise ValueError("an action cannot exist without its preceding observation")
        for action_index in self.action_indices:
            if action_index < 0:
                raise ValueError(f"action_index must be non-negative, got {action_index}")

    def policy_prefix(self, step_index: int) -> AgentTranscript:
        """返回选择第 ``step_index`` 个动作之前的状态。"""

        if not 0 <= step_index < len(self.observation_texts):
            raise IndexError(
                f"step_index {step_index} outside {len(self.observation_texts)} observations"
            )
        if step_index > len(self.action_indices):
            raise ValueError(
                f"step {step_index} needs {step_index} prior actions, "
                f"only {len(self.action_indices)} available"
            )
        return AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=self.observation_texts[: step_index + 1],
            observation_images=self.observation_images[: step_index + 1],
            action_indices=self.action_indices[:step_index],
        )


def bind_image_placeholders(
    messages: Sequence[dict[str, Any]],
    images: Sequence[Any],
) -> list[dict[str, Any]]:
    """按照顺序把文本中的 ``<image>`` 替换为真实图片。"""

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


class NimlothAgentPrompt:
    """在线 Agent、PPO replay 和 SFT2 共用的离散动作 prompt。"""

    def __init__(
        self,
        *,
        latent_token_count: int = 1,
        action_count: int | None = None,
        thought: str = "What should I do next?",
    ) -> None:
        if latent_token_count < 1:
            raise ValueError("latent_token_count must be >= 1")
        token_action_count = len(LatentActionTokens().action_tokens)
        resolved_action_count = token_action_count if action_count is None else action_count
        if not 1 <= resolved_action_count <= token_action_count:
            raise ValueError(
                "action_count must fit the configured action-token vocabulary: "
                f"got {resolved_action_count}, capacity={token_action_count}"
            )
        self.latent_token_count = latent_token_count
        self.action_count = resolved_action_count
        self.thought = thought

    @property
    def version(self) -> str:
        return PROMPT_VERSION

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
                f"action_index must be in [0, {self.action_count}), got {action_index}"
            )
        tokens = LatentActionTokens()
        return (
            f"{self.assistant_prefix(thought=thought)}"
            f"{tokens.action_tokens[action_index]}"
            f"{tokens.action_end}"
        )

    def build_policy_messages(
        self,
        transcript: AgentTranscript,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        """构造最后一个位置用于预测下一动作的精确 prefix。"""

        if len(transcript.observation_texts) != len(transcript.action_indices) + 1:
            raise ValueError(
                "policy prompt requires exactly one unacted observation: "
                f"observations={len(transcript.observation_texts)}, "
                f"actions={len(transcript.action_indices)}"
            )
        messages = self._completed_messages(transcript)
        messages.append(
            {
                "role": "user",
                "content": transcript.observation_texts[-1],
            }
        )
        messages.append(
            {"role": "assistant", "content": self.assistant_prefix()}
        )
        if bind_images:
            return bind_image_placeholders(messages, transcript.observation_images)
        return messages

    def build_supervised_messages(
        self,
        transcript: AgentTranscript,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        """构造 SFT2 与 trajectory 序列化使用的完整动作轮次。"""

        messages = self._completed_messages(transcript)
        acted_images = transcript.observation_images[: len(transcript.action_indices)]
        if bind_images:
            return bind_image_placeholders(messages, acted_images)
        return messages

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
