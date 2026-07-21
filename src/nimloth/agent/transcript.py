"""与模型、环境后端无关的 Agent 对话状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentTranscript:
    """保存一个 episode 中按时间排序的 observation 与动作。"""

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
        if len(self.observation_texts) > len(self.action_indices) + 1:
            raise ValueError(
                "an Agent transcript may contain at most one unacted observation"
            )
        if any(action_index < 0 for action_index in self.action_indices):
            raise ValueError("action indices must be non-negative")

    def policy_prefix(self, step_index: int) -> AgentTranscript:
        """返回选择第 ``step_index`` 个动作之前的状态。"""

        if not 0 <= step_index < len(self.observation_texts):
            raise IndexError(
                f"step_index {step_index} outside "
                f"{len(self.observation_texts)} observations"
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
