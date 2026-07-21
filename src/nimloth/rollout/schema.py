"""统一的 Agent rollout trajectory 及其校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nimloth.agent import (
    PROMPT_VERSION,
    AgentTranscript,
    NimlothAgentPrompt,
    validate_action_log_probs,
)
from nimloth.environment import get_action_space


@dataclass
class RolloutTrajectory:
    """一个完整 Agent episode 及其 behavior policy 来源信息。"""

    record_id: str
    image_paths: list[str] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)
    action_log_probs: list[list[float]] = field(default_factory=list)
    nav_instruction: str = ""
    success: bool = False
    reward: float = 0.0
    split: str = "train"
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    observation_texts: list[str] = field(default_factory=list)
    policy_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    latent_token_count: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    action_space_id: str = "navigation"
    action_space_version: int = 1

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

    def build_policy_messages(
        self,
        step_index: int,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        transcript = AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts),
            observation_images=tuple(self.image_paths),
            action_indices=tuple(self.action_indices),
        )
        action_space = get_action_space(
            self.action_space_id,
            self.action_space_version,
        )
        prompt = NimlothAgentPrompt(
            latent_token_count=self.latent_token_count,
            action_count=len(action_space),
        )
        return prompt.build_policy_messages(
            transcript.policy_prefix(step_index),
            bind_images=bind_images,
        )

    def build_completed_messages(self, *, bind_images: bool) -> list[dict[str, Any]]:
        transcript = AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts),
            observation_images=tuple(self.image_paths),
            action_indices=tuple(self.action_indices),
        )
        action_space = get_action_space(
            self.action_space_id,
            self.action_space_version,
        )
        prompt = NimlothAgentPrompt(
            latent_token_count=self.latent_token_count,
            action_count=len(action_space),
        )
        return prompt.build_supervised_messages(transcript, bind_images=bind_images)

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.record_id,
            "split": self.split,
            "success": self.success,
            "reward": self.reward,
            "messages": self.messages,
            "image_paths": self.image_paths,
            "action_indices": self.action_indices,
            "action_names": self.action_names,
            "action_log_probs": [
                [None if value == float("-inf") else value for value in row]
                for row in self.action_log_probs
            ],
            "nav_instruction": self.nav_instruction,
            "system_prompt": self.system_prompt,
            "observation_texts": self.observation_texts,
            "policy_messages": self.policy_messages,
            "prompt_version": self.prompt_version,
            "latent_token_count": self.latent_token_count,
            "sampling_temperature": self.sampling_temperature,
            "sampling_top_p": self.sampling_top_p,
            "action_space_id": self.action_space_id,
            "action_space_version": self.action_space_version,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=list(record.get("action_indices", [])),
            action_names=list(record.get("action_names", [])),
            action_log_probs=[
                [float("-inf") if value is None else float(value) for value in row]
                for row in record.get("action_log_probs", [])
            ],
            nav_instruction=str(record.get("nav_instruction", "")),
            success=bool(record.get("success", False)),
            reward=float(record.get("reward", 0.0)),
            split=str(record.get("split", "train")),
            messages=list(record.get("messages", [])),
            system_prompt=str(record.get("system_prompt", "")),
            observation_texts=list(record.get("observation_texts", [])),
            policy_messages=list(record.get("policy_messages", [])),
            prompt_version=str(record.get("prompt_version", "")),
            latent_token_count=int(record.get("latent_token_count", 1)),
            sampling_temperature=float(record.get("sampling_temperature", 1.0)),
            sampling_top_p=float(record.get("sampling_top_p", 1.0)),
            # 旧轨迹没有版本字段，按当时唯一存在的 navigation@1 解释。
            action_space_id=str(record.get("action_space_id", "navigation")),
            action_space_version=int(record.get("action_space_version", 1)),
        )


def validate_rollout_trajectory(trajectory: RolloutTrajectory) -> None:
    """在写盘或训练前校验一条结构化 Agent trajectory。"""

    prefix = f"trajectory {trajectory.record_id}"
    if len(trajectory.image_paths) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: images={len(trajectory.image_paths)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.observation_texts) != trajectory.num_steps + 1:
        raise ValueError(
            f"{prefix}: observations={len(trajectory.observation_texts)} "
            f"but actions={trajectory.num_steps}"
        )
    if len(trajectory.action_names) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_names={len(trajectory.action_names)} "
            f"but actions={trajectory.num_steps}"
        )
    action_space = get_action_space(
        trajectory.action_space_id,
        trajectory.action_space_version,
    )
    expected_names = [action_space.key_for(index) for index in trajectory.action_indices]
    if trajectory.action_names != expected_names:
        raise ValueError(f"{prefix}: action names do not match action indices")
    if len(trajectory.action_log_probs) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: action_log_probs={len(trajectory.action_log_probs)} "
            f"but actions={trajectory.num_steps}"
        )
    for step, (action_index, log_probs) in enumerate(
        zip(trajectory.action_indices, trajectory.action_log_probs, strict=True)
    ):
        try:
            validate_action_log_probs(
                action_index,
                log_probs,
                action_count=len(action_space),
            )
        except ValueError as error:
            raise ValueError(
                f"{prefix} step {step} has invalid action probabilities: {error}"
            ) from error
    if len(trajectory.policy_messages) != trajectory.num_steps:
        raise ValueError(
            f"{prefix}: policy_messages={len(trajectory.policy_messages)} "
            f"but actions={trajectory.num_steps}"
        )
    if not trajectory.system_prompt:
        raise ValueError(f"{prefix} has no system prompt")
    if trajectory.prompt_version != PROMPT_VERSION:
        raise ValueError(
            f"{prefix} uses unsupported prompt version {trajectory.prompt_version!r}"
        )
    if not trajectory.nav_instruction:
        raise ValueError(f"{prefix} has no navigation instruction")
    if trajectory.sampling_temperature < 0.0:
        raise ValueError(f"{prefix} has a negative sampling temperature")
    if not 0.0 < trajectory.sampling_top_p <= 1.0:
        raise ValueError(f"{prefix} has sampling_top_p outside (0, 1]")
    for step, policy_messages in enumerate(trajectory.policy_messages):
        expected_messages = trajectory.build_policy_messages(step, bind_images=False)
        if policy_messages != expected_messages:
            raise ValueError(
                f"{prefix} step {step} policy prompt does not match the "
                "shared Agent template"
            )
    expected_completed = trajectory.build_completed_messages(bind_images=False)
    if trajectory.messages != expected_completed:
        raise ValueError(
            f"{prefix} completed messages do not match the shared Agent template"
        )
