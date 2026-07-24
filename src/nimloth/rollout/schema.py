"""统一的 Agent rollout trajectory 及其校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nimloth.agent import (
    NIMLOTH_PROMPT_TEMPLATE_ID,
    PROMPT_VERSION,
    AgentPrompt,
    AgentTranscript,
    PolicyTokenTrace,
    PromptTemplateSpec,
    create_prompt_template,
    prompt_template_spec_from_record,
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
    instruction: str = ""
    success: bool = False
    reward: float = 0.0
    split: str = "train"
    messages: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    observation_texts: list[str] = field(default_factory=list)
    policy_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    policy_credit_assignment: str = "action"
    policy_token_ids: list[list[int]] = field(default_factory=list)
    policy_token_log_probs: list[list[float | None]] = field(default_factory=list)
    policy_loss_masks: list[list[bool]] = field(default_factory=list)
    policy_token_roles: list[list[str]] = field(default_factory=list)
    policy_action_token_ids: list[list[int]] = field(default_factory=list)
    policy_reasoning_texts: list[str | None] = field(default_factory=list)
    policy_finish_reasons: list[str | None] = field(default_factory=list)
    policy_reasoning_truncated: list[bool] = field(default_factory=list)
    prompt_template_spec: PromptTemplateSpec | None = None
    # 下面两个字段只为读取旧 JSONL 保留；新记录以 prompt_template_spec 为准。
    prompt_version: str = PROMPT_VERSION
    latent_token_count: int = 1
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    action_space_id: str = "navigation"
    action_space_version: int = 1

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

    def resolved_prompt_template_spec(self) -> PromptTemplateSpec:
        """返回新式 spec，或把内存中的旧字段显式迁移。"""

        if self.prompt_template_spec is not None:
            return self.prompt_template_spec
        return prompt_template_spec_from_record(
            {
                "prompt_version": self.prompt_version,
                "latent_token_count": self.latent_token_count,
            }
        )

    def resolved_latent_token_count(self) -> int:
        """返回模板声明的 latent 数量，旧模板则读取兼容字段。"""

        spec = self.resolved_prompt_template_spec()
        if spec.identifier == NIMLOTH_PROMPT_TEMPLATE_ID:
            value = int(spec.config.get("latent_token_count", 1))
        else:
            value = self.latent_token_count
        if value < 1:
            raise ValueError("latent_token_count must be >= 1")
        return value

    def build_policy_prompt(self, step_index: int) -> AgentPrompt:
        """通过注册模板重建某一步的 behavior policy prompt。"""

        action_space = get_action_space(
            self.action_space_id,
            self.action_space_version,
        )
        template = create_prompt_template(
            self.resolved_prompt_template_spec(),
            action_count=len(action_space),
        )
        prefix = self.transcript().policy_prefix(step_index)
        if self.policy_credit_assignment == "turn":
            return template.build_response_policy_prompt(prefix)
        return template.build_policy_prompt(prefix)

    def build_state_prompt(self, step_index: int) -> AgentPrompt:
        """RL state replay 等待真实 current/terminal CoT 数据契约。"""

        raise RuntimeError(
            "RL state replay is unfinished: every observation, including terminal, "
            "must have a persisted real CoT before a state prompt can be built"
        )

    def policy_token_trace(self, step_index: int) -> PolicyTokenTrace | None:
        """恢复某一步逐 token behavior provenance；旧 action-only 记录返回 ``None``。"""

        fields = (
            self.policy_token_ids,
            self.policy_token_log_probs,
            self.policy_loss_masks,
            self.policy_token_roles,
            self.policy_action_token_ids,
            self.policy_reasoning_texts,
            self.policy_finish_reasons,
            self.policy_reasoning_truncated,
        )
        if all(not field for field in fields):
            return None
        if not all(len(field) == self.num_steps for field in fields):
            raise ValueError("policy token trace fields do not match trajectory steps")
        return PolicyTokenTrace(
            token_ids=tuple(int(value) for value in self.policy_token_ids[step_index]),
            old_log_probs=tuple(self.policy_token_log_probs[step_index]),
            loss_mask=tuple(bool(value) for value in self.policy_loss_masks[step_index]),
            token_roles=tuple(self.policy_token_roles[step_index]),  # type: ignore[arg-type]
            action_token_ids=tuple(
                int(value) for value in self.policy_action_token_ids[step_index]
            ),
            reasoning_text=self.policy_reasoning_texts[step_index],
            finish_reason=self.policy_finish_reasons[step_index],  # type: ignore[arg-type]
            reasoning_truncated=bool(
                self.policy_reasoning_truncated[step_index]
            ),
        )

    def build_policy_messages(
        self,
        step_index: int,
        *,
        bind_images: bool,
    ) -> list[dict[str, Any]]:
        prompt = self.build_policy_prompt(step_index)
        return prompt.bound_messages() if bind_images else prompt.unbound_messages()

    def build_completed_prompt(self) -> AgentPrompt:
        """通过注册模板重建完整监督 prompt。"""

        action_space = get_action_space(
            self.action_space_id,
            self.action_space_version,
        )
        template = create_prompt_template(
            self.resolved_prompt_template_spec(),
            action_count=len(action_space),
        )
        return template.build_supervised_prompt(self.transcript())

    def build_completed_messages(self, *, bind_images: bool) -> list[dict[str, Any]]:
        prompt = self.build_completed_prompt()
        return prompt.bound_messages() if bind_images else prompt.unbound_messages()

    def transcript(self) -> AgentTranscript:
        """把持久化字段还原为 Agent transcript。"""

        return AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts),
            observation_images=tuple(self.image_paths),
            action_indices=tuple(self.action_indices),
            assistant_responses=tuple(self.assistant_responses),
        )

    def to_record(self) -> dict[str, Any]:
        prompt_spec = self.resolved_prompt_template_spec()
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
            "instruction": self.instruction,
            # 迁移期继续写旧 key，旧训练工具仍可读取。
            "nav_instruction": self.instruction,
            "system_prompt": self.system_prompt,
            "observation_texts": self.observation_texts,
            "policy_messages": self.policy_messages,
            "assistant_responses": self.assistant_responses,
            "policy_credit_assignment": self.policy_credit_assignment,
            "policy_token_ids": self.policy_token_ids,
            "policy_token_log_probs": self.policy_token_log_probs,
            "policy_loss_masks": self.policy_loss_masks,
            "policy_token_roles": self.policy_token_roles,
            "policy_action_token_ids": self.policy_action_token_ids,
            "policy_reasoning_texts": self.policy_reasoning_texts,
            "policy_finish_reasons": self.policy_finish_reasons,
            "policy_reasoning_truncated": self.policy_reasoning_truncated,
            "prompt_template": prompt_spec.to_record(),
            # 同时写出旧字段，便于已有工具在迁移期继续读取。
            "prompt_version": prompt_spec.version,
            "latent_token_count": self.resolved_latent_token_count(),
            "sampling_temperature": self.sampling_temperature,
            "sampling_top_p": self.sampling_top_p,
            "action_space_id": self.action_space_id,
            "action_space_version": self.action_space_version,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        prompt_template_spec = prompt_template_spec_from_record(record)
        latent_token_count = int(
            prompt_template_spec.config.get(
                "latent_token_count",
                record.get("latent_token_count", 1),
            )
        )
        return cls(
            record_id=str(record.get("id", "")),
            image_paths=list(record.get("image_paths", [])),
            action_indices=list(record.get("action_indices", [])),
            action_names=list(record.get("action_names", [])),
            action_log_probs=[
                [float("-inf") if value is None else float(value) for value in row]
                for row in record.get("action_log_probs", [])
            ],
            instruction=str(
                record.get("instruction", record.get("nav_instruction", ""))
            ),
            success=bool(record.get("success", False)),
            reward=float(record.get("reward", 0.0)),
            split=str(record.get("split", "train")),
            messages=list(record.get("messages", [])),
            system_prompt=str(record.get("system_prompt", "")),
            observation_texts=list(record.get("observation_texts", [])),
            policy_messages=list(record.get("policy_messages", [])),
            assistant_responses=list(record.get("assistant_responses", [])),
            policy_credit_assignment=str(
                record.get("policy_credit_assignment", "action")
            ),
            policy_token_ids=[
                [int(value) for value in row]
                for row in record.get("policy_token_ids", [])
            ],
            policy_token_log_probs=[
                [None if value is None else float(value) for value in row]
                for row in record.get("policy_token_log_probs", [])
            ],
            policy_loss_masks=[
                [bool(value) for value in row]
                for row in record.get("policy_loss_masks", [])
            ],
            policy_token_roles=[
                [str(value) for value in row]
                for row in record.get("policy_token_roles", [])
            ],
            policy_action_token_ids=[
                [int(value) for value in row]
                for row in record.get("policy_action_token_ids", [])
            ],
            policy_reasoning_texts=[
                None if value is None else str(value)
                for value in record.get("policy_reasoning_texts", [])
            ],
            policy_finish_reasons=[
                None if value is None else str(value)
                for value in record.get("policy_finish_reasons", [])
            ],
            policy_reasoning_truncated=[
                bool(value)
                for value in record.get("policy_reasoning_truncated", [])
            ],
            prompt_template_spec=prompt_template_spec,
            prompt_version=prompt_template_spec.version,
            latent_token_count=latent_token_count,
            sampling_temperature=float(record.get("sampling_temperature", 1.0)),
            sampling_top_p=float(record.get("sampling_top_p", 1.0)),
            # 旧轨迹没有版本字段，按当时唯一存在的 navigation@1 解释。
            action_space_id=str(record.get("action_space_id", "navigation")),
            action_space_version=int(record.get("action_space_version", 1)),
        )
