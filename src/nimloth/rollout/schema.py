"""统一的 Agent rollout trajectory 及其校验。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from nimloth.agent import (
    ActionTrainingTrace,
    NIMLOTH_PROMPT_TEMPLATE_ID,
    AgentPrompt,
    AgentTranscript,
    PlannerPolicyTrace,
    PolicyTokenTrace,
    PromptTemplateSpec,
    create_prompt_template,
    prompt_template_spec_from_record,
)
from nimloth.environment import get_action_space
from nimloth.latent import LatentActionTokens
from nimloth.rollout.record_format import (
    STRUCTURED_TRAJECTORY_FIELDS,
    TRAJECTORY_RECORD_FORMAT,
    require_trajectory_record,
)


def _encode_log_probabilities(values: Sequence[float]) -> list[float | None]:
    """Encode impossible actions as strict-JSON ``null`` values."""

    return [None if value == float("-inf") else value for value in values]


def _decode_log_probabilities(
    values: Sequence[float | None],
) -> tuple[float, ...]:
    """Restore strict-JSON ``null`` values to impossible-action log probabilities."""

    return tuple(
        float("-inf") if value is None else float(value) for value in values
    )


WorldModelStateRecord = list[float] | list[list[float]]


def _decode_world_model_state(values: list[Any]) -> WorldModelStateRecord:
    if values and isinstance(values[0], list):
        return [[float(value) for value in row] for row in values]
    return [float(value) for value in values]


def _planner_trace_from_record(raw: dict[str, Any]) -> PlannerPolicyTrace:
    action_training_raw = raw["action_training"]
    teacher = action_training_raw["teacher_action_log_probs"]
    sampled_action = action_training_raw["sampled_action_index"]
    action_training = ActionTrainingTrace(
        objective=str(action_training_raw["objective"]),  # type: ignore[arg-type]
        behavior_owner=str(
            action_training_raw["behavior_owner"]
        ),  # type: ignore[arg-type]
        executed_action_index=int(action_training_raw["executed_action_index"]),
        behavior_action_log_probs=_decode_log_probabilities(
            action_training_raw["behavior_action_log_probs"]
        ),
        teacher_action_log_probs=(
            _decode_log_probabilities(teacher) if teacher is not None else None
        ),
        sampled_action_index=(
            int(sampled_action) if sampled_action is not None else None
        ),
    )
    qwen_sampled_action = raw["qwen_sampled_action_index"]
    beam_width = raw["beam_width"]
    return PlannerPolicyTrace(
        qwen_action_log_probs=_decode_log_probabilities(
            raw["qwen_action_log_probs"]
        ),
        candidate_sequences=tuple(
            tuple(int(value) for value in sequence)
            for sequence in raw["candidate_sequences"]
        ),
        candidate_scores=tuple(float(value) for value in raw["candidate_scores"]),
        root_action_scores=_decode_log_probabilities(raw["root_action_scores"]),
        action_training=action_training,
        horizon=int(raw["horizon"]),
        search_mode=str(raw["search_mode"]),
        qwen_sampled_action_index=(
            int(qwen_sampled_action) if qwen_sampled_action is not None else None
        ),
        beam_width=int(beam_width) if beam_width is not None else None,
    )


@dataclass
class RolloutTrajectory:
    """一个完整 Agent episode 及其 behavior policy 来源信息。"""

    record_id: str
    reward_provenance: str
    image_paths: list[str] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)
    action_names: list[str] = field(default_factory=list)
    action_log_probs: list[list[float]] = field(default_factory=list)
    instruction: str = ""
    success: bool = False
    reward: float = 0.0
    rewards: list[float] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    split: str = "train"
    system_prompt: str = ""
    observation_texts: list[str] = field(default_factory=list)
    policy_messages: list[list[dict[str, Any]]] = field(default_factory=list)
    assistant_responses: list[str] = field(default_factory=list)
    terminal_assistant_prefix: str = ""
    state_anchor_steps: list[int] = field(default_factory=list)
    state_latent_hiddens: list[list[list[float]]] = field(default_factory=list)
    world_model_states: list[WorldModelStateRecord] = field(default_factory=list)
    policy_credit_assignment: str = "action"
    policy_step_indices: list[int] = field(default_factory=list)
    policy_token_ids: list[list[int]] = field(default_factory=list)
    policy_token_log_probs: list[list[float | None]] = field(default_factory=list)
    policy_reference_token_log_probs: list[list[float | None]] = field(
        default_factory=list
    )
    policy_loss_masks: list[list[bool]] = field(default_factory=list)
    policy_token_roles: list[list[str]] = field(default_factory=list)
    policy_action_token_ids: list[list[int]] = field(default_factory=list)
    policy_reasoning_texts: list[str | None] = field(default_factory=list)
    policy_finish_reasons: list[str | None] = field(default_factory=list)
    policy_reasoning_truncated: list[bool] = field(default_factory=list)
    planner_policy_traces: list[PlannerPolicyTrace] = field(default_factory=list)
    prompt_template_spec: PromptTemplateSpec | None = None
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    action_space_id: str = "navigation"
    action_space_version: int = 1

    @property
    def num_steps(self) -> int:
        return len(self.action_indices)

    def resolved_prompt_template_spec(self) -> PromptTemplateSpec:
        """返回该 trajectory 明确保存的 prompt spec。"""

        if self.prompt_template_spec is None:
            raise ValueError("trajectory has no prompt_template_spec")
        return self.prompt_template_spec

    def resolved_latent_token_count(self) -> int:
        """返回模板声明的 latent 数量。"""

        spec = self.resolved_prompt_template_spec()
        if spec.identifier != NIMLOTH_PROMPT_TEMPLATE_ID:
            raise ValueError(
                "trajectory prompt template does not declare Nimloth latent states"
            )
        value = int(spec.config["latent_token_count"])
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
        trace = self.policy_token_trace(step_index)
        if self.planner_policy_traces or self.policy_credit_assignment in {
            "turn",
            "token",
        } or (
            trace is not None and "reasoning" in trace.token_roles
        ):
            return template.build_response_policy_prompt(prefix)
        return template.build_policy_prompt(prefix)

    def build_state_prompt(self, step_index: int) -> AgentPrompt:
        """用该 observation 已持久化的真实 CoT 重建 state prompt。"""

        if not 0 <= step_index <= self.num_steps:
            raise IndexError(
                f"state step {step_index} outside [0, {self.num_steps}]"
            )
        if self.state_anchor_steps and step_index not in self.state_anchor_steps:
            raise ValueError(f"state step {step_index} is not a Qwen anchor")
        action_space = get_action_space(
            self.action_space_id,
            self.action_space_version,
        )
        template = create_prompt_template(
            self.resolved_prompt_template_spec(),
            action_count=len(action_space),
        )
        history = AgentTranscript(
            system_prompt=self.system_prompt,
            observation_texts=tuple(self.observation_texts[:step_index]),
            observation_images=tuple(self.image_paths[:step_index]),
            action_indices=tuple(self.action_indices[:step_index]),
            assistant_responses=tuple(self.assistant_responses[:step_index]),
        )
        messages = template.build_supervised_prompt(history).unbound_messages()
        messages.append(
            {"role": "user", "content": self.observation_texts[step_index]}
        )
        messages.append(
            {
                "role": "assistant",
                "content": self._state_assistant_prefix(step_index),
            }
        )
        return AgentPrompt(
            messages=tuple(messages),
            images=tuple(self.image_paths[: step_index + 1]),
            template=template.spec,
        )

    def _state_assistant_prefix(self, step_index: int) -> str:
        tokens = LatentActionTokens()
        if step_index == self.num_steps:
            prefix = self.terminal_assistant_prefix
        else:
            if len(self.assistant_responses) != self.num_steps:
                raise ValueError(
                    "state replay requires a real assistant response for every action"
                )
            response = self.assistant_responses[step_index]
            boundary = response.find(tokens.action_start)
            if boundary < 0:
                raise ValueError(
                    f"step {step_index} assistant response has no action boundary"
                )
            prefix = response[: boundary + len(tokens.action_start)]
        if not prefix.startswith("<think>") or not prefix.endswith(tokens.action_start):
            raise ValueError(
                f"state step {step_index} has no valid persisted real CoT prefix"
            )
        return prefix

    def policy_token_trace(self, step_index: int) -> PolicyTokenTrace | None:
        """恢复某一步逐 token behavior provenance；未记录 trace 时返回 ``None``。"""

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
        row_count = len(self.policy_step_indices) or self.num_steps
        if not all(len(field) == row_count for field in fields):
            raise ValueError("policy token trace fields do not match trajectory steps")
        if self.policy_reference_token_log_probs and len(
            self.policy_reference_token_log_probs
        ) != row_count:
            raise ValueError(
                "policy reference log-probs do not match trajectory steps"
            )
        if self.policy_step_indices:
            try:
                row_index = self.policy_step_indices.index(step_index)
            except ValueError:
                return None
        else:
            if not 0 <= step_index < self.num_steps:
                raise IndexError(f"policy step {step_index} is outside trajectory")
            row_index = step_index
        return PolicyTokenTrace(
            token_ids=tuple(int(value) for value in self.policy_token_ids[row_index]),
            old_log_probs=tuple(self.policy_token_log_probs[row_index]),
            loss_mask=tuple(bool(value) for value in self.policy_loss_masks[row_index]),
            token_roles=tuple(self.policy_token_roles[row_index]),  # type: ignore[arg-type]
            action_token_ids=tuple(
                int(value) for value in self.policy_action_token_ids[row_index]
            ),
            reasoning_text=self.policy_reasoning_texts[row_index],
            finish_reason=self.policy_finish_reasons[row_index],  # type: ignore[arg-type]
            reasoning_truncated=bool(
                self.policy_reasoning_truncated[row_index]
            ),
            reference_log_probs=(
                tuple(self.policy_reference_token_log_probs[row_index])
                if self.policy_reference_token_log_probs
                else None
            ),
        )

    def planner_policy_trace(self, step_index: int) -> PlannerPolicyTrace | None:
        """Return the reconstructable planner teacher trace for one step."""

        if not self.planner_policy_traces:
            return None
        row_count = len(self.policy_step_indices) or self.num_steps
        if len(self.planner_policy_traces) != row_count:
            raise ValueError("planner policy trace count does not match trajectory steps")
        if self.policy_step_indices:
            try:
                row_index = self.policy_step_indices.index(step_index)
            except ValueError:
                return None
        else:
            row_index = step_index
        return self.planner_policy_traces[row_index]

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
            "record_format": TRAJECTORY_RECORD_FORMAT,
            "id": self.record_id,
            "split": self.split,
            "success": self.success,
            "reward": self.reward,
            "reward_provenance": self.reward_provenance,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "image_paths": self.image_paths,
            "action_indices": self.action_indices,
            "action_names": self.action_names,
            "action_log_probs": [
                _encode_log_probabilities(row)
                for row in self.action_log_probs
            ],
            "instruction": self.instruction,
            "system_prompt": self.system_prompt,
            "observation_texts": self.observation_texts,
            "policy_messages": self.policy_messages,
            "assistant_responses": self.assistant_responses,
            "terminal_assistant_prefix": self.terminal_assistant_prefix,
            "state_anchor_steps": self.state_anchor_steps,
            "state_latent_hiddens": self.state_latent_hiddens,
            "world_model_states": self.world_model_states,
            "policy_credit_assignment": self.policy_credit_assignment,
            "policy_step_indices": self.policy_step_indices,
            "policy_token_ids": self.policy_token_ids,
            "policy_token_log_probs": self.policy_token_log_probs,
            "policy_reference_token_log_probs": (
                self.policy_reference_token_log_probs
            ),
            "policy_loss_masks": self.policy_loss_masks,
            "policy_token_roles": self.policy_token_roles,
            "policy_action_token_ids": self.policy_action_token_ids,
            "policy_reasoning_texts": self.policy_reasoning_texts,
            "policy_finish_reasons": self.policy_finish_reasons,
            "policy_reasoning_truncated": self.policy_reasoning_truncated,
            "planner_policy_traces": [
                {
                    "qwen_action_log_probs": _encode_log_probabilities(
                        trace.qwen_action_log_probs
                    ),
                    "candidate_sequences": [
                        list(sequence) for sequence in trace.candidate_sequences
                    ],
                    "candidate_scores": list(trace.candidate_scores),
                    "root_action_scores": _encode_log_probabilities(
                        trace.root_action_scores
                    ),
                    "action_training": {
                        "objective": trace.action_training.objective,
                        "behavior_owner": trace.action_training.behavior_owner,
                        "executed_action_index": (
                            trace.action_training.executed_action_index
                        ),
                        "behavior_action_log_probs": _encode_log_probabilities(
                            trace.action_training.behavior_action_log_probs
                        ),
                        "teacher_action_log_probs": (
                            _encode_log_probabilities(
                                trace.action_training.teacher_action_log_probs
                            )
                            if trace.action_training.teacher_action_log_probs
                            is not None
                            else None
                        ),
                        "sampled_action_index": (
                            trace.action_training.sampled_action_index
                        ),
                    },
                    "horizon": trace.horizon,
                    "search_mode": trace.search_mode,
                    "qwen_sampled_action_index": (
                        trace.qwen_sampled_action_index
                    ),
                    "beam_width": trace.beam_width,
                }
                for trace in self.planner_policy_traces
            ],
            "prompt_template": prompt_spec.to_record(),
            "sampling_temperature": self.sampling_temperature,
            "sampling_top_p": self.sampling_top_p,
            "action_space_id": self.action_space_id,
            "action_space_version": self.action_space_version,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "RolloutTrajectory":
        required_fields = STRUCTURED_TRAJECTORY_FIELDS | frozenset(
            {
                "action_names",
                "action_log_probs",
                "instruction",
                "rewards",
                "terminated",
                "truncated",
                "terminal_assistant_prefix",
                "policy_messages",
                "state_anchor_steps",
                "state_latent_hiddens",
                "world_model_states",
                "policy_credit_assignment",
                "policy_step_indices",
                "policy_token_ids",
                "policy_token_log_probs",
                "policy_reference_token_log_probs",
                "policy_loss_masks",
                "policy_token_roles",
                "policy_action_token_ids",
                "policy_reasoning_texts",
                "policy_finish_reasons",
                "policy_reasoning_truncated",
                "planner_policy_traces",
                "prompt_template",
                "sampling_temperature",
                "sampling_top_p",
            }
        )
        require_trajectory_record(record, required_fields=required_fields)
        prompt_template_spec = prompt_template_spec_from_record(record)
        return cls(
            record_id=str(record["id"]),
            image_paths=list(record["image_paths"]),
            action_indices=list(record["action_indices"]),
            action_names=list(record["action_names"]),
            action_log_probs=[
                list(_decode_log_probabilities(row))
                for row in record["action_log_probs"]
            ],
            instruction=str(record["instruction"]),
            success=bool(record["success"]),
            reward=float(record["reward"]),
            reward_provenance=str(record["reward_provenance"]),
            rewards=[float(value) for value in record["rewards"]],
            terminated=bool(record["terminated"]),
            truncated=bool(record["truncated"]),
            split=str(record["split"]),
            system_prompt=str(record["system_prompt"]),
            observation_texts=list(record["observation_texts"]),
            policy_messages=list(record["policy_messages"]),
            assistant_responses=list(record["assistant_responses"]),
            terminal_assistant_prefix=str(record["terminal_assistant_prefix"]),
            state_anchor_steps=[
                int(value) for value in record["state_anchor_steps"]
            ],
            state_latent_hiddens=[
                [
                    [float(value) for value in hidden]
                    for hidden in state
                ]
                for state in record["state_latent_hiddens"]
            ],
            world_model_states=[
                _decode_world_model_state(state)
                for state in record["world_model_states"]
            ],
            policy_credit_assignment=str(record["policy_credit_assignment"]),
            policy_step_indices=[
                int(value) for value in record["policy_step_indices"]
            ],
            policy_token_ids=[
                [int(value) for value in row]
                for row in record["policy_token_ids"]
            ],
            policy_token_log_probs=[
                [None if value is None else float(value) for value in row]
                for row in record["policy_token_log_probs"]
            ],
            policy_reference_token_log_probs=[
                [None if value is None else float(value) for value in row]
                for row in record["policy_reference_token_log_probs"]
            ],
            policy_loss_masks=[
                [bool(value) for value in row]
                for row in record["policy_loss_masks"]
            ],
            policy_token_roles=[
                [str(value) for value in row]
                for row in record["policy_token_roles"]
            ],
            policy_action_token_ids=[
                [int(value) for value in row]
                for row in record["policy_action_token_ids"]
            ],
            policy_reasoning_texts=[
                None if value is None else str(value)
                for value in record["policy_reasoning_texts"]
            ],
            policy_finish_reasons=[
                None if value is None else str(value)
                for value in record["policy_finish_reasons"]
            ],
            policy_reasoning_truncated=[
                bool(value)
                for value in record["policy_reasoning_truncated"]
            ],
            planner_policy_traces=[
                _planner_trace_from_record(raw)
                for raw in record["planner_policy_traces"]
            ],
            prompt_template_spec=prompt_template_spec,
            sampling_temperature=float(record["sampling_temperature"]),
            sampling_top_p=float(record["sampling_top_p"]),
            action_space_id=str(record["action_space_id"]),
            action_space_version=int(record["action_space_version"]),
        )
