"""把 rollout JSONL 展开为 SFT2 与 RL 共用的逐步 transition。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

from nimloth.agent import bind_image_placeholders
from nimloth.environment import get_action_space
from nimloth.rollout.record_format import (
    STEP_REWARD_PROVENANCE,
    TRAJECTORY_REWARD_PROVENANCE,
    require_trajectory_record,
)

DEFAULT_VALUE_GAMMA = 1.0
TERMINAL_ASSISTANT_PREFIX_FIELD = "terminal_assistant_prefix"


def terminal_assistant_prefix(record: dict[str, Any]) -> str:
    """读取离线生成并持久化的 terminal CoT state prefix。"""

    require_trajectory_record(record)
    value = record.get(TERMINAL_ASSISTANT_PREFIX_FIELD)
    if not isinstance(value, str) or not value.strip():
        record_id = str(record["id"])
        raise ValueError(
            f"record {record_id!r} is missing {TERMINAL_ASSISTANT_PREFIX_FIELD!r}; "
            "generate terminal CoT with the SFT1 initialization checkpoint "
            "before building SFT2 transitions"
        )
    return value


def discounted_action_value_targets(
    record: dict[str, Any],
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
    truncated_bootstrap: float | None = None,
) -> list[float]:
    """计算 trajectory 中每个已执行动作的折扣 Monte Carlo return。

    ``reward_provenance`` 明确区分逐步 reward 与 trajectory 级 terminal reward。
    真正 terminal 从0 bootstrap；truncated trajectory必须由调用方显式提供
    bootstrap值，避免把时间上限猜成terminal。
    """

    action_indices = list(record["action_indices"])
    n = len(action_indices)
    if n == 0:
        return []
    provenance = record["reward_provenance"]
    if provenance == STEP_REWARD_PROVENANCE:
        step_rewards = record["rewards"]
        rewards = [float(value) for value in step_rewards]
        if len(rewards) != n:
            raise ValueError(
                f"trajectory rewards/actions mismatch: {len(rewards)} != {n}"
            )
        terminated = bool(record["terminated"])
        truncated = bool(record["truncated"])
        if terminated == truncated:
            raise ValueError(
                "trajectory with step rewards must be exactly one of "
                "terminated or truncated"
            )
        if truncated and truncated_bootstrap is None:
            raise ValueError(
                "truncated trajectory requires an explicit value bootstrap"
            )
        running = 0.0 if terminated else float(truncated_bootstrap)
        returns = [0.0] * n
        for step in range(n - 1, -1, -1):
            running = rewards[step] + gamma * running
            returns[step] = running
        return returns
    if provenance == TRAJECTORY_REWARD_PROVENANCE:
        terminal = float(record["reward"])
        return [terminal * (gamma ** (n - 1 - t)) for t in range(n)]
    raise ValueError(f"unsupported reward_provenance {provenance!r}")


@dataclass(frozen=True)
class TransitionSample:
    """从一条 rollout 记录派生出的单步 transition。"""

    record_id: str
    step_index: int
    prefix_messages: list[dict[str, Any]]
    prefix_image_paths: list[str]
    action_index: int
    current_image_path: str
    next_image_path: str
    next_prefix_messages: list[dict[str, Any]] | None = None
    next_prefix_image_paths: list[str] | None = None
    action_value_target: float = 0.0
    success: bool = True
    split: str = "train"


@dataclass(frozen=True)
class TransitionContextIndex:
    """DataLoader index carrying one step's real context layout."""

    sample_index: int
    context_length: int
    is_current_step: bool
    loss_weight: float = 1.0


@dataclass(frozen=True)
class ContextualTransitionSample:
    """A transition row annotated with its role in one context window."""

    sample: TransitionSample
    context_length: int
    is_current_step: bool
    loss_weight: float = 1.0


@dataclass(frozen=True)
class TransitionRolloutIndex:
    """DataLoader index for one position in a future-action rollout window."""

    sample_index: int
    prediction_horizon: int
    rollout_position: int
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        if not 0 <= self.rollout_position < self.prediction_horizon:
            raise ValueError(
                "rollout_position must be in [0, prediction_horizon), "
                f"got {self.rollout_position} for T={self.prediction_horizon}"
            )


@dataclass(frozen=True)
class RolloutTransitionSample:
    """A transition annotated with its position in a future rollout window."""

    sample: TransitionSample
    prediction_horizon: int
    rollout_position: int
    loss_weight: float = 1.0


def bind_transition_prompt(sample: TransitionSample) -> list[dict[str, Any]]:
    """绑定单步 transition 当前 prompt 的真实图片。"""

    return bind_image_placeholders(
        sample.prefix_messages,
        sample.prefix_image_paths,
    )


def transition_training_item(sample: TransitionSample) -> dict[str, Any]:
    """把 transition 转为 cache 与 batch assembler 共用的普通字典。"""

    item: dict[str, Any] = {
        "id": f"{sample.record_id}:{sample.step_index}",
        "record_id": sample.record_id,
        "step_index": sample.step_index,
        "messages": bind_transition_prompt(sample),
        "action_index": sample.action_index,
        "action_value_target": sample.action_value_target,
        "success": sample.success,
        "next_image_path": sample.next_image_path,
        "current_image_path": sample.current_image_path,
        "next_messages": None,
    }
    if (
        sample.next_prefix_messages is not None
        and sample.next_prefix_image_paths is not None
    ):
        item["next_messages"] = bind_image_placeholders(
            sample.next_prefix_messages,
            sample.next_prefix_image_paths,
        )
    return item


def collate_transition_training_items(
    batch: list[TransitionSample],
) -> list[dict[str, Any]]:
    """为不需要阶段 target 对齐的调用方批量转换 transition。"""

    return [transition_training_item(sample) for sample in batch]


def load_jsonl_records(path: Path, max_records: int = -1) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_records > 0 and len(records) >= max_records:
                break
    return records


def expand_record_transitions(
    record: dict[str, Any],
    *,
    value_gamma: float = DEFAULT_VALUE_GAMMA,
) -> list[TransitionSample]:
    """把一条 Nimloth JSONL 记录展开为逐步 transition。

    对齐约定：``image_paths[t]`` 是选择 ``action_indices[t]`` 时可见的
    observation，``image_paths[t + 1]`` 是执行该动作后的 observation。
    """

    require_trajectory_record(record)
    image_paths = list(record["image_paths"])
    action_indices = list(record["action_indices"])
    record_id = str(record["id"])
    success = bool(record["success"])
    split = str(record["split"])
    action_space = get_action_space(
        str(record["action_space_id"]),
        int(record["action_space_version"]),
    )

    if not image_paths or not action_indices:
        return []

    return _expand_structured_agent_transitions(
        record=record,
        record_id=record_id,
        system_prompt=str(record["system_prompt"]),
        observation_texts=tuple(str(text) for text in record["observation_texts"]),
        image_paths=tuple(str(path) for path in image_paths),
        action_indices=tuple(int(index) for index in action_indices),
        success=success,
        split=split,
        value_gamma=value_gamma,
        action_count=len(action_space),
    )


def _expand_structured_agent_transitions(
    *,
    record: dict[str, Any],
    record_id: str,
    system_prompt: str,
    observation_texts: tuple[str, ...],
    image_paths: tuple[str, ...],
    action_indices: tuple[int, ...],
    success: bool,
    split: str,
    value_gamma: float,
    action_count: int,
) -> list[TransitionSample]:
    """把公共 Agent runtime 写出的结构化记录转换为 prompt prefix。"""

    expected_observations = len(action_indices) + 1
    if len(observation_texts) != expected_observations:
        raise ValueError(
            f"record {record_id!r}: observations={len(observation_texts)} but "
            f"actions={len(action_indices)}; expected one final observation"
        )
    if len(image_paths) != expected_observations:
        raise ValueError(
            f"record {record_id!r}: images={len(image_paths)} but "
            f"actions={len(action_indices)}; expected one final image"
        )

    assistant_responses = tuple(
        str(response) for response in record["assistant_responses"]
    )
    if len(assistant_responses) != len(action_indices):
        raise ValueError(
            f"record {record_id!r}: assistant_responses={len(assistant_responses)} "
            f"but actions={len(action_indices)}"
        )
    terminal_prefix = terminal_assistant_prefix(record)
    value_targets = discounted_action_value_targets(record, gamma=value_gamma)
    transitions: list[TransitionSample] = []
    for step_index, action_index in enumerate(action_indices):
        if not 0 <= action_index < action_count:
            raise ValueError(
                f"record {record_id!r} step {step_index}: action_index "
                f"{action_index} out of range [0, {action_count})"
            )
        current_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        for history_index in range(step_index + 1):
            current_messages.extend(
                (
                    {
                        "role": "user",
                        "content": observation_texts[history_index],
                    },
                    {
                        "role": "assistant",
                        "content": assistant_responses[history_index],
                    },
                )
            )
        next_messages = [dict(message) for message in current_messages]
        next_messages.append(
            {
                "role": "user",
                "content": observation_texts[step_index + 1],
            }
        )
        next_messages.append(
            {
                "role": "assistant",
                "content": (
                    assistant_responses[step_index + 1]
                    if step_index + 1 < len(action_indices)
                    else terminal_prefix
                ),
            }
        )
        transitions.append(
            TransitionSample(
                record_id=record_id,
                step_index=step_index,
                prefix_messages=current_messages,
                prefix_image_paths=list(image_paths[: step_index + 1]),
                action_index=action_index,
                current_image_path=image_paths[step_index],
                next_image_path=image_paths[step_index + 1],
                next_prefix_messages=next_messages,
                next_prefix_image_paths=list(image_paths[: step_index + 2]),
                action_value_target=float(value_targets[step_index]),
                success=success,
                split=split,
            )
        )
    return transitions


def iter_transitions_from_jsonl(
    path: Path,
    *,
    max_records: int = -1,
    success_only: bool = False,
    split: str | None = None,
    value_gamma: float = DEFAULT_VALUE_GAMMA,
) -> Iterator[TransitionSample]:
    for record in load_jsonl_records(path, max_records=max_records):
        require_trajectory_record(record)
        if success_only and not record["success"]:
            continue
        if split is not None and str(record["split"]) != split:
            continue
        yield from expand_record_transitions(record, value_gamma=value_gamma)


class TransitionJsonlDataset(Dataset[TransitionSample]):
    """对展开后 transition 提供 PyTorch 风格的随机访问。"""

    def __init__(
        self,
        path: Path,
        *,
        max_records: int = -1,
        success_only: bool = False,
        split: str | None = None,
        value_gamma: float = DEFAULT_VALUE_GAMMA,
    ) -> None:
        self.samples = list(
            iter_transitions_from_jsonl(
                path,
                max_records=max_records,
                success_only=success_only,
                split=split,
                value_gamma=value_gamma,
            )
        )

    @classmethod
    def from_samples(cls, samples: list[TransitionSample]) -> TransitionJsonlDataset:
        dataset = cls.__new__(cls)
        dataset.samples = samples
        return dataset

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int | TransitionContextIndex | TransitionRolloutIndex,
    ) -> TransitionSample | ContextualTransitionSample | RolloutTransitionSample:
        if isinstance(index, TransitionContextIndex):
            return ContextualTransitionSample(
                sample=self.samples[index.sample_index],
                context_length=index.context_length,
                is_current_step=index.is_current_step,
                loss_weight=index.loss_weight,
            )
        if isinstance(index, TransitionRolloutIndex):
            return RolloutTransitionSample(
                sample=self.samples[index.sample_index],
                prediction_horizon=index.prediction_horizon,
                rollout_position=index.rollout_position,
                loss_weight=index.loss_weight,
            )
        return self.samples[index]
