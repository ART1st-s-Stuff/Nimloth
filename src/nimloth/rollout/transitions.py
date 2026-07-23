"""把 rollout JSONL 展开为 SFT2 与 RL 共用的逐步 transition。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

from nimloth.agent import (
    AgentTranscript,
    bind_image_placeholders,
    create_prompt_template,
    prompt_template_spec_from_record,
)
from nimloth.environment import get_action_space

DEFAULT_VALUE_GAMMA = 1.0


def discounted_action_value_targets(
    record: dict[str, Any],
    *,
    gamma: float = DEFAULT_VALUE_GAMMA,
) -> list[float]:
    """计算 trajectory 中每个已执行动作的折扣 Monte Carlo return。

    当前数据只有 trajectory 级稀疏 reward，因此第 ``t`` 步的目标为
    ``reward * gamma ** (T - 1 - t)``。
    """

    action_indices = list(record.get("action_indices", []))
    n = len(action_indices)
    if n == 0:
        return []
    terminal = float(record.get("reward", 0.0) or 0.0)
    return [terminal * (gamma ** (n - 1 - t)) for t in range(n)]


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

    image_paths = list(record.get("image_paths", []))
    action_indices = list(record.get("action_indices", []))
    record_id = str(record.get("id", ""))
    success = bool(record.get("success", False))
    split = str(record.get("split", "train"))
    action_space = get_action_space(
        str(record.get("action_space_id", "navigation")),
        int(record.get("action_space_version", 1)),
    )

    if not image_paths or not action_indices:
        return []

    system_prompt = str(record.get("system_prompt", ""))
    observation_texts = tuple(str(text) for text in record.get("observation_texts", []))
    if system_prompt and observation_texts:
        return _expand_structured_agent_transitions(
            record=record,
            record_id=record_id,
            system_prompt=system_prompt,
            observation_texts=observation_texts,
            image_paths=tuple(str(path) for path in image_paths),
            action_indices=tuple(int(index) for index in action_indices),
            success=success,
            split=split,
            value_gamma=value_gamma,
            action_count=len(action_space),
        )

    messages = list(record.get("messages", []))
    if not messages:
        return []

    value_targets = discounted_action_value_targets(record, gamma=value_gamma)
    transitions: list[TransitionSample] = []
    assistant_turn = 0
    assistant_msg_indices: list[int] = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    for msg_index, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        if assistant_turn >= len(action_indices):
            break
        if assistant_turn + 1 >= len(image_paths):
            break

        action_index = int(action_indices[assistant_turn])
        if not 0 <= action_index < len(action_space):
            raise ValueError(
                f"record {record_id!r} step {assistant_turn}: action_index {action_index} "
                f"out of range [0, {len(action_space)})"
            )

        next_prefix_messages: list[dict[str, Any]] | None = None
        next_prefix_image_paths: list[str] | None = None
        if assistant_turn + 1 < len(assistant_msg_indices):
            next_msg_index = assistant_msg_indices[assistant_turn + 1]
            if assistant_turn + 2 < len(image_paths):
                next_prefix_messages = [dict(m) for m in messages[: next_msg_index + 1]]
                next_prefix_image_paths = [str(p) for p in image_paths[: assistant_turn + 2]]

        transitions.append(
            TransitionSample(
                record_id=record_id,
                step_index=assistant_turn,
                prefix_messages=[dict(m) for m in messages[: msg_index + 1]],
                prefix_image_paths=[str(p) for p in image_paths[: assistant_turn + 1]],
                action_index=action_index,
                current_image_path=str(image_paths[assistant_turn]),
                next_image_path=str(image_paths[assistant_turn + 1]),
                next_prefix_messages=next_prefix_messages,
                next_prefix_image_paths=next_prefix_image_paths,
                action_value_target=float(value_targets[assistant_turn]),
                success=success,
                split=split,
            )
        )
        assistant_turn += 1

    return transitions


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

    prompt = create_prompt_template(
        prompt_template_spec_from_record(record),
        action_count=action_count,
    )
    value_targets = discounted_action_value_targets(record, gamma=value_gamma)
    transitions: list[TransitionSample] = []
    for step_index, action_index in enumerate(action_indices):
        current = AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[: step_index + 1],
            observation_images=image_paths[: step_index + 1],
            action_indices=action_indices[: step_index + 1],
        )
        next_state = AgentTranscript(
            system_prompt=system_prompt,
            observation_texts=observation_texts[: step_index + 2],
            observation_images=image_paths[: step_index + 2],
            action_indices=action_indices[: step_index + 1],
        )
        transitions.append(
            TransitionSample(
                record_id=record_id,
                step_index=step_index,
                prefix_messages=(
                    prompt.build_supervised_prompt(current).unbound_messages()
                ),
                prefix_image_paths=list(image_paths[: step_index + 1]),
                action_index=action_index,
                current_image_path=image_paths[step_index],
                next_image_path=image_paths[step_index + 1],
                next_prefix_messages=(
                    prompt.build_policy_prompt(next_state).unbound_messages()
                ),
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
        if success_only and not record.get("success", False):
            continue
        if split is not None and str(record.get("split", "")) != split:
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
        index: int | TransitionContextIndex,
    ) -> TransitionSample | ContextualTransitionSample:
        if isinstance(index, TransitionContextIndex):
            return ContextualTransitionSample(
                sample=self.samples[index.sample_index],
                context_length=index.context_length,
                is_current_step=index.is_current_step,
                loss_weight=index.loss_weight,
            )
        return self.samples[index]
