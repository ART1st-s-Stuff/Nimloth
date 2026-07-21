"""Expand Nimloth SFT jsonl trajectories into WM transition samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset

from nimloth.agent.prompt import (
    PROMPT_VERSION,
    AgentTranscript,
    NimlothAgentPrompt,
)

# Matches vagen.envs.navigation.utils.nimloth_format.ACTION_NAMES length.
NUM_NAVIGATION_ACTIONS = 8

DEFAULT_VALUE_GAMMA = 1.0


def discounted_action_value_targets(record: dict[str, Any], *, gamma: float = DEFAULT_VALUE_GAMMA) -> list[float]:
    """Discounted Monte Carlo return for each taken action in a trajectory.

    Uses trajectory-level ``reward`` as terminal return (sparse). Step ``t`` receives
    ``reward * gamma ** (T - 1 - t)`` where ``T`` is the number of actions.
    """

    action_indices = list(record.get("action_indices", []))
    n = len(action_indices)
    if n == 0:
        return []
    terminal = float(record.get("reward", 0.0) or 0.0)
    return [terminal * (gamma ** (n - 1 - t)) for t in range(n)]


@dataclass(frozen=True)
class TransitionSample:
    """One environment step transition derived from a rollout record."""

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


def expand_record_transitions(record: dict[str, Any], *, value_gamma: float = DEFAULT_VALUE_GAMMA) -> list[TransitionSample]:
    """Expand one Nimloth jsonl record into per-step transitions.

    Alignment convention (matches convert_sft1_rollouts_to_nimloth):
      - image_paths[t] is the observation visible when choosing action_indices[t]
      - image_paths[t + 1] is the observation after executing action_indices[t]
    """

    image_paths = list(record.get("image_paths", []))
    action_indices = list(record.get("action_indices", []))
    record_id = str(record.get("id", ""))
    success = bool(record.get("success", False))
    split = str(record.get("split", "train"))

    if not image_paths or not action_indices:
        return []

    system_prompt = str(record.get("system_prompt", ""))
    observation_texts = tuple(str(text) for text in record.get("observation_texts", []))
    if system_prompt and observation_texts:
        prompt_version = str(record.get("prompt_version", PROMPT_VERSION))
        if prompt_version != PROMPT_VERSION:
            raise ValueError(
                f"record {record_id!r}: unsupported prompt_version {prompt_version!r}; "
                f"expected {PROMPT_VERSION!r}"
            )
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
        if not 0 <= action_index < NUM_NAVIGATION_ACTIONS:
            raise ValueError(
                f"record {record_id!r} step {assistant_turn}: action_index {action_index} "
                f"out of range [0, {NUM_NAVIGATION_ACTIONS})"
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
) -> list[TransitionSample]:
    """Expand records written by the shared Agent runtime into SFT2 prefixes."""

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

    prompt = NimlothAgentPrompt(
        latent_token_count=int(record.get("latent_token_count", 1))
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
                prefix_messages=prompt.build_supervised_messages(
                    current,
                    bind_images=False,
                ),
                prefix_image_paths=list(image_paths[: step_index + 1]),
                action_index=action_index,
                current_image_path=image_paths[step_index],
                next_image_path=image_paths[step_index + 1],
                next_prefix_messages=prompt.build_policy_messages(
                    next_state,
                    bind_images=False,
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
    """PyTorch-style indexable dataset over expanded transitions."""

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

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]
