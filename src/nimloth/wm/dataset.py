"""Expand Nimloth SFT jsonl trajectories into WM transition samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

# Matches vagen.envs.navigation.utils.nimloth_format.ACTION_NAMES length.
NUM_NAVIGATION_ACTIONS = 8

DEFAULT_VALUE_GAMMA = 0.99


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


def expand_record_transitions(record: dict[str, Any]) -> list[TransitionSample]:
    """Expand one Nimloth jsonl record into per-step transitions.

    Alignment convention (matches convert_sft1_rollouts_to_nimloth):
      - image_paths[t] is the observation visible when choosing action_indices[t]
      - image_paths[t + 1] is the observation after executing action_indices[t]
    """

    messages = list(record.get("messages", []))
    image_paths = list(record.get("image_paths", []))
    action_indices = list(record.get("action_indices", []))
    record_id = str(record.get("id", ""))
    success = bool(record.get("success", False))
    split = str(record.get("split", "train"))

    if not messages or not image_paths or not action_indices:
        return []

    value_targets = discounted_action_value_targets(record)
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


def iter_transitions_from_jsonl(
    path: Path,
    *,
    max_records: int = -1,
    success_only: bool = False,
    split: str | None = None,
) -> Iterator[TransitionSample]:
    for record in load_jsonl_records(path, max_records=max_records):
        if success_only and not record.get("success", False):
            continue
        if split is not None and str(record.get("split", "")) != split:
            continue
        yield from expand_record_transitions(record)


class TransitionJsonlDataset:
    """PyTorch-style indexable dataset over expanded transitions."""

    def __init__(
        self,
        path: Path,
        *,
        max_records: int = -1,
        success_only: bool = False,
        split: str | None = None,
    ) -> None:
        self.samples = list(
            iter_transitions_from_jsonl(
                path,
                max_records=max_records,
                success_only=success_only,
                split=split,
            )
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TransitionSample:
        return self.samples[index]
