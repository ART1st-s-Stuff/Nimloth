"""持久化 trajectory 的版本与最小结构契约。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TRAJECTORY_RECORD_FORMAT = "nimloth_trajectory_v1"
STEP_REWARD_PROVENANCE = "step_rewards"
TRAJECTORY_REWARD_PROVENANCE = "trajectory_terminal_reward"
REWARD_PROVENANCE_VALUES = frozenset(
    {STEP_REWARD_PROVENANCE, TRAJECTORY_REWARD_PROVENANCE}
)

STRUCTURED_TRAJECTORY_FIELDS = frozenset(
    {
        "id",
        "split",
        "success",
        "reward",
        "reward_provenance",
        "image_paths",
        "action_indices",
        "system_prompt",
        "observation_texts",
        "assistant_responses",
        "action_space_id",
        "action_space_version",
    }
)


def require_trajectory_record(
    record: Mapping[str, Any],
    *,
    required_fields: frozenset[str] = STRUCTURED_TRAJECTORY_FIELDS,
) -> None:
    """在训练边界拒绝未迁移或不完整的持久化记录。"""

    record_format = record.get("record_format")
    if record_format != TRAJECTORY_RECORD_FORMAT:
        raise ValueError(
            "trajectory record must be migrated to "
            f"{TRAJECTORY_RECORD_FORMAT!r}; got {record_format!r}"
        )
    missing = sorted(required_fields - record.keys())
    if missing:
        raise ValueError(
            f"{TRAJECTORY_RECORD_FORMAT} record is missing field {missing[0]!r}"
        )
    provenance = record["reward_provenance"]
    if provenance not in REWARD_PROVENANCE_VALUES:
        raise ValueError(f"unsupported reward_provenance {provenance!r}")


__all__ = [
    "REWARD_PROVENANCE_VALUES",
    "STEP_REWARD_PROVENANCE",
    "STRUCTURED_TRAJECTORY_FIELDS",
    "TRAJECTORY_RECORD_FORMAT",
    "TRAJECTORY_REWARD_PROVENANCE",
    "require_trajectory_record",
]
