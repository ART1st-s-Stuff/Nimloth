from __future__ import annotations

import json

import pytest

from nimloth.rollout.migration import migrate_trajectory_jsonl
from nimloth.rollout.migration_validation import validate_trajectory_migration
from nimloth.rollout.record_format import TRAJECTORY_REWARD_PROVENANCE


def _legacy_record(record_id: str = "legacy-1") -> dict:
    terminal_cot = "<think>terminal real thought</think>"
    return {
        "id": record_id,
        "split": "train",
        "success": True,
        "reward": 1.0,
        "nav_instruction": "move forward",
        "image_paths": ["before.png", "after.png"],
        "action_indices": [0],
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "before <image>"},
            {
                "role": "assistant",
                "content": (
                    "<think>real thought</think><|latent_state|>"
                    "<|action_start|><|action_(0)|><|action_end|>"
                ),
            },
            {"role": "user", "content": "after <image>"},
        ],
        "terminal_assistant_prefix": terminal_cot,
    }


def _migrate(source, output) -> None:
    migrate_trajectory_jsonl(
        source_path=source,
        output_path=output,
        missing_action_space_id="navigation",
        missing_action_space_version=1,
        missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
    )


def _validate(source, output) -> dict:
    return validate_trajectory_migration(
        source_path=source,
        output_path=output,
        missing_action_space_id="navigation",
        missing_action_space_version=1,
        missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
    )


def test_validation_audits_all_records_actions_terminal_cot_and_transitions(
    tmp_path,
) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "migrated.jsonl"
    source.write_text(
        "".join(
            json.dumps(_legacy_record(f"legacy-{index}")) + "\n"
            for index in range(2)
        ),
        encoding="utf-8",
    )
    _migrate(source, output)

    result = _validate(source, output)

    assert result["record_count"] == 2
    assert result["transition_count"] == 2
    assert result["ids_unique"] is True
    assert result["actions_equal"] is True
    assert result["terminal_cot_equal"] is True


def test_validation_rejects_modified_migrated_record(tmp_path) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "migrated.jsonl"
    source.write_text(json.dumps(_legacy_record()) + "\n", encoding="utf-8")
    _migrate(source, output)
    record = json.loads(output.read_text(encoding="utf-8"))
    record["terminal_assistant_prefix"] = "<think>changed</think>"
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic migration"):
        _validate(source, output)
