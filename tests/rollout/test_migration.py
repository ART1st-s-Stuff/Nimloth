from __future__ import annotations

import json

import pytest

from nimloth.rollout.migration import (
    migrate_trajectory_jsonl,
    migrate_trajectory_record,
)
from nimloth.rollout.record_format import (
    TRAJECTORY_RECORD_FORMAT,
    TRAJECTORY_REWARD_PROVENANCE,
)
from nimloth.rollout.transitions import expand_record_transitions


def _legacy_record() -> dict:
    return {
        "id": "legacy-1",
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
    }


def _migrate(record: dict) -> dict:
    return migrate_trajectory_record(
        record,
        missing_action_space_id="navigation",
        missing_action_space_version=1,
        missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
    )


def test_migration_extracts_real_transcript_and_removes_legacy_fields() -> None:
    migrated = _migrate(_legacy_record())

    assert migrated["record_format"] == TRAJECTORY_RECORD_FORMAT
    assert migrated["system_prompt"] == "system"
    assert migrated["observation_texts"] == ["before <image>", "after <image>"]
    assert migrated["assistant_responses"][0].startswith("<think>real thought")
    assert migrated["instruction"] == "move forward"
    assert migrated["reward_provenance"] == TRAJECTORY_REWARD_PROVENANCE
    assert "messages" not in migrated
    assert "nav_instruction" not in migrated
    assert "terminal_assistant_prefix" not in migrated
    assert "policy_token_ids" not in migrated
    assert "state_latent_hiddens" not in migrated


def test_migration_requires_explicit_missing_semantics() -> None:
    with pytest.raises(ValueError, match="action-space identity"):
        migrate_trajectory_record(
            _legacy_record(),
            missing_action_space_id=None,
            missing_action_space_version=None,
            missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
        )
    with pytest.raises(ValueError, match="reward provenance"):
        migrate_trajectory_record(
            {**_legacy_record(), "rewards": [1.0]},
            missing_action_space_id="navigation",
            missing_action_space_version=1,
            missing_reward_provenance=None,
        )


def test_migration_converts_legacy_prompt_identity() -> None:
    record = _legacy_record()
    record["prompt_version"] = "nimloth-agent-v1"
    record["latent_token_count"] = 4

    migrated = _migrate(record)

    assert migrated["prompt_template"] == {
        "identifier": "nimloth-latent-action",
        "version": "nimloth-agent-v1",
        "config": {"latent_token_count": 4},
    }
    assert "prompt_version" not in migrated
    assert "latent_token_count" not in migrated


def test_training_reader_rejects_unmigrated_record() -> None:
    with pytest.raises(ValueError, match="must be migrated"):
        expand_record_transitions(_legacy_record())


def test_legacy_planner_semantics_must_be_declared_during_migration() -> None:
    record = _legacy_record()
    record["planner_policy_traces"] = [
        {
            "qwen_action_log_probs": [-1.0, -2.0],
            "candidate_sequences": [[0]],
            "candidate_scores": [1.0],
            "root_action_scores": [0.0, None],
            "behavior_action_log_probs": [0.0, None],
            "teacher_action_log_probs": [0.0, None],
            "horizon": 1,
            "search_mode": "greedy",
        }
    ]

    with pytest.raises(ValueError, match="legacy planner trace"):
        _migrate(record)

    migrated = migrate_trajectory_record(
        record,
        missing_action_space_id="navigation",
        missing_action_space_version=1,
        missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
        legacy_planner_semantics="distillation_world_model",
    )
    action_training = migrated["planner_policy_traces"][0]["action_training"]
    assert action_training == {
        "objective": "distillation",
        "behavior_owner": "world_model",
        "executed_action_index": 0,
        "behavior_action_log_probs": [0.0, None],
        "teacher_action_log_probs": [0.0, None],
        "sampled_action_index": None,
    }


def test_jsonl_migration_writes_auditable_manifest_without_overwrite(tmp_path) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "structured.jsonl"
    source.write_text(json.dumps(_legacy_record()) + "\n", encoding="utf-8")

    manifest = migrate_trajectory_jsonl(
        source_path=source,
        output_path=output,
        missing_action_space_id="navigation",
        missing_action_space_version=1,
        missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
    )

    migrated = json.loads(output.read_text(encoding="utf-8"))
    assert migrated["record_format"] == TRAJECTORY_RECORD_FORMAT
    assert manifest["record_count"] == 1
    assert manifest["source_sha256"] != manifest["output_sha256"]
    assert output.with_suffix(".jsonl.manifest.json").is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        migrate_trajectory_jsonl(
            source_path=source,
            output_path=output,
            missing_action_space_id="navigation",
            missing_action_space_version=1,
            missing_reward_provenance=TRAJECTORY_REWARD_PROVENANCE,
        )
