from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from experiments.training.sft1 import vagen_step60_data


def _source_row(
    eval_set: str,
    seed: int,
    *,
    split: str = "train",
) -> dict[str, object]:
    return {
        "data_source": "navigation",
        "prompt": [{"role": "user", "content": "<image>"}],
        "extra_info": {
            "env_name": "navigation",
            "seed": seed,
            "split": split,
            "env_config": {
                "render_mode": "vision",
                "prompt_format": "grounding_worldmodeling",
                "use_state_reward": False,
                "eval_set": eval_set,
                "max_actions_per_step": 1,
                "format_reward": 0.02,
                "invalid_action_penalty": -0.2,
                "success_threshold": 1.5,
            },
        },
    }


def _source_rows() -> list[dict[str, object]]:
    seeds = list(range(10_000))
    return [
        *(_source_row("base", seed) for seed in seeds),
        *(_source_row("common_sense", seed) for seed in seeds),
    ]


def _source_record() -> dict[str, object]:
    source_prompt = (
        "Navigate to the target. Respond in this format:\n"
        "<think>...</think><answer>one_action</answer>"
    )
    ordinary_responses = [
        "<think>inspect the hallway</think><answer>moveahead</answer>",
        "<think>turn toward the doorway</think><answer>rotateright</answer>",
    ]
    terminal_response = (
        "<think>the target is now visible</think><answer>moveleft</answer>"
    )
    messages = [
        {"role": "system", "content": source_prompt},
        {
            "role": "user",
            "content": (
                "first observation <image>\n"
                "Human Instruction: navigate to the target"
            ),
        },
        {"role": "assistant", "content": ordinary_responses[0]},
        {"role": "user", "content": "second observation <image>"},
        {"role": "assistant", "content": ordinary_responses[1]},
        {"role": "user", "content": "terminal observation <image>"},
        {"role": "assistant", "content": terminal_response},
    ]
    record = {
        "record_format": "vagen_step60_source_trajectory_v1",
        "id": "batch1/base/000000",
        "batch": 1,
        "split": "train",
        "source_index": 0,
        "source_key": "base:0",
        "eval_set": "base",
        "seed": 0,
        "system_prompt": source_prompt,
        "messages": messages,
        "observation_texts": [
            "first observation <image>\nHuman Instruction: navigate to the target",
            "second observation <image>",
            "terminal observation <image>",
        ],
        "assistant_responses": ordinary_responses,
        "executed_action_names": ["moveahead", "rotateright"],
        "image_paths": ["first.png", "second.png", "terminal.png"],
        "image_artifacts": [
            {"path": name, "size_bytes": 1, "sha256": str(index) * 64}
            for index, name in enumerate(
                ("first.png", "second.png", "terminal.png"),
                start=1,
            )
        ],
        "turns": [
            {
                "response": response,
                "reward": reward,
                "done": done,
                "info": ({"episode_reward": 1.0} if done else {}),
            }
            for response, reward, done in zip(
                ordinary_responses,
                (0.0, 1.0),
                (False, True),
                strict=True,
            )
        ],
        "rewards": [],
        "environment_reward_events": [0.0, 1.0],
        "success": True,
        "reward": 1.0,
        "reward_provenance": "trajectory_terminal_reward",
        "environment_done": True,
        "source_runtime_commit": "fee3ffac036a599b0ae979a6dd1ce2b21f7dec49",
        "source_runtime_contract": {
            "format": "fixture",
            "reward_provenance": "trajectory_terminal_reward",
            "trajectory_reward_info_key": "episode_reward",
        },
        "policy_artifact": {"artifact_manifest_sha256": "a" * 64},
        "policy_runtime_contract": {"backend": "fixture"},
        "policy_requests": [],
        "terminal_generation": {
            "assistant_response": terminal_response,
            "draft_action_name": "moveleft",
            "format_valid": True,
            "executed": False,
        },
    }
    record["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return record


def _rehash_source_record(record: dict[str, object]) -> None:
    payload = {
        key: value for key, value in record.items() if key != "raw_record_sha256"
    }
    record["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_partition_manifest_covers_ten_balanced_disjoint_batches() -> None:
    manifest = vagen_step60_data.build_partition_manifest(
        _source_rows(),
        source_path="/source/train.parquet",
        source_sha256="3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6",
    )

    entries = manifest["rows"]
    assert len(entries) == 20_000
    assert {entry["source_index"] for entry in entries} == set(range(20_000))
    assert Counter(entry["batch"] for entry in entries) == Counter(
        {batch: 2_000 for batch in range(1, 11)}
    )
    for batch in range(1, 11):
        selected = [entry for entry in entries if entry["batch"] == batch]
        assert Counter(entry["eval_set"] for entry in selected) == {
            "base": 1_000,
            "common_sense": 1_000,
        }


def test_batch1_internal_heldout_keeps_shared_seeds_together() -> None:
    manifest = vagen_step60_data.build_partition_manifest(
        _source_rows(),
        source_path="/source/train.parquet",
        source_sha256="3c8161bd45adc4cde5d67157cf4db225753ed3925cb9a52e3a57d1dd11dbe9d6",
    )
    batch1 = [entry for entry in manifest["rows"] if entry["batch"] == 1]
    counts = Counter(entry["dataset_split"] for entry in batch1)

    assert counts == {"train": 1_800, "heldout": 200}
    train_seeds = {
        entry["seed"] for entry in batch1 if entry["dataset_split"] == "train"
    }
    heldout_seeds = {
        entry["seed"] for entry in batch1 if entry["dataset_split"] == "heldout"
    }
    assert train_seeds.isdisjoint(heldout_seeds)
    for seed in range(1_000):
        rows = [entry for entry in batch1 if entry["seed"] == seed]
        assert len(rows) == 2
        assert len({entry["dataset_split"] for entry in rows}) == 1


def test_partition_rejects_source_hash_or_order_drift() -> None:
    rows = _source_rows()
    with pytest.raises(ValueError, match="SHA256 drift"):
        vagen_step60_data.build_partition_manifest(
            rows,
            source_path="/source/train.parquet",
            source_sha256="0" * 64,
        )

    rows[0], rows[10_000] = rows[10_000], rows[0]
    with pytest.raises(ValueError, match="row order drift"):
        vagen_step60_data.build_partition_manifest(
            rows,
            source_path="/source/train.parquet",
            source_sha256=(
                "3c8161bd45adc4cde5d67157cf4db2257"
                "53ed3925cb9a52e3a57d1dd11dbe9d6"
            ),
        )


def test_overlapping_source_test_cannot_be_classified_as_heldout() -> None:
    train = _source_rows()
    overlapping_test = [
        _source_row("base", seed, split="test") for seed in range(64)
    ] + [
        _source_row("common_sense", seed, split="test") for seed in range(64)
    ]

    evidence = vagen_step60_data.measure_identity_overlap(
        train, overlapping_test
    )
    assert evidence["eval_set_seed_overlap_count"] == 128
    assert evidence["seed_overlap_count"] == 64
    with pytest.raises(ValueError, match="not held-out"):
        vagen_step60_data.require_nonoverlapping_heldout(
            train,
            overlapping_test,
            candidate_name="source test.parquet",
        )


def test_partition_consumer_recomputes_contract_and_rejects_tampering() -> None:
    manifest = vagen_step60_data.build_partition_manifest(
        _source_rows(),
        source_path="/source/train.parquet",
        source_sha256=vagen_step60_data.SOURCE_TRAIN_SHA256,
    )
    manifest["source"]["size_bytes"] = 123
    for batch in manifest["batches"]:
        batch["parquet"] = f"batch_{batch['batch']:02d}.parquet"
        batch["parquet_sha256"] = "a" * 64
        batch["parquet_size_bytes"] = 1
    manifest["manifest_payload_sha256"] = (
        vagen_step60_data.partition_manifest_payload_sha256(manifest)
    )
    vagen_step60_data.validate_partition_manifest(
        manifest,
        require_published=True,
    )

    manifest["rows"][0]["seed"] = 999_999
    with pytest.raises(ValueError, match="source key drift|ordered seeds|row hash"):
        vagen_step60_data.validate_partition_manifest(
            manifest,
            require_published=True,
        )


def test_atomic_directory_publish_never_replaces_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        vagen_step60_data.atomic_publish_directory(source, target)

    assert (source / "payload").read_text(encoding="utf-8") == "new"
    assert (target / "existing").read_text(encoding="utf-8") == "keep"


def test_conversion_is_k16_compatible_and_preserves_verbatim_source_chat() -> None:
    source = _source_record()
    converted = vagen_step60_data.convert_source_record(
        source, latent_token_count=16
    )

    for view_name in ("sft1", "sft2"):
        view = converted[view_name]
        assert "<answer>" not in view["system_prompt"]
        assert "<|action_start|>" in view["system_prompt"]
        assistant_text = "".join(
            message["content"]
            for message in view.get("messages", [])
            if message["role"] == "assistant"
        ) or "".join(view.get("assistant_responses", []))
        assert "inspect the hallway" in assistant_text
        assert "turn toward the doorway" in assistant_text
        assert assistant_text.count("<|latent_state|>") == 2
        assert assistant_text.count("<|latent_state_") == 30
        assert "<answer>" not in assistant_text

    assert converted["sft2"]["observation_texts"] == source[
        "observation_texts"
    ]
    audit = converted["source_audit"]
    assert audit["system_prompt"] == source["system_prompt"]
    assert audit["messages"] == source["messages"]
    assert audit["terminal_assistant_response"] == source["terminal_generation"][
        "assistant_response"
    ]
    assert audit["source_sha256"] == source["raw_record_sha256"]
    assert audit["source_identity"] == {
        "source_index": 0,
        "source_key": "base:0",
        "eval_set": "base",
        "seed": 0,
        "batch": 1,
        "split": "train",
    }
    assert audit["reward_provenance"] == "trajectory_terminal_reward"
    assert audit["policy_artifact"] == source["policy_artifact"]
    assert converted["sft1"]["source_audit"] == audit
    assert converted["sft2"]["source_audit"] == audit
    assert converted["sft1"]["conversion_provenance"]["source_sha256"] == audit[
        "source_sha256"
    ]
    assert converted["sft2"]["conversion_provenance"]["source_sha256"] == audit[
        "source_sha256"
    ]


def test_terminal_draft_is_audited_but_not_supervised_or_executed() -> None:
    from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory
    from nimloth.rollout.transitions import expand_record_transitions

    source = _source_record()
    converted = vagen_step60_data.convert_source_record(
        source, latent_token_count=16
    )
    sft1 = converted["sft1"]
    sft2 = converted["sft2"]

    supervised_assistant = [
        message["content"]
        for message in sft1["messages"]
        if message["role"] == "assistant"
    ]
    assert len(supervised_assistant) == 2
    assert all("target is now visible" not in text for text in supervised_assistant)

    assert len(sft2["assistant_responses"]) == 2
    assert len(sft2["action_indices"]) == 2
    assert "target is now visible" in sft2["terminal_assistant_prefix"]
    assert sft2["terminal_assistant_prefix"].endswith("<|action_start|>")
    assert sft2["terminal_generation_audit"] == {
        "source_response": source["terminal_generation"]["assistant_response"],
        "draft_action_name": "moveleft",
        "draft_action_index": 3,
        "format_valid": True,
        "executed": False,
        "environment_step_after_generation": False,
    }

    trajectory = RolloutTrajectory.from_record(sft2)
    validate_rollout_trajectory(trajectory)
    assert trajectory.to_record()["source_identity"] == sft2["source_identity"]
    assert len(expand_record_transitions(sft2)) == 2
    assert sft2["policy_credit_assignment"] == "none"
    assert sft2["action_log_probs"] == []


def test_conversion_rejects_identity_or_step_reward_drift() -> None:
    source = _source_record()
    source.pop("source_key")
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="no source_key"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["reward_provenance"] = "step_rewards"
    source["source_runtime_contract"]["reward_provenance"] = "step_rewards"
    source["source_runtime_contract"]["trajectory_reward_info_key"] = None
    source["rewards"] = [0.0, 1.0]
    source["reward"] = 99.0
    source["terminated"] = True
    source["truncated"] = False
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="aggregate reward does not equal step rewards"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["rewards"] = [1.0]
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="cannot expose values as step rewards"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["reward"] = float("nan")
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="aggregate reward is non-finite"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["environment_reward_events"][0] = float("nan")
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="environment reward events"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["turns"][0]["reward"] = 2.0
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="turn rewards and environment events"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["source_runtime_contract"]["reward_provenance"] = "step_rewards"
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="row/runtime reward provenance"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["turns"][-1]["info"]["episode_reward"] = 2.0
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="differs from terminal info"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)


def test_unavailable_behavior_provenance_requires_verified_offline_contract() -> None:
    from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory

    sft2 = vagen_step60_data.convert_source_record(
        _source_record(),
        latent_token_count=16,
    )["sft2"]
    sft2["conversion_provenance"]["format"] = "unverified"
    trajectory = RolloutTrajectory.from_record(sft2)

    with pytest.raises(ValueError, match="verified offline source conversion"):
        validate_rollout_trajectory(trajectory)


def test_nonempty_shard_without_complete_manifest_is_rejected(tmp_path: Path) -> None:
    shard = tmp_path / "shard_000"
    shard.mkdir()
    (shard / "raw.jsonl").write_text(
        json.dumps(_source_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="COMPLETE"):
        vagen_step60_data.validate_complete_shard(
            shard, expected_source_indices={0}
        )
