from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
        "<think><observation>hallway</observation><reasoning>inspect the hallway</reasoning><prediction>closer view</prediction></think><answer>moveahead</answer>",
        "<think><observation>doorway</observation><reasoning>turn toward the doorway</reasoning><prediction>new view</prediction></think><answer>rotateright</answer>",
    ]
    terminal_response = (
        "<think><observation>target visible</observation><reasoning>the target is now visible</reasoning><prediction>closer target</prediction></think><answer>moveleft</answer>"
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

    def policy_request(
        kind: str,
        response: str,
        token_ids: list[int],
        message_window: list[dict[str, str]],
    ) -> dict[str, object]:
        rendered_prompt = f"rendered-{kind}-{token_ids[0]}"
        return {
            "kind": kind,
            "message_window": message_window,
            "first_observation_index": 0,
            "rendered_prompt": rendered_prompt,
            "rendered_prompt_sha256": hashlib.sha256(
                rendered_prompt.encode()
            ).hexdigest(),
            "response": response,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            "finish_reason": "stop",
            "stop_reason": None,
            "token_ids": token_ids,
            "eos_token_id": token_ids[-1],
        }

    record = {
        "record_format": "vagen_step60_source_trajectory_v3",
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
                "parsed_response": vagen_step60_data.parse_source_response(response),
                "environment_extracted_actions": [
                    vagen_step60_data.parse_source_response(response)["action_name"]
                ],
                "reward": reward,
                "done": done,
                "info": {"task_success": done},
                "generation_exclusion_reason": None,
            }
            for response, reward, done in zip(
                ordinary_responses,
                (0.02, 10.02),
                (False, True),
                strict=True,
            )
        ],
        "rewards": [0.02, 10.02],
        "environment_reward_events": [0.02, 10.02],
        "success": True,
        "reward": 10.04,
        "reward_provenance": "step_rewards",
        "environment_done": True,
        "terminated": True,
        "truncated": False,
        "unavailable_source_commit": "fee3ffac036a599b0ae979a6dd1ce2b21f7dec49",
        "reconstruction_identity": {"runtime_head": "a" * 40},
        "source_runtime_contract": {
            "format": "vagen_step60_reconstruction_runtime_contract_v3",
            "reconstruction_identity": {"runtime_head": "a" * 40},
            "source_generation_package_evidence": {
                "packages": {
                    "vllm": "0.8.5.post1",
                    "transformers": "4.49.0",
                    "torch": "2.6.0",
                },
                "evidence": "source_wandb_requirements_2q620nss",
            },
            "executable_generation_packages": {
                "vllm": "0.8.2",
                "transformers": "4.49.0",
                "torch": "2.6.0",
            },
            "reward_provenance": "step_rewards",
            "trajectory_reward_info_key": None,
        },
        "policy_artifact": {"artifact_manifest_sha256": "a" * 64},
        "policy_runtime_contract": {
            "backend": "fixture",
            "source_generation_package_evidence": {
                "packages": {
                    "vllm": "0.8.5.post1",
                    "transformers": "4.49.0",
                    "torch": "2.6.0",
                },
                "evidence": "source_wandb_requirements_2q620nss",
            },
            "executable_generation_packages": {
                "vllm": "0.8.2",
                "transformers": "4.49.0",
                "torch": "2.6.0",
            },
            "package_versions": {
                "vllm": "0.8.2",
                "transformers": "4.49.0",
                "torch": "2.6.0",
            },
        },
        "policy_requests": [
            policy_request("ordinary", ordinary_responses[0], [101, 102], messages[:2]),
            policy_request("ordinary", ordinary_responses[1], [111, 112], messages[:4]),
            policy_request("terminal", terminal_response, [201, 202], messages[:-1]),
        ],
        "conversion_eligible": True,
        "exclusion_reasons": [],
        "terminal_generation": {
            "assistant_response": terminal_response,
            "parsed": vagen_step60_data.parse_source_response(terminal_response),
            "finish_reason": "stop",
            "stop_reason": None,
            "token_ids": [201, 202],
            "eos_token_id": 202,
            "generation_exclusion_reason": None,
            "executed": False,
            "environment_step_after_generation": False,
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


def _publication_source(path: Path, marker: str = "COMPLETE") -> Path:
    path.mkdir()
    (path / "payload").write_text("new", encoding="utf-8")
    (path / marker).write_text("ready\n", encoding="utf-8")
    return path


def test_published_partition_loader_rehashes_sibling_parquets(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "partition"
    directory.mkdir()
    manifest = vagen_step60_data.build_partition_manifest(
        _source_rows(),
        source_path="/source/train.parquet",
        source_sha256=vagen_step60_data.SOURCE_TRAIN_SHA256,
    )
    manifest["source"]["size_bytes"] = 1
    for batch in manifest["batches"]:
        name = f"batch_{int(batch['batch']):02d}.parquet"
        parquet = directory / name
        parquet.write_bytes(bytes([int(batch["batch"])]))
        batch["parquet"] = name
        batch["parquet_sha256"] = vagen_step60_data._file_sha256(parquet)
        batch["parquet_size_bytes"] = parquet.stat().st_size
    manifest["manifest_payload_sha256"] = (
        vagen_step60_data.partition_manifest_payload_sha256(manifest)
    )
    marker = directory / "partition_manifest.json"
    marker.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = vagen_step60_data.load_published_partition_manifest(marker)
    assert loaded["checks"]["batch1_train_count"] == 1_800

    (directory / "batch_01.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="parquet (size|hash) mismatch"):
        vagen_step60_data.load_published_partition_manifest(marker)


def test_reserved_publication_never_replaces_existing_target(
    tmp_path: Path,
) -> None:
    source = _publication_source(tmp_path / "source")
    target = tmp_path / "target"
    target.mkdir()
    (target / "existing").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        vagen_step60_data.publish_reserved_directory(
            source,
            target,
            readiness_marker="COMPLETE",
        )

    assert (source / "payload").read_text(encoding="utf-8") == "new"
    assert (target / "existing").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("target_kind", ["file", "symlink"])
def test_reserved_publication_rejects_lexical_existing_targets(
    tmp_path: Path,
    target_kind: str,
) -> None:
    source = _publication_source(tmp_path / "source")
    target = tmp_path / "target"
    if target_kind == "file":
        target.write_text("keep", encoding="utf-8")
    else:
        target.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        vagen_step60_data.publish_reserved_directory(
            source,
            target,
            readiness_marker="COMPLETE",
        )
    assert target.is_symlink() if target_kind == "symlink" else target.is_file()


def test_reserved_publication_requires_real_sibling_staging(tmp_path: Path) -> None:
    real = _publication_source(tmp_path / "real")
    staging = tmp_path / "staging"
    staging.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="staging"):
        vagen_step60_data.publish_reserved_directory(
            staging,
            tmp_path / "target",
            readiness_marker="COMPLETE",
        )


@pytest.mark.parametrize(
    "marker_problem", ["missing", "symlink", "directory", "nested", "other"]
)
def test_reserved_publication_rejects_invalid_marker_ownership(
    tmp_path: Path,
    marker_problem: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    if marker_problem == "symlink":
        (source / "COMPLETE").symlink_to(source / "payload")
    elif marker_problem == "directory":
        (source / "COMPLETE").mkdir()
    elif marker_problem == "nested":
        nested = source / "nested"
        nested.mkdir()
        (nested / "COMPLETE").write_text("ready\n", encoding="utf-8")
    elif marker_problem == "other":
        (source / "COMPLETE").write_text("ready\n", encoding="utf-8")
        (source / "partition_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="marker"):
        vagen_step60_data.publish_reserved_directory(
            source,
            tmp_path / "target",
            readiness_marker="COMPLETE",
        )
    assert not (tmp_path / "target").exists()


def test_reserved_publication_publishes_marker_last(tmp_path: Path) -> None:
    source = _publication_source(tmp_path / "source")
    target = tmp_path / "target"
    vagen_step60_data.publish_reserved_directory(
        source,
        target,
        readiness_marker="COMPLETE",
    )
    assert not source.exists()
    assert (target / "payload").read_text(encoding="utf-8") == "new"
    vagen_step60_data.validate_published_directory(
        target,
        readiness_marker="COMPLETE",
    )


def test_reserved_publication_allows_only_one_concurrent_winner(
    tmp_path: Path,
) -> None:
    sources = [
        _publication_source(tmp_path / f"source-{index}") for index in range(2)
    ]
    target = tmp_path / "target"

    def publish(source: Path) -> str:
        try:
            vagen_step60_data.publish_reserved_directory(
                source,
                target,
                readiness_marker="COMPLETE",
            )
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, sources))
    assert sorted(outcomes) == ["lost", "won"]
    vagen_step60_data.validate_published_directory(
        target,
        readiness_marker="COMPLETE",
    )


@pytest.mark.parametrize("failure_point", ["payload", "marker"])
def test_reserved_publication_retains_interrupted_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    source = _publication_source(tmp_path / "source")
    target = tmp_path / "target"
    real_rename = vagen_step60_data.os.rename

    def fail_selected(old: Path, new: Path) -> None:
        is_payload_failure = failure_point == "payload" and Path(old).name == "payload"
        is_marker_failure = failure_point == "marker" and Path(new).name == "COMPLETE"
        if is_payload_failure or is_marker_failure:
            raise OSError("injected publication failure")
        real_rename(old, new)

    monkeypatch.setattr(vagen_step60_data.os, "rename", fail_selected)
    with pytest.raises(OSError, match="injected"):
        vagen_step60_data.publish_reserved_directory(
            source,
            target,
            readiness_marker="COMPLETE",
        )
    assert target.is_dir()
    assert not (target / "COMPLETE").exists()
    if failure_point == "payload":
        assert (target / vagen_step60_data.PUBLISHING_SENTINEL).is_file()
    else:
        assert not (target / vagen_step60_data.PUBLISHING_SENTINEL).exists()
    with pytest.raises(ValueError):
        vagen_step60_data.validate_published_directory(
            target,
            readiness_marker="COMPLETE",
        )


def test_readiness_rename_is_the_final_fallible_publication_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _publication_source(tmp_path / "source")
    target = tmp_path / "target"
    committed = False
    real_rename = vagen_step60_data.os.rename
    real_fsync = vagen_step60_data._fsync_directory

    def track_rename(old: Path, new: Path) -> None:
        nonlocal committed
        real_rename(old, new)
        if Path(new).name == "COMPLETE":
            committed = True

    def reject_post_commit_fsync(path: Path) -> None:
        if committed:
            raise AssertionError("fallible operation occurred after readiness commit")
        real_fsync(path)

    monkeypatch.setattr(vagen_step60_data.os, "rename", track_rename)
    monkeypatch.setattr(vagen_step60_data, "_fsync_directory", reject_post_commit_fsync)
    vagen_step60_data.publish_reserved_directory(
        source,
        target,
        readiness_marker="COMPLETE",
    )
    assert committed


def test_partition_rejects_dangling_output_before_source_read(tmp_path: Path) -> None:
    output = tmp_path / "partition"
    output.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        vagen_step60_data.partition_source_parquet(tmp_path / "absent.parquet", output)
    assert output.is_symlink()
    assert not (tmp_path / "missing").exists()


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
    assert audit["contract_version"] == (
        "vagen_step60_reconstruction_audit_v3"
    )
    assert audit["reward_provenance"] == "step_rewards"
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
    source["reward"] = 99.0
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="aggregate reward does not equal step rewards"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["reward_provenance"] = "trajectory_terminal_reward"
    source["source_runtime_contract"]["reward_provenance"] = (
        "trajectory_terminal_reward"
    )
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="must be step_rewards"):
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
    with pytest.raises(ValueError, match="parser/success class"):
        vagen_step60_data.convert_source_record(source, latent_token_count=16)

    source = _source_record()
    source["source_runtime_contract"]["reward_provenance"] = "invalid"
    _rehash_source_record(source)
    with pytest.raises(ValueError, match="row/runtime reward provenance"):
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
