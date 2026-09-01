from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from experiments.training.sft1 import vagen_step60_collect as collect_module
from experiments.training.sft1.vagen_step60_collect import (
    EpisodeSpec,
    GeneratedTurn,
    SourceShardCollector,
    parse_source_response,
    windowed_source_messages,
)
from experiments.training.sft1.vagen_step60_data import validate_complete_shard


def _image(offset: int) -> Image.Image:
    image = Image.new("RGB", (4, 4), color=(offset, 0, 0))
    image.putpixel((0, 0), (255, 255, 255))
    return image


def _initial_prompt(instruction: str) -> str:
    return (
        "[Initial Observation]:\n<image>\n"
        f"Human Instruction: {instruction}\n"
        "Decide your next action."
    )


def _reconstruction_identity() -> dict[str, object]:
    return {
        "base_commit": collect_module.RECONSTRUCTION_BASE_COMMIT,
        "runtime_head": collect_module.APPROVED_RECONSTRUCTION_HEAD,
        "runtime_parent": collect_module.RECONSTRUCTION_BASE_COMMIT,
        "runtime_tree": collect_module.APPROVED_RECONSTRUCTION_TREE,
        "commit_count": 1,
        "parent_count": 1,
        "diff_sha256": collect_module.APPROVED_RECONSTRUCTION_DIFF_SHA256,
        "git_version": "git version test",
    }


def _source_runtime_evidence(
    *,
    reward_provenance: str = "step_rewards",
) -> dict[str, object]:
    contract = {
        "format": collect_module.SOURCE_RUNTIME_CONTRACT_FORMAT,
        "runtime_root": "/source/VAGEN",
        "unavailable_source_commit": collect_module.SOURCE_VAGEN_COMMIT,
        "reconstruction_identity": _reconstruction_identity(),
        "evidence_artifact": {
            "sha256": collect_module.RECONSTRUCTION_EVIDENCE_FILE_SHA256,
            "manifest_sha256": (
                collect_module.RECONSTRUCTION_EVIDENCE_MANIFEST_SHA256
            ),
        },
        "environment_assets": collect_module.RECONSTRUCTION_ENVIRONMENT_ASSETS,
        "service_api_contract": "legacy_batch_environment_v1",
        "service_routes": collect_module.RECONSTRUCTION_SERVICE_ROUTES,
        "reward_provenance": reward_provenance,
        "trajectory_reward_info_key": None,
        "http_timeout_seconds": 500,
        "prompt_hashes": {
            "system_prompt_sha256": collect_module.SOURCE_SYSTEM_PROMPT_SHA256,
            "initial_prompt_normalized_sha256": (
                collect_module.SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256
            ),
            "step_prompt_normalized_sha256": (
                collect_module.SOURCE_STEP_PROMPT_NORMALIZED_SHA256
            ),
        },
        "environment_config": {
            **collect_module.SOURCE_ENV_BASE_CONFIG,
            "step_length": 0.5,
            "success_reward": 10.0,
            "action_names": list(collect_module.SOURCE_ACTION_NAMES),
        },
    }
    contract["contract_payload_sha256"] = (
        collect_module.source_runtime_contract_payload_sha256(contract)
    )
    return contract


def _policy_artifact_evidence() -> dict[str, str]:
    return {
        "merge_manifest_path": "/policy/nimloth_merge_manifest.json",
        "merge_manifest_file_sha256": "a" * 64,
        "merge_manifest_payload_sha256": "b" * 64,
        "artifact_manifest_sha256": "c" * 64,
        "model_config_artifacts": {
            "config.json": {"size_bytes": 1, "sha256": "d" * 64},
            "tokenizer_config.json": {"size_bytes": 1, "sha256": "e" * 64},
        },
        "source_actor_dir": "/source/global_step_60/actor",
    }


def _step_prompt(action: str | None) -> str:
    extracted = [] if action is None else [action]
    return (
        f"After your answer, the extracted valid action is {extracted!r}.\n"
        "The environment feedback is: Last action is executed successfully.\n"
        "reward: 10.02\ndone: 1.0\nAfter that, the observation is:\n"
        "<image>\nHuman Instruction: task\nDecide your next action."
    )


class _FakePolicy:
    def __init__(self) -> None:
        self.runtime_contract = {
            **collect_module.SOURCE_SAMPLING_CONTRACT,
            "backend": "vllm",
            "engine_seed": 7,
            "tokenizer_eos_token_id": 102,
            "package_versions": {
                "vllm": "0.8.5.post1",
                "transformers": "4.49.0",
                "torch": "2.6.0",
            },
            "model_config_artifacts": {
                "config.json": {"size_bytes": 1, "sha256": "d" * 64},
                "tokenizer_config.json": {
                    "size_bytes": 1,
                    "sha256": "e" * 64,
                },
            },
        }
        self.calls: list[list[list[dict[str, str]]]] = []

    def generate(self, requests):
        self.calls.append([messages for messages, _images in requests])
        terminal = len(self.calls) == 2
        rows = []
        for index, (_messages, _images) in enumerate(requests):
            action = "moveleft" if terminal else (
                "moveahead" if index == 0 else "rotateright"
            )
            response = (
                f"<think><observation>observation {index}</observation>"
                f"<reasoning>real {'terminal ' if terminal else ''}reasoning {index}</reasoning>"
                f"<prediction>prediction {index}</prediction></think>"
                f"<answer>{action}</answer>"
            )
            rows.append(
                GeneratedTurn(
                    response=response,
                    rendered_prompt=f"rendered-{len(self.calls)}-{index}",
                    finish_reason="stop",
                    stop_reason=None,
                    token_ids=(101, 102),
                    eos_token_id=102,
                    prompt_tokens=10,
                    completion_tokens=5,
                )
            )
        return rows


class _InvalidOrdinaryPolicy(_FakePolicy):
    def generate(self, requests):
        rows = super().generate(requests)
        if len(self.calls) == 1:
            invalid = (
                "<think><observation>observation</observation>"
                "<reasoning>reasoning</reasoning>"
                "<prediction>prediction</prediction></think><answer>stay</answer>"
            )
            return [replace(row, response=invalid) for row in rows]
        return rows


class _LengthOrdinaryPolicy(_FakePolicy):
    def generate(self, requests):
        rows = super().generate(requests)
        if len(self.calls) == 1:
            return [replace(row, finish_reason="length") for row in rows]
        return rows


class _LengthTerminalPolicy(_FakePolicy):
    def generate(self, requests):
        rows = super().generate(requests)
        if len(self.calls) == 2:
            return [replace(row, finish_reason="length") for row in rows]
        return rows


class _FakeClient:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.configs = {}
        self.step_payloads: list[dict[str, str]] = []
        self.closed: list[str] = []

    def create_environments_batch(self, ids2configs):
        self.configs.update(ids2configs)

    def reset_batch(self, ids2seeds):
        return {
            env_id: (
                {
                    "obs_str": _initial_prompt("task"),
                    "multi_modal_data": {"<image>": [_image(index + 1)]},
                },
                {"seed": seed},
            )
            for index, (env_id, seed) in enumerate(ids2seeds.items())
        }

    def get_system_prompts_batch(self, env_ids):
        return {env_id: self.system_prompt for env_id in env_ids}

    def step_batch(self, ids2actions):
        self.step_payloads.append(dict(ids2actions))
        rows = {}
        for index, (env_id, response) in enumerate(ids2actions.items()):
            parsed = parse_source_response(response)
            action = parsed["action_name"] if parsed["format_valid"] else None
            valid = action is not None
            rows[env_id] = (
                {
                    "obs_str": _step_prompt(action),
                    "multi_modal_data": {"<image>": [_image(index + 10)]},
                },
                10.02 if valid else -0.2,
                valid,
                {
                    "task_success": valid,
                    "last_action_success": True,
                    "episode_reward": 10.02,
                },
            )
        return rows

    def close_batch(self, env_ids=None):
        self.closed.extend(sorted(env_ids or self.configs))


def test_source_response_parser_is_strict_and_preserves_real_thought() -> None:
    parsed = parse_source_response(
        "<think><observation>real</observation><reasoning>reason</reasoning>"
        "<prediction>prediction</prediction></think><answer>moveahead</answer>"
    )
    assert parsed == {
        "format_valid": True,
        "thought": (
            "<observation>real</observation><reasoning>reason</reasoning>"
            "<prediction>prediction</prediction>"
        ),
        "action_name": "moveahead",
        "action_index": 0,
    }
    assert not parse_source_response(
        "prefix <think><observation>real</observation><reasoning>reason</reasoning>"
        "<prediction>prediction</prediction></think><answer>moveahead</answer>"
    )["format_valid"]
    assert not parse_source_response(
        "<think><observation>real</observation><reasoning>reason</reasoning>"
        "<prediction>prediction</prediction></think><answer>unknown</answer>"
    )["format_valid"]
    assert not parse_source_response(
        "<think>plain text</think><answer>moveahead</answer>"
    )["format_valid"]
    assert not parse_source_response(
        "<think><observation>real</observation><reasoning>reason</reasoning>"
        "<prediction>prediction</prediction></think><answer> moveahead </answer>"
    )["format_valid"]
    assert not parse_source_response(
        "<think><observation></observation><reasoning>reason</reasoning>"
        "<prediction>prediction</prediction></think><answer>moveahead</answer>"
    )["format_valid"]


def test_source_window_keeps_five_history_turns_plus_current_observation() -> None:
    messages = [{"role": "system", "content": "system"}]
    for turn in range(7):
        messages.extend(
            [
                {"role": "user", "content": f"observation {turn} <image>"},
                {"role": "assistant", "content": f"response {turn}"},
            ]
        )
    messages.append({"role": "user", "content": "current <image>"})

    window, first_turn = windowed_source_messages(messages, window_size=5)

    assert first_turn == 2
    assert window[0] == {"role": "system", "content": "system"}
    assert window[1]["content"] == "observation 2 <image>"
    assert window[-1]["content"] == "current <image>"
    assert sum(message["content"].count("<image>") for message in window) == 6


def test_collector_saves_terminal_generation_without_environment_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    initial_normalized = _initial_prompt("<INSTRUCTION>")
    monkeypatch.setattr(
        collect_module,
        "SOURCE_SYSTEM_PROMPT_SHA256",
        hashlib.sha256(system_prompt.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(initial_normalized.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_STEP_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(
            collect_module.normalized_step_prompt(_step_prompt("moveahead")).encode()
        ).hexdigest(),
    )
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    client = _FakeClient(system_prompt)
    policy = _FakePolicy()
    collector = SourceShardCollector(
        client=client,
        policy=policy,
        run_id="unit",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=2,
    )
    output = tmp_path / "shard_000"

    manifest = collector.collect(specs, output_dir=output, max_steps=20)

    assert manifest["counts"] == {
        "records": 2,
        "eligible": 2,
        "excluded": 0,
        "transitions": 2,
        "images": 4,
        "terminal_generations": 2,
        "terminal_environment_steps": 0,
    }
    assert len(client.step_payloads) == 1
    assert len(client.step_payloads[0]) == 2
    assert len(policy.calls) == 2  # ordinary batch, then terminal batch
    assert len(set(client.configs)) == 2
    assert all(
        config["env_name"] == "navigation"
        and config["env_config"]["prompt_format"]
        == collect_module.RECONSTRUCTION_MODE
        for config in client.configs.values()
    )
    rows = [
        json.loads(line)
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        assert len(row["assistant_responses"]) == 1
        assert len(row["turns"]) == 1
        assert len(row["observation_texts"]) == 2
        assert len(row["image_paths"]) == 2
        assert len(row["image_artifacts"]) == 2
        assert len(row["messages"]) == 5
        assert "real terminal reasoning" in row["messages"][-1]["content"]
        assert row["terminal_generation"]["executed"] is False
        assert row["terminal_generation"]["environment_step_after_generation"] is False
        assert row["terminated"] is True
        assert row["truncated"] is False
        assert row["unavailable_source_commit"] == collect_module.SOURCE_VAGEN_COMMIT
        assert row["reconstruction_identity"] == _reconstruction_identity()
        assert row["system_prompt"] == system_prompt
        assert len(row["policy_requests"]) == 2
        assert all(request["rendered_prompt"] for request in row["policy_requests"])
        raw_payload = {
            key: value for key, value in row.items() if key != "raw_record_sha256"
        }
        assert row["raw_record_sha256"] == hashlib.sha256(
            json.dumps(
                raw_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    validate_complete_shard(output, expected_source_indices={0, 10_000})


def test_invalid_ordinary_response_publishes_a_complete_excluded_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    monkeypatch.setattr(
        collect_module,
        "SOURCE_SYSTEM_PROMPT_SHA256",
        hashlib.sha256(system_prompt.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(_initial_prompt("<INSTRUCTION>").encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_STEP_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(
            collect_module.normalized_step_prompt(_step_prompt("moveahead")).encode()
        ).hexdigest(),
    )
    collector = SourceShardCollector(
        client=_FakeClient(system_prompt),
        policy=_InvalidOrdinaryPolicy(),
        run_id="invalid-ordinary",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )
    output = tmp_path / "invalid-ordinary-shard"
    manifest = collector.collect(
        [EpisodeSpec(0, "base", 100, "train", "base:100")],
        output_dir=output,
    )
    row = json.loads((output / "raw.jsonl").read_text(encoding="utf-8"))
    assert manifest["counts"]["eligible"] == 0
    assert row["conversion_eligible"] is False
    assert row["exclusion_reasons"] == ["source_response_format_invalid"]
    assert row["turns"][0]["environment_extracted_actions"] == []
    validate_complete_shard(output, expected_source_indices={0})


def test_ordinary_length_fails_before_environment_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    monkeypatch.setattr(
        collect_module,
        "SOURCE_SYSTEM_PROMPT_SHA256",
        hashlib.sha256(system_prompt.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(_initial_prompt("<INSTRUCTION>").encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_STEP_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(
            collect_module.normalized_step_prompt(_step_prompt("moveahead")).encode()
        ).hexdigest(),
    )
    client = _FakeClient(system_prompt)
    collector = SourceShardCollector(
        client=client,
        policy=_LengthOrdinaryPolicy(),
        run_id="ordinary-length",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )
    with pytest.raises(ValueError, match="cannot be stepped"):
        collector.collect(
            [EpisodeSpec(0, "base", 100, "train", "base:100")],
            output_dir=tmp_path / "ordinary-length-shard",
        )
    assert client.step_payloads == []


def test_terminal_length_excludes_the_whole_linked_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    monkeypatch.setattr(
        collect_module,
        "SOURCE_SYSTEM_PROMPT_SHA256",
        hashlib.sha256(system_prompt.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(_initial_prompt("<INSTRUCTION>").encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_STEP_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(
            collect_module.normalized_step_prompt(_step_prompt("moveahead")).encode()
        ).hexdigest(),
    )
    collector = SourceShardCollector(
        client=_FakeClient(system_prompt),
        policy=_LengthTerminalPolicy(),
        run_id="length",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )
    output = tmp_path / "length-shard"
    manifest = collector.collect(
        [EpisodeSpec(0, "base", 100, "train", "base:100")],
        output_dir=output,
    )
    row = json.loads((output / "raw.jsonl").read_text(encoding="utf-8"))
    assert manifest["counts"]["eligible"] == 0
    assert row["conversion_eligible"] is False
    assert row["exclusion_reasons"] == ["generation_length_truncated"]
    assert row["terminal_generation"]["executed"] is False
    assert row["terminal_generation"]["environment_step_after_generation"] is False


def test_runtime_contract_rejects_terminal_aggregate_reward() -> None:
    with pytest.raises(ValueError, match="must be step_rewards"):
        SourceShardCollector(
            client=_FakeClient("system"),
            policy=_FakePolicy(),
            run_id="aggregate",
            shard_index=0,
            reconstruction_identity=_reconstruction_identity(),
            source_runtime_evidence=_source_runtime_evidence(
                reward_provenance="trajectory_terminal_reward"
            ),
            policy_artifact_evidence=_policy_artifact_evidence(),
            format_failure_policy="exclude_trajectory",
            concurrency=1,
        )


def test_complete_shard_rejects_raw_semantic_tamper_even_if_outer_hashes_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    monkeypatch.setattr(
        collect_module,
        "SOURCE_SYSTEM_PROMPT_SHA256",
        hashlib.sha256(system_prompt.encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_INITIAL_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(_initial_prompt("<INSTRUCTION>").encode()).hexdigest(),
    )
    monkeypatch.setattr(
        collect_module,
        "SOURCE_STEP_PROMPT_NORMALIZED_SHA256",
        hashlib.sha256(
            collect_module.normalized_step_prompt(_step_prompt("moveahead")).encode()
        ).hexdigest(),
    )
    collector = SourceShardCollector(
        client=_FakeClient(system_prompt),
        policy=_FakePolicy(),
        run_id="tamper",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )
    output = tmp_path / "shard"
    collector.collect(
        [EpisodeSpec(0, "base", 100, "train", "base:100")],
        output_dir=output,
    )

    raw_path = output / "raw.jsonl"
    original = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest_path = output / "shard_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def republish(row: dict[str, object]) -> None:
        raw_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        manifest["raw_jsonl"]["sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (output / "COMPLETE").write_text(
            json.dumps(
                {
                    "format": collect_module.COMPLETE_MARKER_FORMAT,
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest()
                }
            ),
            encoding="utf-8",
        )

    provenance_tamper = json.loads(json.dumps(original))
    provenance_tamper["policy_artifact"]["artifact_manifest_sha256"] = "f" * 64
    provenance_payload = {
        key: value
        for key, value in provenance_tamper.items()
        if key != "raw_record_sha256"
    }
    provenance_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            provenance_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    republish(provenance_tamper)
    with pytest.raises(ValueError, match="policy_artifact"):
        validate_complete_shard(output, expected_source_indices={0})

    window_tamper = json.loads(json.dumps(original))
    window_tamper["policy_requests"][0]["message_window"][-1]["content"] = (
        "different observation <image>"
    )
    window_payload = {
        key: value
        for key, value in window_tamper.items()
        if key != "raw_record_sha256"
    }
    window_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            window_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    republish(window_tamper)
    with pytest.raises(ValueError, match="message window"):
        validate_complete_shard(output, expected_source_indices={0})

    eligibility_tamper = json.loads(json.dumps(original))
    eligibility_tamper["terminal_generation"]["finish_reason"] = "length"
    eligibility_tamper["terminal_generation"]["generation_exclusion_reason"] = (
        "generation_length_truncated"
    )
    eligibility_tamper["policy_requests"][-1]["finish_reason"] = "length"
    eligibility_tamper["conversion_eligible"] = True
    eligibility_tamper["exclusion_reasons"] = []
    eligibility_payload = {
        key: value
        for key, value in eligibility_tamper.items()
        if key != "raw_record_sha256"
    }
    eligibility_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            eligibility_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    republish(eligibility_tamper)
    with pytest.raises(ValueError, match="conversion eligibility"):
        validate_complete_shard(output, expected_source_indices={0})

    coordinated_reward_tamper = json.loads(json.dumps(original))
    coordinated_reward_tamper["turns"][0]["reward"] = 5.0
    coordinated_reward_tamper["environment_reward_events"] = [5.0]
    coordinated_reward_tamper["rewards"] = [5.0]
    coordinated_reward_tamper["reward"] = 5.0
    coordinated_payload = {
        key: value
        for key, value in coordinated_reward_tamper.items()
        if key != "raw_record_sha256"
    }
    coordinated_reward_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            coordinated_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    republish(coordinated_reward_tamper)
    with pytest.raises(ValueError, match="parser/success class"):
        validate_complete_shard(output, expected_source_indices={0})

    semantic_tamper = json.loads(json.dumps(original))
    semantic_tamper["reward"] = -999.0
    semantic_payload = {
        key: value
        for key, value in semantic_tamper.items()
        if key != "raw_record_sha256"
    }
    semantic_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            semantic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    republish(semantic_tamper)
    with pytest.raises(ValueError, match="aggregate reward does not equal step rewards"):
        validate_complete_shard(output, expected_source_indices={0})

    runtime_tamper = json.loads(json.dumps(original))
    tampered_identity = dict(runtime_tamper["reconstruction_identity"])
    tampered_identity["runtime_head"] = "0" * 40
    runtime_tamper["reconstruction_identity"] = tampered_identity
    runtime_tamper["source_runtime_contract"]["reconstruction_identity"] = (
        tampered_identity
    )
    runtime_tamper["source_runtime_contract"]["contract_payload_sha256"] = (
        collect_module.source_runtime_contract_payload_sha256(
            runtime_tamper["source_runtime_contract"]
        )
    )
    runtime_payload = {
        key: value
        for key, value in runtime_tamper.items()
        if key != "raw_record_sha256"
    }
    runtime_tamper["raw_record_sha256"] = hashlib.sha256(
        json.dumps(
            runtime_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest["reconstruction_identity"] = tampered_identity
    manifest["source_runtime_contract"] = runtime_tamper[
        "source_runtime_contract"
    ]
    republish(runtime_tamper)
    with pytest.raises(ValueError, match="runtime_head"):
        validate_complete_shard(output, expected_source_indices={0})


def test_collector_rejects_reconstruction_identity_relabel() -> None:
    relabeled = _reconstruction_identity()
    relabeled["runtime_head"] = collect_module.SOURCE_VAGEN_COMMIT
    with pytest.raises(ValueError, match="runtime_head"):
        SourceShardCollector(
            client=_FakeClient("system"),
            policy=_FakePolicy(),
            run_id="unit",
            shard_index=0,
            reconstruction_identity=relabeled,
            source_runtime_evidence=_source_runtime_evidence(),
            policy_artifact_evidence=_policy_artifact_evidence(),
            format_failure_policy="fail_shard",
            concurrency=1,
        )
