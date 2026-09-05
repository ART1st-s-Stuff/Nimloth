from __future__ import annotations

import hashlib
import json
import sys
import types
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


def _write_inspection_handoff(
    path: Path, *, bindings: dict[str, object], output_dir: Path
) -> str:
    payload = {
        "bindings": bindings,
        "inspection_context": {
            "policy_artifact_evidence": _policy_artifact_evidence(),
            "source_runtime_evidence": _source_runtime_evidence(),
            "reconstruction_identity": _reconstruction_identity(),
            "inspected_policy_contract": _FakePolicy().runtime_contract,
        },
        "items": [
            {
                "label": "shard-0",
                "output_dir": str(output_dir.absolute()),
                "run_id": "gate-0",
                "selector": "shard-index",
                "index": 0,
                "format_failure_policy": "exclude_trajectory",
                "concurrency": 4,
                "ordered_episode_specs": [
                    {
                        "source_index": 0,
                        "eval_set": "base",
                        "seed": 100,
                        "dataset_split": "train",
                        "source_key": "base:100",
                    }
                ],
            }
        ],
    }
    envelope = {
        "format": collect_module.INSPECTION_HANDOFF_FORMAT,
        "payload": payload,
        "payload_sha256": collect_module._canonical_sha256(payload),
    }
    content = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def test_inspection_handoff_is_hash_schema_cli_and_output_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = {"model_path": str((tmp_path / "model").absolute())}
    output = tmp_path / "shard"
    handoff = tmp_path / "handoff.json"
    digest = _write_inspection_handoff(handoff, bindings=bindings, output_dir=output)
    monkeypatch.setattr(
        collect_module,
        "prepare_collection_inspection_context",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("inspection repeated")),
    )
    collector, specs = collect_module.build_collection_from_inspection_handoff(
        path=handoff,
        expected_file_sha256=digest,
        expected_bindings=bindings,
        output_dir=output,
        run_id="gate-0",
        selector="shard-index",
        index=0,
        shard_index=0,
        format_failure_policy="exclude_trajectory",
        concurrency=4,
    )
    assert [spec.source_index for spec in specs] == [0]
    assert collector.policy.runtime_contract == _FakePolicy().runtime_contract

    with pytest.raises(ValueError, match="output"):
        collect_module.build_collection_from_inspection_handoff(
            path=handoff,
            expected_file_sha256=digest,
            expected_bindings=bindings,
            output_dir=tmp_path / "wrong-output",
            run_id="gate-0",
            selector="shard-index",
            index=0,
            shard_index=0,
            format_failure_policy="exclude_trajectory",
            concurrency=4,
        )
    with pytest.raises(ValueError, match="binding"):
        collect_module.load_inspection_handoff(
            handoff,
            expected_file_sha256=digest,
            expected_bindings={"model_path": "stale"},
        )
    symlink = tmp_path / "handoff-link.json"
    symlink.symlink_to(handoff)
    with pytest.raises(ValueError, match="regular file"):
        collect_module.load_inspection_handoff(
            symlink,
            expected_file_sha256=digest,
            expected_bindings=bindings,
        )
    handoff.write_bytes(handoff.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        collect_module.load_inspection_handoff(
            handoff,
            expected_file_sha256=digest,
            expected_bindings=bindings,
        )


def test_gpu_engine_is_constructed_before_live_runtime_contract_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class LLM:
        def __init__(self, **kwargs):
            events.append("engine")

    class SamplingParams:
        def __init__(self, **kwargs):
            events.append("sampling")

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoProcessor=AutoProcessor)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
    )
    expected = {"verified": True}

    def runtime_contract(**kwargs):
        assert events == ["engine"]
        return expected

    monkeypatch.setattr(
        collect_module.VLLMSourcePolicy,
        "_runtime_contract",
        staticmethod(runtime_contract),
    )
    policy = collect_module.VLLMSourcePolicy(
        model_path=tmp_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        engine_seed=42,
        expected_runtime_contract=expected,
    )
    assert policy.runtime_contract == expected
    assert events == ["engine", "sampling"]


def test_collector_rejects_dangling_output_before_rollout(tmp_path: Path) -> None:
    output = tmp_path / "shard"
    output.symlink_to(tmp_path / "missing")
    collector = object.__new__(SourceShardCollector)
    with pytest.raises(FileExistsError):
        collector.collect([], output_dir=output, max_steps=20)
    assert output.is_symlink()
    assert not (tmp_path / "missing").exists()


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
        "source_generation_package_evidence": (
            collect_module.SOURCE_GENERATION_PACKAGE_EVIDENCE
        ),
        "executable_generation_packages": (
            collect_module.EXECUTABLE_GENERATION_PACKAGES
        ),
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
                "vllm": "0.8.2",
                "transformers": "4.49.0",
                "torch": "2.6.0",
            },
            "source_generation_package_evidence": (
                collect_module.SOURCE_GENERATION_PACKAGE_EVIDENCE
            ),
            "executable_generation_packages": (
                collect_module.EXECUTABLE_GENERATION_PACKAGES
            ),
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


def _patch_prompt_hashes(monkeypatch: pytest.MonkeyPatch, system_prompt: str) -> None:
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


class _InterruptSecondEpisodePolicy(_FakePolicy):
    def generate(self, requests):
        if len(self.calls) == 2:
            raise RuntimeError("injected interruption")
        return super().generate(requests)


class _ObserveCheckpointThenInterruptPolicy(_FakePolicy):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.checkpoint = checkpoint

    def generate(self, requests):
        if len(self.calls) == 2:
            envelope = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            assert envelope["record"]["source_index"] == 0
            raise RuntimeError("injected interruption")
        return super().generate(requests)


class _AlwaysSingleEpisodePolicy(_FakePolicy):
    def generate(self, requests):
        rows = super().generate(requests)
        return rows


class _ObserveCheckpointDurabilityThenInterruptPolicy(_FakePolicy):
    def __init__(self, observe) -> None:
        super().__init__()
        self.observe = observe

    def generate(self, requests):
        if len(self.calls) == 2:
            self.observe()
            raise RuntimeError("injected interruption after durability check")
        return super().generate(requests)


def _collector(client: _FakeClient, policy: _FakePolicy, *, run_id: str = "resume"):
    return SourceShardCollector(
        client=client,
        policy=policy,
        run_id=run_id,
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )


def test_interrupted_shard_checkpoints_completed_rows_and_resumes_only_unfinished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / "shard_000"
    first_client = _FakeClient(system_prompt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(
            first_client,
            _ObserveCheckpointThenInterruptPolicy(
                tmp_path / "shard_000.inprogress" / "records" / "00000000.json"
            ),
        ).collect(specs, output_dir=output)

    staging = tmp_path / "shard_000.inprogress"
    checkpoints = sorted((staging / "records").glob("*.json"))
    assert [path.name for path in checkpoints] == ["00000000.json"]
    first_record = json.loads(checkpoints[0].read_text(encoding="utf-8"))["record"]
    assert first_record["source_index"] == 0
    assert first_record["image_paths"]
    assert all(path.startswith("attempts/") for path in first_record["image_paths"])

    resumed_client = _FakeClient(system_prompt)
    manifest = _collector(resumed_client, _AlwaysSingleEpisodePolicy()).collect(
        specs, output_dir=output, resume=True
    )
    assert len(resumed_client.configs) == 1
    assert next(iter(resumed_client.configs)).startswith(
        "v60_resume_s000_r10000_common_sense_100_a"
    )
    assert manifest["source_indices"] == [0, 10_000]
    rows = [
        json.loads(line)
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["source_index"] for row in rows] == [0, 10_000]
    assert (output / "IN_PROGRESS.json").is_file()
    assert len(list((output / "attempts").iterdir())) == 2


def test_orchestrator_state_inspection_reuses_full_resume_and_complete_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    interrupted = tmp_path / "inspected-interrupted"
    collector = _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy())
    with pytest.raises(RuntimeError, match="injected interruption"):
        collector.collect(specs, output_dir=interrupted)
    assert collector.inspect_output_state(specs, output_dir=interrupted) == "resume"

    checkpoint = next(
        interrupted.with_name("inspected-interrupted.inprogress")
        .joinpath("records")
        .glob("*.json")
    )
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    image = interrupted.with_name("inspected-interrupted.inprogress") / envelope[
        "record"
    ]["image_paths"][0]
    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="image"):
        collector.inspect_output_state(specs, output_dir=interrupted)

    complete = tmp_path / "inspected-complete"
    complete_collector = _collector(_FakeClient(system_prompt), _FakePolicy())
    complete_collector.collect(specs, output_dir=complete)
    assert complete_collector.inspect_output_state(specs, output_dir=complete) == "complete"
    changed_artifact = _policy_artifact_evidence()
    changed_artifact["artifact_manifest_sha256"] = "f" * 64
    mismatched = SourceShardCollector(
        client=_FakeClient(system_prompt),
        policy=_FakePolicy(),
        run_id="resume",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=_source_runtime_evidence(),
        policy_artifact_evidence=changed_artifact,
        format_failure_policy="exclude_trajectory",
        concurrency=1,
    )
    with pytest.raises(ValueError, match="policy_artifact"):
        mismatched.inspect_output_state(specs, output_dir=complete)


def test_checkpoint_file_and_directory_are_fsynced_before_next_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / "durable"
    checkpoint = output.with_name("durable.inprogress") / "records" / "00000000.json"
    events: list[tuple[str, str]] = []
    real_fsync = collect_module.os.fsync
    real_link = collect_module.os.link

    def tracked_fsync(descriptor: int) -> None:
        events.append(("fsync", collect_module.os.readlink(f"/proc/self/fd/{descriptor}")))
        real_fsync(descriptor)

    def tracked_link(source, destination) -> None:
        events.append(("link", str(destination)))
        real_link(source, destination)

    monkeypatch.setattr(collect_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(collect_module.os, "link", tracked_link)

    def assert_durable_checkpoint() -> None:
        checkpoint_link = events.index(("link", str(checkpoint)))
        checkpoint_file_fsync = max(
            index
            for index, event in enumerate(events[:checkpoint_link])
            if event[0] == "fsync" and ".00000000.json.tmp-" in event[1]
        )
        records_directory_fsync = next(
            index
            for index, event in enumerate(events[checkpoint_link + 1 :], checkpoint_link + 1)
            if event == ("fsync", str(checkpoint.parent))
        )
        assert checkpoint_file_fsync < checkpoint_link < records_directory_fsync
        assert checkpoint.is_file()

    policy = _ObserveCheckpointDurabilityThenInterruptPolicy(
        assert_durable_checkpoint
    )
    with pytest.raises(RuntimeError, match="durability check"):
        _collector(_FakeClient(system_prompt), policy).collect(
            specs, output_dir=output
        )


def test_fresh_and_resume_modes_fail_closed_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    spec = [EpisodeSpec(0, "base", 100, "train", "base:100")]
    output = tmp_path / "shard"
    client = _FakeClient(system_prompt)
    with pytest.raises(FileNotFoundError, match="in-progress"):
        _collector(client, _FakePolicy()).collect(spec, output_dir=output, resume=True)
    assert client.configs == {}

    (tmp_path / "shard.inprogress").mkdir()
    client = _FakeClient(system_prompt)
    with pytest.raises(FileExistsError, match="in-progress"):
        _collector(client, _FakePolicy()).collect(spec, output_dir=output)
    assert client.configs == {}

    output.mkdir()
    client = _FakeClient(system_prompt)
    with pytest.raises(FileExistsError, match="output"):
        _collector(client, _FakePolicy()).collect(spec, output_dir=output, resume=True)
    assert client.configs == {}


def test_resume_validation_precedes_runtime_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / "activation-order"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=output
        )
    metadata_path = output.with_name(f"{output.name}.inprogress") / "IN_PROGRESS.json"
    metadata_path.write_text("{}", encoding="utf-8")
    activated = False

    def activate_runtime():
        nonlocal activated
        activated = True
        raise AssertionError("runtime activation must follow resume validation")

    with pytest.raises(ValueError, match="metadata"):
        _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
            specs,
            output_dir=output,
            resume=True,
            activate_runtime=activate_runtime,
        )
    assert not activated


def test_resume_rejects_identity_checkpoint_and_image_drift_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]

    def interrupted(name: str) -> tuple[Path, Path]:
        output = tmp_path / name
        with pytest.raises(RuntimeError, match="injected interruption"):
            _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
                specs, output_dir=output
            )
        return output, tmp_path / f"{name}.inprogress"

    output, staging = interrupted("identity")
    metadata_path = staging / "IN_PROGRESS.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["payload"]["max_steps"] = 19
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="metadata"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}

    output, staging = interrupted("truncated")
    checkpoint = next((staging / "records").glob("*.json"))
    checkpoint.write_text("{", encoding="utf-8")
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="checkpoint JSON"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}

    output, staging = interrupted("record-hash")
    checkpoint = next((staging / "records").glob("*.json"))
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["record"]["seed"] = 999
    checkpoint_payload = {
        key: value
        for key, value in envelope.items()
        if key != "checkpoint_payload_sha256"
    }
    envelope["checkpoint_payload_sha256"] = collect_module._canonical_sha256(
        checkpoint_payload
    )
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="record hash"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}

    output, staging = interrupted("image")
    checkpoint = next((staging / "records").glob("*.json"))
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    image_path = staging / envelope["record"]["image_paths"][0]
    image_path.write_bytes(b"tampered")
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="image"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}


@pytest.mark.parametrize(
    "drift",
    ["ordered_specs", "runtime", "policy", "max_steps", "format"],
)
def test_resume_rejects_collection_contract_drift_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / f"shard-{drift}"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=output
        )

    client = _FakeClient(system_prompt)
    policy = _FakePolicy()
    runtime = _source_runtime_evidence()
    format_policy = "exclude_trajectory"
    resumed_specs = specs
    max_steps = 20
    if drift == "ordered_specs":
        resumed_specs = list(reversed(specs))
    elif drift == "runtime":
        runtime["runtime_root"] = "/different/source/VAGEN"
        runtime["contract_payload_sha256"] = (
            collect_module.source_runtime_contract_payload_sha256(runtime)
        )
    elif drift == "policy":
        policy.runtime_contract["engine_seed"] = 8
    elif drift == "max_steps":
        max_steps = 19
    elif drift == "format":
        format_policy = "fail_shard"
    collector = SourceShardCollector(
        client=client,
        policy=policy,
        run_id="resume",
        shard_index=0,
        reconstruction_identity=_reconstruction_identity(),
        source_runtime_evidence=runtime,
        policy_artifact_evidence=_policy_artifact_evidence(),
        format_failure_policy=format_policy,
        concurrency=1,
    )
    with pytest.raises(ValueError, match="metadata|exactly 20"):
        collector.collect(
            resumed_specs,
            output_dir=output,
            max_steps=max_steps,
            resume=True,
        )
    assert client.configs == {}


def test_resume_rejects_duplicate_and_unknown_checkpoint_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / "shard"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=output
        )
    records = tmp_path / "shard.inprogress" / "records"
    original = next(records.glob("*.json"))
    (records / "99999999.json").write_bytes(original.read_bytes())
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="unknown checkpoint"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}


@pytest.mark.parametrize(
    "artifact_name",
    ["raw.jsonl", "shard_manifest.json", "COMPLETE"],
)
def test_resume_never_overwrites_tampered_finalization_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [EpisodeSpec(0, "base", 100, "train", "base:100")]
    output = tmp_path / f"shard-{artifact_name.replace('.', '-')}"
    original_publish = collect_module.publish_reserved_directory

    def interrupt_publication(*_args, **_kwargs):
        raise RuntimeError("injected publication interruption")

    monkeypatch.setattr(
        collect_module,
        "publish_reserved_directory",
        interrupt_publication,
    )
    with pytest.raises(RuntimeError, match="publication interruption"):
        _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
            specs, output_dir=output
        )

    staging = output.with_name(f"{output.name}.inprogress")
    artifact = staging / artifact_name
    artifact.write_bytes(b"tampered-finalization-evidence\n")
    client = _FakeClient(system_prompt)
    monkeypatch.setattr(
        collect_module,
        "publish_reserved_directory",
        original_publish,
    )
    def unexpected_activation():
        raise AssertionError("complete resume must validate/finalize without runtime activation")

    with pytest.raises(ValueError, match="finalization artifact"):
        _collector(client, _FakePolicy()).collect(
            specs,
            output_dir=output,
            resume=True,
            activate_runtime=unexpected_activation,
        )
    assert artifact.read_bytes() == b"tampered-finalization-evidence\n"
    assert client.configs == {}
    assert not output.exists()


@pytest.mark.parametrize("entry", ["IN_PROGRESS.json", "records", "attempts"])
def test_resume_rejects_symlinked_control_entries_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / f"symlink-{entry.replace('.', '-')}"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=output
        )
    staging = output.with_name(f"{output.name}.inprogress")
    path = staging / entry
    backup = staging / f"{entry}.real"
    path.rename(backup)
    path.symlink_to(backup, target_is_directory=backup.is_dir())

    client = _FakeClient(system_prompt)
    with pytest.raises((ValueError, FileNotFoundError), match="symlink|real|invalid"):
        _collector(client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    assert client.configs == {}


@pytest.mark.parametrize("entry", ["IN_PROGRESS.json", "records", "attempts"])
def test_complete_validation_rejects_symlinked_control_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    output = tmp_path / f"final-control-{entry.replace('.', '-')}"
    _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
        [EpisodeSpec(0, "base", 100, "train", "base:100")],
        output_dir=output,
    )
    path = output / entry
    backup = output / f"{entry}.real"
    path.rename(backup)
    path.symlink_to(backup, target_is_directory=backup.is_dir())
    with pytest.raises(ValueError, match="symlink"):
        validate_complete_shard(output, expected_source_indices={0})


def test_resume_and_complete_validation_reject_symlinked_image_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    interrupted_output = tmp_path / "resume-image-ancestor"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=interrupted_output
        )
    staging = interrupted_output.with_name(f"{interrupted_output.name}.inprogress")
    checkpoint = next((staging / "records").glob("*.json"))
    relative = Path(
        json.loads(checkpoint.read_text(encoding="utf-8"))["record"]["image_paths"][0]
    )
    ancestor = staging / relative.parts[0] / relative.parts[1]
    backing = ancestor.with_name(f"{ancestor.name}.real")
    ancestor.rename(backing)
    ancestor.symlink_to(backing, target_is_directory=True)
    client = _FakeClient(system_prompt)
    with pytest.raises(ValueError, match="symlink"):
        _collector(client, _FakePolicy()).collect(
            specs, output_dir=interrupted_output, resume=True
        )
    assert client.configs == {}

    final_output = tmp_path / "final-image-ancestor"
    manifest = _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
        [specs[0]], output_dir=final_output
    )
    relative = Path(manifest["images"][0]["path"])
    ancestor = final_output / relative.parts[0] / relative.parts[1]
    backing = ancestor.with_name(f"{ancestor.name}.real")
    ancestor.rename(backing)
    ancestor.symlink_to(backing, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate_complete_shard(final_output, expected_source_indices={0})


def test_resume_uses_attempt_unique_service_id_but_stable_record_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    specs = [
        EpisodeSpec(0, "base", 100, "train", "base:100"),
        EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
    ]
    output = tmp_path / "attempt-identities"
    first_client = _FakeClient(system_prompt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(first_client, _InterruptSecondEpisodePolicy()).collect(
            specs, output_dir=output
        )
    first_ids = set(first_client.configs)
    resumed_client = _FakeClient(system_prompt)
    _collector(resumed_client, _FakePolicy()).collect(specs, output_dir=output, resume=True)
    resumed_ids = set(resumed_client.configs)
    assert first_ids.isdisjoint(resumed_ids)
    rows = [
        json.loads(line)
        for line in (output / "raw.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == [
        "v60_resume_s000_r00000_base_100",
        "v60_resume_s000_r10000_common_sense_100",
    ]
    assert all(row["id"] not in first_ids | resumed_ids for row in rows)


def test_staging_creation_fsyncs_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    calls: list[Path] = []
    original = collect_module._fsync_directory

    def observe(path: Path) -> None:
        calls.append(path)
        original(path)

    monkeypatch.setattr(collect_module, "_fsync_directory", observe)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            [
                EpisodeSpec(0, "base", 100, "train", "base:100"),
                EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
            ],
            output_dir=tmp_path / "fsync-parent",
        )
    assert calls[0] == tmp_path


def test_failure_recording_io_never_obscures_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    original = collect_module._write_json_atomic

    def fail_only_for_failure_marker(path: Path, payload: dict[str, object]) -> None:
        if path.name == "FAILED.json":
            raise OSError("injected failure-recording I/O")
        original(path, payload)

    monkeypatch.setattr(collect_module, "_write_json_atomic", fail_only_for_failure_marker)
    with pytest.raises(RuntimeError, match="injected interruption"):
        _collector(_FakeClient(system_prompt), _InterruptSecondEpisodePolicy()).collect(
            [
                EpisodeSpec(0, "base", 100, "train", "base:100"),
                EpisodeSpec(10_000, "common_sense", 100, "train", "common_sense:100"),
            ],
            output_dir=tmp_path / "failure-io",
        )


def test_checkpoint_atomic_writer_never_replaces_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_bytes(b"winner")
    with pytest.raises(FileExistsError):
        collect_module._write_json_exclusive_atomic(path, {"loser": True})
    assert path.read_bytes() == b"winner"


def test_collection_lock_rejects_concurrent_owner_before_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    output = tmp_path / "locked"
    client = _FakeClient(system_prompt)
    with (
        collect_module._exclusive_collection_lock(output),
        pytest.raises(RuntimeError, match="locked|collector"),
    ):
        _collector(client, _FakePolicy()).collect(
            [EpisodeSpec(0, "base", 100, "train", "base:100")],
            output_dir=output,
        )
    assert client.configs == {}


def test_collection_lock_is_held_through_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    output = tmp_path / "locked-through-publication"
    original_publish = collect_module.publish_reserved_directory

    def assert_locked(source: Path, target: Path, *, readiness_marker: str) -> None:
        with (
            pytest.raises(RuntimeError, match="locked|collector"),
            collect_module._exclusive_collection_lock(output),
        ):
            raise AssertionError("concurrent lock unexpectedly acquired")
        original_publish(source, target, readiness_marker=readiness_marker)

    monkeypatch.setattr(collect_module, "publish_reserved_directory", assert_locked)
    _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
        [EpisodeSpec(0, "base", 100, "train", "base:100")],
        output_dir=output,
    )
    assert (output / "COMPLETE").is_file()


def test_post_publication_validation_error_is_not_obscured_or_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_prompt = "exact source system"
    _patch_prompt_hashes(monkeypatch, system_prompt)
    spec = [EpisodeSpec(0, "base", 100, "train", "base:100")]
    output = tmp_path / "published-then-invalid"
    original_validate = collect_module.validate_complete_shard

    def fail_only_after_publication(shard_dir, *, expected_source_indices):
        result = original_validate(
            shard_dir,
            expected_source_indices=expected_source_indices,
        )
        if Path(shard_dir) == output:
            raise ValueError("injected post-publication validation failure")
        return result

    monkeypatch.setattr(
        collect_module,
        "validate_complete_shard",
        fail_only_after_publication,
    )
    with pytest.raises(ValueError, match="post-publication validation failure"):
        _collector(_FakeClient(system_prompt), _FakePolicy()).collect(
            spec, output_dir=output
        )
    assert output.is_dir()
    assert (output / "COMPLETE").is_file()
    assert not list(output.glob("**/FAILED.json"))
    assert not (output / "FAILED_VALIDATION.json").exists()


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
