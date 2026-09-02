from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from pathlib import Path

import pytest

from experiments.training.sft1 import vagen_step60_collect as collect_module
from experiments.training.sft1 import vagen_step60_data as data_module

BASE_COMMIT = "3003c2e5e4ad84565627e6aa7f6ad5ca731dad1a"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git(path: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(path), *args],
        input=input_bytes,
    )


def _commit(path: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            message,
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    return _git(path, "rev-parse", "HEAD").decode().strip()


def test_reconstruction_persisted_contract_versions_are_explicit() -> None:
    assert collect_module.SOURCE_RUNTIME_CONTRACT_FORMAT == (
        "vagen_step60_reconstruction_runtime_contract_v3"
    )
    assert collect_module.RAW_RECORD_FORMAT == "vagen_step60_source_trajectory_v3"
    assert collect_module.SHARD_MANIFEST_FORMAT == "vagen_step60_complete_shard_v3"
    assert collect_module.COMPLETE_MARKER_FORMAT == "vagen_step60_complete_shard_v3"
    assert data_module.CONVERSION_FORMAT == "vagen_step60_dual_view_conversion_v3"
    assert data_module.REJECTION_FORMAT == "vagen_step60_rejections_v3"
    assert data_module.SOURCE_AUDIT_CONTRACT_VERSION == (
        "vagen_step60_reconstruction_audit_v3"
    )
    assert data_module.PARTITION_FORMAT == "vagen_step60_partition_v1"


@pytest.mark.parametrize(
    ("surface", "old_format"),
    [
        ("runtime_contract", "vagen_step60_source_runtime_contract_v1"),
        ("raw_row", "vagen_step60_source_trajectory_v1"),
        ("shard_manifest", "vagen_step60_complete_shard_v1"),
        ("complete_marker", "vagen_step60_complete_shard_v1"),
        ("conversion_manifest", "vagen_step60_dual_view_conversion_v1"),
        ("rejection_envelope", "vagen_step60_rejections_v1"),
        ("source_audit", "vagen_step60_source_audit_v1"),
        ("runtime_contract", "vagen_step60_reconstruction_runtime_contract_v2"),
        ("raw_row", "vagen_step60_source_trajectory_v2"),
        ("shard_manifest", "vagen_step60_complete_shard_v2"),
        ("complete_marker", "vagen_step60_complete_shard_v2"),
        ("conversion_manifest", "vagen_step60_dual_view_conversion_v2"),
        ("rejection_envelope", "vagen_step60_rejections_v2"),
        ("source_audit", "vagen_step60_reconstruction_audit_v2"),
    ],
)
def test_reconstruction_rejects_every_consumed_old_surface(
    surface: str,
    old_format: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported reconstruction format"):
        data_module.validate_reconstruction_format(surface, old_format)


def test_unchanged_partition_and_merge_v1_surfaces_remain_supported() -> None:
    data_module.validate_reconstruction_format(
        "partition_manifest",
        "vagen_step60_partition_v1",
    )
    data_module.validate_reconstruction_format(
        "hf_merge_manifest",
        "nimloth_vagen_step60_hf_export_v1",
    )


@pytest.mark.parametrize(
    "answer",
    ("moveahead,moveright,", ",moveahead,moveright", "moveahead,,moveright"),
)
def test_nimloth_malformed_comma_reward_class_matches_vagen(answer: str) -> None:
    response = (
        "<think><observation>obs</observation><reasoning>reason</reasoning>"
        f"<prediction>prediction</prediction></think><answer>{answer}</answer>"
    )
    assert data_module._source_response_error_kind(response) == (
        "missing_or_malformed_tags"
    )


def test_reconstruction_git_identity_is_computed_not_declared(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "runtime"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "runtime.py").write_text("BASE = True\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "runtime.py").write_text(
        "BASE = True\nRECONSTRUCTION = True\n",
        encoding="utf-8",
    )
    head = _commit(repo, "patch")
    tree = _git(repo, "rev-parse", "HEAD^{tree}").decode().strip()
    diff = _git(
        repo,
        "--no-pager",
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        f"{base}..HEAD",
        "--",
    )
    identity = collect_module.reconstruction_git_identity(repo, base_commit=base)
    assert identity["runtime_head"] == head
    assert identity["runtime_parent"] == base
    assert identity["runtime_tree"] == tree
    assert identity["commit_count"] == 1
    assert identity["parent_count"] == 1
    assert identity["diff_sha256"] == _sha256_bytes(diff)

    expected = dict(identity)
    collect_module.validate_reconstruction_git_identity(identity, expected=expected)
    expected["runtime_head"] = BASE_COMMIT
    with pytest.raises(ValueError, match="runtime_head"):
        collect_module.validate_reconstruction_git_identity(identity, expected=expected)


def test_runtime_contract_builder_is_canonical_and_identity_bound(
    tmp_path: Path,
) -> None:
    identity = {
        "base_commit": BASE_COMMIT,
        "runtime_head": "a" * 40,
        "runtime_parent": BASE_COMMIT,
        "runtime_tree": "b" * 40,
        "commit_count": 1,
        "parent_count": 1,
        "diff_sha256": "c" * 64,
        "git_version": "git version test",
    }
    contract = collect_module.build_source_runtime_contract(
        runtime_root=tmp_path,
        reconstruction_identity=identity,
    )
    assert contract["source_generation_package_evidence"] == {
        "packages": {
            "vllm": "0.8.5.post1",
            "transformers": "4.49.0",
            "torch": "2.6.0",
        },
        "evidence": "source_wandb_requirements_2q620nss",
    }
    assert contract["executable_generation_packages"] == {
        "vllm": "0.8.2",
        "transformers": "4.49.0",
        "torch": "2.6.0",
    }
    assert contract["contract_payload_sha256"] == (
        collect_module.source_runtime_contract_payload_sha256(contract)
    )
    collect_module.validate_source_runtime_contract(
        contract,
        expected_reconstruction_identity=identity,
        expected_runtime_root=tmp_path,
    )
    missing_source = json.loads(json.dumps(contract))
    missing_source.pop("source_generation_package_evidence")
    missing_source["contract_payload_sha256"] = (
        collect_module.source_runtime_contract_payload_sha256(missing_source)
    )
    with pytest.raises(ValueError, match="fields drift"):
        collect_module.validate_source_runtime_contract(
            missing_source,
            expected_reconstruction_identity=identity,
        )
    overloaded = json.loads(json.dumps(contract))
    overloaded["executable_generation_packages"] = overloaded[
        "source_generation_package_evidence"
    ]["packages"]
    overloaded["contract_payload_sha256"] = (
        collect_module.source_runtime_contract_payload_sha256(overloaded)
    )
    with pytest.raises(ValueError, match="executable generation package"):
        collect_module.validate_source_runtime_contract(
            overloaded,
            expected_reconstruction_identity=identity,
        )
    legacy_extra = json.loads(json.dumps(contract))
    legacy_extra["package_versions"] = dict(
        legacy_extra["executable_generation_packages"]
    )
    legacy_extra["contract_payload_sha256"] = (
        collect_module.source_runtime_contract_payload_sha256(legacy_extra)
    )
    with pytest.raises(ValueError, match="fields drift"):
        collect_module.validate_source_runtime_contract(
            legacy_extra,
            expected_reconstruction_identity=identity,
        )
    service_identity = collect_module.expected_service_runtime_identity(contract)
    collect_module.validate_service_runtime_identity(
        service_identity,
        contract=contract,
    )
    service_identity["environment_config"]["step_length"] = 0.3
    with pytest.raises(ValueError, match="service reconstruction identity"):
        collect_module.validate_service_runtime_identity(
            service_identity,
            contract=contract,
        )


def test_reconstruction_git_identity_rejects_dirty_runtime(tmp_path: Path) -> None:
    repo = tmp_path / "runtime"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "runtime.py").write_text("BASE = True\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "runtime.py").write_text("PATCH = True\n", encoding="utf-8")
    _commit(repo, "patch")
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        collect_module.reconstruction_git_identity(repo, base_commit=base)


def test_generation_boundary_requires_source_eos_evidence() -> None:
    stop = collect_module.GeneratedTurn(
        response="<RESPONSE_PAYLOAD>",
        rendered_prompt="prompt",
        finish_reason="stop",
        stop_reason=None,
        token_ids=(101, 102),
        eos_token_id=102,
    )
    assert collect_module.generation_exclusion_reason(stop) is None

    length = collect_module.GeneratedTurn(
        response="<RESPONSE_PAYLOAD>",
        rendered_prompt="prompt",
        finish_reason="length",
        stop_reason=None,
        token_ids=(101, 102),
        eos_token_id=102,
    )
    assert collect_module.generation_exclusion_reason(length) == (
        "generation_length_truncated"
    )

    custom_stop = collect_module.GeneratedTurn(
        response="<RESPONSE_PAYLOAD>",
        rendered_prompt="prompt",
        finish_reason="stop",
        stop_reason="custom-stop",
        token_ids=(101, 102),
        eos_token_id=102,
    )
    assert collect_module.generation_exclusion_reason(custom_stop) == (
        "generation_custom_stop"
    )
    missing_eos = collect_module.GeneratedTurn(
        response="<RESPONSE_PAYLOAD>",
        rendered_prompt="prompt",
        finish_reason="stop",
        stop_reason=None,
        token_ids=(101, 102),
        eos_token_id=999,
    )
    assert collect_module.generation_exclusion_reason(missing_eos) == (
        "generation_eos_token_missing"
    )


def test_extractor_cli_rejects_noncanonical_prompt_fixture() -> None:
    extract_module = importlib.import_module(
        "experiments.training.sft1.extract_vagen_step60_evidence"
    )
    fixture = extract_module.CANONICAL_FIXTURE
    source = Path("/tmp") / fixture["filename"]
    extract_module.validate_canonical_cli_fixture(
        source_table=source,
        source_sha256=fixture["sha256"],
        row_index=fixture["row_index"],
        transcript_column=fixture["transcript_column"],
        prompt_hashes=fixture["prompt_hashes"],
    )
    with pytest.raises(ValueError, match="canonical source"):
        extract_module.validate_canonical_cli_fixture(
            source_table=source,
            source_sha256=fixture["sha256"],
            row_index=1,
            transcript_column=fixture["transcript_column"],
            prompt_hashes=fixture["prompt_hashes"],
        )


def test_hash_pinned_extractor_is_non_overwriting_and_excludes_assistant_text(
    tmp_path: Path,
) -> None:
    extract_module = importlib.import_module(
        "experiments.training.sft1.extract_vagen_step60_evidence"
    )
    system = "SYSTEM_PROMPT"
    initial = (
        "[Initial Observation]:\n<image>\n"
        "Human Instruction: find target\nDecide your next action."
    )
    step = (
        "After your answer, the extracted valid action is ['moveahead'].\n"
        "The environment feedback is: Last action is executed successfully.\n"
        "reward: 0.02\ndone: 0.0\nAfter that, the observation is:\n"
        "<image>\nHuman Instruction: find target\nDecide your next action."
    )
    assistant_marker = "ASSISTANT_RESPONSE_MUST_NOT_PERSIST"
    transcript = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{initial}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_marker}<|im_end|>\n"
        f"<|im_start|>user\n{step}<|im_end|>"
    )
    table = {
        "columns": ["step", "output_1"],
        "data": [[5, transcript]],
    }
    source = tmp_path / "table.json"
    source.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "evidence.json"
    initial_normalized = initial.replace(
        "Human Instruction: find target",
        "Human Instruction: <INSTRUCTION>",
    )
    step_normalized = (
        "After your answer, the extracted valid action is <ACTIONS>.\n"
        "The environment feedback is: <FEEDBACK>\n"
        "reward: <REWARD>\ndone: <DONE>\nAfter that, the observation is:\n"
        "<image>\nHuman Instruction: <INSTRUCTION>\nDecide your next action."
    )
    evidence = extract_module.extract_reconstruction_evidence(
        source,
        output,
        expected_table_sha256=_sha256_bytes(source.read_bytes()),
        row_index=0,
        transcript_column="output_1",
        expected_prompt_hashes={
            "system_prompt_sha256": _sha256_bytes(system.encode()),
            "initial_prompt_normalized_sha256": _sha256_bytes(
                initial_normalized.encode()
            ),
            "step_prompt_normalized_sha256": _sha256_bytes(
                step_normalized.encode()
            ),
        },
    )
    assert evidence["format"] == "vagen_step60_reconstruction_evidence_v1"
    persisted = output.read_text(encoding="utf-8")
    assert assistant_marker not in persisted
    assert evidence["prompt_templates"]["system"] == system
    assert evidence["prompt_templates"]["initial"] == initial_normalized
    assert evidence["prompt_templates"]["post_step"] == step_normalized
    with pytest.raises(ValueError, match="duplicate reward table"):
        extract_module.extract_reconstruction_evidence(
            source,
            tmp_path / "duplicate.json",
            expected_table_sha256=_sha256_bytes(source.read_bytes()),
            row_index=0,
            transcript_column="output_1",
            expected_prompt_hashes=evidence["prompt_hashes"],
            reward_table_paths=[source, source],
            expected_reward_table_sha256={
                source.name: _sha256_bytes(source.read_bytes())
            },
        )
    with pytest.raises(FileExistsError):
        extract_module.extract_reconstruction_evidence(
            source,
            output,
            expected_table_sha256=_sha256_bytes(source.read_bytes()),
            row_index=0,
            transcript_column="output_1",
            expected_prompt_hashes=evidence["prompt_hashes"],
        )
