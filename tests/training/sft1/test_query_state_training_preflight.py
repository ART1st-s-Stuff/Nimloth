from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nimloth.latent import LatentActionTokens, latent_state_tokens
from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
)
from nimloth.training.sft1.query_state_training_preflight import (
    assert_query_state_training_backend_ready,
    verify_query_state_training_preflight,
)
from tests.training.sft1.test_query_state_training_config import _raw


def _run(*argv: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_ALLOW_PROTOCOL": "file"},
    )
    return result.stdout.strip()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path.resolve()


def _git_repo(tmp_path: Path) -> tuple[Path, str, dict[str, str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    child = tmp_path / "child"
    child.mkdir()
    _run("git", "init", "-q", cwd=child)
    _run("git", "config", "user.email", "test@example.com", cwd=child)
    _run("git", "config", "user.name", "Test", cwd=child)
    _write(child / "child.txt", "child\n")
    _run("git", "add", ".", cwd=child)
    _run("git", "commit", "-qm", "child", cwd=child)
    child_commit = _run("git", "rev-parse", "HEAD", cwd=child)

    repo = tmp_path / "repo"
    repo.mkdir()
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "test@example.com", cwd=repo)
    _run("git", "config", "user.name", "Test", cwd=repo)
    _run("git", "checkout", "-qb", "exp/sft1-query-state-formal", cwd=repo)
    _write(repo / "tracked.txt", "source\n")
    _write(
        repo / "experiments/training/sft1/query_state_train.py",
        "#!/usr/bin/env python3\n",
    )
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(child),
        "external/child",
        cwd=repo,
    )
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "source", cwd=repo)
    return repo.resolve(), _run("git", "rev-parse", "HEAD", cwd=repo), {
        "external/child": child_commit
    }


def _resolved_contract(tmp_path: Path, *, launch_locked: bool) -> tuple[dict, dict[str, str]]:
    repo, commit, submodules = _git_repo(tmp_path)
    contracts = tmp_path / "contracts"
    data = tmp_path / "data"
    actor = tmp_path / "checkpoints" / "id176"
    dino = tmp_path / "models" / "dinov2-large"

    source_manifest = _write(contracts / "source.json", json.dumps({"identity": "1" * 64}) + "\n")
    train_source = _write(data / "train.jsonl", "{}\n")
    validation_source = _write(data / "validation.jsonl", "{}\n")
    train_manifest = _write(contracts / "train.json", json.dumps({"identity": "2" * 64, "entries": []}) + "\n")
    validation_manifest = _write(contracts / "validation.json", json.dumps({"identity": "3" * 64, "calibration_row_identities": [], "holdout_row_identities": []}) + "\n")
    generation_manifest = _write(
        contracts / "generation-format.json",
        json.dumps({"identity": "4" * 64}) + "\n",
    )

    completion = _write(actor.parent / "complete.marker", "complete\n")
    actor_config = _write(actor / "config.json", json.dumps({
        "hidden_size": 2048,
        "nimloth_latent_token_count": 16,
        "nimloth_latent_query_mode": "inject",
        "nimloth_action_token_ids": list(range(151683, 151691)),
    }) + "\n")
    shard = _write(actor / "model-00001-of-00001.safetensors", b"model-shard")
    model_index = _write(actor / "model.safetensors.index.json", json.dumps({
        "weight_map": {"model.embed_tokens.weight": shard.name, "lm_head.weight": shard.name}
    }) + "\n")
    action_head = _write(actor / "action_head_repair.pt", b"action-head")

    token_contract = LatentActionTokens()
    tokens = [*latent_state_tokens(16, token_contract)]
    tokens += [token_contract.action_start, token_contract.action_end]
    tokens += list(token_contract.action_tokens)
    added = {token: 151665 + index for index, token in enumerate(tokens)}
    added.update({token: 151683 + index for index, token in enumerate(token_contract.action_tokens)})
    processor_files = {
        "preprocessor_config.json": "{}\n",
        "video_preprocessor_config.json": "{}\n",
        "tokenizer.json": "{}\n",
        "tokenizer_config.json": "{}\n",
        "vocab.json": "{}\n",
        "merges.txt": "#version\n",
        "added_tokens.json": json.dumps(added) + "\n",
        "special_tokens_map.json": "{}\n",
        "chat_template.jinja": "{{ messages }}\n",
    }
    processor_paths = [_write(actor / name, value) for name, value in processor_files.items()]
    processor_identity = audit_id176_processor_identity(actor)

    dino_paths = [
        _write(dino / "config.json", "{}\n"),
        _write(dino / "preprocessor_config.json", "{}\n"),
        _write(dino / "model.safetensors", b"dino-weights"),
    ]
    command_manifest = _write(contracts / "command.json", "{}\n")
    config_path = (contracts / "resolved.json").resolve()
    script = (repo / "experiments/training/sft1/query_state_train.py").resolve()
    python_executable = Path(sys.executable).resolve()
    child_argv = [
        str(python_executable),
        str(script),
        "--config",
        str(config_path),
        "--phase",
        "run",
    ]
    command_payload = {
        "schema": "nimloth_sft1_query_state_training_command_v1",
        "child_argv": child_argv,
        "topology": {"backend": "nccl", "nodes": 1, "gpus_per_node": 2, "world_size": 2},
    }
    command_manifest.write_text(json.dumps(command_payload, sort_keys=True) + "\n", encoding="utf-8")

    files = [
        source_manifest,
        train_source,
        validation_source,
        train_manifest,
        validation_manifest,
        generation_manifest,
        command_manifest,
        completion,
        actor_config,
        model_index,
        shard,
        action_head,
        *processor_paths,
        *dino_paths,
    ]
    raw = _raw(mode="pilot")
    raw["lifecycle"] = {"preflight_locked": True, "launch_locked": launch_locked}
    raw["authorization"]["launch_authorized"] = launch_locked
    raw["source"].update({
        "repo_root": str(repo),
        "branch": "exp/sft1-query-state-formal",
        "commit": commit,
        "submodule_commits": submodules,
        "source_manifest_path": str(source_manifest),
        "source_manifest_identity": "1" * 64,
    })
    raw["data"].update({
        "train_source_path": str(train_source),
        "validation_source_path": str(validation_source),
        "train_manifest_path": str(train_manifest),
        "train_manifest_identity": "2" * 64,
        "validation_manifest_path": str(validation_manifest),
        "validation_manifest_identity": "3" * 64,
    })
    raw["validation"].update({
        "generation_format_manifest_path": str(generation_manifest),
        "generation_format_manifest_identity": "4" * 64,
    })
    actor_files = [
        completion,
        actor_config,
        model_index,
        shard,
        action_head,
        *processor_paths,
    ]
    actor_identity = hashlib.sha256(
        json.dumps(
            {str(path): _sha(path) for path in sorted(actor_files)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    raw["model"].update({
        "initialization_identity": "id176:" + actor_identity,
        "dino_snapshot_path": str(dino),
        "processor_path": str(actor),
        "processor_identity": processor_identity.processor_sha256,
        "tokenizer_identity": processor_identity.tokenizer_sha256,
        "template_identity": processor_identity.prompt_template_sha256,
        "token_table_identity": processor_identity.token_table_sha256,
        "action_token_ids": list(processor_identity.action_token_ids),
    })
    raw["initialization"].update({
        "actor_checkpoint": str(actor),
        "actor_checkpoint_identity": actor_identity,
    })
    raw["output"].update({
        "run_root": str(tmp_path / "outputs" / "pilot"),
        "controller_root": str(tmp_path / "controllers" / "pilot"),
        "resolved_config_path": str(config_path),
        "command_manifest_path": str(command_manifest),
        "minimum_free_bytes": 1,
    })
    raw["environment"].update({
        "python_executable": str(python_executable),
        "hf_home": str((tmp_path / "hf").resolve()),
        "hf_hub_cache": str((tmp_path / "hf" / "hub").resolve()),
        "pycache_prefix": str((tmp_path / "pycache").resolve()),
        "python_hash_seed": "7",
        "python_version": sys.version.split()[0],
        "torch_version": __import__("torch").__version__,
        "transformers_version": __import__("transformers").__version__,
    })
    raw["command"].update({"argv": child_argv, "identity": _sha(command_manifest)})
    raw["artifacts"] = {"file_sha256": {str(path): _sha(path) for path in files}}
    config_path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": raw["environment"]["pycache_prefix"],
        "PYTHONHASHSEED": "7",
        "HF_HOME": raw["environment"]["hf_home"],
        "HF_HUB_CACHE": raw["environment"]["hf_hub_cache"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    return raw, environment


def test_live_preflight_hashes_real_files_and_returns_launch_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, environment = _resolved_contract(tmp_path, launch_locked=False)
    config = parse_query_state_training_config(raw)
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight._verify_training_data_contract",
        lambda _config: None,
    )
    evidence = verify_query_state_training_preflight(
        config,
        repo_root=Path(config.source["repo_root"]),
        current_argv=config.command["argv"],
        environ=environment,
    )
    assert evidence.config_identity == config.identity
    assert evidence.cuda_entered is False
    Path(config.validation["generation_format_manifest_path"]).write_text(
        json.dumps({"identity": "changed"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_query_state_training_preflight(
            config,
            repo_root=Path(config.source["repo_root"]),
            current_argv=config.command["argv"],
            environ=environment,
        )
    with pytest.raises(PermissionError, match="launch-locked"):
        assert_query_state_training_backend_ready(config)


def test_live_preflight_rejects_dirty_source_hash_and_command_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, environment = _resolved_contract(tmp_path, launch_locked=False)
    config = parse_query_state_training_config(raw)
    shard = Path(config.initialization["actor_checkpoint"]) / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_query_state_training_preflight(
            config,
            repo_root=Path(config.source["repo_root"]),
            current_argv=config.command["argv"],
            environ=environment,
        )

    raw, environment = _resolved_contract(tmp_path / "second", launch_locked=False)
    config = parse_query_state_training_config(raw)
    Path(config.source["repo_root"], "untracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="dirty"):
        verify_query_state_training_preflight(
            config,
            repo_root=Path(config.source["repo_root"]),
            current_argv=config.command["argv"],
            environ=environment,
        )
    Path(config.source["repo_root"], "untracked.txt").unlink()
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight._verify_training_data_contract",
        lambda _config: None,
    )
    with pytest.raises(ValueError, match="command"):
        verify_query_state_training_preflight(
            config,
            repo_root=Path(config.source["repo_root"]),
            current_argv=[*config.command["argv"], "--extra"],
            environ=environment,
        )


def test_live_preflight_accepts_explicit_first_boundary_crash_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, environment = _resolved_contract(tmp_path, launch_locked=True)
    raw["initialization"].update(
        {"resume_mode": "crash_replay", "resume_checkpoint": "none"}
    )
    run_root = Path(raw["output"]["run_root"])
    controller_root = Path(raw["output"]["controller_root"])
    (run_root / "durable").mkdir(parents=True)
    controller_root.mkdir(parents=True)
    (run_root / "durable" / "authoritative_index.json").write_text(
        json.dumps({"mode": "pilot", "entries": []}) + "\n",
        encoding="utf-8",
    )
    (run_root / "actor_baseline_id176.json").write_text("{}\n", encoding="utf-8")
    (run_root / "validation_update_00000000.json").write_text(
        "{}\n", encoding="utf-8"
    )
    config = parse_query_state_training_config(raw)
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight._verify_training_data_contract",
        lambda _config: None,
    )
    evidence = verify_query_state_training_preflight(
        config,
        repo_root=Path(config.source["repo_root"]),
        current_argv=config.command["argv"],
        environ=environment,
    )
    assert evidence.output_ownership_verified is True


def test_backend_gate_accepts_same_config_launch_locked_preflight_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw, environment = _resolved_contract(tmp_path, launch_locked=True)
    config = parse_query_state_training_config(raw)
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight._verify_training_data_contract",
        lambda _config: None,
    )
    evidence = verify_query_state_training_preflight(
        config,
        repo_root=Path(config.source["repo_root"]),
        current_argv=config.command["argv"],
        environ=environment,
    )
    assert_query_state_training_backend_ready(config, preflight=evidence)
    with pytest.raises(PermissionError, match="evidence"):
        assert_query_state_training_backend_ready(config, preflight=None)
