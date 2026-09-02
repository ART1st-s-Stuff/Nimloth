from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nimloth.latent import LatentActionTokens, latent_state_tokens
from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.query_state_training_config import (
    parse_query_state_training_config,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_preflight import (
    _authenticate_prior_pause_receipt,
    _authenticated_exact_restart_update,
    _reject_forensic_failure_restart,
    _required_output_free_bytes,
    _verify_training_data_contract,
    assert_query_state_training_backend_ready,
    validate_query_state_distributed_topology,
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
        "topology": {
            "backend": "nccl",
            "nodes": 1,
            "gpus_per_node": 2,
            "world_size": 2,
            "nccl_socket_ifname": "test0",
            "nccl_ib_disable": "1",
        },
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
        "NCCL_SOCKET_IFNAME": raw["environment"]["nccl_socket_ifname"],
        "NCCL_IB_DISABLE": raw["environment"]["nccl_ib_disable"],
    }
    return raw, environment


def test_ws8_topology_gate_requires_two_complete_four_rank_nodes() -> None:
    resources = {
        "world_size": 8,
        "nodes": 2,
        "gpus_per_node": 4,
    }
    environment = {"nccl_socket_ifname": "ibp24s0", "nccl_ib_disable": "1"}
    records = tuple(
        {
            "rank": rank,
            "group_rank": rank // 4,
            "local_rank": rank % 4,
            "hostname": f"dgx-{rank // 4}",
            "cuda_visible_devices": "0,1,2,3",
            "nccl_socket_ifname": "ibp24s0",
            "nccl_ib_disable": "1",
        }
        for rank in range(8)
    )
    assert validate_query_state_distributed_topology(
        records, resources=resources, environment=environment
    ) == records

    duplicate_host = [dict(record) for record in records]
    duplicate_host[4]["hostname"] = "dgx-0"
    with pytest.raises(ValueError, match="node identity|physical node"):
        validate_query_state_distributed_topology(
            duplicate_host, resources=resources, environment=environment
        )
    wrong_network = [dict(record) for record in records]
    wrong_network[7]["nccl_socket_ifname"] = "eth0"
    with pytest.raises(ValueError, match="NCCL environment"):
        validate_query_state_distributed_topology(
            wrong_network, resources=resources, environment=environment
        )


def test_data_preflight_uses_post_index_exact_audit_not_missing_selection_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = parse_query_state_training_config(_raw())

    def fail_after_index(_contract: object, *, enforce_approved_counts: bool) -> object:
        assert enforce_approved_counts is False
        raise RuntimeError("index-call-proved")

    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight.index_early4_rows",
        fail_after_index,
    )
    with pytest.raises(RuntimeError, match="index-call-proved"):
        _verify_training_data_contract(config)


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


def test_forensic_failure_evidence_independently_blocks_restart(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    failure_root = run_root / "durable" / "failures"
    failure_root.mkdir(parents=True)
    _reject_forensic_failure_restart(run_root)
    (failure_root / "unsafe_00000000_00000321.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="forensic safety failure.*cannot restart"):
        _reject_forensic_failure_restart(run_root)


def test_prior_pause_receipt_is_mirrored_and_identity_bound(tmp_path: Path) -> None:
    raw = _raw(mode="formal")
    raw["schedule"]["approved_pause_update"] = 3210
    command_manifest = {
        "schema": "nimloth_sft1_query_state_training_command_v1",
        "child_argv": raw["command"]["argv"],
    }
    command_manifest_text = json.dumps(command_manifest, sort_keys=True) + "\n"
    command_digest = hashlib.sha256(command_manifest_text.encode()).hexdigest()
    raw["command"]["identity"] = command_digest
    raw["artifacts"]["file_sha256"][
        raw["output"]["command_manifest_path"]
    ] = command_digest
    config = parse_query_state_training_config(raw)
    run_root = tmp_path / "run"
    controller_root = tmp_path / "controller"
    checkpoint = run_root / "checkpoints" / "update_00001605"
    checkpoint.mkdir(parents=True)
    (checkpoint / "control.json").write_text("{}\n", encoding="utf-8")
    process_identity = "c" * 64
    prior_raw = json.loads(json.dumps(raw))
    prior_raw["schedule"]["approved_pause_update"] = 1605
    prior_config = parse_query_state_training_config(prior_raw)
    process_path = controller_root / "processes" / f"process_{process_identity}.json"
    process_path.parent.mkdir(parents=True)
    process_path.write_text(
        json.dumps(
            {
                "run_identity": query_state_training_run_identity(config),
                "mode": "formal",
                "process_identity": process_identity,
                "config_identity": prior_config.identity,
                "command_identity": prior_config.command["identity"],
                "resume_mode": "fresh",
                "approved_pause_update": 1605,
                "resume_checkpoint": None,
                "resolved_config": prior_raw,
                "command_manifest_text": command_manifest_text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approved boundary lacks.*pause receipt"):
        _authenticate_prior_pause_receipt(
            config,
            completed_update=1605,
            checkpoint=checkpoint,
            run_root=run_root,
            controller_root=controller_root,
        )
    payload = {
        "run_identity": query_state_training_run_identity(config),
        "mode": "formal",
        "status": "paused",
        "update": 1605,
        "checkpoint": str(checkpoint),
        "checkpoint_control_hash": _sha(checkpoint / "control.json"),
        "terminal_primary": False,
        "automatic_formal_extension": False,
        "automatic_sft2_authorization": False,
        "automatic_export": False,
    }
    filename = "pause_update_00001605.json"
    for root in (run_root, controller_root):
        path = root / "pauses" / filename
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert _authenticate_prior_pause_receipt(
        config,
        completed_update=1605,
        checkpoint=checkpoint,
        run_root=run_root,
        controller_root=controller_root,
    )
    process_payload = process_path.read_text(encoding="utf-8")
    tampered_process = json.loads(process_payload)
    tampered_manifest = json.loads(tampered_process["command_manifest_text"])
    tampered_manifest["topology"] = {"world_size": 999}
    tampered_process["command_manifest_text"] = (
        json.dumps(tampered_manifest, sort_keys=True) + "\n"
    )
    process_path.write_text(json.dumps(tampered_process) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prior process evidence provenance"):
        _authenticate_prior_pause_receipt(
            config,
            completed_update=1605,
            checkpoint=checkpoint,
            run_root=run_root,
            controller_root=controller_root,
        )
    process_path.write_text(process_payload, encoding="utf-8")
    process_path.unlink()
    with pytest.raises(ValueError, match="lacks prior process evidence"):
        _authenticate_prior_pause_receipt(
            config,
            completed_update=1605,
            checkpoint=checkpoint,
            run_root=run_root,
            controller_root=controller_root,
        )
    process_path.write_text(process_payload, encoding="utf-8")
    (controller_root / "pauses" / filename).unlink()
    with pytest.raises(ValueError, match="mirrors are incomplete"):
        _authenticate_prior_pause_receipt(
            config,
            completed_update=1605,
            checkpoint=checkpoint,
            run_root=run_root,
            controller_root=controller_root,
        )

    preempt_run = tmp_path / "preempt-run"
    preempt_controller = tmp_path / "preempt-controller"
    preempt_checkpoint = preempt_run / "checkpoints" / "update_00001605"
    preempt_checkpoint.mkdir(parents=True)
    (preempt_checkpoint / "control.json").write_text("{}\n", encoding="utf-8")
    preempt_process = (
        preempt_controller / "processes" / f"process_{process_identity}.json"
    )
    preempt_process.parent.mkdir(parents=True)
    preempt_process.write_text(
        json.dumps(
            {
                "run_identity": query_state_training_run_identity(config),
                "mode": "formal",
                "process_identity": process_identity,
                "config_identity": config.identity,
                "command_identity": config.command["identity"],
                "resume_mode": "fresh",
                "approved_pause_update": 3210,
                "resume_checkpoint": None,
                "resolved_config": raw,
                "command_manifest_text": command_manifest_text,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not _authenticate_prior_pause_receipt(
        config,
        completed_update=1605,
        checkpoint=preempt_checkpoint,
        run_root=preempt_run,
        controller_root=preempt_controller,
    )


def test_exact_restart_authority_authenticates_marker_control_and_index(
    tmp_path: Path,
) -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    checkpoint = tmp_path / "run" / "checkpoints" / "update_00000321"
    checkpoint.mkdir(parents=True)
    run_identity = query_state_training_run_identity(config)
    base = {
        "identity": {
            "config_identity": run_identity,
            "source_commit": config.source["commit"],
            "source_manifest_identity": config.source["source_manifest_identity"],
            "world_size": 8,
            "experiment_mode": "formal",
            "run_identity": run_identity,
        },
        "config_identity": run_identity,
        "source_commit": config.source["commit"],
        "global_step": 321,
        "data_cursor": {"next_update": 322},
    }
    checkpoint_identity = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with_checkpoint = {**base, "checkpoint_identity": checkpoint_identity}
    control_hash = hashlib.sha256(
        json.dumps(
            with_checkpoint,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    control_path = checkpoint / "control.json"
    control_path.write_text(
        json.dumps({**with_checkpoint, "control_hash": control_hash}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    control_sha256 = _sha(control_path)
    (checkpoint / "COMPLETED").write_text(
        f"control_sha256={control_sha256}\n",
        encoding="utf-8",
    )
    index_path = checkpoint.parents[1] / "durable" / "authoritative_index.json"
    index_path.parent.mkdir()
    index = {
        "mode": "formal",
        "run_identity": run_identity,
        "entries": [{
            "run_identity": run_identity,
            "end_update": 321,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_control_hash": control_sha256,
        }],
    }
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
    assert _authenticated_exact_restart_update(
        config,
        checkpoint=checkpoint,
        run_root=checkpoint.parents[1],
    ) == 321

    (checkpoint / "COMPLETED").write_text(
        "control_sha256=" + ("0" * 64) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="marker/control hash"):
        _authenticated_exact_restart_update(
            config,
            checkpoint=checkpoint,
            run_root=checkpoint.parents[1],
        )

    (checkpoint / "COMPLETED").write_text(
        f"control_sha256={control_sha256}\n",
        encoding="utf-8",
    )
    tampered_control = json.loads(control_path.read_text(encoding="utf-8"))
    tampered_control["control_hash"] = "0" * 64
    control_path.write_text(json.dumps(tampered_control) + "\n", encoding="utf-8")
    (checkpoint / "COMPLETED").write_text(
        f"control_sha256={_sha(control_path)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="control identity"):
        _authenticated_exact_restart_update(
            config,
            checkpoint=checkpoint,
            run_root=checkpoint.parents[1],
        )

    control_path.write_text(
        json.dumps({**with_checkpoint, "control_hash": control_hash}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    control_sha256 = _sha(control_path)
    (checkpoint / "COMPLETED").write_text(
        f"control_sha256={control_sha256}\n",
        encoding="utf-8",
    )
    index["entries"][0]["end_update"] = 322
    index["entries"][0]["checkpoint_control_hash"] = control_sha256
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint/index identity"):
        _authenticated_exact_restart_update(
            config,
            checkpoint=checkpoint,
            run_root=checkpoint.parents[1],
        )


def test_exact_restart_free_space_budget_counts_only_remaining_commits() -> None:
    config = parse_query_state_training_config(_raw(mode="formal"))
    estimate = int(config.output["checkpoint_estimated_bytes"])
    fresh_required = _required_output_free_bytes(
        config,
        completed_checkpoint_update=0,
    )
    assert fresh_required == int(config.output["minimum_free_bytes"]) + (
        50 * estimate
    )
    paused_raw = _raw(mode="formal")
    paused_raw["schedule"]["approved_pause_update"] = 1605
    paused = parse_query_state_training_config(paused_raw)
    assert _required_output_free_bytes(
        paused,
        completed_checkpoint_update=0,
    ) == 402_500_000_000
    with pytest.raises(ValueError, match="must exceed restored update"):
        _required_output_free_bytes(paused, completed_checkpoint_update=1605)
    continued_raw = _raw(mode="formal")
    continued_raw["schedule"]["approved_pause_update"] = 3210
    continued = parse_query_state_training_config(continued_raw)
    assert _required_output_free_bytes(
        continued,
        completed_checkpoint_update=1605,
    ) == 402_500_000_000
    terminal_raw = _raw(mode="formal")
    terminal_raw["schedule"]["approved_pause_update"] = 16050
    terminal_window = parse_query_state_training_config(terminal_raw)
    assert _required_output_free_bytes(
        terminal_window,
        completed_checkpoint_update=14445,
    ) == 402_500_000_000
    unbounded_restart = parse_query_state_training_config(_raw(mode="formal"))
    with pytest.raises(ValueError, match="approved pause.*required for restart"):
        _required_output_free_bytes(
            unbounded_restart,
            completed_checkpoint_update=1605,
        )
    with pytest.raises(ValueError, match="commit boundary"):
        _required_output_free_bytes(config, completed_checkpoint_update=400)


def test_live_preflight_requires_checkpoint_budget_plus_free_space_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, environment = _resolved_contract(tmp_path, launch_locked=False)
    raw["output"]["checkpoint_estimated_bytes"] = 1_000_000
    raw["output"]["checkpoint_budget_bytes"] = 8_000_000
    config = parse_query_state_training_config(raw)
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight._verify_training_data_contract",
        lambda _config: None,
    )
    monkeypatch.setattr(
        "nimloth.training.sft1.query_state_training_preflight.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=8_000_000),
    )
    with pytest.raises(OSError, match="checkpoint budget.*minimum-free"):
        verify_query_state_training_preflight(
            config,
            repo_root=Path(config.source["repo_root"]),
            current_argv=config.command["argv"],
            environ=environment,
        )


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
