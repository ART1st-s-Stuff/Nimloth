"""Live CPU-only preflight for the pilot/formal Query-State owner.

The gate verifies the exact source checkout and every launch-owned asset before
CUDA or process-group initialization.  It does not resolve fields, create an
output directory, submit Slurm, or initialize tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import transformers

from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.query_state_training_config import (
    QueryStateTrainingConfig,
    parse_query_state_training_config,
    query_state_training_lineage_identity,
    query_state_training_run_identity,
)
from nimloth.training.sft1.query_state_training_migration import (
    parse_legacy_prior_process_config,
    validate_query_state_execution_migration_contract,
)
from nimloth.training.sft1.query_state_training_manifest import (
    deserialize_generation_format_manifest,
    deserialize_query_state_training_manifest,
    deserialize_query_state_validation_split,
    rows_for_training_mode,
    rows_for_validation_mode,
    validate_query_state_row_audit,
)
from nimloth.training.sft1.real_rows import index_early4_rows


@dataclass(frozen=True)
class QueryStateTrainingPreflightEvidence:
    config_identity: str
    lifecycle_state: str
    source_commit: str
    clean_source: bool
    recursive_submodule_commits: Mapping[str, str]
    verified_file_count: int
    processor_identity: str
    tokenizer_identity: str
    dino_identity: str
    output_ownership_verified: bool
    resource_contract_verified: bool
    cuda_entered: bool = False


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _required_output_free_bytes(
    config: QueryStateTrainingConfig,
    *,
    completed_checkpoint_update: int,
) -> int:
    cadence = int(config.schedule["checkpoint_cadence_updates"])
    terminal_update = int(config.schedule["max_updates"])
    schedule_start = int(config.schedule["schedule_start_update"])
    if config.mode == "visual_only_forensic_fork":
        resume_mode = config.initialization["resume_mode"]
        if (
            completed_checkpoint_update < schedule_start
            or completed_checkpoint_update > terminal_update
            or (completed_checkpoint_update - schedule_start) % cadence
            or (resume_mode == "fresh" and completed_checkpoint_update != schedule_start)
            or (resume_mode == "exact_restart" and completed_checkpoint_update <= schedule_start)
        ):
            raise ValueError("visual fork launch/restart boundary is invalid")
        return int(config.output["minimum_free_bytes"]) + int(
            config.output["checkpoint_budget_bytes"]
        )
    approved_pause_update = int(config.schedule["approved_pause_update"])
    authorized_stop_update = approved_pause_update or terminal_update
    if (
        isinstance(completed_checkpoint_update, bool)
        or not isinstance(completed_checkpoint_update, int)
        or completed_checkpoint_update < 0
        or completed_checkpoint_update > terminal_update
        or completed_checkpoint_update % cadence
    ):
        raise ValueError("completed checkpoint update is not a commit boundary")
    if completed_checkpoint_update > 0 and approved_pause_update == 0:
        raise ValueError("an approved pause boundary is required for restart")
    if approved_pause_update and completed_checkpoint_update >= approved_pause_update:
        raise ValueError("approved pause update must exceed restored update")
    remaining_commits = (authorized_stop_update - completed_checkpoint_update) // cadence
    return int(config.output["minimum_free_bytes"]) + (
        remaining_commits * int(config.output["checkpoint_estimated_bytes"])
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_process_requires_pause_receipt(
    config: QueryStateTrainingConfig,
    *,
    completed_update: int,
    controller_root: Path,
) -> bool:
    process_root = Path(controller_root) / "processes"
    process_paths = sorted(process_root.glob("process_*.json"))
    if not process_paths:
        raise ValueError("Query-State exact restart lacks prior process evidence")
    run_identity = query_state_training_run_identity(config)
    requires_pause = False
    for path in process_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Query-State prior process evidence is invalid") from error
        process_identity = payload.get("process_identity") if isinstance(payload, dict) else None
        resolved_config = payload.get("resolved_config") if isinstance(payload, dict) else None
        command_manifest_text = (
            payload.get("command_manifest_text") if isinstance(payload, dict) else None
        )
        try:
            prior_config = (
                parse_query_state_training_config(resolved_config)
                if isinstance(resolved_config, Mapping)
                and "execution_migration" in resolved_config
                else parse_legacy_prior_process_config(resolved_config)
            )
            if not isinstance(command_manifest_text, str):
                raise ValueError("command manifest text is absent")
            command_manifest = json.loads(command_manifest_text)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Query-State prior process resolved config is invalid"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("run_identity") != run_identity
            or payload.get("mode") != "formal"
            or not _is_sha256(process_identity)
            or path.name != f"process_{process_identity}.json"
            or not _is_sha256(payload.get("config_identity"))
            or prior_config.identity != payload.get("config_identity")
            or query_state_training_run_identity(prior_config) != run_identity
            or not _is_sha256(payload.get("command_identity"))
            or prior_config.command["identity"] != payload.get("command_identity")
            or hashlib.sha256(command_manifest_text.encode()).hexdigest()
            != payload.get("command_identity")
            or not isinstance(command_manifest, Mapping)
            or command_manifest.get("schema")
            != "nimloth_sft1_query_state_training_command_v1"
            or command_manifest.get("child_argv")
            != list(prior_config.command["argv"])
            or payload.get("resume_mode")
            not in {"fresh", "crash_replay", "exact_restart"}
            or isinstance(payload.get("approved_pause_update"), bool)
            or not isinstance(payload.get("approved_pause_update"), int)
            or payload.get("approved_pause_update") < 0
            or prior_config.schedule["approved_pause_update"]
            != payload.get("approved_pause_update")
        ):
            raise ValueError("Query-State prior process evidence provenance is invalid")
        requires_pause = requires_pause or (
            payload["approved_pause_update"] == completed_update
        )
    return requires_pause


def _authenticate_prior_pause_receipt(
    config: QueryStateTrainingConfig,
    *,
    completed_update: int,
    checkpoint: Path,
    run_root: Path,
    controller_root: Path,
) -> bool:
    pause_required = _prior_process_requires_pause_receipt(
        config,
        completed_update=completed_update,
        controller_root=controller_root,
    )
    filename = f"pause_update_{completed_update:08d}.json"
    run_receipt = Path(run_root) / "pauses" / filename
    controller_receipt = Path(controller_root) / "pauses" / filename
    exists = (run_receipt.is_file(), controller_receipt.is_file())
    if exists == (False, False):
        if pause_required:
            raise ValueError("Query-State approved boundary lacks its pause receipt")
        return False
    if exists != (True, True):
        raise ValueError("Query-State prior pause receipt mirrors are incomplete")
    try:
        run_payload = json.loads(run_receipt.read_text(encoding="utf-8"))
        controller_payload = json.loads(
            controller_receipt.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Query-State prior pause receipt is invalid") from error
    expected_run_identity = query_state_training_run_identity(config)
    control_hash = _sha256_file(Path(checkpoint) / "control.json")
    if (
        run_payload != controller_payload
        or not isinstance(run_payload, dict)
        or run_payload.get("run_identity") != expected_run_identity
        or run_payload.get("mode") != "formal"
        or run_payload.get("status") != "paused"
        or run_payload.get("update") != completed_update
        or Path(str(run_payload.get("checkpoint"))).resolve()
        != Path(checkpoint).resolve()
        or run_payload.get("checkpoint_control_hash") != control_hash
        or run_payload.get("terminal_primary") is not False
        or run_payload.get("automatic_formal_extension") is not False
        or run_payload.get("automatic_sft2_authorization") is not False
        or run_payload.get("automatic_export") is not False
    ):
        raise ValueError("Query-State prior pause receipt provenance is invalid")
    return True


def _authenticated_exact_restart_update(
    config: QueryStateTrainingConfig,
    *,
    checkpoint: Path,
    run_root: Path,
) -> int:
    control_path = checkpoint / "control.json"
    marker_path = checkpoint / "COMPLETED"
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
        marker = marker_path.read_text(encoding="utf-8")
        index = json.loads(
            (run_root / "durable" / "authoritative_index.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Query-State exact restart authority is unreadable"
        ) from error
    if not isinstance(index, dict):
        raise ValueError("Query-State exact restart authoritative index is invalid")
    control_sha256 = _sha256_file(control_path)
    if marker != f"control_sha256={control_sha256}\n" or not isinstance(
        control, dict
    ):
        raise ValueError("Query-State exact restart marker/control hash mismatch")
    control_hash = control.get("control_hash")
    checkpoint_identity = control.get("checkpoint_identity")
    canonical_with_checkpoint = {
        key: value for key, value in control.items() if key != "control_hash"
    }
    canonical_base = {
        key: value
        for key, value in canonical_with_checkpoint.items()
        if key != "checkpoint_identity"
    }
    if (
        not isinstance(control_hash, str)
        or hashlib.sha256(
            json.dumps(
                canonical_with_checkpoint,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        != control_hash
        or not isinstance(checkpoint_identity, str)
        or hashlib.sha256(
            json.dumps(
                canonical_base,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        != checkpoint_identity
    ):
        raise ValueError("Query-State exact restart control identity mismatch")
    identity = control.get("identity")
    expected_run_identity = query_state_training_lineage_identity(config)
    expected_source_commit = (
        config.execution_migration["anchor_source_commit"]
        if config.execution_migration["enabled"] is True
        else config.source["commit"]
    )
    expected_source_manifest = (
        config.execution_migration["anchor_source_manifest_identity"]
        if config.execution_migration["enabled"] is True
        else config.source["source_manifest_identity"]
    )
    update = control.get("global_step")
    data_cursor = control.get("data_cursor")
    entries = index.get("entries") if isinstance(index, dict) else None
    latest = entries[-1] if isinstance(entries, list) and entries else None
    if (
        not isinstance(identity, dict)
        or identity.get("config_identity") != expected_run_identity
        or identity.get("run_identity") != expected_run_identity
        or identity.get("source_commit") != expected_source_commit
        or identity.get("source_manifest_identity") != expected_source_manifest
        or identity.get("world_size") != config.resources["world_size"]
        or identity.get("experiment_mode") != config.mode
        or control.get("config_identity") != expected_run_identity
        or control.get("source_commit") != expected_source_commit
        or not isinstance(update, int)
        or isinstance(update, bool)
        or not isinstance(data_cursor, dict)
        or data_cursor.get("next_update") != update + 1
        or index.get("mode") != config.mode
        or index.get("run_identity") != identity.get("run_identity")
        or not isinstance(latest, dict)
        or latest.get("run_identity") != identity.get("run_identity")
        or latest.get("end_update") != update
        or Path(str(latest.get("checkpoint_path"))).resolve()
        != checkpoint.resolve()
        or latest.get("checkpoint_control_hash") != control_sha256
        or latest.get("checkpoint_payload_present", True) is not True
        or latest.get("resumable", True) is not True
    ):
        raise ValueError(
            "Query-State exact restart checkpoint/index identity mismatch"
        )
    provenance = control.get("execution_provenance")
    anchor_checkpoint = Path(
        str(config.execution_migration["anchor_checkpoint_path"])
    ).resolve() if config.execution_migration["enabled"] is True else None
    if config.execution_migration["enabled"] is True:
        if checkpoint.resolve() == anchor_checkpoint:
            if provenance is not None:
                raise ValueError("migration anchor checkpoint unexpectedly has execution provenance")
        elif not isinstance(provenance, Mapping):
            raise ValueError("future restart lost execution migration provenance")
        else:
            anchor = provenance.get("anchor")
            chain = provenance.get("execution_chain")
            if (
                not isinstance(anchor, Mapping)
                or anchor.get("run_identity") != expected_run_identity
                or anchor.get("source_commit") != expected_source_commit
                or anchor.get("source_manifest_path")
                != config.execution_migration["anchor_source_manifest_path"]
                or anchor.get("source_manifest_identity") != expected_source_manifest
                or anchor.get("partition") != "preempt"
                or not isinstance(chain, list)
                or not chain
                or not isinstance(chain[-1], Mapping)
                or chain[-1].get("source_commit")
                != config.execution_migration["execution_source_commit"]
                or chain[-1].get("source_manifest_path")
                != config.execution_migration["execution_source_manifest_path"]
                or chain[-1].get("source_manifest_identity")
                != config.execution_migration["execution_source_manifest_identity"]
                or chain[-1].get("partition")
                != config.execution_migration["execution_partition"]
            ):
                raise ValueError("future restart execution migration chain is invalid")
    return update


def _authenticate_execution_migration(
    config: QueryStateTrainingConfig,
    *,
    checkpoint: Path,
    run_root: Path,
    environ: Mapping[str, str],
    require_actual_partition: bool = True,
) -> None:
    migration = config.execution_migration
    if migration["enabled"] is not True:
        return
    process_path = Path(str(migration["prior_process_path"])).resolve()
    anchor_checkpoint = Path(str(migration["anchor_checkpoint_path"])).resolve()
    anchor_control_path = anchor_checkpoint / "control.json"
    index_path = Path(str(migration["anchor_index_path"])).resolve()
    live_index_path = (Path(run_root) / "durable" / "authoritative_index.json").resolve()
    if (
        process_path.is_symlink()
        or not process_path.is_file()
        or _sha256_file(process_path) != migration["prior_process_sha256"]
        or anchor_checkpoint.parent
        != (Path(run_root).resolve() / "checkpoints")
        or anchor_checkpoint.name != "update_00004815"
        or not (anchor_checkpoint / "COMPLETED").is_file()
        or (anchor_checkpoint / "COMPLETED").read_text(encoding="utf-8")
        != f"control_sha256={migration['anchor_control_sha256']}\n"
        or _sha256_file(anchor_control_path) != migration["anchor_control_sha256"]
        or index_path == live_index_path
        or index_path.is_symlink()
        or not index_path.is_file()
        or _sha256_file(index_path) != migration["anchor_index_sha256"]
    ):
        raise ValueError("execution migration immutable anchor evidence mismatch")
    try:
        process = json.loads(process_path.read_text(encoding="utf-8"))
        anchor_index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("execution migration prior evidence is invalid") from error
    anchor_entries = (
        anchor_index.get("entries") if isinstance(anchor_index, Mapping) else None
    )
    anchor_latest = (
        anchor_entries[-1] if isinstance(anchor_entries, list) and anchor_entries else None
    )
    if (
        not isinstance(anchor_index, Mapping)
        or anchor_index.get("mode") != "visual_only_forensic_fork"
        or anchor_index.get("run_identity") != migration["anchor_run_identity"]
        or not isinstance(anchor_latest, Mapping)
        or anchor_latest.get("run_identity") != migration["anchor_run_identity"]
        or anchor_latest.get("end_update") != 4815
        or Path(str(anchor_latest.get("checkpoint_path"))).resolve()
        != anchor_checkpoint
        or anchor_latest.get("checkpoint_control_hash")
        != migration["anchor_control_sha256"]
        or anchor_latest.get("checkpoint_payload_present", True) is not True
        or anchor_latest.get("resumable", True) is not True
    ):
        raise ValueError("execution migration anchor index snapshot is invalid")
    prior_raw = process.get("resolved_config") if isinstance(process, Mapping) else None
    actual_partition = environ.get("SLURM_JOB_PARTITION")
    if require_actual_partition and actual_partition != migration["execution_partition"]:
        raise ValueError("actual Slurm partition does not match execution migration")
    prior = validate_query_state_execution_migration_contract(
        config,
        prior_raw,
        actual_partition=actual_partition if require_actual_partition else None,
    )
    if (
        process.get("run_identity") != migration["anchor_run_identity"]
        or process.get("mode") != "visual_only_forensic_fork"
        or process.get("config_identity") != prior.identity
    ):
        raise ValueError("execution migration prior process provenance mismatch")


def _reject_visual_fixed_budget_completion_restart(
    config: QueryStateTrainingConfig,
    *,
    run_root: Path,
    controller_root: Path,
) -> None:
    if (
        config.mode != "visual_only_forensic_fork"
        or config.initialization["resume_mode"] != "exact_restart"
    ):
        return
    marker = "VISUAL_FIXED_BUDGET_COMPLETED.json"
    if (Path(run_root) / marker).exists() or (Path(controller_root) / marker).exists():
        raise RuntimeError("visual fixed-budget completed run cannot restart")


def _reject_forensic_failure_restart(run_root: Path) -> None:
    failure_root = Path(run_root) / "durable" / "failures"
    if not failure_root.exists():
        return
    blockers = sorted(
        path
        for pattern in ("unsafe_*.json", "forensic_save_failed_*.json")
        for path in failure_root.glob(pattern)
        if path.is_file()
    )
    if blockers:
        raise RuntimeError(
            "Query-State run has forensic safety failure evidence and cannot restart: "
            + ", ".join(path.name for path in blockers)
        )


def _run(repo: Path, *argv: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *argv),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Query-State preflight JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Query-State preflight JSON must be an object: {path}")
    return value


def _verify_manifest_identity(path: Path, expected: str, *, label: str) -> None:
    value = _json(path)
    if value.get("identity") != expected:
        raise ValueError(f"Query-State {label} manifest identity mismatch")


def _verify_training_data_contract(config: QueryStateTrainingConfig) -> None:
    """Re-index the immutable archives and rebuild both canonical manifests."""

    hashes = config.artifacts["file_sha256"]
    contract = SimpleNamespace(data=SimpleNamespace(
        train_jsonl=config.data["train_source_path"],
        validation_jsonl=config.data["validation_source_path"],
        train_sha256=hashes[config.data["train_source_path"]],
        validation_sha256=hashes[config.data["validation_source_path"]],
        train_split="train",
        validation_split="val",
    ))
    rows, audit = index_early4_rows(
        contract,
        enforce_approved_counts=False,
    )
    validate_query_state_row_audit(audit)
    training = deserialize_query_state_training_manifest(
        Path(str(config.data["train_manifest_path"])),
        rows=rows,
        expected_identity=str(config.data["train_manifest_identity"]),
        expected_mode=config.mode,
        expected_rows=int(config.data["train_rows"]),
        expected_seed=int(config.schedule["seed"]),
    )
    validation = deserialize_query_state_validation_split(
        Path(str(config.data["validation_manifest_path"])),
        rows=rows,
        expected_identity=str(config.data["validation_manifest_identity"]),
    )
    if len(rows_for_training_mode(training, mode=config.mode)) != int(config.data["train_rows"]):
        raise ValueError("Query-State mode-specific training manifest row count changed")
    selected_validation = rows_for_validation_mode(validation, mode=config.mode)
    if not selected_validation:
        raise ValueError("Query-State mode-specific validation split is empty")
    generation = deserialize_generation_format_manifest(
        Path(str(config.validation["generation_format_manifest_path"])),
        rows=rows,
        validation_split=validation,
        expected_identity=str(config.validation["generation_format_manifest_identity"]),
        expected_mode=config.mode,
    )
    if not generation.entries:
        raise ValueError("Query-State production generation-format rows are empty")


def _submodule_commits(repo: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    output = _run(repo, "submodule", "status", "--recursive")
    for line in output.splitlines():
        if not line:
            continue
        marker = line[0] if line[0] in {"-", "+", "U", " "} else " "
        fields = (line[1:] if marker != " " or line.startswith(" ") else line).split()
        if marker in {"+", "U"} or len(fields) < 2:
            raise ValueError("Query-State source submodule checkout is dirty or invalid")
        commit, path = fields[0], fields[1]
        values[path] = commit
    return values


def _verify_environment(config: QueryStateTrainingConfig, environ: Mapping[str, str]) -> None:
    expected = config.environment
    paths = {
        "PYTHONPYCACHEPREFIX": expected["pycache_prefix"],
        "HF_HOME": expected["hf_home"],
        "HF_HUB_CACHE": expected["hf_hub_cache"],
    }
    for name, value in paths.items():
        actual = environ.get(name)
        if not actual or Path(actual).resolve() != Path(str(value)).resolve():
            raise ValueError(f"Query-State environment {name} identity mismatch")
    expected_values = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": str(expected["python_hash_seed"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NCCL_SOCKET_IFNAME": str(expected["nccl_socket_ifname"]),
        "NCCL_IB_DISABLE": str(expected["nccl_ib_disable"]),
    }
    for name, value in expected_values.items():
        if environ.get(name) != value:
            raise ValueError(f"Query-State environment {name} must equal {value}")
    if environ.get("TRANSFORMERS_CACHE"):
        raise ValueError("Query-State environment rejects TRANSFORMERS_CACHE")
    if Path(sys.executable).resolve() != Path(str(expected["python_executable"])).resolve():
        raise ValueError("Query-State Python executable identity mismatch")
    actual_versions = (platform.python_version(), torch.__version__, transformers.__version__)
    expected_versions = (
        expected["python_version"], expected["torch_version"], expected["transformers_version"]
    )
    if actual_versions != expected_versions:
        raise ValueError(
            f"Query-State package identity mismatch: {actual_versions} != {expected_versions}"
        )


def validate_query_state_distributed_topology(
    records: Sequence[Mapping[str, object]],
    *,
    resources: Mapping[str, object],
    environment: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    """Validate the exact two-node/four-local-rank runtime assignment."""

    world_size = int(resources["world_size"])
    nodes = int(resources["nodes"])
    local_world_size = int(resources["gpus_per_node"])
    expected_fields = {
        "rank",
        "group_rank",
        "local_rank",
        "hostname",
        "cuda_visible_devices",
        "nccl_socket_ifname",
        "nccl_ib_disable",
    }
    if len(records) != world_size:
        raise ValueError("Query-State topology gate rank count mismatch")
    normalized: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_fields:
            raise ValueError("Query-State topology gate record fields mismatch")
        value = dict(record)
        for name in ("rank", "group_rank", "local_rank"):
            if isinstance(value[name], bool) or not isinstance(value[name], int):
                raise ValueError("Query-State topology gate rank type mismatch")
        if any(
            not isinstance(value[name], str) or not value[name]
            for name in (
                "hostname",
                "cuda_visible_devices",
                "nccl_socket_ifname",
                "nccl_ib_disable",
            )
        ):
            raise ValueError("Query-State topology gate string identity is invalid")
        if value["nccl_socket_ifname"] != environment["nccl_socket_ifname"] or value[
            "nccl_ib_disable"
        ] != environment["nccl_ib_disable"]:
            raise ValueError("Query-State topology gate NCCL environment mismatch")
        normalized.append(value)
    normalized.sort(key=lambda value: int(value["rank"]))
    if [value["rank"] for value in normalized] != list(range(world_size)):
        raise ValueError("Query-State topology gate global ranks are incomplete")
    hostnames: set[str] = set()
    for group_rank in range(nodes):
        group = [value for value in normalized if value["group_rank"] == group_rank]
        if len(group) != local_world_size:
            raise ValueError("Query-State topology gate node rank count mismatch")
        hosts = {str(value["hostname"]) for value in group}
        visible = {str(value["cuda_visible_devices"]) for value in group}
        if len(hosts) != 1 or len(visible) != 1:
            raise ValueError("Query-State topology gate node identity is inconsistent")
        if sorted(int(value["local_rank"]) for value in group) != list(
            range(local_world_size)
        ):
            raise ValueError("Query-State topology gate local ranks are incomplete")
        if sorted(int(value["rank"]) for value in group) != list(
            range(group_rank * local_world_size, (group_rank + 1) * local_world_size)
        ):
            raise ValueError("Query-State topology gate global/local rank mapping changed")
        if len(next(iter(visible)).split(",")) != local_world_size:
            raise ValueError("Query-State topology gate CUDA visibility count mismatch")
        hostnames.update(hosts)
    if len(hostnames) != nodes:
        raise ValueError("Query-State topology gate physical node count mismatch")
    return tuple(normalized)


def _verify_id176_and_dino(config: QueryStateTrainingConfig) -> tuple[str, str]:
    actor = Path(str(config.initialization["actor_checkpoint"])).resolve()
    actor_config = _json(actor / "config.json")
    if (
        actor_config.get("hidden_size") != 2048
        or actor_config.get("nimloth_latent_token_count") != 16
        or actor_config.get("nimloth_latent_query_mode") != "inject"
        or tuple(actor_config.get("nimloth_action_token_ids", ()))
        != tuple(config.model["action_token_ids"])
    ):
        raise ValueError("Query-State ID176 K16/action/hidden contract mismatch")
    index = _json(actor / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Query-State ID176 model index is invalid")
    for name in set(weight_map.values()):
        if not isinstance(name, str) or Path(name).name != name or not (actor / name).is_file():
            raise ValueError("Query-State ID176 model shard index is unsafe or incomplete")
    completion = actor.parent / "complete.marker"
    action_head = actor / "action_head_repair.pt"
    if not completion.is_file() or not action_head.is_file():
        raise FileNotFoundError("Query-State ID176 completion/action-head evidence is incomplete")
    processor_names = (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    actor_files = [
        completion,
        actor / "config.json",
        actor / "model.safetensors.index.json",
        action_head,
        *(actor / str(name) for name in sorted(set(weight_map.values()))),
        *(actor / name for name in processor_names),
    ]
    actor_identity = hashlib.sha256(
        json.dumps(
            {str(path): _sha256(path) for path in sorted(actor_files)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if actor_identity != config.initialization["actor_checkpoint_identity"]:
        raise ValueError("Query-State ID176 checkpoint identity mismatch")
    processor = audit_id176_processor_identity(actor)
    expected = (
        config.model["processor_identity"],
        config.model["tokenizer_identity"],
        config.model["template_identity"],
        config.model["token_table_identity"],
        tuple(config.model["action_token_ids"]),
    )
    actual = (
        processor.processor_sha256,
        processor.tokenizer_sha256,
        processor.prompt_template_sha256,
        processor.token_table_sha256,
        processor.action_token_ids,
    )
    if actual != expected:
        raise ValueError("Query-State ID176 processor/token identity mismatch")

    dino = Path(str(config.model["dino_snapshot_path"])).resolve()
    if not dino.is_dir() or not (dino / "config.json").is_file() or not (
        dino / "preprocessor_config.json"
    ).is_file():
        raise FileNotFoundError("Query-State pinned DINO snapshot metadata is incomplete")
    weights = tuple(dino.glob("*.safetensors")) + tuple(dino.glob("pytorch_model*.bin"))
    if not weights or any(not path.is_file() for path in weights):
        raise FileNotFoundError("Query-State pinned DINO snapshot weights are incomplete")
    required_dino = (
        "facebook/dinov2-large@47b73eefe95e8d44ec3623f8890bd894b6ea2d6c:"
        "7d65a7de8788e87d:1024:grid4"
    )
    if config.model["dino_identity"] != required_dino:
        raise ValueError("Query-State DINO identity differs from the reviewed owner")
    return processor.processor_sha256, processor.tokenizer_sha256


def verify_query_state_training_preflight(
    config: QueryStateTrainingConfig,
    *,
    repo_root: Path,
    current_argv: Sequence[str],
    environ: Mapping[str, str] | None = None,
    require_runtime_partition: bool = False,
) -> QueryStateTrainingPreflightEvidence:
    """Verify exact source/assets/command/output/resources without entering CUDA."""

    if not isinstance(config, QueryStateTrainingConfig) or config.lifecycle_state == "template":
        raise PermissionError("Query-State live preflight requires a preflight-locked config")
    repo = Path(repo_root).resolve()
    if repo != Path(str(config.source["repo_root"])).resolve():
        raise ValueError("Query-State source repository path mismatch")
    if _run(repo, "rev-parse", "HEAD") != config.source["commit"]:
        raise ValueError("Query-State source commit mismatch")
    if _run(repo, "symbolic-ref", "--short", "HEAD") != config.source["branch"]:
        raise ValueError("Query-State source branch mismatch")
    if _run(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Query-State source checkout is dirty")
    submodules = _submodule_commits(repo)
    if submodules != dict(config.source["submodule_commits"]):
        raise ValueError("Query-State recursive submodule commit mismatch")

    artifacts = config.artifacts["file_sha256"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("Query-State artifact hash inventory is empty")
    for raw_path, expected in artifacts.items():
        path = Path(str(raw_path))
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"Query-State hashed artifact is missing: {path}")
        if _sha256(path) != expected:
            raise ValueError(f"Query-State artifact hash mismatch: {path}")

    _verify_manifest_identity(
        Path(str(config.source["source_manifest_path"])),
        str(config.source["source_manifest_identity"]),
        label="source",
    )
    _verify_manifest_identity(
        Path(str(config.data["train_manifest_path"])),
        str(config.data["train_manifest_identity"]),
        label="train",
    )
    _verify_manifest_identity(
        Path(str(config.data["validation_manifest_path"])),
        str(config.data["validation_manifest_identity"]),
        label="validation",
    )
    _verify_manifest_identity(
        Path(str(config.validation["generation_format_manifest_path"])),
        str(config.validation["generation_format_manifest_identity"]),
        label="production generation-format",
    )
    _verify_training_data_contract(config)
    if config.mode == "visual_only_forensic_fork":
        from nimloth.training.sft1.query_state_visual_forensic_fork import (
            authenticate_visual_fork_ancestor,
        )

        authenticate_visual_fork_ancestor(
            config,
            verify_payload_hashes=False,
        )
    processor_identity, tokenizer_identity = _verify_id176_and_dino(config)
    _verify_environment(config, environ or os.environ)

    command_path = Path(str(config.output["command_manifest_path"])).resolve()
    if _sha256(command_path) != config.command["identity"]:
        raise ValueError("Query-State approved command manifest hash mismatch")
    command = _json(command_path)
    if command.get("schema") != "nimloth_sft1_query_state_training_command_v1":
        raise ValueError("Query-State approved command manifest schema mismatch")
    if command.get("child_argv") != list(config.command["argv"]):
        raise ValueError("Query-State approved child command differs from config")
    expected_entry = (repo / "experiments/training/sft1/query_state_train.py").resolve()
    if (
        len(config.command["argv"]) != 6
        or Path(str(config.command["argv"][0])).resolve() != Path(sys.executable).resolve()
        or Path(str(config.command["argv"][1])).resolve() != expected_entry
        or not expected_entry.is_file()
        or _run(repo, "ls-files", "--error-unmatch", str(expected_entry.relative_to(repo)))
        != str(expected_entry.relative_to(repo))
    ):
        raise ValueError("Query-State approved command source/interpreter identity mismatch")
    if list(current_argv) != list(config.command["argv"]):
        raise ValueError("Query-State runtime command differs from approval")
    topology = command.get("topology")
    expected_topology = {
        "backend": config.resources["backend"],
        "nodes": config.resources["nodes"],
        "gpus_per_node": config.resources["gpus_per_node"],
        "world_size": config.resources["world_size"],
        "nccl_socket_ifname": config.environment["nccl_socket_ifname"],
        "nccl_ib_disable": config.environment["nccl_ib_disable"],
    }
    if topology != expected_topology or config.resources["backend"] != "nccl":
        raise ValueError("Query-State command/resource topology mismatch")
    allowlist = config.resources["gpu_model_allowlist"]
    if not isinstance(allowlist, (tuple, list)) or not allowlist or any(
        not isinstance(value, str) or not value for value in allowlist
    ):
        raise ValueError("Query-State GPU resource allowlist is invalid")

    resume_mode = config.initialization["resume_mode"]
    completed_checkpoint_update = int(config.schedule["schedule_start_update"])
    run_root = Path(str(config.output["run_root"]))
    controller_root = Path(str(config.output["controller_root"]))
    if resume_mode == "fresh":
        for field, path in (("run_root", run_root), ("controller_root", controller_root)):
            if path.exists():
                raise FileExistsError(f"Query-State output ownership collision: {field}")
    else:
        if not run_root.is_dir() or not controller_root.is_dir():
            raise FileNotFoundError(
                "Query-State restart/replay requires the existing run/controller owner"
            )
        _reject_visual_fixed_budget_completion_restart(
            config,
            run_root=run_root,
            controller_root=controller_root,
        )
        terminal_names = (
            "COMPLETED.json", "FAILED.json", "PREEMPTED.json", "VALIDATOR_FAILED.json"
        )
        if any(
            (run_root / name).exists() or (controller_root / name).exists()
            for name in terminal_names
        ):
            raise RuntimeError("Query-State terminal run cannot restart or replay")
        _reject_forensic_failure_restart(run_root)
        if resume_mode == "exact_restart":
            checkpoint = Path(str(config.initialization["resume_checkpoint"]))
            control_path = checkpoint / "control.json"
            if not (checkpoint / "COMPLETED").is_file() or not control_path.is_file():
                raise FileNotFoundError("Query-State exact restart checkpoint is incomplete")
            _authenticate_execution_migration(
                config,
                checkpoint=checkpoint,
                run_root=run_root,
                environ=environ or os.environ,
                require_actual_partition=require_runtime_partition,
            )
            completed_checkpoint_update = _authenticated_exact_restart_update(
                config,
                checkpoint=checkpoint,
                run_root=run_root,
            )
            if config.mode == "formal":
                _authenticate_prior_pause_receipt(
                    config,
                    completed_update=completed_checkpoint_update,
                    checkpoint=checkpoint,
                    run_root=run_root,
                    controller_root=controller_root,
                )
            cadence = int(config.schedule["checkpoint_cadence_updates"])
            terminal_update = int(config.schedule["max_updates"])
            if (
                isinstance(completed_checkpoint_update, bool)
                or not isinstance(completed_checkpoint_update, int)
                or completed_checkpoint_update < 1
                or completed_checkpoint_update > terminal_update
                or completed_checkpoint_update % cadence
            ):
                raise ValueError(
                    "Query-State exact restart checkpoint update is not a commit boundary"
                )
        else:
            index_path = run_root / "durable" / "authoritative_index.json"
            baseline_path = run_root / "actor_baseline_id176.json"
            validation_path = run_root / "validation_update_00000000.json"
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    "Query-State crash replay authoritative index is invalid"
                ) from error
            if (
                index.get("mode") != config.mode
                or index.get("entries") != []
                or not baseline_path.is_file()
                or not validation_path.is_file()
            ):
                raise ValueError(
                    "Query-State crash replay requires an empty authoritative index "
                    "and immutable update-zero evidence"
                )
    output_parent = run_root.parent
    while not output_parent.exists() and output_parent != output_parent.parent:
        output_parent = output_parent.parent
    required_free_bytes = _required_output_free_bytes(
        config,
        completed_checkpoint_update=completed_checkpoint_update,
    )
    if shutil.disk_usage(output_parent).free < required_free_bytes:
        raise OSError(
            "Query-State output filesystem lacks the locked checkpoint budget "
            "plus minimum-free reserve"
        )

    return QueryStateTrainingPreflightEvidence(
        config_identity=config.identity,
        lifecycle_state=config.lifecycle_state,
        source_commit=str(config.source["commit"]),
        clean_source=True,
        recursive_submodule_commits=submodules,
        verified_file_count=len(artifacts),
        processor_identity=processor_identity,
        tokenizer_identity=tokenizer_identity,
        dino_identity=str(config.model["dino_identity"]),
        output_ownership_verified=True,
        resource_contract_verified=True,
        cuda_entered=False,
    )


def assert_query_state_training_backend_ready(
    config: QueryStateTrainingConfig,
    *,
    preflight: QueryStateTrainingPreflightEvidence | None = None,
) -> None:
    if config.lifecycle_state != "launch_locked" or not config.authorization["launch_authorized"]:
        raise PermissionError("Query-State backend requires a launch-locked approved config")
    if preflight is None or preflight.config_identity != config.identity:
        raise PermissionError("Query-State backend requires evidence from the same live preflight")
    if preflight.cuda_entered or not preflight.clean_source or not preflight.output_ownership_verified:
        raise PermissionError("Query-State backend live preflight evidence is invalid")


__all__ = [
    "QueryStateTrainingPreflightEvidence",
    "assert_query_state_training_backend_ready",
    "validate_query_state_distributed_topology",
    "verify_query_state_training_preflight",
]
