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

import torch
import transformers

from nimloth.training.sft1.identity import audit_id176_processor_identity
from nimloth.training.sft1.query_state_training_config import QueryStateTrainingConfig
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
    rows, audit = index_early4_rows(contract)
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
    }
    if topology != expected_topology or config.resources["backend"] != "nccl":
        raise ValueError("Query-State command/resource topology mismatch")
    allowlist = config.resources["gpu_model_allowlist"]
    if not isinstance(allowlist, (tuple, list)) or not allowlist or any(
        not isinstance(value, str) or not value for value in allowlist
    ):
        raise ValueError("Query-State GPU resource allowlist is invalid")

    resume_mode = config.initialization["resume_mode"]
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
        terminal_names = (
            "COMPLETED.json", "FAILED.json", "PREEMPTED.json", "VALIDATOR_FAILED.json"
        )
        if any(
            (run_root / name).exists() or (controller_root / name).exists()
            for name in terminal_names
        ):
            raise RuntimeError("Query-State terminal run cannot restart or replay")
        if resume_mode == "exact_restart":
            checkpoint = Path(str(config.initialization["resume_checkpoint"]))
            if not (checkpoint / "COMPLETED").is_file() or not (checkpoint / "control.json").is_file():
                raise FileNotFoundError("Query-State exact restart checkpoint is incomplete")
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
    if shutil.disk_usage(output_parent).free < int(config.output["minimum_free_bytes"]):
        raise OSError("Query-State output filesystem free-space gate failed")

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
    "verify_query_state_training_preflight",
]
