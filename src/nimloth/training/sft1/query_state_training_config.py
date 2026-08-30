"""Strict pilot/formal configuration and coordinated W&B identity state.

This owner is intentionally schema-distinct from the Query-State code canary,
mechanics smoke, and retired seven-loss SFT1-v2 experiment.  Parsing is pure and
CPU-only; it neither initializes W&B nor enters CUDA or submits a job.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
)

QUERY_STATE_TRAINING_CONFIG_SCHEMA = "nimloth_sft1_query_state_training_v1"
_HEX = frozenset("0123456789abcdef")

_SECTION_FIELDS: Mapping[str, frozenset[str]] = {
    "lifecycle": frozenset({"preflight_locked", "launch_locked"}),
    "source": frozenset({"repo_root", "branch", "commit", "submodule_commits", "source_manifest_path", "source_manifest_identity"}),
    "data": frozenset({"train_source_path", "validation_source_path", "train_manifest_path", "train_manifest_identity", "validation_manifest_path", "validation_manifest_identity", "train_rows", "external_rows"}),
    "model": frozenset({"initialization_identity", "dino_identity", "dino_snapshot_path", "processor_path", "processor_identity", "tokenizer_identity", "template_identity", "token_table_identity", "action_token_ids", "state_schema", "objective_version", "query_count", "hidden_size", "state_dim", "llm_tune", "vision_tune", "query_tune", "direct_head_bias"}),
    "objective": frozenset({"state_weight", "lm_weight", "state_target", "lm_target"}),
    "optimizer": frozenset({"name", "language_learning_rate", "direct_state_learning_rate", "weight_decay", "betas", "epsilon", "scheduler", "warmup_updates"}),
    "runtime": frozenset({"max_sequence_length", "min_pixels", "max_pixels", "attention_implementation", "model_dtype", "dino_dtype", "dino_batch_size", "max_padded_tokens", "max_rows_per_micro_batch", "max_grad_norm", "gradient_checkpointing", "fsdp_sharding", "fsdp_use_orig_params", "fsdp_wrap_policy"}),
    "schedule": frozenset({"seed", "epochs", "max_updates", "rows_per_rank_update", "checkpoint_cadence_updates", "validation_updates", "forced_restart_update"}),
    "validation": frozenset({"split", "baseline_update", "terminal_update", "generation_format_manifest_path", "generation_format_manifest_identity", "generation_format_updates", "actor_tolerances", "effective_rank_formula", "effective_rank_collapse_threshold"}),
    "output": frozenset({"run_root", "controller_root", "overwrite", "resolved_config_path", "command_manifest_path", "minimum_free_bytes"}),
    "resources": frozenset({"world_size", "nodes", "gpus_per_node", "cpus_per_task", "memory_gib", "walltime", "partition", "backend", "gpu_model_allowlist"}),
    "authorization": frozenset({"approval_id", "approval_sha256", "launch_authorized"}),
    "initialization": frozenset({"actor_checkpoint", "actor_checkpoint_identity", "direct_head_initialization", "resume_checkpoint", "resume_mode"}),
    "tracking": frozenset({"enabled", "entity", "project", "group", "run_name", "run_id", "resume"}),
    "environment": frozenset({"python_executable", "hf_home", "hf_hub_cache", "offline", "dont_write_bytecode", "pycache_prefix", "python_hash_seed", "python_version", "torch_version", "transformers_version"}),
    "command": frozenset({"argv", "identity"}),
    "artifacts": frozenset({"file_sha256"}),
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _is_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= _HEX


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _int(value: object, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    invalid = not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0)
    if invalid:
        raise ValueError(f"{field} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _absolute(value: object, field: str) -> str:
    text = _text(value, field)
    if not Path(text).is_absolute():
        raise ValueError(f"{field} must be an absolute canonical path")
    return text


def _strict_section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing Query-State training section: {name}")
    fields = _SECTION_FIELDS[name]
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown Query-State training field: {name}.{unknown[0]}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"missing Query-State training field: {name}.{missing[0]}")
    null = sorted(key for key in fields if value[key] is None)
    if null:
        raise ValueError(f"Query-State training field may not be null: {name}.{null[0]}")
    return deepcopy(dict(value))


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            frozen[key] = _frozen_mapping(item)
        elif isinstance(item, list):
            frozen[key] = tuple(item)
        else:
            frozen[key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class QueryStateTrackingConfig:
    enabled: bool
    entity: str
    project: str
    group: str
    run_name: str
    run_id: str
    resume: str

    @property
    def identity(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


@dataclass(frozen=True)
class QueryStateTrainingConfig:
    schema: str
    mode: str
    lifecycle_state: str
    lifecycle: Mapping[str, Any]
    source: Mapping[str, Any]
    data: Mapping[str, Any]
    model: Mapping[str, Any]
    objective: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    runtime: Mapping[str, Any]
    schedule: Mapping[str, Any]
    validation: Mapping[str, Any]
    output: Mapping[str, Any]
    resources: Mapping[str, Any]
    authorization: Mapping[str, Any]
    initialization: Mapping[str, Any]
    tracking: QueryStateTrackingConfig
    environment: Mapping[str, Any]
    command: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    identity: str


@dataclass(frozen=True)
class QueryStateWandbStart:
    operation: str
    identity: str
    resume: str


@dataclass(frozen=True)
class QueryStateTrackingInitResult:
    rank: int
    success: bool
    identity: str | None
    url: str | None
    error: str | None


def _canonical_identity(raw: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def parse_query_state_training_config(raw: Mapping[str, Any]) -> QueryStateTrainingConfig:
    """Parse every field and bind all mode/lifecycle semantics without defaults."""

    if not isinstance(raw, Mapping):
        raise ValueError("Query-State training config must be a mapping")
    top = frozenset({"schema", "mode", *_SECTION_FIELDS})
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"unknown Query-State training section: {unknown[0]}")
    missing = sorted(top - set(raw))
    if missing:
        raise ValueError(f"missing Query-State training section: {missing[0]}")
    if raw["schema"] != QUERY_STATE_TRAINING_CONFIG_SCHEMA:
        raise ValueError("unsupported or legacy Query-State training schema")
    mode = _text(raw["mode"], "mode")
    if mode not in {"pilot", "formal"}:
        raise ValueError("Query-State training mode must be pilot or formal")
    sections = {name: _strict_section(raw, name) for name in _SECTION_FIELDS}

    lifecycle = sections["lifecycle"]
    preflight = _bool(lifecycle["preflight_locked"], "lifecycle.preflight_locked")
    launch = _bool(lifecycle["launch_locked"], "lifecycle.launch_locked")
    if launch and not preflight:
        raise ValueError("Query-State lifecycle cannot launch-lock before preflight lock")
    lifecycle_state = "launch_locked" if launch else "preflight_locked" if preflight else "template"

    if lifecycle_state == "template":
        authorization = sections["authorization"]
        if authorization["launch_authorized"] is not False:
            raise ValueError("template Query-State config cannot authorize launch")
        if sections["output"]["overwrite"] is not False:
            raise ValueError("template Query-State output overwrite is forbidden")
        model = sections["model"]
        if (
            model["state_schema"],
            model["objective_version"],
            model["query_count"],
            model["hidden_size"],
            model["state_dim"],
            model["llm_tune"],
            model["vision_tune"],
            model["query_tune"],
            model["direct_head_bias"],
        ) != (
            DIRECT_STATE_ARTIFACT_SCHEMA,
            QUERY_STATE_OBJECTIVE_VERSION,
            16,
            2048,
            1024,
            "full",
            "freeze",
            "freeze",
            False,
        ):
            raise ValueError("template Query-State canonical state/model contract changed")
        objective = sections["objective"]
        if (
            objective["state_weight"],
            objective["lm_weight"],
            objective["state_target"],
            objective["lm_target"],
        ) != (
            2.0,
            1.0,
            "online_original_observation_dino",
            "real_final_current_assistant",
        ):
            raise ValueError("template Query-State canonical objective changed")
        tracking_raw = sections["tracking"]
        tracking = QueryStateTrackingConfig(
            enabled=_bool(tracking_raw["enabled"], "tracking.enabled"),
            entity=_text(tracking_raw["entity"], "tracking.entity"),
            project=_text(tracking_raw["project"], "tracking.project"),
            group=_text(tracking_raw["group"], "tracking.group"),
            run_name=_text(tracking_raw["run_name"], "tracking.run_name"),
            run_id=_text(tracking_raw["run_id"], "tracking.run_id"),
            resume=_text(tracking_raw["resume"], "tracking.resume"),
        )
        if tracking.enabled:
            raise ValueError("template Query-State config cannot enable W&B")
        canonical = deepcopy(dict(raw))
        return QueryStateTrainingConfig(
            schema=QUERY_STATE_TRAINING_CONFIG_SCHEMA,
            mode=mode,
            lifecycle_state=lifecycle_state,
            lifecycle=_frozen_mapping(lifecycle),
            source=_frozen_mapping(sections["source"]),
            data=_frozen_mapping(sections["data"]),
            model=_frozen_mapping(model),
            objective=_frozen_mapping(objective),
            optimizer=_frozen_mapping(sections["optimizer"]),
            runtime=_frozen_mapping(sections["runtime"]),
            schedule=_frozen_mapping(sections["schedule"]),
            validation=_frozen_mapping(sections["validation"]),
            output=_frozen_mapping(sections["output"]),
            resources=_frozen_mapping(sections["resources"]),
            authorization=_frozen_mapping(authorization),
            initialization=_frozen_mapping(sections["initialization"]),
            tracking=tracking,
            environment=_frozen_mapping(sections["environment"]),
            command=_frozen_mapping(sections["command"]),
            artifacts=_frozen_mapping(sections["artifacts"]),
            identity=_canonical_identity(canonical),
        )

    source = sections["source"]
    for field in ("repo_root", "source_manifest_path"):
        _absolute(source[field], f"source.{field}")
    if not _is_git_sha(source["commit"]):
        raise ValueError("source.commit must be a lowercase Git SHA")
    submodules = source["submodule_commits"]
    if not isinstance(submodules, Mapping) or not submodules or any(
        not isinstance(path, str) or not path or not _is_git_sha(commit)
        for path, commit in submodules.items()
    ):
        raise ValueError("source.submodule_commits must bind exact Git SHAs")
    if not _is_sha256(source["source_manifest_identity"]):
        raise ValueError("source.source_manifest_identity must be SHA256")

    data = sections["data"]
    for field in (
        "train_source_path",
        "validation_source_path",
        "train_manifest_path",
        "validation_manifest_path",
    ):
        _absolute(data[field], f"data.{field}")
    for field in ("train_manifest_identity", "validation_manifest_identity"):
        if not _is_sha256(data[field]):
            raise ValueError(f"data.{field} must be SHA256")
    train_rows = _int(data["train_rows"], "data.train_rows")
    if _int(data["external_rows"], "data.external_rows") != 1413:
        raise ValueError("Query-State external manifest must contain exactly 1,413 rows")
    if mode == "formal" and train_rows != 12836:
        raise ValueError("formal Query-State manifest must contain all 12,836 valid train rows")
    if mode == "pilot" and train_rows >= 12836:
        raise ValueError("pilot Query-State manifest must be a locked coverage subset")

    model = sections["model"]
    for field in ("dino_snapshot_path", "processor_path"):
        _absolute(model[field], f"model.{field}")
    for field in ("processor_identity", "tokenizer_identity"):
        if not _is_sha256(model[field]):
            raise ValueError(f"model.{field} must be SHA256")
    expected_model = (
        model["state_schema"], model["objective_version"], model["query_count"],
        model["hidden_size"], model["state_dim"], model["llm_tune"],
        model["vision_tune"], model["query_tune"], model["direct_head_bias"],
    )
    if expected_model != (
        DIRECT_STATE_ARTIFACT_SCHEMA, QUERY_STATE_OBJECTIVE_VERSION, 16, 2048,
        1024, "full", "freeze", "freeze", False,
    ):
        raise ValueError("Query-State canonical state/model contract changed")
    if not str(model["initialization_identity"]).startswith("id176:"):
        raise ValueError("Query-State training must initialize from exact ID176")
    _text(model["dino_identity"], "model.dino_identity")

    objective = sections["objective"]
    if (
        _number(objective["state_weight"], "objective.state_weight", positive=True),
        _number(objective["lm_weight"], "objective.lm_weight", positive=True),
        objective["state_target"], objective["lm_target"],
    ) != (2.0, 1.0, "online_original_observation_dino", "real_final_current_assistant"):
        raise ValueError("Query-State canonical 2:1 same-forward objective changed")

    optimizer = sections["optimizer"]
    if str(optimizer["name"]).lower() != "adamw":
        raise ValueError("Query-State optimizer must be AdamW")
    for field in ("language_learning_rate", "direct_state_learning_rate", "epsilon"):
        _number(optimizer[field], f"optimizer.{field}", positive=True)
    _number(optimizer["weight_decay"], "optimizer.weight_decay", positive=False)
    betas = optimizer["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2 or any(
        not 0.0 <= _number(value, "optimizer.betas", positive=False) < 1.0 for value in betas
    ):
        raise ValueError("optimizer.betas must contain two values in [0,1)")
    if optimizer["scheduler"] not in {"constant", "cosine", "linear"}:
        raise ValueError("optimizer.scheduler must be explicitly supported")
    _int(optimizer["warmup_updates"], "optimizer.warmup_updates", minimum=0)

    runtime = sections["runtime"]
    max_length = _int(runtime["max_sequence_length"], "runtime.max_sequence_length")
    min_pixels = _int(runtime["min_pixels"], "runtime.min_pixels")
    max_pixels = _int(runtime["max_pixels"], "runtime.max_pixels")
    if min_pixels > max_pixels or max_length < 1:
        raise ValueError("runtime sequence/pixel limits are invalid")
    if runtime["attention_implementation"] not in {"flash_attention_2", "sdpa"}:
        raise ValueError("runtime attention implementation is unsupported")
    if runtime["model_dtype"] not in {"bfloat16", "float32"} or runtime["dino_dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("runtime model/DINO dtype is unsupported")
    for field in ("dino_batch_size", "max_padded_tokens", "max_rows_per_micro_batch"):
        _int(runtime[field], f"runtime.{field}")
    _number(runtime["max_grad_norm"], "runtime.max_grad_norm", positive=True)
    if (
        runtime["gradient_checkpointing"] is not True
        or runtime["fsdp_sharding"] != "full_shard"
        or runtime["fsdp_use_orig_params"] is not True
        or not isinstance(runtime["fsdp_wrap_policy"], Mapping)
        or not runtime["fsdp_wrap_policy"]
    ):
        raise ValueError("runtime requires gradient checkpointing and official FULL_SHARD use_orig_params")

    schedule = sections["schedule"]
    _int(schedule["seed"], "schedule.seed", minimum=0)
    epochs = _int(schedule["epochs"], "schedule.epochs")
    terminal = _int(schedule["max_updates"], "schedule.max_updates")
    rows_per_rank_update = _int(
        schedule["rows_per_rank_update"], "schedule.rows_per_rank_update"
    )
    schedule_world_size = _int(
        sections["resources"]["world_size"], "resources.world_size"
    )
    rows_per_rank_epoch = math.ceil(train_rows / schedule_world_size)
    exact_updates = epochs * math.ceil(rows_per_rank_epoch / rows_per_rank_update)
    if terminal != exact_updates:
        raise ValueError(
            "schedule/topology max_updates must equal the exact deterministic "
            f"schedule cardinality: {terminal} != {exact_updates}"
        )
    cadence = _int(schedule["checkpoint_cadence_updates"], "schedule.checkpoint_cadence_updates")
    forced = _int(schedule["forced_restart_update"], "schedule.forced_restart_update", minimum=0)
    updates = schedule["validation_updates"]
    if not isinstance(updates, (list, tuple)) or not updates or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in updates
    ) or tuple(updates) != tuple(sorted(set(updates))):
        raise ValueError("schedule.validation_updates must be sorted unique update indices")
    if tuple(updates)[0] != 0 or tuple(updates)[-1] != terminal or any(
        value != 0 and value % cadence for value in updates
    ):
        raise ValueError("validation must start at update 0 and otherwise use checkpoint boundaries")
    if terminal % cadence:
        raise ValueError("terminal update must be a resumable commit boundary")
    if mode == "pilot" and (epochs != 1 or not 0 < forced < terminal or forced % cadence):
        raise ValueError("pilot requires one coverage pass and a forced restart commit boundary")
    if mode == "formal" and forced != 0:
        raise ValueError("formal training has no pilot forced-restart boundary")

    validation = sections["validation"]
    if validation["baseline_update"] != 0 or validation["terminal_update"] != terminal:
        raise ValueError("validation baseline/terminal identity disagrees with schedule")
    expected_split = "calibration" if mode == "pilot" else "holdout"
    if validation["split"] != expected_split:
        raise ValueError(f"{mode} Query-State validation must use {expected_split}")
    _absolute(
        validation["generation_format_manifest_path"],
        "validation.generation_format_manifest_path",
    )
    if not _is_sha256(validation["generation_format_manifest_identity"]):
        raise ValueError("validation.generation_format_manifest_identity must be SHA256")
    generation_updates = validation["generation_format_updates"]
    if (
        not isinstance(generation_updates, (list, tuple))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in generation_updates
        )
        or tuple(generation_updates) != tuple(sorted(set(generation_updates)))
        or not {0, terminal} <= set(generation_updates)
        or not set(generation_updates) <= set(updates)
    ):
        raise ValueError(
            "generation-format cadence must explicitly include update 0 and terminal "
            "and may only use registered validation updates"
        )
    tolerances = validation["actor_tolerances"]
    expected_tolerances = {
        "kl_max",
        "top1_min",
        "logit_rms_ratio_min",
        "logit_rms_ratio_max",
    }
    if not isinstance(tolerances, Mapping) or set(tolerances) != expected_tolerances:
        raise ValueError("validation.actor_tolerances must be pre-registered")
    kl_max = _number(
        tolerances["kl_max"],
        "validation.actor_tolerances.kl_max",
        positive=True,
    )
    top1 = _number(
        tolerances["top1_min"],
        "validation.actor_tolerances.top1_min",
        positive=True,
    )
    ratio_min = _number(
        tolerances["logit_rms_ratio_min"],
        "validation.actor_tolerances.logit_rms_ratio_min",
        positive=True,
    )
    ratio_max = _number(
        tolerances["logit_rms_ratio_max"],
        "validation.actor_tolerances.logit_rms_ratio_max",
        positive=True,
    )
    if top1 > 1.0 or ratio_min > ratio_max or kl_max <= 0.0:
        raise ValueError("validation actor tolerances are inconsistent")
    if validation["effective_rank_formula"] != "entropy_rank_rows_slots_centered_float64_eps1e-12":
        raise ValueError("validation effective-rank formula is not the reviewed formula")
    _number(
        validation["effective_rank_collapse_threshold"],
        "validation.effective_rank_collapse_threshold",
        positive=True,
    )

    output = sections["output"]
    for field in ("run_root", "controller_root", "resolved_config_path", "command_manifest_path"):
        _absolute(output[field], f"output.{field}")
    if _bool(output["overwrite"], "output.overwrite"):
        raise ValueError("Query-State training output overwrite is forbidden")
    _int(output["minimum_free_bytes"], "output.minimum_free_bytes")

    resources = sections["resources"]
    world = _int(resources["world_size"], "resources.world_size")
    nodes = _int(resources["nodes"], "resources.nodes")
    gpus = _int(resources["gpus_per_node"], "resources.gpus_per_node")
    if nodes * gpus != world:
        raise ValueError("resources topology nodes*gpus_per_node must equal world_size")
    _int(resources["cpus_per_task"], "resources.cpus_per_task")
    _int(resources["memory_gib"], "resources.memory_gib")
    _text(resources["walltime"], "resources.walltime")
    _text(resources["partition"], "resources.partition")
    if resources["backend"] != "nccl":
        raise ValueError("Query-State production backend must be nccl")
    allowlist = resources["gpu_model_allowlist"]
    if not isinstance(allowlist, (list, tuple)) or not allowlist or any(
        not isinstance(value, str) or not value.strip() for value in allowlist
    ):
        raise ValueError("resources.gpu_model_allowlist must be explicit")

    authorization = sections["authorization"]
    _text(authorization["approval_id"], "authorization.approval_id")
    if not _is_sha256(authorization["approval_sha256"]):
        raise ValueError("authorization.approval_sha256 must be SHA256")
    authorized = _bool(authorization["launch_authorized"], "authorization.launch_authorized")
    if authorized != launch:
        raise ValueError("launch authorization must exactly match launch-locked lifecycle")

    initialization = sections["initialization"]
    actor_checkpoint = _absolute(initialization["actor_checkpoint"], "initialization.actor_checkpoint")
    if "id176" not in actor_checkpoint.lower():
        raise ValueError("pilot/formal initialization must use ID176, never a pilot checkpoint")
    if not _is_sha256(initialization["actor_checkpoint_identity"]):
        raise ValueError("initialization.actor_checkpoint_identity must be SHA256")
    if model["initialization_identity"] != "id176:" + initialization["actor_checkpoint_identity"]:
        raise ValueError("model and initialization ID176 identities disagree")
    if Path(model["processor_path"]).resolve() != Path(actor_checkpoint).resolve():
        raise ValueError("ID176 actor and processor owners must be the same exact bundle")
    if initialization["direct_head_initialization"] != "fresh_seeded_no_bias":
        raise ValueError("pilot/formal direct head must initialize fresh without bias")
    resume_mode = initialization["resume_mode"]
    if resume_mode not in {"fresh", "crash_replay", "exact_restart"}:
        raise ValueError(
            "initialization.resume_mode must be fresh, crash_replay, or exact_restart"
        )
    if resume_mode in {"fresh", "crash_replay"} and initialization["resume_checkpoint"] != "none":
        raise ValueError(
            "fresh/crash-replay initialization cannot consume a resume checkpoint"
        )
    if resume_mode == "exact_restart":
        _absolute(initialization["resume_checkpoint"], "initialization.resume_checkpoint")

    tracking_raw = sections["tracking"]
    tracking = QueryStateTrackingConfig(
        enabled=_bool(tracking_raw["enabled"], "tracking.enabled"),
        entity=_text(tracking_raw["entity"], "tracking.entity"),
        project=_text(tracking_raw["project"], "tracking.project"),
        group=_text(tracking_raw["group"], "tracking.group"),
        run_name=_text(tracking_raw["run_name"], "tracking.run_name"),
        run_id=_text(tracking_raw["run_id"], "tracking.run_id"),
        resume=_text(tracking_raw["resume"], "tracking.resume"),
    )
    if mode == "pilot":
        if tracking.enabled or set((tracking.entity, tracking.project, tracking.group, tracking.run_name, tracking.run_id, tracking.resume)) != {"disabled"}:
            raise ValueError("pilot W&B must be completely disabled")
    else:
        if not tracking.enabled:
            raise ValueError("formal W&B tracking must be enabled")
        expected_resume = "never" if resume_mode == "fresh" else "must"
        if tracking.resume != expected_resume or "disabled" in {
            tracking.entity, tracking.project, tracking.group, tracking.run_name, tracking.run_id
        }:
            raise ValueError("formal W&B fresh/restart identity is invalid")

    environment = sections["environment"]
    for field in ("python_executable", "hf_home", "hf_hub_cache", "pycache_prefix"):
        _absolute(environment[field], f"environment.{field}")
    if not _bool(environment["offline"], "environment.offline"):
        raise ValueError("Query-State launch requires locked HF offline mode")
    if not _bool(environment["dont_write_bytecode"], "environment.dont_write_bytecode"):
        raise ValueError("Query-State launch requires bytecode writes disabled")

    command = sections["command"]
    argv = command["argv"]
    expected_argv = [
        environment["python_executable"],
        str(Path(source["repo_root"]) / "experiments/training/sft1/query_state_train.py"),
        "--config",
        output["resolved_config_path"],
        "--phase",
        "run",
    ]
    if not isinstance(argv, (list, tuple)) or list(argv) != expected_argv:
        raise ValueError("Query-State command does not match the resolved command parity contract")
    if not _is_sha256(command["identity"]):
        raise ValueError("command.identity must be SHA256")

    artifacts = sections["artifacts"]["file_sha256"]
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("artifacts.file_sha256 must bind every launch-owned file")
    if any(
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not _is_sha256(digest)
        for path, digest in artifacts.items()
    ):
        raise ValueError("artifacts.file_sha256 requires absolute paths and SHA256 values")
    required_hashed_paths = {
        source["source_manifest_path"],
        data["train_source_path"],
        data["validation_source_path"],
        data["train_manifest_path"],
        data["validation_manifest_path"],
        validation["generation_format_manifest_path"],
        output["command_manifest_path"],
    }
    if not required_hashed_paths <= set(artifacts):
        raise ValueError("artifacts.file_sha256 omits a source/data/command owner")

    canonical = deepcopy(dict(raw))
    return QueryStateTrainingConfig(
        schema=QUERY_STATE_TRAINING_CONFIG_SCHEMA,
        mode=mode,
        lifecycle_state=lifecycle_state,
        lifecycle=_frozen_mapping(lifecycle), source=_frozen_mapping(source),
        data=_frozen_mapping(data), model=_frozen_mapping(model),
        objective=_frozen_mapping(objective), optimizer=_frozen_mapping(optimizer),
        runtime=_frozen_mapping(runtime), schedule=_frozen_mapping(schedule),
        validation=_frozen_mapping(validation),
        output=_frozen_mapping(output), resources=_frozen_mapping(resources),
        authorization=_frozen_mapping(authorization), initialization=_frozen_mapping(initialization),
        tracking=tracking, environment=_frozen_mapping(environment),
        command=_frozen_mapping(command),
        artifacts=_frozen_mapping(sections["artifacts"]),
        identity=_canonical_identity(canonical),
    )


def resolve_wandb_start(
    config: QueryStateTrainingConfig,
    *,
    remote_exists: bool,
    remote_identity: str | None,
) -> QueryStateWandbStart:
    """Resolve fresh/restart before W&B init; identity errors are hard failures."""

    if config.mode != "formal" or not config.tracking.enabled:
        raise ValueError("W&B start resolution is formal-only")
    mode = config.initialization["resume_mode"]
    identity = config.tracking.identity
    if mode == "fresh":
        if remote_exists:
            raise ValueError("fresh formal W&B identity already exists")
        if remote_identity is not None:
            raise ValueError("fresh formal W&B query returned an unexpected identity")
        return QueryStateWandbStart("fresh", identity, "never")
    if not remote_exists or remote_identity is None:
        raise ValueError("restart/replay requires a matching existing W&B identity")
    if remote_identity != identity:
        raise ValueError("restart/replay W&B identity mismatch")
    return QueryStateWandbStart(str(mode), identity, "must")


def reapply_locked_wandb_environment(
    config: QueryStateTrainingConfig,
    sourced_environment: Mapping[str, str],
) -> dict[str, str]:
    """Apply run-owned values after shared credentials/environment are sourced."""

    if config.mode != "formal" or not config.tracking.enabled:
        raise ValueError("locked W&B environment is formal-only")
    effective = dict(sourced_environment)
    effective.update({
        "WANDB_ENTITY": config.tracking.entity,
        "WANDB_PROJECT": config.tracking.project,
        "WANDB_NAME": config.tracking.run_name,
        "WANDB_RUN_ID": config.tracking.run_id,
        "WANDB_RESUME": config.tracking.resume,
    })
    return effective


def coordinate_tracking_init(
    results: Sequence[QueryStateTrackingInitResult],
) -> QueryStateTrackingInitResult:
    """Require one successful identical rank result before any rank trains."""

    if not results or tuple(sorted(item.rank for item in results)) != tuple(range(len(results))):
        raise RuntimeError("W&B all-rank init results have invalid rank coverage")
    if any(not item.success or item.error is not None for item in results):
        failures = "; ".join(item.error or "unknown" for item in results if not item.success or item.error)
        raise RuntimeError("W&B all-rank initialization failed: " + failures)
    identities = {item.identity for item in results}
    urls = {item.url for item in results}
    if None in identities or None in urls or len(identities) != 1 or len(urls) != 1:
        raise RuntimeError("W&B ranks disagree on initialized identity or URL")
    return results[0]


__all__ = [
    "QUERY_STATE_TRAINING_CONFIG_SCHEMA",
    "QueryStateTrackingConfig",
    "QueryStateTrackingInitResult",
    "QueryStateTrainingConfig",
    "QueryStateWandbStart",
    "coordinate_tracking_init",
    "parse_query_state_training_config",
    "reapply_locked_wandb_environment",
    "resolve_wandb_start",
]
