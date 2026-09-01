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

QUERY_STATE_TRAINING_CONFIG_SCHEMA = "nimloth_sft1_query_state_training_v4"
FORMAL_CHECKPOINT_ESTIMATED_BYTES = 20_500_000_000
VISUAL_FORK_CHECKPOINT_ESTIMATED_BYTES = 23_370_000_000
_HEX = frozenset("0123456789abcdef")

_SECTION_FIELDS: Mapping[str, frozenset[str]] = {
    "lifecycle": frozenset({"preflight_locked", "launch_locked"}),
    "source": frozenset({"repo_root", "branch", "commit", "submodule_commits", "source_manifest_path", "source_manifest_identity"}),
    "data": frozenset({"train_source_path", "validation_source_path", "train_manifest_path", "train_manifest_identity", "validation_manifest_path", "validation_manifest_identity", "train_rows", "external_rows"}),
    "model": frozenset({"initialization_identity", "dino_identity", "dino_snapshot_path", "processor_path", "processor_identity", "tokenizer_identity", "template_identity", "token_table_identity", "action_token_ids", "state_schema", "objective_version", "query_count", "hidden_size", "state_dim", "llm_tune", "vision_tune", "query_tune", "direct_head_bias"}),
    "objective": frozenset({"state_weight", "lm_weight", "state_target", "lm_target"}),
    "optimizer": frozenset({"name", "language_learning_rate", "direct_state_learning_rate", "weight_decay", "betas", "epsilon", "scheduler", "warmup_updates"}),
    "runtime": frozenset({"max_sequence_length", "min_pixels", "max_pixels", "attention_implementation", "model_dtype", "dino_dtype", "dino_batch_size", "max_padded_tokens", "max_rows_per_micro_batch", "max_grad_norm", "gradient_checkpointing", "fsdp_sharding", "fsdp_use_orig_params", "fsdp_wrap_policy"}),
    "schedule": frozenset({"seed", "epochs", "schedule_start_update", "max_updates", "rows_per_rank_update", "epoch_updates", "checkpoint_cadence_updates", "validation_updates", "forced_restart_update", "approved_pause_update"}),
    "early_stopping": frozenset({"enabled", "metric", "min_epochs", "max_epochs", "patience_epochs", "min_relative_improvement", "calibration_split", "holdout_controls_early_stop", "actual_terminal_primary"}),
    "validation": frozenset({"split", "baseline_update", "terminal_update", "calibration_cadence_updates", "holdout_updates", "holdout_at_actual_terminal", "generation_format_manifest_path", "generation_format_manifest_identity", "generation_format_updates", "generation_format_at_actual_terminal", "actor_tolerances", "effective_rank_formula", "effective_rank_collapse_threshold", "bootstrap_seed", "bootstrap_resamples", "ordinary_cluster_unit", "ordinary_bootstrap_formula", "natural_pair_unit", "natural_pair_formula", "terminal_state_gates"}),
    "output": frozenset({"run_root", "controller_root", "overwrite", "resolved_config_path", "command_manifest_path", "minimum_free_bytes", "checkpoint_estimated_bytes", "checkpoint_budget_bytes"}),
    "resources": frozenset({"world_size", "nodes", "gpus_per_node", "cpus_per_task", "memory_gib", "walltime", "partition", "backend", "gpu_model_allowlist"}),
    "authorization": frozenset({"approval_id", "approval_sha256", "launch_authorized"}),
    "initialization": frozenset({"actor_checkpoint", "actor_checkpoint_identity", "direct_head_initialization", "resume_checkpoint", "resume_mode"}),
    "tracking": frozenset({"enabled", "entity", "project", "group", "run_name", "run_id", "resume"}),
    "environment": frozenset({"python_executable", "hf_home", "hf_hub_cache", "offline", "dont_write_bytecode", "pycache_prefix", "python_hash_seed", "python_version", "torch_version", "transformers_version", "nccl_socket_ifname", "nccl_ib_disable"}),
    "forensic_fork": frozenset({"enabled", "ancestor_checkpoint_path", "ancestor_failure_manifest_path", "id176_actor_baseline_path", "id176_actor_baseline_sha256", "ancestor_control_sha256", "ancestor_source_commit", "ancestor_source_manifest_identity", "ancestor_run_identity", "ancestor_source_config_identity", "ancestor_update", "initialization_kind", "actor_policy", "generation_policy", "retention_policy", "ancestor_protected", "parity_relative_tolerance", "parity_absolute_tolerance"}),
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


def _path_trees_overlap(left: Path, right: Path) -> bool:
    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


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
        return (
            f"{self.entity}/{self.project}/{self.group}/"
            f"{self.run_name}/{self.run_id}"
        )


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
    early_stopping: Mapping[str, Any]
    validation: Mapping[str, Any]
    output: Mapping[str, Any]
    resources: Mapping[str, Any]
    authorization: Mapping[str, Any]
    initialization: Mapping[str, Any]
    tracking: QueryStateTrackingConfig
    environment: Mapping[str, Any]
    forensic_fork: Mapping[str, Any]
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


def _plain_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_identity_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_identity_value(item) for item in value]
    return value


def query_state_training_run_identity(config: QueryStateTrainingConfig) -> str:
    """Bind resume-critical semantics while excluding process approval horizons."""

    schedule_identity = dict(config.schedule)
    schedule_identity.pop("approved_pause_update", None)
    payload = {
        "schema": config.schema,
        "mode": config.mode,
        "source": _plain_identity_value(config.source),
        "data": _plain_identity_value(config.data),
        "model": _plain_identity_value(config.model),
        "objective": _plain_identity_value(config.objective),
        "optimizer": _plain_identity_value(config.optimizer),
        "runtime": _plain_identity_value(config.runtime),
        "schedule": _plain_identity_value(schedule_identity),
        "early_stopping": _plain_identity_value(config.early_stopping),
        "validation": _plain_identity_value(config.validation),
        "run_root": config.output["run_root"],
        "controller_root": config.output["controller_root"],
        "output_storage": {
            "minimum_free_bytes": config.output["minimum_free_bytes"],
            "checkpoint_estimated_bytes": config.output[
                "checkpoint_estimated_bytes"
            ],
            "checkpoint_budget_bytes": config.output["checkpoint_budget_bytes"],
        },
        "resources": _plain_identity_value(config.resources),
        "environment": _plain_identity_value(config.environment),
        "forensic_fork": _plain_identity_value(config.forensic_fork),
        "actor_checkpoint": config.initialization["actor_checkpoint"],
        "actor_checkpoint_identity": config.initialization[
            "actor_checkpoint_identity"
        ],
        "direct_head_initialization": config.initialization[
            "direct_head_initialization"
        ],
        "tracking_identity": (
            None if config.mode == "pilot" else config.tracking.identity
        ),
    }
    return _canonical_identity(payload)


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
    if mode not in {"pilot", "formal", "visual_only_forensic_fork"}:
        raise ValueError(
            "Query-State training mode must be pilot, formal, or visual_only_forensic_fork"
        )
    sections = {name: _strict_section(raw, name) for name in _SECTION_FIELDS}
    visual_fork = mode == "visual_only_forensic_fork"

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
            early_stopping=_frozen_mapping(sections["early_stopping"]),
            validation=_frozen_mapping(sections["validation"]),
            output=_frozen_mapping(sections["output"]),
            resources=_frozen_mapping(sections["resources"]),
            authorization=_frozen_mapping(authorization),
            initialization=_frozen_mapping(sections["initialization"]),
            tracking=tracking,
            environment=_frozen_mapping(sections["environment"]),
            forensic_fork=_frozen_mapping(sections["forensic_fork"]),
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
    if mode != "pilot" and train_rows != 12836:
        raise ValueError("production Query-State manifest must contain all 12,836 valid train rows")
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
    expected_model_initialization = (
        "formal38_forensic_model_only:"
        + str(sections["forensic_fork"]["ancestor_control_sha256"])
        if visual_fork
        else "id176:" + str(sections["initialization"]["actor_checkpoint_identity"])
    )
    if model["initialization_identity"] != expected_model_initialization:
        raise ValueError("Query-State model initialization identity is invalid")
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
    warmup_updates = _int(
        optimizer["warmup_updates"], "optimizer.warmup_updates", minimum=0
    )
    if mode != "pilot" and (
        float(optimizer["language_learning_rate"]) != 1e-6
        or float(optimizer["direct_state_learning_rate"]) != 1e-4
        or float(optimizer["weight_decay"]) != 0.0
        or tuple(float(value) for value in optimizer["betas"]) != (0.9, 0.95)
        or float(optimizer["epsilon"]) != 1e-8
        or optimizer["scheduler"] != "constant"
        or warmup_updates != 0
    ):
        raise ValueError("formal Query-State optimizer/LR contract changed")

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
    max_grad_norm = _number(
        runtime["max_grad_norm"], "runtime.max_grad_norm", positive=True
    )
    if mode != "pilot" and max_grad_norm != 1.0:
        raise ValueError("production Query-State max_grad_norm must remain 1.0")
    if (
        runtime["gradient_checkpointing"] is not True
        or runtime["fsdp_sharding"] != "full_shard"
        or runtime["fsdp_use_orig_params"] is not True
        or not isinstance(runtime["fsdp_wrap_policy"], Mapping)
        or not runtime["fsdp_wrap_policy"]
    ):
        raise ValueError("runtime requires gradient checkpointing and official FULL_SHARD use_orig_params")

    schedule = sections["schedule"]
    schedule_seed = _int(schedule["seed"], "schedule.seed", minimum=0)
    epochs = _int(schedule["epochs"], "schedule.epochs")
    schedule_start = _int(
        schedule["schedule_start_update"],
        "schedule.schedule_start_update",
        minimum=0,
    )
    terminal = _int(schedule["max_updates"], "schedule.max_updates")
    rows_per_rank_update = _int(
        schedule["rows_per_rank_update"], "schedule.rows_per_rank_update"
    )
    schedule_world_size = _int(
        sections["resources"]["world_size"], "resources.world_size"
    )
    rows_per_rank_epoch = math.ceil(train_rows / schedule_world_size)
    exact_updates = epochs * math.ceil(rows_per_rank_epoch / rows_per_rank_update)
    if terminal != schedule_start + exact_updates:
        raise ValueError(
            "schedule/topology max_updates must equal start plus the exact deterministic "
            f"schedule cardinality: {terminal} != {schedule_start + exact_updates}"
        )
    epoch_updates = _int(schedule["epoch_updates"], "schedule.epoch_updates")
    exact_epoch_updates = math.ceil(rows_per_rank_epoch / rows_per_rank_update)
    if epoch_updates != exact_epoch_updates:
        raise ValueError(
            "schedule.epoch_updates must equal the exact deterministic epoch "
            f"update count: {epoch_updates} != {exact_epoch_updates}"
        )
    cadence = _int(
        schedule["checkpoint_cadence_updates"],
        "schedule.checkpoint_cadence_updates",
    )
    if epoch_updates % cadence:
        raise ValueError("checkpoint cadence must divide the exact epoch updates")
    forced = _int(schedule["forced_restart_update"], "schedule.forced_restart_update", minimum=0)
    approved_pause = _int(
        schedule["approved_pause_update"],
        "schedule.approved_pause_update",
        minimum=0,
    )
    updates = schedule["validation_updates"]
    if not isinstance(updates, (list, tuple)) or not updates or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in updates
    ) or tuple(updates) != tuple(sorted(set(updates))):
        raise ValueError("schedule.validation_updates must be sorted unique update indices")
    if tuple(updates)[0] != schedule_start or tuple(updates)[-1] != terminal or any(
        value != schedule_start and (value - schedule_start) % cadence for value in updates
    ):
        raise ValueError(
            "validation must start at the schedule offset and otherwise use checkpoint boundaries"
        )
    if (terminal - schedule_start) % cadence:
        raise ValueError("terminal update must be a resumable commit boundary")
    if mode == "pilot" and (
        schedule_start != 0
        or epochs != 1
        or not 0 < forced < terminal
        or forced % cadence
    ):
        raise ValueError("pilot requires one coverage pass and a forced restart commit boundary")
    if mode == "pilot" and approved_pause != 0:
        raise ValueError("pilot cannot use a formal approved pause boundary")
    if mode == "formal":
        if forced != 0:
            raise ValueError("formal training has no pilot forced-restart boundary")
        if approved_pause and (
            approved_pause > terminal or approved_pause % epoch_updates
        ):
            raise ValueError(
                "formal approved pause must be an exact epoch boundary at or before terminal"
            )
        if (
            schedule_start != 0
            or schedule_seed != 3335631237
            or epochs != 10
            or rows_per_rank_update != 1
            or terminal != 16050
            or epoch_updates != 1605
            or cadence != 321
            or tuple(updates) != (0, 3210, 8025, 16050)
        ):
            raise ValueError("formal WS8 max10 schedule contract changed")
    elif visual_fork:
        if (
            schedule_seed != 3335631237
            or schedule_start != 1605
            or epochs != 4
            or rows_per_rank_update != 1
            or terminal != 8025
            or epoch_updates != 1605
            or cadence != 321
            or forced != 0
            or approved_pause != 0
            or tuple(updates) != (1605, 3210, 4815, 6420, 8025)
        ):
            raise ValueError("visual forensic fork schedule must cover fixed epochs 2-5")

    early_stopping = sections["early_stopping"]
    early_enabled = _bool(early_stopping["enabled"], "early_stopping.enabled")
    early_metric = _text(early_stopping["metric"], "early_stopping.metric")
    min_epochs = _int(early_stopping["min_epochs"], "early_stopping.min_epochs")
    early_max_epochs = _int(early_stopping["max_epochs"], "early_stopping.max_epochs")
    patience = _int(
        early_stopping["patience_epochs"],
        "early_stopping.patience_epochs",
        minimum=0,
    )
    relative_improvement = _number(
        early_stopping["min_relative_improvement"],
        "early_stopping.min_relative_improvement",
        positive=False,
    )
    if _text(
        early_stopping["calibration_split"],
        "early_stopping.calibration_split",
    ) != "calibration":
        raise ValueError("early stopping must use the locked calibration split")
    if _bool(
        early_stopping["holdout_controls_early_stop"],
        "early_stopping.holdout_controls_early_stop",
    ):
        raise ValueError("holdout must never control early stopping")
    actual_terminal_primary = _bool(
        early_stopping["actual_terminal_primary"],
        "early_stopping.actual_terminal_primary",
    )
    if mode == "pilot":
        if (
            early_enabled
            or early_metric != "disabled"
            or (min_epochs, early_max_epochs, patience, relative_improvement)
            != (1, 1, 0, 0.0)
            or actual_terminal_primary
        ):
            raise ValueError("pilot must keep formal early stopping disabled")
    elif visual_fork:
        if (
            early_enabled
            or early_metric != "disabled"
            or (min_epochs, early_max_epochs, patience, relative_improvement)
            != (1, 4, 0, 0.0)
            or actual_terminal_primary
        ):
            raise ValueError("visual forensic fork must use a fixed budget without early stopping")
    elif (
        not early_enabled
        or early_metric != "calibration_2x_dino_mse_plus_assistant_ce"
        or min_epochs < 2
        or early_max_epochs != epochs
        or min_epochs > early_max_epochs
        or patience < 1
        or relative_improvement <= 0.0
        or relative_improvement >= 1.0
        or not actual_terminal_primary
    ):
        raise ValueError("formal early-stopping contract is invalid")

    validation = sections["validation"]
    if (
        validation["baseline_update"] != schedule_start
        or validation["terminal_update"] != terminal
    ):
        raise ValueError("validation baseline/terminal identity disagrees with schedule")
    expected_split = (
        "calibration"
        if mode == "pilot"
        else "visual_fork_calibration_trend_final_holdout"
        if visual_fork
        else "dual_calibration_control_holdout_primary"
    )
    if validation["split"] != expected_split:
        raise ValueError(f"{mode} Query-State validation must use {expected_split}")
    calibration_cadence = _int(
        validation["calibration_cadence_updates"],
        "validation.calibration_cadence_updates",
    )
    expected_calibration_cadence = cadence if mode == "pilot" else epoch_updates
    if calibration_cadence != expected_calibration_cadence:
        raise ValueError("calibration validation must run at every epoch commit boundary")
    holdout_updates = validation["holdout_updates"]
    expected_holdout_updates = (terminal,) if visual_fork else tuple(updates)
    if (
        not isinstance(holdout_updates, (list, tuple))
        or tuple(holdout_updates) != expected_holdout_updates
    ):
        raise ValueError("holdout validation updates differ from the registered cadence")
    holdout_at_terminal = _bool(
        validation["holdout_at_actual_terminal"],
        "validation.holdout_at_actual_terminal",
    )
    if holdout_at_terminal != (mode == "formal"):
        raise ValueError("only formal validation may add the dynamic actual terminal holdout")
    _absolute(
        validation["generation_format_manifest_path"],
        "validation.generation_format_manifest_path",
    )
    if not _is_sha256(validation["generation_format_manifest_identity"]):
        raise ValueError("validation.generation_format_manifest_identity must be SHA256")
    generation_updates = validation["generation_format_updates"]
    generation_at_terminal = _bool(
        validation["generation_format_at_actual_terminal"],
        "validation.generation_format_at_actual_terminal",
    )
    if generation_at_terminal != (mode == "formal"):
        raise ValueError("only formal validation may add dynamic terminal generation")
    if (
        not isinstance(generation_updates, (list, tuple))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in generation_updates
        )
        or tuple(generation_updates) != tuple(sorted(set(generation_updates)))
        or not {schedule_start, terminal} <= set(generation_updates)
        or not set(generation_updates) <= set(updates)
    ):
        raise ValueError(
            "generation-format cadence must explicitly include update 0 and terminal "
            "(the visual fork baseline is update 1605) "
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
    _int(validation["bootstrap_seed"], "validation.bootstrap_seed", minimum=0)
    _int(validation["bootstrap_resamples"], "validation.bootstrap_resamples")
    if validation["ordinary_cluster_unit"] != "record_id":
        raise ValueError("validation ordinary cluster unit must be record_id")
    if validation["ordinary_bootstrap_formula"] != (
        "record_cluster_percentile_95_row_weighted_mean_v1"
    ):
        raise ValueError("validation ordinary bootstrap formula is unsupported")
    if validation["natural_pair_unit"] != "natural_group_mean":
        raise ValueError("validation natural pair unit must be natural_group_mean")
    if validation["natural_pair_formula"] != (
        "equal_group_mean_percentile_95_group_bootstrap_v1"
    ):
        raise ValueError("validation natural pair formula is unsupported")
    state_gates = validation["terminal_state_gates"]
    expected_gate_fields = {
        "terminal_primary_only",
        "canonical_effective_rank_min",
        "raw_query_effective_rank_min",
        "dino_mse_max_fraction_of_update0",
        "dino_cosine_min_increase_from_update0",
        "instruction_relation_max_decrease_from_update0",
    }
    if not isinstance(state_gates, Mapping) or set(state_gates) != expected_gate_fields:
        raise ValueError("validation.terminal_state_gates must be fully pre-registered")
    if _bool(
        state_gates["terminal_primary_only"],
        "validation.terminal_state_gates.terminal_primary_only",
    ) is not True:
        raise ValueError("formal state gate primary must be the terminal checkpoint")
    for field in ("canonical_effective_rank_min", "raw_query_effective_rank_min"):
        _number(
            state_gates[field],
            f"validation.terminal_state_gates.{field}",
            positive=True,
        )
    mse_fraction = _number(
        state_gates["dino_mse_max_fraction_of_update0"],
        "validation.terminal_state_gates.dino_mse_max_fraction_of_update0",
        positive=True,
    )
    cosine_increase = _number(
        state_gates["dino_cosine_min_increase_from_update0"],
        "validation.terminal_state_gates.dino_cosine_min_increase_from_update0",
        positive=False,
    )
    instruction_decrease = _number(
        state_gates["instruction_relation_max_decrease_from_update0"],
        "validation.terminal_state_gates.instruction_relation_max_decrease_from_update0",
        positive=False,
    )
    if mse_fraction > 1.0 or cosine_increase > 2.0 or instruction_decrease > 2.0:
        raise ValueError("validation terminal state gate bounds are inconsistent")

    forensic = sections["forensic_fork"]
    if visual_fork:
        for field in (
            "ancestor_checkpoint_path",
            "ancestor_failure_manifest_path",
            "id176_actor_baseline_path",
        ):
            _absolute(forensic[field], f"forensic_fork.{field}")
        if (
            _bool(forensic["enabled"], "forensic_fork.enabled") is not True
            or not _is_sha256(forensic["id176_actor_baseline_sha256"])
            or not _is_sha256(forensic["ancestor_control_sha256"])
            or not _is_git_sha(forensic["ancestor_source_commit"])
            or forensic["ancestor_source_commit"] == source["commit"]
            or not _is_sha256(forensic["ancestor_source_manifest_identity"])
            or forensic["ancestor_source_manifest_identity"]
            == source["source_manifest_identity"]
            or not _is_sha256(forensic["ancestor_run_identity"])
            or not _is_sha256(forensic["ancestor_source_config_identity"])
            or forensic["ancestor_update"] != 1605
            or forensic["initialization_kind"]
            != "forensic_model_only_fresh_optimizer"
            or forensic["actor_policy"] != "report_only"
            or forensic["generation_policy"] != "report_only"
            or forensic["retention_policy"]
            != "successor_first_non_epoch_final_payload_v1"
            or forensic["ancestor_protected"] is not True
            or _number(
                forensic["parity_relative_tolerance"],
                "forensic_fork.parity_relative_tolerance",
                positive=True,
            ) <= 0.0
            or _number(
                forensic["parity_absolute_tolerance"],
                "forensic_fork.parity_absolute_tolerance",
                positive=False,
            ) < 0.0
        ):
            if forensic.get("ancestor_source_commit") == source["commit"]:
                raise ValueError(
                    "visual forensic fork ancestor source commit must differ from current source.commit"
                )
            if forensic.get("ancestor_source_manifest_identity") == source[
                "source_manifest_identity"
            ]:
                raise ValueError(
                    "visual forensic fork ancestor source manifest must differ from current source manifest"
                )
            raise ValueError("visual forensic fork ancestor/fresh-runtime contract changed")
    elif forensic != {
        "enabled": False,
        "ancestor_checkpoint_path": "disabled",
        "ancestor_failure_manifest_path": "disabled",
        "id176_actor_baseline_path": "disabled",
        "id176_actor_baseline_sha256": "disabled",
        "ancestor_control_sha256": "disabled",
        "ancestor_source_commit": "disabled",
        "ancestor_source_manifest_identity": "disabled",
        "ancestor_run_identity": "disabled",
        "ancestor_source_config_identity": "disabled",
        "ancestor_update": 0,
        "initialization_kind": "disabled",
        "actor_policy": "disabled",
        "generation_policy": "disabled",
        "retention_policy": "disabled",
        "ancestor_protected": False,
        "parity_relative_tolerance": 0.0,
        "parity_absolute_tolerance": 0.0,
    }:
        raise ValueError("pilot/formal configs cannot carry forensic fork authority")

    output = sections["output"]
    for field in ("run_root", "controller_root", "resolved_config_path", "command_manifest_path"):
        _absolute(output[field], f"output.{field}")
    if _bool(output["overwrite"], "output.overwrite"):
        raise ValueError("Query-State training output overwrite is forbidden")
    checkpoint_estimated_bytes = _int(
        output["checkpoint_estimated_bytes"],
        "output.checkpoint_estimated_bytes",
    )
    minimum_free_bytes = _int(
        output["minimum_free_bytes"],
        "output.minimum_free_bytes",
    )
    checkpoint_budget_bytes = _int(
        output["checkpoint_budget_bytes"],
        "output.checkpoint_budget_bytes",
    )
    expected_checkpoint_budget = (
        5 * checkpoint_estimated_bytes
        if visual_fork
        else (terminal // cadence) * checkpoint_estimated_bytes
    )
    if checkpoint_budget_bytes != expected_checkpoint_budget:
        raise ValueError(
            "output.checkpoint_budget_bytes must cover every max-budget commit"
        )
    if visual_fork:
        if minimum_free_bytes != 150_000_000_000:
            raise ValueError(
                "visual fork output.minimum_free_bytes must equal the approved 150GB"
            )
    elif mode != "pilot" and minimum_free_bytes != 300_000_000_000:
        raise ValueError("production output.minimum_free_bytes must equal 300GB")
    expected_estimate = (
        VISUAL_FORK_CHECKPOINT_ESTIMATED_BYTES
        if visual_fork
        else FORMAL_CHECKPOINT_ESTIMATED_BYTES
    )
    if mode != "pilot" and checkpoint_estimated_bytes != expected_estimate:
        if visual_fork:
            raise ValueError(
                "visual fork checkpoint estimate must equal the locked 23.37GB"
            )
        raise ValueError("formal checkpoint estimate must equal the locked 20.5GB")
    if visual_fork:
        ancestor_checkpoint = Path(str(forensic["ancestor_checkpoint_path"])).resolve()
        ancestor_failure = Path(
            str(forensic["ancestor_failure_manifest_path"])
        ).resolve()
        protected_run_root = ancestor_checkpoint.parent.parent
        if (
            ancestor_checkpoint.parent.name != "forensics"
            or ancestor_failure.parent.name != "failures"
            or ancestor_failure.parent.parent.name != "durable"
            or ancestor_failure.parent.parent.parent != protected_run_root
        ):
            raise ValueError(
                "visual fork ancestor evidence paths must share the protected Formal38 run root"
            )
        for field in ("run_root", "controller_root"):
            output_root = Path(str(output[field])).resolve()
            if _path_trees_overlap(output_root, protected_run_root):
                raise ValueError(
                    f"visual fork output.{field} overlaps the protected Formal38 ancestor tree"
                )

    resources = sections["resources"]
    world = _int(resources["world_size"], "resources.world_size")
    nodes = _int(resources["nodes"], "resources.nodes")
    gpus = _int(resources["gpus_per_node"], "resources.gpus_per_node")
    if nodes * gpus != world:
        raise ValueError("resources topology nodes*gpus_per_node must equal world_size")
    if mode != "pilot" and (
        world != 8
        or nodes != 2
        or gpus != 4
        or resources["partition"] not in {"normal", "preempt"}
    ):
        raise ValueError(
            "production Query-State topology must be normal or preempt 2x4 WS8"
        )
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
    actor_checkpoint = _absolute(
        initialization["actor_checkpoint"], "initialization.actor_checkpoint"
    )
    if not _is_sha256(initialization["actor_checkpoint_identity"]):
        raise ValueError("initialization.actor_checkpoint_identity must be SHA256")
    if not visual_fork and (
        model["initialization_identity"]
        != "id176:" + initialization["actor_checkpoint_identity"]
    ):
        raise ValueError("model and initialization ID176 identities disagree")
    if Path(model["processor_path"]).resolve() != Path(actor_checkpoint).resolve():
        raise ValueError("ID176 actor and processor owners must be the same exact bundle")
    expected_direct_initialization = (
        "forensic_model_only" if visual_fork else "fresh_seeded_no_bias"
    )
    if initialization["direct_head_initialization"] != expected_direct_initialization:
        raise ValueError("Query-State direct-head initialization owner is invalid")
    resume_mode = initialization["resume_mode"]
    if resume_mode not in {"fresh", "crash_replay", "exact_restart"}:
        raise ValueError(
            "initialization.resume_mode must be fresh, crash_replay, or exact_restart"
        )
    if visual_fork and resume_mode == "crash_replay":
        raise ValueError("visual forensic fork supports fresh start or its own exact resume")
    if resume_mode in {"fresh", "crash_replay"} and initialization["resume_checkpoint"] != "none":
        raise ValueError(
            "fresh/crash-replay initialization cannot consume a resume checkpoint"
        )
    if resume_mode == "exact_restart":
        resume_checkpoint = Path(
            _absolute(
                initialization["resume_checkpoint"],
                "initialization.resume_checkpoint",
            )
        ).resolve()
        if visual_fork and (
            resume_checkpoint.parent
            != (Path(str(sections["output"]["run_root"])).resolve() / "checkpoints")
            or resume_checkpoint
            == Path(str(forensic["ancestor_checkpoint_path"])).resolve()
        ):
            raise ValueError(
                "visual fork exact resume must consume only its run-owned checkpoint"
            )

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
    nccl_socket_ifname = _text(
        environment["nccl_socket_ifname"], "environment.nccl_socket_ifname"
    )
    nccl_ib_disable = _text(
        environment["nccl_ib_disable"], "environment.nccl_ib_disable"
    )
    if mode != "pilot" and (
        nccl_socket_ifname != "ibp24s0" or nccl_ib_disable != "1"
    ):
        raise ValueError("formal Query-State NCCL network contract changed")
    if nccl_ib_disable not in {"0", "1"}:
        raise ValueError("environment.nccl_ib_disable must be 0 or 1")

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
    if visual_fork:
        ancestor_checkpoint_path = Path(
            str(forensic["ancestor_checkpoint_path"])
        )
        required_hashed_paths.update({
            str(ancestor_checkpoint_path / "control.json"),
            str(forensic["ancestor_failure_manifest_path"]),
            str(forensic["id176_actor_baseline_path"]),
            *(
                str(ancestor_checkpoint_path / f"rank_{rank:05d}_of_00008{suffix}")
                for rank in range(8)
                for suffix in (".pt", ".json")
            ),
        })
    if not required_hashed_paths <= set(artifacts):
        raise ValueError("artifacts.file_sha256 omits a source/data/command/ancestor owner")

    canonical = deepcopy(dict(raw))
    return QueryStateTrainingConfig(
        schema=QUERY_STATE_TRAINING_CONFIG_SCHEMA,
        mode=mode,
        lifecycle_state=lifecycle_state,
        lifecycle=_frozen_mapping(lifecycle), source=_frozen_mapping(source),
        data=_frozen_mapping(data), model=_frozen_mapping(model),
        objective=_frozen_mapping(objective), optimizer=_frozen_mapping(optimizer),
        runtime=_frozen_mapping(runtime), schedule=_frozen_mapping(schedule),
        early_stopping=_frozen_mapping(sections["early_stopping"]),
        validation=_frozen_mapping(validation),
        output=_frozen_mapping(output), resources=_frozen_mapping(resources),
        authorization=_frozen_mapping(authorization), initialization=_frozen_mapping(initialization),
        tracking=tracking, environment=_frozen_mapping(environment),
        forensic_fork=_frozen_mapping(sections["forensic_fork"]),
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

    if config.mode == "pilot" or not config.tracking.enabled:
        raise ValueError("W&B start resolution requires a tracked production mode")
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

    if config.mode == "pilot" or not config.tracking.enabled:
        raise ValueError("locked W&B environment requires a tracked production mode")
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
