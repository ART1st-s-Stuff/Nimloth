"""Strict, non-launching configuration identity for Query-State code canaries.

The schema has no field defaults and deliberately contains no command, output,
resource, W&B, epoch, or update-budget field.  Supplying values proves only that
a local code-path check was configured explicitly; it is not a launch contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
)


QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA = "nimloth_sft1_query_state_code_canary_v1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class QueryStateOptimizerConfig:
    name: str
    language_learning_rate: float
    direct_state_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    scheduler: str


@dataclass(frozen=True)
class QueryStateRuntimeConfig:
    max_padded_tokens: int
    max_rows_per_micro_batch: int
    rows_per_rank_update: int
    max_grad_norm: float
    world_size: int
    gradient_checkpointing: bool
    train_mode: bool
    fsdp_sharding: str
    fsdp_use_orig_params: bool
    launch_authorized: bool


@dataclass(frozen=True)
class QueryStateCheckpointConfig:
    cadence_updates: int
    at_update_boundary: bool
    exact_resume: bool
    immutable_rank_shards: bool
    save_optimizer: bool
    save_rng: bool
    save_data_cursor: bool
    save_metric_cursor: bool


@dataclass(frozen=True)
class QueryStateValidationConfig:
    cadence_updates: int
    report_only: bool
    model_quality_gate: bool
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class QueryStateCodeCanaryConfig:
    schema: str
    optimizer: QueryStateOptimizerConfig
    runtime: QueryStateRuntimeConfig
    checkpoint: QueryStateCheckpointConfig
    validation: QueryStateValidationConfig

    @property
    def identity(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class QueryStateCodeCanaryIdentity:
    source_commit: str
    source_manifest_identity: str
    config_identity: str
    run_identity: str
    world_size: int
    stage: str
    training_schema: str
    objective_version: str
    state_artifact_schema: str

    def __post_init__(self) -> None:
        if (
            len(self.source_commit) != 40
            or any(char not in _HEX for char in self.source_commit)
        ):
            raise ValueError("Query-State code-canary source commit must be a Git SHA")
        for name in (
            "source_manifest_identity",
            "config_identity",
            "run_identity",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in _HEX for char in value):
                raise ValueError(f"Query-State code-canary {name} must be SHA256")
        if self.world_size < 1:
            raise ValueError("Query-State code-canary world size must be positive")
        if (
            self.stage,
            self.training_schema,
            self.objective_version,
            self.state_artifact_schema,
        ) != (
            "sft1_query_state",
            QUERY_STATE_SCHEMA,
            QUERY_STATE_OBJECTIVE_VERSION,
            DIRECT_STATE_ARTIFACT_SCHEMA,
        ):
            raise ValueError("Query-State code-canary stage/schema identity mismatch")


def _strict_section(
    raw: Mapping[str, Any],
    name: str,
    fields: set[str],
) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Query-State config section {name!r} is required")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown Query-State config field: {name}.{unknown[0]}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"missing Query-State config field: {name}.{missing[0]}")
    null = sorted(field for field in fields if value[field] is None)
    if null:
        raise ValueError(f"Query-State config field may not be null: {name}.{null[0]}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _number(value: Any, field: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return result


def bind_query_state_code_canary_identity(
    config: QueryStateCodeCanaryConfig,
    *,
    source_commit: str,
    source_manifest_identity: str,
    run_identity: str,
) -> QueryStateCodeCanaryIdentity:
    """Bind source/run identity to the exact parsed config and its world size."""

    if not isinstance(config, QueryStateCodeCanaryConfig):
        raise TypeError("Query-State code-canary identity requires a parsed config")
    return QueryStateCodeCanaryIdentity(
        source_commit=source_commit,
        source_manifest_identity=source_manifest_identity,
        config_identity=config.identity,
        run_identity=run_identity,
        world_size=config.runtime.world_size,
        stage="sft1_query_state",
        training_schema=QUERY_STATE_SCHEMA,
        objective_version=QUERY_STATE_OBJECTIVE_VERSION,
        state_artifact_schema=DIRECT_STATE_ARTIFACT_SCHEMA,
    )


def parse_query_state_code_canary_config(
    raw: Mapping[str, Any],
) -> QueryStateCodeCanaryConfig:
    """Parse every local code-canary field without inferring any value."""

    if not isinstance(raw, Mapping):
        raise ValueError("Query-State code-canary config must be a mapping")
    top = {"schema", "optimizer", "runtime", "checkpoint", "validation"}
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"unknown Query-State config section: {unknown[0]}")
    missing = sorted(top - set(raw))
    if missing:
        raise ValueError(f"missing Query-State config section: {missing[0]}")
    if raw["schema"] != QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA:
        raise ValueError("unsupported or legacy Query-State code-canary schema")

    optimizer_raw = _strict_section(
        raw, "optimizer", set(QueryStateOptimizerConfig.__dataclass_fields__)
    )
    betas = optimizer_raw["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    parsed_betas = tuple(
        _number(value, "optimizer.betas", positive=False) for value in betas
    )
    if any(value >= 1.0 for value in parsed_betas):
        raise ValueError("optimizer.betas must lie in [0,1)")
    optimizer = QueryStateOptimizerConfig(
        name=_text(optimizer_raw["name"], "optimizer.name"),
        language_learning_rate=_number(
            optimizer_raw["language_learning_rate"],
            "optimizer.language_learning_rate",
            positive=True,
        ),
        direct_state_learning_rate=_number(
            optimizer_raw["direct_state_learning_rate"],
            "optimizer.direct_state_learning_rate",
            positive=True,
        ),
        weight_decay=_number(
            optimizer_raw["weight_decay"], "optimizer.weight_decay", positive=False
        ),
        betas=parsed_betas,  # type: ignore[arg-type]
        epsilon=_number(
            optimizer_raw["epsilon"], "optimizer.epsilon", positive=True
        ),
        scheduler=_text(optimizer_raw["scheduler"], "optimizer.scheduler"),
    )
    if optimizer.name.lower() != "adamw":
        raise ValueError("Query-State code-canary optimizer must be AdamW")

    runtime_raw = _strict_section(
        raw, "runtime", set(QueryStateRuntimeConfig.__dataclass_fields__)
    )
    runtime = QueryStateRuntimeConfig(
        max_padded_tokens=_int(
            runtime_raw["max_padded_tokens"], "runtime.max_padded_tokens"
        ),
        max_rows_per_micro_batch=_int(
            runtime_raw["max_rows_per_micro_batch"],
            "runtime.max_rows_per_micro_batch",
        ),
        rows_per_rank_update=_int(
            runtime_raw["rows_per_rank_update"], "runtime.rows_per_rank_update"
        ),
        max_grad_norm=_number(
            runtime_raw["max_grad_norm"], "runtime.max_grad_norm", positive=True
        ),
        world_size=_int(runtime_raw["world_size"], "runtime.world_size"),
        gradient_checkpointing=_bool(
            runtime_raw["gradient_checkpointing"],
            "runtime.gradient_checkpointing",
        ),
        train_mode=_bool(runtime_raw["train_mode"], "runtime.train_mode"),
        fsdp_sharding=_text(runtime_raw["fsdp_sharding"], "runtime.fsdp_sharding"),
        fsdp_use_orig_params=_bool(
            runtime_raw["fsdp_use_orig_params"], "runtime.fsdp_use_orig_params"
        ),
        launch_authorized=_bool(
            runtime_raw["launch_authorized"], "runtime.launch_authorized"
        ),
    )
    if (
        not runtime.gradient_checkpointing
        or not runtime.train_mode
        or runtime.fsdp_sharding != "full_shard"
        or not runtime.fsdp_use_orig_params
    ):
        raise ValueError(
            "Query-State code-canary requires train-mode checkpointing and "
            "official FULL_SHARD use_orig_params"
        )
    if runtime.launch_authorized:
        raise ValueError("Query-State code-canary config is strictly non-launching")

    checkpoint_raw = _strict_section(
        raw, "checkpoint", set(QueryStateCheckpointConfig.__dataclass_fields__)
    )
    checkpoint = QueryStateCheckpointConfig(
        cadence_updates=_int(
            checkpoint_raw["cadence_updates"], "checkpoint.cadence_updates"
        ),
        **{
            name: _bool(checkpoint_raw[name], f"checkpoint.{name}")
            for name in QueryStateCheckpointConfig.__dataclass_fields__
            if name != "cadence_updates"
        },
    )
    if not all(
        value
        for name, value in asdict(checkpoint).items()
        if name != "cadence_updates"
    ):
        raise ValueError(
            "Query-State checkpoint must save immutable exact optimizer/RNG/data/metric state"
        )

    validation_raw = _strict_section(
        raw, "validation", set(QueryStateValidationConfig.__dataclass_fields__)
    )
    diagnostics_raw = validation_raw["diagnostics"]
    if not isinstance(diagnostics_raw, (list, tuple)):
        raise ValueError("validation.diagnostics must be a sequence")
    diagnostics = tuple(
        _text(value, "validation.diagnostics") for value in diagnostics_raw
    )
    expected_diagnostics = (
        "raw_query_hidden",
        "canonical_state",
        "lm_ce",
        "action_logits",
    )
    if diagnostics != expected_diagnostics:
        raise ValueError(
            "Query-State code-canary diagnostics must explicitly list the four direct views"
        )
    validation = QueryStateValidationConfig(
        cadence_updates=_int(
            validation_raw["cadence_updates"], "validation.cadence_updates"
        ),
        report_only=_bool(validation_raw["report_only"], "validation.report_only"),
        model_quality_gate=_bool(
            validation_raw["model_quality_gate"], "validation.model_quality_gate"
        ),
        diagnostics=diagnostics,
    )
    if not validation.report_only or validation.model_quality_gate:
        raise ValueError(
            "Query-State code-canary validation is diagnostic-only, not a model-quality gate"
        )

    return QueryStateCodeCanaryConfig(
        schema=QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA,
        optimizer=optimizer,
        runtime=runtime,
        checkpoint=checkpoint,
        validation=validation,
    )


__all__ = [
    "QUERY_STATE_CODE_CANARY_CONFIG_SCHEMA",
    "QueryStateCheckpointConfig",
    "QueryStateCodeCanaryConfig",
    "QueryStateCodeCanaryIdentity",
    "QueryStateOptimizerConfig",
    "QueryStateRuntimeConfig",
    "QueryStateValidationConfig",
    "bind_query_state_code_canary_identity",
    "parse_query_state_code_canary_config",
]
