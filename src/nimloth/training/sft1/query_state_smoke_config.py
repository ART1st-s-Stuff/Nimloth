"""Strict identity for the two-process Query-State production-path smoke.

This schema is deliberately distinct from both the legacy SFT1-v2 experiment
and the local Query-State code-canary schema.  Preparation files may carry the
single explicit unresolved sentinel while ``launch_locked`` is false.  CUDA
execution requires a fully resolved, externally immutable identity-bound config
artifact and the exact approved commands; parsing never constitutes launch
authorization.  The artifact is resolved after the source commit is known, so
it must not claim the impossible property of containing its own Git commit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from nimloth.training.sft1.query_state import (
    DIRECT_STATE_ARTIFACT_SCHEMA,
    QUERY_STATE_OBJECTIVE_VERSION,
    QUERY_STATE_SCHEMA,
)


QUERY_STATE_SMOKE_CONFIG_SCHEMA = "nimloth_sft1_query_state_smoke_v1"
QUERY_STATE_SMOKE_UNRESOLVED = "_UNRESOLVED_BEFORE_LAUNCH_"
_HEX = frozenset("0123456789abcdef")
_PINNED_VAGEN = "9f1e89eb8c9839a406b6e62aa75703494a79e5b5"
_PINNED_VERL = "494f264494b2525f2c13595f63ac4912963e6d2f"
_PINNED_DINO_REVISION = "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"


@dataclass(frozen=True)
class QueryStateSmokeStateContract:
    training_schema: str
    objective_version: str
    direct_state_artifact_schema: str
    latent_query_mode: str
    grid_tokens: int
    qwen_hidden_dim: int
    state_dim: int
    direct_state_bias: bool
    state_weight: float
    lm_weight: float
    llm_tune: str
    vision_tune: str
    query_tune: str
    lora: bool
    dino_frozen: bool


@dataclass(frozen=True)
class QueryStateSmokeSource:
    repo: str
    expected_commit: str
    vagen_commit: str
    verl_commit: str
    interpreter: str
    python_version: str
    torch_version: str
    transformers_version: str


@dataclass(frozen=True)
class QueryStateSmokeSelection:
    steps: tuple[int, ...]
    train_records: int
    train_rows: int
    excluded_train_empty_cot_rows: int
    validation_records: int
    raw_validation_rows: int
    excluded_validation_empty_cot_rows: int
    external_validation_rows: int
    cross_split_image_hashes: int
    same_image_multi_instruction_groups: int
    same_instruction_multi_image_groups: int


@dataclass(frozen=True)
class QueryStateSmokeRowDescriptor:
    phase: str
    rank: int
    ordinal: int
    record_id: str
    step_index: int
    row_identity: str
    original_image_sha256: str
    rendered_token_count: int
    valid_lm_token_count: int
    split: str


@dataclass(frozen=True)
class QueryStateSmokeData:
    train_jsonl: str
    train_sha256: str
    validation_jsonl: str
    validation_sha256: str
    record_format: str
    train_split: str
    validation_split: str
    overlap_key: str
    source_manifest_identity: str
    smoke_rows: tuple[QueryStateSmokeRowDescriptor, ...]


@dataclass(frozen=True)
class QueryStateSmokeInitialization:
    actor_checkpoint: str
    actor_completion_sha256: str
    actor_config_sha256: str
    actor_model_index_sha256: str
    actor_model_shards_sha256: tuple[str, ...]
    actor_action_head_sha256: str
    processor_sha256: str
    tokenizer_sha256: str
    prompt_template_sha256: str
    token_table_sha256: str
    action_token_ids: tuple[int, ...]
    dino_source: str
    dino_revision: str
    dino_processor_fingerprint: str
    dino_hidden_size: int
    dino_grid_size: int
    fresh_original_observation_targets: bool


@dataclass(frozen=True)
class QueryStateSmokeOptimizer:
    name: str
    language_learning_rate: float
    direct_state_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    scheduler: str
    max_grad_norm: float


@dataclass(frozen=True)
class QueryStateSmokeRuntime:
    world_size: int | str
    nodes: int | str
    ranks_per_node: int | str
    backend: str
    model_dtype: str
    mixed_precision_param_dtype: str
    mixed_precision_reduce_dtype: str
    mixed_precision_buffer_dtype: str
    attention_implementation: str
    gradient_checkpointing: bool
    train_mode: bool
    fsdp_sharding: str
    fsdp_use_orig_params: bool
    fsdp_wrap_policy: Mapping[str, Any]
    model_parallel_size: int
    max_sequence_length: int | str
    max_pixels: int | str
    max_padded_tokens: int | str
    max_rows_per_micro_batch: int
    rows_per_rank_update: int
    fresh_updates: int
    resume_updates: int
    seed: int | str
    rng_schedule_version: str


@dataclass(frozen=True)
class QueryStateSmokeCheckpoint:
    immutable_rank_shards: bool
    same_world_size_resume: bool
    same_rank_resume: bool
    save_optimizer: bool
    save_scheduler: bool
    save_rng: bool
    save_data_cursor: bool
    save_metric_cursor: bool
    fresh_checkpoint_name: str
    resume_checkpoint_name: str
    completion_marker: str
    forbid_cross_stage_resume: bool
    forbid_legacy_resume: bool


@dataclass(frozen=True)
class QueryStateSmokeOutput:
    experiment_group: str
    run_root: str
    controller_log_dir: str
    fresh_child: str
    resume_child: str
    minimum_free_bytes: int | str
    overwrite: bool
    tracking_mode: str


@dataclass(frozen=True)
class QueryStateSmokeResources:
    account: str
    partition: str
    node_count: int | str
    gpus_per_node: int | str
    total_gpus: int | str
    gpu_model_allowlist: tuple[str, ...]
    cpus_per_task: int | str
    memory_gib: int | str
    walltime: str


@dataclass(frozen=True)
class QueryStateSmokeAuthorization:
    preflight_locked: bool
    launch_locked: bool
    approval_evidence: str
    approved_command_sha256: str


@dataclass(frozen=True)
class QueryStateSmokeConfig:
    schema: str
    state_contract: QueryStateSmokeStateContract
    source: QueryStateSmokeSource
    selection: QueryStateSmokeSelection
    data: QueryStateSmokeData
    initialization: QueryStateSmokeInitialization
    optimizer: QueryStateSmokeOptimizer
    runtime: QueryStateSmokeRuntime
    checkpoint: QueryStateSmokeCheckpoint
    output: QueryStateSmokeOutput
    resources: QueryStateSmokeResources
    authorization: QueryStateSmokeAuthorization

    @property
    def launch_locked(self) -> bool:
        return self.authorization.launch_locked

    @property
    def preflight_locked(self) -> bool:
        return self.authorization.preflight_locked

    @property
    def identity(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def _strict_section(raw: Mapping[str, Any], name: str, cls: type[Any]) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Query-State smoke section {name!r} is required")
    fields = set(cls.__dataclass_fields__)
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown Query-State smoke field: {name}.{unknown[0]}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"missing Query-State smoke field: {name}.{missing[0]}")
    null = sorted(field for field in fields if value[field] is None)
    if null:
        raise ValueError(f"Query-State smoke field may not be null: {name}.{null[0]}")
    return value


def _text(value: Any, field: str, *, allow_unresolved: bool = False) -> str:
    if value == QUERY_STATE_SMOKE_UNRESOLVED and allow_unresolved:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _int(
    value: Any,
    field: str,
    *,
    minimum: int = 1,
    allow_unresolved: bool = False,
) -> int | str:
    if value == QUERY_STATE_SMOKE_UNRESOLVED and allow_unresolved:
        return value
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, field: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(
            f"{field} must be finite and {'positive' if positive else 'non-negative'}"
        )
    return result


def _sha(value: Any, field: str, *, length: int = 64, allow_unresolved: bool = False) -> str:
    text = _text(value, field, allow_unresolved=allow_unresolved)
    if text == QUERY_STATE_SMOKE_UNRESOLVED:
        return text
    if len(text) != length or any(char not in _HEX for char in text):
        raise ValueError(f"{field} must be a lowercase {'Git SHA' if length == 40 else 'SHA256'}")
    return text


def _exact(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise ValueError(f"Query-State smoke {field} violates the state contract")


def _contains_unresolved(value: Any) -> bool:
    if value == QUERY_STATE_SMOKE_UNRESOLVED:
        return True
    if isinstance(value, Mapping):
        return any(_contains_unresolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unresolved(item) for item in value)
    return False


def _parse_descriptors(value: Any, *, locked: bool, world_size: int | str) -> tuple[QueryStateSmokeRowDescriptor, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("data.smoke_rows must be a sequence")
    if not locked:
        if value:
            raise ValueError("unlocked Query-State smoke prep may not guess smoke rows")
        return ()
    if not isinstance(world_size, int):
        raise ValueError("resolved Query-State smoke world size is invalid")
    expected_count = 2 * world_size
    if len(value) != expected_count:
        raise ValueError("Query-State smoke requires exactly two real rows per rank")
    descriptors: list[QueryStateSmokeRowDescriptor] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("data.smoke_rows entries must be mappings")
        fields = set(QueryStateSmokeRowDescriptor.__dataclass_fields__)
        if set(item) != fields:
            raise ValueError("data.smoke_rows entry has unknown or missing fields")
        descriptor = QueryStateSmokeRowDescriptor(
            phase=_text(item["phase"], f"data.smoke_rows[{index}].phase"),
            rank=int(_int(item["rank"], f"data.smoke_rows[{index}].rank", minimum=0)),
            ordinal=int(_int(item["ordinal"], f"data.smoke_rows[{index}].ordinal", minimum=0)),
            record_id=_text(item["record_id"], f"data.smoke_rows[{index}].record_id"),
            step_index=int(_int(item["step_index"], f"data.smoke_rows[{index}].step_index", minimum=0)),
            row_identity=_sha(item["row_identity"], f"data.smoke_rows[{index}].row_identity"),
            original_image_sha256=_sha(item["original_image_sha256"], f"data.smoke_rows[{index}].original_image_sha256"),
            rendered_token_count=int(_int(item["rendered_token_count"], f"data.smoke_rows[{index}].rendered_token_count")),
            valid_lm_token_count=int(_int(item["valid_lm_token_count"], f"data.smoke_rows[{index}].valid_lm_token_count")),
            split=_text(item["split"], f"data.smoke_rows[{index}].split"),
        )
        if descriptor.phase not in {"fresh", "resume"} or descriptor.split != "train":
            raise ValueError("Query-State smoke rows must be real train rows for fresh/resume")
        if descriptor.rank >= world_size or descriptor.step_index not in (0, 1, 2, 3):
            raise ValueError("Query-State smoke row rank/step mapping is invalid")
        if descriptor.valid_lm_token_count > descriptor.rendered_token_count:
            raise ValueError("Query-State smoke LM-token count exceeds rendered tokens")
        descriptors.append(descriptor)
    if len({row.ordinal for row in descriptors}) != len(descriptors):
        raise ValueError("Query-State smoke rows require unique ordinals")
    if len({row.row_identity for row in descriptors}) != len(descriptors):
        raise ValueError("Query-State smoke rows require unique row identities")
    if len({row.original_image_sha256 for row in descriptors}) != len(descriptors):
        raise ValueError("Query-State smoke rows require unique original images")
    for phase in ("fresh", "resume"):
        ranks = {row.rank for row in descriptors if row.phase == phase}
        if ranks != set(range(world_size)):
            raise ValueError("Query-State smoke rows must assign one row per rank and phase")
    return tuple(descriptors)


def parse_query_state_smoke_preparation(raw: Mapping[str, Any]) -> QueryStateSmokeConfig:
    """Parse a resolved or explicitly unlocked preparation without inferring fields."""

    if not isinstance(raw, Mapping):
        raise ValueError("Query-State smoke config must be a mapping")
    top = {
        "schema", "state_contract", "source", "selection", "data",
        "initialization", "optimizer", "runtime", "checkpoint", "output",
        "resources", "authorization",
    }
    unknown = sorted(set(raw) - top)
    if unknown:
        raise ValueError(f"unknown Query-State smoke section: {unknown[0]}")
    missing = sorted(top - set(raw))
    if missing:
        raise ValueError(f"missing Query-State smoke section: {missing[0]}")
    if raw["schema"] != QUERY_STATE_SMOKE_CONFIG_SCHEMA:
        raise ValueError("unsupported or legacy Query-State smoke schema")

    authorization_raw = _strict_section(raw, "authorization", QueryStateSmokeAuthorization)
    preflight_locked = _bool(
        authorization_raw["preflight_locked"],
        "authorization.preflight_locked",
    )
    locked = _bool(authorization_raw["launch_locked"], "authorization.launch_locked")
    if locked and not preflight_locked:
        raise ValueError("Query-State smoke launch requires completed preflight locking")
    resolved = preflight_locked
    authorization = QueryStateSmokeAuthorization(
        preflight_locked=preflight_locked,
        launch_locked=locked,
        approval_evidence=_text(
            authorization_raw["approval_evidence"],
            "authorization.approval_evidence",
            allow_unresolved=not locked,
        ),
        approved_command_sha256=_sha(
            authorization_raw["approved_command_sha256"],
            "authorization.approved_command_sha256",
            allow_unresolved=not locked,
        ),
    )

    state_raw = _strict_section(raw, "state_contract", QueryStateSmokeStateContract)
    state = QueryStateSmokeStateContract(
        training_schema=_text(state_raw["training_schema"], "state_contract.training_schema"),
        objective_version=_text(state_raw["objective_version"], "state_contract.objective_version"),
        direct_state_artifact_schema=_text(state_raw["direct_state_artifact_schema"], "state_contract.direct_state_artifact_schema"),
        latent_query_mode=_text(state_raw["latent_query_mode"], "state_contract.latent_query_mode"),
        grid_tokens=int(_int(state_raw["grid_tokens"], "state_contract.grid_tokens")),
        qwen_hidden_dim=int(_int(state_raw["qwen_hidden_dim"], "state_contract.qwen_hidden_dim")),
        state_dim=int(_int(state_raw["state_dim"], "state_contract.state_dim")),
        direct_state_bias=_bool(state_raw["direct_state_bias"], "state_contract.direct_state_bias"),
        state_weight=_number(state_raw["state_weight"], "state_contract.state_weight", positive=True),
        lm_weight=_number(state_raw["lm_weight"], "state_contract.lm_weight", positive=True),
        llm_tune=_text(state_raw["llm_tune"], "state_contract.llm_tune"),
        vision_tune=_text(state_raw["vision_tune"], "state_contract.vision_tune"),
        query_tune=_text(state_raw["query_tune"], "state_contract.query_tune"),
        lora=_bool(state_raw["lora"], "state_contract.lora"),
        dino_frozen=_bool(state_raw["dino_frozen"], "state_contract.dino_frozen"),
    )
    _exact(
        asdict(state),
        {
            "training_schema": QUERY_STATE_SCHEMA,
            "objective_version": QUERY_STATE_OBJECTIVE_VERSION,
            "direct_state_artifact_schema": DIRECT_STATE_ARTIFACT_SCHEMA,
            "latent_query_mode": "inject",
            "grid_tokens": 16,
            "qwen_hidden_dim": 2048,
            "state_dim": 1024,
            "direct_state_bias": False,
            "state_weight": 2.0,
            "lm_weight": 1.0,
            "llm_tune": "full",
            "vision_tune": "freeze",
            "query_tune": "freeze",
            "lora": False,
            "dino_frozen": True,
        },
        "state contract",
    )

    source_raw = _strict_section(raw, "source", QueryStateSmokeSource)
    source = QueryStateSmokeSource(
        repo=_text(source_raw["repo"], "source.repo", allow_unresolved=not resolved),
        expected_commit=_sha(source_raw["expected_commit"], "source.expected_commit", length=40, allow_unresolved=not resolved),
        vagen_commit=_sha(source_raw["vagen_commit"], "source.vagen_commit", length=40),
        verl_commit=_sha(source_raw["verl_commit"], "source.verl_commit", length=40),
        interpreter=_text(source_raw["interpreter"], "source.interpreter", allow_unresolved=not resolved),
        python_version=_text(source_raw["python_version"], "source.python_version"),
        torch_version=_text(source_raw["torch_version"], "source.torch_version"),
        transformers_version=_text(source_raw["transformers_version"], "source.transformers_version"),
    )
    if (source.vagen_commit, source.verl_commit) != (_PINNED_VAGEN, _PINNED_VERL):
        raise ValueError("Query-State smoke source submodule identity mismatch")
    if resolved and (
        not Path(source.repo).is_absolute()
        or not Path(source.interpreter).is_absolute()
    ):
        raise ValueError("resolved Query-State smoke source paths must be absolute")

    selection_raw = _strict_section(raw, "selection", QueryStateSmokeSelection)
    steps_raw = selection_raw["steps"]
    if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
        raise ValueError("selection.steps must be a sequence")
    selection = QueryStateSmokeSelection(
        steps=tuple(int(_int(value, "selection.steps", minimum=0)) for value in steps_raw),
        **{
            name: int(_int(selection_raw[name], f"selection.{name}", minimum=0))
            for name in QueryStateSmokeSelection.__dataclass_fields__
            if name != "steps"
        },
    )
    expected_selection = {
        "steps": (0, 1, 2, 3), "train_records": 3211, "train_rows": 12836,
        "excluded_train_empty_cot_rows": 5, "validation_records": 355,
        "raw_validation_rows": 1420, "excluded_validation_empty_cot_rows": 0,
        "external_validation_rows": 1413, "cross_split_image_hashes": 5,
        "same_image_multi_instruction_groups": 42,
        "same_instruction_multi_image_groups": 101,
    }
    if asdict(selection) != expected_selection:
        raise ValueError("Query-State smoke selection violates the audited data contract")

    runtime_raw = _strict_section(raw, "runtime", QueryStateSmokeRuntime)
    wrap_policy_raw = runtime_raw["fsdp_wrap_policy"]
    if not isinstance(wrap_policy_raw, Mapping):
        raise ValueError("runtime.fsdp_wrap_policy must be a mapping")
    world_size = _int(
        runtime_raw["world_size"],
        "runtime.world_size",
        allow_unresolved=not resolved,
    )
    runtime = QueryStateSmokeRuntime(
        world_size=world_size,
        nodes=_int(runtime_raw["nodes"], "runtime.nodes", allow_unresolved=not resolved),
        ranks_per_node=_int(runtime_raw["ranks_per_node"], "runtime.ranks_per_node", allow_unresolved=not resolved),
        backend=_text(runtime_raw["backend"], "runtime.backend"),
        model_dtype=_text(runtime_raw["model_dtype"], "runtime.model_dtype"),
        mixed_precision_param_dtype=_text(runtime_raw["mixed_precision_param_dtype"], "runtime.mixed_precision_param_dtype"),
        mixed_precision_reduce_dtype=_text(runtime_raw["mixed_precision_reduce_dtype"], "runtime.mixed_precision_reduce_dtype"),
        mixed_precision_buffer_dtype=_text(runtime_raw["mixed_precision_buffer_dtype"], "runtime.mixed_precision_buffer_dtype"),
        attention_implementation=_text(runtime_raw["attention_implementation"], "runtime.attention_implementation", allow_unresolved=not resolved),
        gradient_checkpointing=_bool(runtime_raw["gradient_checkpointing"], "runtime.gradient_checkpointing"),
        train_mode=_bool(runtime_raw["train_mode"], "runtime.train_mode"),
        fsdp_sharding=_text(runtime_raw["fsdp_sharding"], "runtime.fsdp_sharding"),
        fsdp_use_orig_params=_bool(runtime_raw["fsdp_use_orig_params"], "runtime.fsdp_use_orig_params"),
        fsdp_wrap_policy=dict(wrap_policy_raw),
        model_parallel_size=int(_int(runtime_raw["model_parallel_size"], "runtime.model_parallel_size")),
        max_sequence_length=_int(runtime_raw["max_sequence_length"], "runtime.max_sequence_length", allow_unresolved=not resolved),
        max_pixels=_int(runtime_raw["max_pixels"], "runtime.max_pixels", allow_unresolved=not resolved),
        max_padded_tokens=_int(runtime_raw["max_padded_tokens"], "runtime.max_padded_tokens", allow_unresolved=not resolved),
        max_rows_per_micro_batch=int(_int(runtime_raw["max_rows_per_micro_batch"], "runtime.max_rows_per_micro_batch")),
        rows_per_rank_update=int(_int(runtime_raw["rows_per_rank_update"], "runtime.rows_per_rank_update")),
        fresh_updates=int(_int(runtime_raw["fresh_updates"], "runtime.fresh_updates")),
        resume_updates=int(_int(runtime_raw["resume_updates"], "runtime.resume_updates")),
        seed=_int(runtime_raw["seed"], "runtime.seed", minimum=0, allow_unresolved=not resolved),
        rng_schedule_version=_text(runtime_raw["rng_schedule_version"], "runtime.rng_schedule_version"),
    )
    if (
        runtime.backend != "nccl" or runtime.model_dtype != "bfloat16"
        or runtime.mixed_precision_param_dtype != "bfloat16"
        or runtime.mixed_precision_reduce_dtype != "float32"
        or runtime.mixed_precision_buffer_dtype != "float32"
        or not runtime.gradient_checkpointing or not runtime.train_mode
        or runtime.fsdp_sharding != "full_shard" or not runtime.fsdp_use_orig_params
        or runtime.model_parallel_size != 1 or runtime.max_rows_per_micro_batch != 1
        or runtime.rows_per_rank_update != 1
    ):
        raise ValueError("Query-State smoke runtime violates full-language FSDP contract")
    if (runtime.fresh_updates, runtime.resume_updates) != (1, 1):
        raise ValueError("Query-State smoke requires exactly one fresh and one resume update")
    if resolved:
        if not all(
            isinstance(value, int)
            for value in (
                runtime.world_size,
                runtime.nodes,
                runtime.ranks_per_node,
            )
        ):
            raise ValueError("resolved Query-State smoke topology remains unresolved")
        if runtime.world_size != runtime.nodes * runtime.ranks_per_node:
            raise ValueError("Query-State smoke runtime topology is inconsistent")
        if not runtime.fsdp_wrap_policy:
            raise ValueError("resolved Query-State smoke requires an explicit FSDP wrap policy")
        if not all(isinstance(value, int) for value in (runtime.max_sequence_length, runtime.max_pixels, runtime.max_padded_tokens, runtime.seed)):
            raise ValueError("resolved Query-State smoke runtime remains unresolved")
        if runtime.seed > 2**32 - 1:
            raise ValueError("Query-State smoke seed exceeds NumPy's exact range")
        if runtime.max_padded_tokens < runtime.max_sequence_length:
            raise ValueError("Query-State smoke padded-token budget cannot hold one maximum row")

    data_raw = _strict_section(raw, "data", QueryStateSmokeData)
    data = QueryStateSmokeData(
        train_jsonl=_text(data_raw["train_jsonl"], "data.train_jsonl", allow_unresolved=not resolved),
        train_sha256=_sha(data_raw["train_sha256"], "data.train_sha256", allow_unresolved=not resolved),
        validation_jsonl=_text(data_raw["validation_jsonl"], "data.validation_jsonl", allow_unresolved=not resolved),
        validation_sha256=_sha(data_raw["validation_sha256"], "data.validation_sha256", allow_unresolved=not resolved),
        record_format=_text(data_raw["record_format"], "data.record_format"),
        train_split=_text(data_raw["train_split"], "data.train_split"),
        validation_split=_text(data_raw["validation_split"], "data.validation_split"),
        overlap_key=_text(data_raw["overlap_key"], "data.overlap_key"),
        source_manifest_identity=_sha(data_raw["source_manifest_identity"], "data.source_manifest_identity", allow_unresolved=not resolved),
        smoke_rows=_parse_descriptors(data_raw["smoke_rows"], locked=resolved, world_size=world_size),
    )
    if (data.record_format, data.train_split, data.validation_split, data.overlap_key) != (
        "nimloth_trajectory_v1", "train", "val",
        "record_initial_and_current_next_original_image_sha256",
    ):
        raise ValueError("Query-State smoke data identity mismatch")
    if resolved and (
        not Path(data.train_jsonl).is_absolute()
        or not Path(data.validation_jsonl).is_absolute()
    ):
        raise ValueError("resolved Query-State smoke data paths must be absolute")
    if resolved and any(
        row.rendered_token_count > int(runtime.max_sequence_length)
        or row.rendered_token_count > int(runtime.max_padded_tokens)
        for row in data.smoke_rows
    ):
        raise ValueError("Query-State smoke row exceeds the locked token budget")

    init_raw = _strict_section(raw, "initialization", QueryStateSmokeInitialization)
    shard_hashes = init_raw["actor_model_shards_sha256"]
    action_ids = init_raw["action_token_ids"]
    if not isinstance(shard_hashes, (list, tuple)) or (resolved and not shard_hashes):
        raise ValueError("initialization.actor_model_shards_sha256 must be a sequence")
    if not isinstance(action_ids, (list, tuple)):
        raise ValueError("initialization.action_token_ids must be a sequence")
    initialization = QueryStateSmokeInitialization(
        actor_checkpoint=_text(init_raw["actor_checkpoint"], "initialization.actor_checkpoint", allow_unresolved=not resolved),
        actor_completion_sha256=_sha(init_raw["actor_completion_sha256"], "initialization.actor_completion_sha256", allow_unresolved=not resolved),
        actor_config_sha256=_sha(init_raw["actor_config_sha256"], "initialization.actor_config_sha256", allow_unresolved=not resolved),
        actor_model_index_sha256=_sha(init_raw["actor_model_index_sha256"], "initialization.actor_model_index_sha256", allow_unresolved=not resolved),
        actor_model_shards_sha256=tuple(_sha(value, "initialization.actor_model_shards_sha256", allow_unresolved=not resolved) for value in shard_hashes),
        actor_action_head_sha256=_sha(init_raw["actor_action_head_sha256"], "initialization.actor_action_head_sha256", allow_unresolved=not resolved),
        processor_sha256=_sha(init_raw["processor_sha256"], "initialization.processor_sha256", allow_unresolved=not resolved),
        tokenizer_sha256=_sha(init_raw["tokenizer_sha256"], "initialization.tokenizer_sha256", allow_unresolved=not resolved),
        prompt_template_sha256=_sha(init_raw["prompt_template_sha256"], "initialization.prompt_template_sha256", allow_unresolved=not resolved),
        token_table_sha256=_sha(init_raw["token_table_sha256"], "initialization.token_table_sha256", allow_unresolved=not resolved),
        action_token_ids=tuple(int(_int(value, "initialization.action_token_ids", minimum=0)) for value in action_ids),
        dino_source=_text(init_raw["dino_source"], "initialization.dino_source"),
        dino_revision=_text(init_raw["dino_revision"], "initialization.dino_revision"),
        dino_processor_fingerprint=_text(init_raw["dino_processor_fingerprint"], "initialization.dino_processor_fingerprint"),
        dino_hidden_size=int(_int(init_raw["dino_hidden_size"], "initialization.dino_hidden_size")),
        dino_grid_size=int(_int(init_raw["dino_grid_size"], "initialization.dino_grid_size")),
        fresh_original_observation_targets=_bool(init_raw["fresh_original_observation_targets"], "initialization.fresh_original_observation_targets"),
    )
    if len(set(initialization.actor_model_shards_sha256)) != len(
        initialization.actor_model_shards_sha256
    ):
        raise ValueError("Query-State smoke actor model shard identities are duplicated")
    if resolved and not Path(initialization.actor_checkpoint).is_absolute():
        raise ValueError("resolved Query-State smoke actor path must be absolute")
    if (
        initialization.action_token_ids != tuple(range(151683, 151691))
        or initialization.dino_source != "facebook/dinov2-large"
        or initialization.dino_revision != _PINNED_DINO_REVISION
        or initialization.dino_processor_fingerprint != "7d65a7de8788e87d"
        or (initialization.dino_hidden_size, initialization.dino_grid_size) != (1024, 4)
        or not initialization.fresh_original_observation_targets
    ):
        raise ValueError("Query-State smoke initialization/teacher identity mismatch")

    optimizer_raw = _strict_section(raw, "optimizer", QueryStateSmokeOptimizer)
    betas_raw = optimizer_raw["betas"]
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    optimizer = QueryStateSmokeOptimizer(
        name=_text(optimizer_raw["name"], "optimizer.name"),
        language_learning_rate=_number(optimizer_raw["language_learning_rate"], "optimizer.language_learning_rate", positive=True),
        direct_state_learning_rate=_number(optimizer_raw["direct_state_learning_rate"], "optimizer.direct_state_learning_rate", positive=True),
        weight_decay=_number(optimizer_raw["weight_decay"], "optimizer.weight_decay", positive=False),
        betas=tuple(_number(value, "optimizer.betas", positive=False) for value in betas_raw),  # type: ignore[arg-type]
        epsilon=_number(optimizer_raw["epsilon"], "optimizer.epsilon", positive=True),
        scheduler=_text(optimizer_raw["scheduler"], "optimizer.scheduler"),
        max_grad_norm=_number(optimizer_raw["max_grad_norm"], "optimizer.max_grad_norm", positive=True),
    )
    if (
        optimizer.name.lower() != "adamw"
        or optimizer.scheduler != "constant_lambda_1"
        or any(value >= 1.0 for value in optimizer.betas)
    ):
        raise ValueError("Query-State smoke optimizer contract is invalid")

    checkpoint_raw = _strict_section(raw, "checkpoint", QueryStateSmokeCheckpoint)
    checkpoint = QueryStateSmokeCheckpoint(
        **{
            name: _bool(checkpoint_raw[name], f"checkpoint.{name}")
            for name in QueryStateSmokeCheckpoint.__dataclass_fields__
            if name not in {"fresh_checkpoint_name", "resume_checkpoint_name", "completion_marker"}
        },
        fresh_checkpoint_name=_text(checkpoint_raw["fresh_checkpoint_name"], "checkpoint.fresh_checkpoint_name"),
        resume_checkpoint_name=_text(checkpoint_raw["resume_checkpoint_name"], "checkpoint.resume_checkpoint_name"),
        completion_marker=_text(checkpoint_raw["completion_marker"], "checkpoint.completion_marker"),
    )
    bool_values = [value for value in asdict(checkpoint).values() if isinstance(value, bool)]
    checkpoint_names = (
        checkpoint.fresh_checkpoint_name,
        checkpoint.resume_checkpoint_name,
    )
    if (
        not all(bool_values)
        or checkpoint.completion_marker != "COMPLETED"
        or checkpoint.fresh_checkpoint_name == checkpoint.resume_checkpoint_name
        or any(Path(name).name != name or name in {".", ".."} for name in checkpoint_names)
    ):
        raise ValueError("Query-State smoke checkpoint contract is not immutable/exact")

    output_raw = _strict_section(raw, "output", QueryStateSmokeOutput)
    output = QueryStateSmokeOutput(
        experiment_group=_text(output_raw["experiment_group"], "output.experiment_group"),
        run_root=_text(output_raw["run_root"], "output.run_root", allow_unresolved=not resolved),
        controller_log_dir=_text(output_raw["controller_log_dir"], "output.controller_log_dir", allow_unresolved=not resolved),
        fresh_child=_text(output_raw["fresh_child"], "output.fresh_child"),
        resume_child=_text(output_raw["resume_child"], "output.resume_child"),
        minimum_free_bytes=_int(output_raw["minimum_free_bytes"], "output.minimum_free_bytes", allow_unresolved=not resolved),
        overwrite=_bool(output_raw["overwrite"], "output.overwrite"),
        tracking_mode=_text(output_raw["tracking_mode"], "output.tracking_mode"),
    )
    if (
        output.experiment_group != "outputs/experiments/training/sft1_query_state_smoke"
        or output.overwrite or output.tracking_mode != "disabled"
        or output.fresh_child == output.resume_child
    ):
        raise ValueError("Query-State smoke output/tracking contract is invalid")
    if resolved:
        run_root = Path(output.run_root)
        controller_root = Path(output.controller_log_dir)
        group_parts = Path(output.experiment_group).parts
        if (
            not run_root.is_absolute()
            or not controller_root.is_absolute()
            or run_root.parent != controller_root.parent
            or run_root.parent.parts[-len(group_parts) :] != group_parts
        ):
            raise ValueError(
                "Query-State smoke outputs must be absolute siblings directly "
                "inside the canonical experiment_group"
            )
        if any(
            Path(name).name != name or name in {".", ".."}
            for name in (output.fresh_child, output.resume_child)
        ):
            raise ValueError("Query-State smoke output child names must be simple paths")
        if run_root == controller_root:
            raise ValueError(
                "Query-State smoke controller logs must be a sibling of the guarded run root"
            )

    resources_raw = _strict_section(raw, "resources", QueryStateSmokeResources)
    allowlist = resources_raw["gpu_model_allowlist"]
    if not isinstance(allowlist, (list, tuple)):
        raise ValueError("resources.gpu_model_allowlist must be a sequence")
    resources = QueryStateSmokeResources(
        account=_text(resources_raw["account"], "resources.account"),
        partition=_text(resources_raw["partition"], "resources.partition", allow_unresolved=not resolved),
        node_count=_int(resources_raw["node_count"], "resources.node_count", allow_unresolved=not resolved),
        gpus_per_node=_int(resources_raw["gpus_per_node"], "resources.gpus_per_node", allow_unresolved=not resolved),
        total_gpus=_int(resources_raw["total_gpus"], "resources.total_gpus", allow_unresolved=not resolved),
        gpu_model_allowlist=tuple(_text(value, "resources.gpu_model_allowlist") for value in allowlist),
        cpus_per_task=_int(resources_raw["cpus_per_task"], "resources.cpus_per_task", allow_unresolved=not resolved),
        memory_gib=_int(resources_raw["memory_gib"], "resources.memory_gib", allow_unresolved=not resolved),
        walltime=_text(resources_raw["walltime"], "resources.walltime", allow_unresolved=not resolved),
    )
    if resources.account != "peilab":
        raise ValueError("Query-State smoke resource account identity mismatch")
    if resolved:
        if not all(
            isinstance(value, int)
            for value in (
                resources.node_count,
                resources.gpus_per_node,
                resources.total_gpus,
            )
        ):
            raise ValueError("resolved Query-State smoke resources remain unresolved")
        if (
            resources.total_gpus != resources.node_count * resources.gpus_per_node
            or resources.total_gpus != runtime.world_size
            or resources.node_count != runtime.nodes
            or resources.gpus_per_node != runtime.ranks_per_node
        ):
            raise ValueError("Query-State smoke resource topology is inconsistent")
        if (
            not resources.gpu_model_allowlist
            or len(set(resources.gpu_model_allowlist)) != len(resources.gpu_model_allowlist)
            or not re.fullmatch(r"\d{2}:\d{2}:\d{2}", resources.walltime)
        ):
            raise ValueError("resolved Query-State smoke resource fields are invalid")

    config = QueryStateSmokeConfig(
        schema=QUERY_STATE_SMOKE_CONFIG_SCHEMA,
        state_contract=state,
        source=source,
        selection=selection,
        data=data,
        initialization=initialization,
        optimizer=optimizer,
        runtime=runtime,
        checkpoint=checkpoint,
        output=output,
        resources=resources,
        authorization=authorization,
    )
    payload = asdict(config)
    authorization_payload = payload.pop("authorization")
    operational_unresolved = _contains_unresolved(payload)
    authorization_unresolved = _contains_unresolved(authorization_payload)
    if resolved and operational_unresolved:
        raise ValueError("preflight-locked Query-State smoke config contains unresolved field")
    if not resolved and not operational_unresolved:
        raise ValueError(
            "unlocked Query-State smoke template must remain explicitly unresolved"
        )
    if locked and authorization_unresolved:
        raise ValueError("launch-locked Query-State smoke config contains unresolved approval")
    if not locked and (
        authorization.approval_evidence != QUERY_STATE_SMOKE_UNRESOLVED
        or authorization.approved_command_sha256 != QUERY_STATE_SMOKE_UNRESOLVED
    ):
        raise ValueError(
            "non-launching Query-State smoke config may not claim approval evidence"
        )
    return config


def parse_query_state_smoke_preflight_config(
    raw: Mapping[str, Any],
) -> QueryStateSmokeConfig:
    """Parse an operationally resolved config without granting CUDA launch."""

    config = parse_query_state_smoke_preparation(raw)
    if not config.preflight_locked:
        raise PermissionError("Query-State smoke CPU preflight config is not locked")
    return config


def parse_query_state_smoke_config(raw: Mapping[str, Any]) -> QueryStateSmokeConfig:
    """Parse only a fully resolved launch-locked smoke config."""

    config = parse_query_state_smoke_preparation(raw)
    if not config.launch_locked:
        raise PermissionError("Query-State smoke CUDA config is not launch-locked")
    return config


def assert_query_state_smoke_cuda_ready(
    config: QueryStateSmokeConfig,
    *,
    approved_command: str | None = None,
) -> None:
    """Fail closed before any CUDA/process-group/model/output side effect."""

    if (
        not isinstance(config, QueryStateSmokeConfig)
        or not config.preflight_locked
        or not config.launch_locked
    ):
        raise PermissionError("Query-State smoke is not launch-locked")
    if approved_command is None:
        raise PermissionError("Query-State smoke requires the approved command manifest")
    actual = hashlib.sha256(approved_command.encode()).hexdigest()
    if actual != config.authorization.approved_command_sha256:
        raise ValueError("Query-State smoke approved command identity mismatch")


__all__ = [
    "QUERY_STATE_SMOKE_CONFIG_SCHEMA",
    "QUERY_STATE_SMOKE_UNRESOLVED",
    "QueryStateSmokeAuthorization",
    "QueryStateSmokeCheckpoint",
    "QueryStateSmokeConfig",
    "QueryStateSmokeData",
    "QueryStateSmokeInitialization",
    "QueryStateSmokeOptimizer",
    "QueryStateSmokeOutput",
    "QueryStateSmokeResources",
    "QueryStateSmokeRowDescriptor",
    "QueryStateSmokeRuntime",
    "QueryStateSmokeSelection",
    "QueryStateSmokeSource",
    "QueryStateSmokeStateContract",
    "assert_query_state_smoke_cuda_ready",
    "parse_query_state_smoke_config",
    "parse_query_state_smoke_preflight_config",
    "parse_query_state_smoke_preparation",
]
