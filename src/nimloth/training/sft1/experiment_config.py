"""Strict early-4 SFT1-v2 report-first experiment contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from nimloth.config import load_yaml_config
from nimloth.training.sft1.objective import (
    OBSERVED_MOVEMENT_ACTION_INDICES,
    SFT1V2LossWeights,
)



SFT1_V2_CONFIG_SCHEMA = "nimloth_sft1_state_v2_experiment_v1"
STATE_INTERFACE_OBJECTIVE_VERSION = "nimloth_state_interface_v2_canary"
EARLY4_STEPS = (0, 1, 2, 3)
APPROVED_COUNTS = {
    "train_records": 3211,
    "train_rows": 12836,
    "excluded_train_empty_cot_rows": 5,
    "validation_records": 355,
    "raw_validation_rows": 1420,
    "excluded_validation_empty_cot_rows": 0,
    "external_validation_rows": 1413,
    "cross_split_image_hashes": 5,
    "same_image_multi_instruction_groups": 42,
    "same_instruction_multi_image_groups": 101,
}


@dataclass(frozen=True)
class SFT1V2StateConfig:
    objective_version: str
    latent_query_mode: str
    query_tune: str
    grid_tokens: int
    qwen_hidden_dim: int
    state_dim: int
    projector_hidden_dim: int
    instruction_teacher_dim: int
    action_dim: int
    movement_action_indices: tuple[int, int, int]


@dataclass(frozen=True)
class SFT1V2ObjectiveConfig:
    weights: SFT1V2LossWeights
    policy_temperature: float
    contrastive_temperature: float


@dataclass(frozen=True)
class SFT1V2FreezeConfig:
    freeze_qwen_language_body: bool
    freeze_lm_head: bool
    freeze_vision_tower: bool
    freeze_dino_teacher: bool
    freeze_instruction_action_teacher: bool
    train_query_adapter: bool
    train_fresh_projector: bool
    train_readouts: bool


@dataclass(frozen=True)
class SFT1V2SelectionConfig:
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
class SFT1V2SourceConfig:
    repo: str
    expected_commit: str
    vagen_commit: str
    verl_commit: str
    interpreter: str


@dataclass(frozen=True)
class SFT1V2DataConfig:
    train_jsonl: str
    train_sha256: str
    validation_jsonl: str
    validation_sha256: str
    record_format: str
    train_split: str
    validation_split: str
    overlap_key: str


@dataclass(frozen=True)
class SFT1V2TeacherConfig:
    actor_checkpoint: str
    actor_completion_sha256: str
    actor_config_sha256: str
    actor_model_index_sha256: str
    actor_model_shards_sha256: tuple[str, ...]
    actor_action_head_sha256: str
    action_token_ids: tuple[int, ...]
    processor_sha256: str
    tokenizer_sha256: str
    prompt_template_sha256: str
    token_table_sha256: str
    dino_source: str
    dino_revision: str
    dino_processor_fingerprint: str
    fresh_targets_only: bool


@dataclass(frozen=True)
class SFT1V2CacheConfig:
    output_dir: str
    shard_count: int
    row_schema: str
    parity_dino_path: str
    parity_dino_sha256: str
    parity_instruction_path: str
    parity_instruction_sha256: str


@dataclass(frozen=True)
class SFT1V2OptimizerConfig:
    name: str
    query_learning_rate: float
    projector_readout_learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    epsilon: float
    scheduler: str
    warmup_steps: int


@dataclass(frozen=True)
class SFT1V2RuntimeConfig:
    epochs: int
    max_grad_norm: float
    fsdp_sharding: str
    fsdp_use_orig_params: bool
    gradient_checkpointing: bool
    train_mode: bool
    attention_implementation: str
    model_dtype: str
    max_sequence_length: int
    max_pixels: int
    teacher_batch_size: int
    max_padded_tokens: int
    max_rows_per_micro_batch: int
    rows_per_rank_update: int
    world_size: int
    launch_locked: bool


@dataclass(frozen=True)
class SFT1V2CheckpointConfig:
    cadence_steps: int
    at_epoch_boundary: bool
    exact_resume: bool
    immutable_shards: bool


@dataclass(frozen=True)
class SFT1V2ValidationConfig:
    epochs: tuple[int, ...]
    report_first: bool
    bootstrap_seed: int
    bootstrap_resamples: int
    actor_mean_kl_stop: float
    actor_top1_agreement_stop: float
    external_only: bool


@dataclass(frozen=True)
class SFT1V2OutputConfig:
    experiment_group: str
    run_dir: str
    wandb_project: str
    wandb_run_name: str
    wandb_run_id: str
    minimum_free_bytes: int
    overwrite: bool


@dataclass(frozen=True)
class SFT1V2Config:
    schema: str
    state: SFT1V2StateConfig
    objective: SFT1V2ObjectiveConfig
    freeze: SFT1V2FreezeConfig
    selection: SFT1V2SelectionConfig
    source: SFT1V2SourceConfig
    data: SFT1V2DataConfig
    teacher: SFT1V2TeacherConfig
    cache: SFT1V2CacheConfig
    optimizer: SFT1V2OptimizerConfig
    runtime: SFT1V2RuntimeConfig
    checkpoint: SFT1V2CheckpointConfig
    validation: SFT1V2ValidationConfig
    output: SFT1V2OutputConfig

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def _strict_section(raw: Mapping[str, Any], name: str, fields: set[str]) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"SFT1-v2 config section {name!r} is required")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown SFT1-v2 config field: {name}.{unknown[0]}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"missing SFT1-v2 config field: {name}.{missing[0]}")
    for field in sorted(fields):
        if value[field] is None:
            raise ValueError(f"SFT1-v2 config field may not be null: {name}.{field}")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return int(value)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    if (positive and result <= 0.0) or (not positive and result < 0.0):
        raise ValueError(f"{field} must be {'positive' if positive else 'non-negative'}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    text = _text(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return text


def _exact(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(f"{field} must equal the approved value {expected!r}")


def parse_sft1_v2_config(raw: Mapping[str, Any]) -> SFT1V2Config:
    """Parse the fail-closed canary schema; no experiment value is inferred."""

    if not isinstance(raw, Mapping):
        raise ValueError("SFT1-v2 config must be a mapping")
    top_fields = {
        "schema", "state", "objective", "freeze", "selection", "source", "data",
        "teacher", "cache", "optimizer", "runtime", "checkpoint",
        "validation", "output",
    }
    unknown = sorted(set(raw) - top_fields)
    if unknown:
        raise ValueError(f"unknown SFT1-v2 config section: {unknown[0]}")
    missing = sorted(top_fields - set(raw))
    if missing:
        raise ValueError(f"missing SFT1-v2 config section: {missing[0]}")
    if raw["schema"] != SFT1_V2_CONFIG_SCHEMA:
        raise ValueError(f"unsupported SFT1-v2 config schema {raw['schema']!r}; legacy state config cannot resume")

    state_raw = _strict_section(raw, "state", {
        "objective_version", "latent_query_mode", "query_tune", "grid_tokens",
        "qwen_hidden_dim", "state_dim", "projector_hidden_dim",
        "instruction_teacher_dim", "action_dim", "movement_action_indices",
    })
    movement_raw = state_raw["movement_action_indices"]
    if not isinstance(movement_raw, (list, tuple)):
        raise ValueError("state.movement_action_indices must be a sequence")
    state = SFT1V2StateConfig(
        objective_version=_text(state_raw["objective_version"], "state.objective_version"),
        latent_query_mode=_text(state_raw["latent_query_mode"], "state.latent_query_mode"),
        query_tune=_text(state_raw["query_tune"], "state.query_tune"),
        grid_tokens=_integer(state_raw["grid_tokens"], "state.grid_tokens"),
        qwen_hidden_dim=_integer(state_raw["qwen_hidden_dim"], "state.qwen_hidden_dim"),
        state_dim=_integer(state_raw["state_dim"], "state.state_dim"),
        projector_hidden_dim=_integer(state_raw["projector_hidden_dim"], "state.projector_hidden_dim"),
        instruction_teacher_dim=_integer(state_raw["instruction_teacher_dim"], "state.instruction_teacher_dim"),
        action_dim=_integer(state_raw["action_dim"], "state.action_dim"),
        movement_action_indices=tuple(movement_raw),  # type: ignore[arg-type]
    )
    _exact(state.objective_version, STATE_INTERFACE_OBJECTIVE_VERSION, "state.objective_version")
    if state.latent_query_mode != "inject" or state.query_tune != "adapter":
        raise ValueError("v2 canary requires injected queries with additive adapter tuning")
    if (state.grid_tokens, state.state_dim, state.action_dim) != (16, 1024, 8):
        raise ValueError("v2 deployed contract is exactly K16 x 1024 with eight actions")
    if (state.qwen_hidden_dim, state.instruction_teacher_dim) != (2048, 2048):
        raise ValueError("v2 ID176 interface requires 2048-dimensional Qwen hidden and exact-instruction teacher targets")
    if state.movement_action_indices != OBSERVED_MOVEMENT_ACTION_INDICES:
        raise ValueError("state.movement_action_indices must equal the identity-bound move_forward/move_right/move_left mapping")

    objective_raw = _strict_section(raw, "objective", {
        "visual_weight", "visual_relation_coefficient", "instruction_weight",
        "instruction_contrastive_coefficient", "observed_feasibility_weight",
        "actor_preservation_weight", "state_policy_weight", "policy_temperature",
        "contrastive_temperature",
    })
    objective = SFT1V2ObjectiveConfig(
        weights=SFT1V2LossWeights(
            visual=_number(objective_raw["visual_weight"], "objective.visual_weight"),
            visual_relation_coefficient=_number(objective_raw["visual_relation_coefficient"], "objective.visual_relation_coefficient"),
            instruction=_number(objective_raw["instruction_weight"], "objective.instruction_weight"),
            instruction_contrastive_coefficient=_number(objective_raw["instruction_contrastive_coefficient"], "objective.instruction_contrastive_coefficient"),
            observed_feasibility=_number(objective_raw["observed_feasibility_weight"], "objective.observed_feasibility_weight"),
            actor_preservation=_number(objective_raw["actor_preservation_weight"], "objective.actor_preservation_weight"),
            state_policy=_number(objective_raw["state_policy_weight"], "objective.state_policy_weight"),
        ),
        policy_temperature=_number(objective_raw["policy_temperature"], "objective.policy_temperature", positive=True),
        contrastive_temperature=_number(objective_raw["contrastive_temperature"], "objective.contrastive_temperature", positive=True),
    )
    expected_weights = (1.0, 1.0, 1.0, 0.25, 1.0, 10.0, 1.0)
    _exact(tuple(asdict(objective.weights).values()), expected_weights, "objective weights")
    _exact(objective.policy_temperature, 1.0, "objective.policy_temperature")
    _exact(objective.contrastive_temperature, 0.1, "objective.contrastive_temperature")

    freeze_raw = _strict_section(raw, "freeze", {
        "freeze_qwen_language_body", "freeze_lm_head", "freeze_vision_tower",
        "freeze_dino_teacher", "freeze_instruction_action_teacher",
        "train_query_adapter", "train_fresh_projector", "train_readouts",
    })
    freeze = SFT1V2FreezeConfig(**{
        field: _boolean(freeze_raw[field], f"freeze.{field}")
        for field in SFT1V2FreezeConfig.__dataclass_fields__
    })
    if not all(asdict(freeze).values()):
        raise ValueError("v2 canary freeze contract permits only query adapter, fresh projector, and readouts")

    selection_raw = _strict_section(raw, "selection", {"steps", *APPROVED_COUNTS})
    steps = selection_raw["steps"]
    if not isinstance(steps, (list, tuple)):
        raise ValueError("selection.steps must be a sequence")
    selection = SFT1V2SelectionConfig(
        steps=tuple(_integer(value, "selection.steps", minimum=0) for value in steps),
        **{name: _integer(selection_raw[name], f"selection.{name}", minimum=0) for name in APPROVED_COUNTS},
    )
    _exact(selection.steps, EARLY4_STEPS, "selection.steps")
    for name, expected in APPROVED_COUNTS.items():
        _exact(getattr(selection, name), expected, f"selection.{name}")

    source_raw = _strict_section(raw, "source", set(SFT1V2SourceConfig.__dataclass_fields__))
    source = SFT1V2SourceConfig(
        repo=_text(source_raw["repo"], "source.repo"),
        expected_commit=_text(source_raw["expected_commit"], "source.expected_commit"),
        vagen_commit=_text(source_raw["vagen_commit"], "source.vagen_commit"),
        verl_commit=_text(source_raw["verl_commit"], "source.verl_commit"),
        interpreter=_text(source_raw["interpreter"], "source.interpreter"),
    )
    _exact(source.vagen_commit, "9f1e89eb8c9839a406b6e62aa75703494a79e5b5", "source.vagen_commit")
    _exact(source.verl_commit, "494f264494b2525f2c13595f63ac4912963e6d2f", "source.verl_commit")
    if not source.interpreter.endswith("/.venv-vagen-main/bin/python3"):
        raise ValueError("source.interpreter must be the explicit .venv-vagen-main Python")

    data_raw = _strict_section(raw, "data", set(SFT1V2DataConfig.__dataclass_fields__))
    data = SFT1V2DataConfig(
        train_jsonl=_text(data_raw["train_jsonl"], "data.train_jsonl"),
        train_sha256=_sha256(data_raw["train_sha256"], "data.train_sha256"),
        validation_jsonl=_text(data_raw["validation_jsonl"], "data.validation_jsonl"),
        validation_sha256=_sha256(data_raw["validation_sha256"], "data.validation_sha256"),
        record_format=_text(data_raw["record_format"], "data.record_format"),
        train_split=_text(data_raw["train_split"], "data.train_split"),
        validation_split=_text(data_raw["validation_split"], "data.validation_split"),
        overlap_key=_text(data_raw["overlap_key"], "data.overlap_key"),
    )
    _exact(data.record_format, "nimloth_trajectory_v1", "data.record_format")
    _exact((data.train_split, data.validation_split), ("train", "val"), "data splits")
    _exact(
        data.overlap_key,
        "record_initial_and_current_next_original_image_sha256",
        "data.overlap_key",
    )

    teacher_raw = _strict_section(raw, "teacher", set(SFT1V2TeacherConfig.__dataclass_fields__))
    shard_hashes = teacher_raw["actor_model_shards_sha256"]
    action_token_ids = teacher_raw["action_token_ids"]
    if not isinstance(shard_hashes, (list, tuple)) or not shard_hashes:
        raise ValueError("teacher.actor_model_shards_sha256 must be a non-empty sequence")
    if not isinstance(action_token_ids, (list, tuple)):
        raise ValueError("teacher.action_token_ids must be a sequence")
    teacher = SFT1V2TeacherConfig(
        actor_checkpoint=_text(teacher_raw["actor_checkpoint"], "teacher.actor_checkpoint"),
        actor_completion_sha256=_sha256(teacher_raw["actor_completion_sha256"], "teacher.actor_completion_sha256"),
        actor_config_sha256=_sha256(teacher_raw["actor_config_sha256"], "teacher.actor_config_sha256"),
        actor_model_index_sha256=_sha256(teacher_raw["actor_model_index_sha256"], "teacher.actor_model_index_sha256"),
        actor_model_shards_sha256=tuple(_sha256(value, "teacher.actor_model_shards_sha256") for value in shard_hashes),
        actor_action_head_sha256=_sha256(teacher_raw["actor_action_head_sha256"], "teacher.actor_action_head_sha256"),
        action_token_ids=tuple(_integer(value, "teacher.action_token_ids", minimum=0) for value in action_token_ids),
        processor_sha256=_sha256(teacher_raw["processor_sha256"], "teacher.processor_sha256"),
        tokenizer_sha256=_sha256(teacher_raw["tokenizer_sha256"], "teacher.tokenizer_sha256"),
        prompt_template_sha256=_sha256(teacher_raw["prompt_template_sha256"], "teacher.prompt_template_sha256"),
        token_table_sha256=_sha256(teacher_raw["token_table_sha256"], "teacher.token_table_sha256"),
        dino_source=_text(teacher_raw["dino_source"], "teacher.dino_source"),
        dino_revision=_text(teacher_raw["dino_revision"], "teacher.dino_revision"),
        dino_processor_fingerprint=_text(teacher_raw["dino_processor_fingerprint"], "teacher.dino_processor_fingerprint"),
        fresh_targets_only=_boolean(teacher_raw["fresh_targets_only"], "teacher.fresh_targets_only"),
    )
    _exact(teacher.dino_source, "facebook/dinov2-large", "teacher.dino_source")
    _exact(teacher.dino_revision, "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c", "teacher.dino_revision")
    _exact(teacher.action_token_ids, tuple(range(151683, 151691)), "teacher.action_token_ids")
    if not teacher.fresh_targets_only:
        raise ValueError("teacher targets must be generated fresh from original rows")

    cache_raw = _strict_section(raw, "cache", set(SFT1V2CacheConfig.__dataclass_fields__))
    cache = SFT1V2CacheConfig(
        output_dir=_text(cache_raw["output_dir"], "cache.output_dir"),
        shard_count=_integer(cache_raw["shard_count"], "cache.shard_count"),
        row_schema=_text(cache_raw["row_schema"], "cache.row_schema"),
        parity_dino_path=_text(cache_raw["parity_dino_path"], "cache.parity_dino_path"),
        parity_dino_sha256=_sha256(cache_raw["parity_dino_sha256"], "cache.parity_dino_sha256"),
        parity_instruction_path=_text(cache_raw["parity_instruction_path"], "cache.parity_instruction_path"),
        parity_instruction_sha256=_sha256(cache_raw["parity_instruction_sha256"], "cache.parity_instruction_sha256"),
    )
    _exact(cache.row_schema, "nimloth_sft1_state_v2_teacher_row_v2", "cache.row_schema")

    optimizer_raw = _strict_section(raw, "optimizer", set(SFT1V2OptimizerConfig.__dataclass_fields__))
    betas_raw = optimizer_raw["betas"]
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ValueError("optimizer.betas must contain exactly two values")
    optimizer = SFT1V2OptimizerConfig(
        name=_text(optimizer_raw["name"], "optimizer.name"),
        query_learning_rate=_number(optimizer_raw["query_learning_rate"], "optimizer.query_learning_rate", positive=True),
        projector_readout_learning_rate=_number(optimizer_raw["projector_readout_learning_rate"], "optimizer.projector_readout_learning_rate", positive=True),
        weight_decay=_number(optimizer_raw["weight_decay"], "optimizer.weight_decay"),
        betas=tuple(_number(value, "optimizer.betas") for value in betas_raw),  # type: ignore[arg-type]
        epsilon=_number(optimizer_raw["epsilon"], "optimizer.epsilon", positive=True),
        scheduler=_text(optimizer_raw["scheduler"], "optimizer.scheduler"),
        warmup_steps=_integer(optimizer_raw["warmup_steps"], "optimizer.warmup_steps", minimum=0),
    )
    _exact((optimizer.name, optimizer.query_learning_rate, optimizer.projector_readout_learning_rate, optimizer.weight_decay, optimizer.betas, optimizer.epsilon, optimizer.scheduler, optimizer.warmup_steps), ("adamw", 1e-4, 1e-3, 0.0, (0.9, 0.95), 1e-8, "none", 0), "optimizer contract")

    runtime_raw = _strict_section(raw, "runtime", set(SFT1V2RuntimeConfig.__dataclass_fields__))
    runtime = SFT1V2RuntimeConfig(
        epochs=_integer(runtime_raw["epochs"], "runtime.epochs"),
        max_grad_norm=_number(runtime_raw["max_grad_norm"], "runtime.max_grad_norm", positive=True),
        fsdp_sharding=_text(runtime_raw["fsdp_sharding"], "runtime.fsdp_sharding"),
        fsdp_use_orig_params=_boolean(runtime_raw["fsdp_use_orig_params"], "runtime.fsdp_use_orig_params"),
        gradient_checkpointing=_boolean(runtime_raw["gradient_checkpointing"], "runtime.gradient_checkpointing"),
        train_mode=_boolean(runtime_raw["train_mode"], "runtime.train_mode"),
        attention_implementation=_text(
            runtime_raw["attention_implementation"],
            "runtime.attention_implementation",
        ),
        model_dtype=_text(runtime_raw["model_dtype"], "runtime.model_dtype"),
        max_sequence_length=_integer(
            runtime_raw["max_sequence_length"], "runtime.max_sequence_length"
        ),
        max_pixels=_integer(runtime_raw["max_pixels"], "runtime.max_pixels"),
        teacher_batch_size=_integer(
            runtime_raw["teacher_batch_size"], "runtime.teacher_batch_size"
        ),
        max_padded_tokens=_integer(runtime_raw["max_padded_tokens"], "runtime.max_padded_tokens"),
        max_rows_per_micro_batch=_integer(runtime_raw["max_rows_per_micro_batch"], "runtime.max_rows_per_micro_batch"),
        rows_per_rank_update=_integer(
            runtime_raw["rows_per_rank_update"], "runtime.rows_per_rank_update"
        ),
        world_size=_integer(runtime_raw["world_size"], "runtime.world_size"),
        launch_locked=_boolean(runtime_raw["launch_locked"], "runtime.launch_locked"),
    )
    _exact(
        (
            runtime.epochs,
            runtime.max_grad_norm,
            runtime.fsdp_sharding,
            runtime.fsdp_use_orig_params,
            runtime.gradient_checkpointing,
            runtime.train_mode,
            runtime.attention_implementation,
            runtime.model_dtype,
        ),
        (3, 1.0, "full_shard", True, True, True, "flash_attention_2", "bfloat16"),
        "runtime contract",
    )
    if runtime.max_rows_per_micro_batch < 2 or runtime.rows_per_rank_update < 2:
        raise ValueError(
            "instruction contrastive supervision requires at least two rows per micro-batch/update"
        )
    if runtime.max_padded_tokens < 2 * runtime.max_sequence_length:
        raise ValueError(
            "runtime.max_padded_tokens must keep two maximum-length rows together"
        )

    checkpoint_raw = _strict_section(raw, "checkpoint", set(SFT1V2CheckpointConfig.__dataclass_fields__))
    checkpoint = SFT1V2CheckpointConfig(
        cadence_steps=_integer(checkpoint_raw["cadence_steps"], "checkpoint.cadence_steps"),
        at_epoch_boundary=_boolean(checkpoint_raw["at_epoch_boundary"], "checkpoint.at_epoch_boundary"),
        exact_resume=_boolean(checkpoint_raw["exact_resume"], "checkpoint.exact_resume"),
        immutable_shards=_boolean(checkpoint_raw["immutable_shards"], "checkpoint.immutable_shards"),
    )
    if not (checkpoint.at_epoch_boundary and checkpoint.exact_resume and checkpoint.immutable_shards):
        raise ValueError("checkpoint contract requires immutable exact resume at epoch boundaries")

    validation_raw = _strict_section(raw, "validation", set(SFT1V2ValidationConfig.__dataclass_fields__))
    epochs_raw = validation_raw["epochs"]
    if not isinstance(epochs_raw, (list, tuple)):
        raise ValueError("validation.epochs must be a sequence")
    validation = SFT1V2ValidationConfig(
        epochs=tuple(_integer(value, "validation.epochs", minimum=0) for value in epochs_raw),
        report_first=_boolean(validation_raw["report_first"], "validation.report_first"),
        bootstrap_seed=_integer(validation_raw["bootstrap_seed"], "validation.bootstrap_seed", minimum=0),
        bootstrap_resamples=_integer(validation_raw["bootstrap_resamples"], "validation.bootstrap_resamples"),
        actor_mean_kl_stop=_number(validation_raw["actor_mean_kl_stop"], "validation.actor_mean_kl_stop", positive=True),
        actor_top1_agreement_stop=_number(validation_raw["actor_top1_agreement_stop"], "validation.actor_top1_agreement_stop"),
        external_only=_boolean(validation_raw["external_only"], "validation.external_only"),
    )
    _exact((validation.epochs, validation.report_first, validation.actor_mean_kl_stop, validation.actor_top1_agreement_stop, validation.external_only), ((0, 1, 2, 3), True, 0.1, 0.90, True), "validation contract")

    output_raw = _strict_section(raw, "output", set(SFT1V2OutputConfig.__dataclass_fields__))
    output = SFT1V2OutputConfig(
        experiment_group=_text(output_raw["experiment_group"], "output.experiment_group"),
        run_dir=_text(output_raw["run_dir"], "output.run_dir"),
        wandb_project=_text(output_raw["wandb_project"], "output.wandb_project"),
        wandb_run_name=_text(output_raw["wandb_run_name"], "output.wandb_run_name"),
        wandb_run_id=_text(output_raw["wandb_run_id"], "output.wandb_run_id"),
        minimum_free_bytes=_integer(
            output_raw["minimum_free_bytes"], "output.minimum_free_bytes"
        ),
        overwrite=_boolean(output_raw["overwrite"], "output.overwrite"),
    )
    _exact(output.experiment_group, "outputs/experiments/training/sft1_state_interface_v2", "output.experiment_group")
    _exact(output.wandb_project, "nimloth-sft1", "output.wandb_project")
    if output.overwrite:
        raise ValueError("SFT1-v2 output overwrite is forbidden")

    if runtime.launch_locked:
        unresolved = {
            "source.repo": source.repo,
            "source.expected_commit": source.expected_commit,
            "cache.output_dir": cache.output_dir,
            "output.run_dir": output.run_dir,
            "output.wandb_run_name": output.wandb_run_name,
            "output.wandb_run_id": output.wandb_run_id,
        }
        bad = sorted(
            name for name, value in unresolved.items()
            if "LOCK_BEFORE_LAUNCH" in value
        )
        if len(source.expected_commit) != 40 or any(
            char not in "0123456789abcdef" for char in source.expected_commit
        ):
            bad.append("source.expected_commit")
        identity_hashes = {
            "teacher.processor_sha256": teacher.processor_sha256,
            "teacher.tokenizer_sha256": teacher.tokenizer_sha256,
            "teacher.prompt_template_sha256": teacher.prompt_template_sha256,
            "teacher.token_table_sha256": teacher.token_table_sha256,
        }
        bad.extend(
            name for name, value in identity_hashes.items() if value == "0" * 64
        )
        if bad:
            raise ValueError(
                "launch-locked config contains unresolved field: " + sorted(set(bad))[0]
            )
        if (
            not output.wandb_run_id.startswith("nimloth-sft1-id")
            or not output.wandb_run_name.split("_", 1)[0].isdigit()
        ):
            raise ValueError("launch W&B identity is outside the nimloth-sft1 numeric namespace")
        group = output.experiment_group.strip("/")
        if group not in Path(output.run_dir).as_posix() or group not in Path(
            cache.output_dir
        ).as_posix():
            raise ValueError(
                "launch outputs must remain inside output.experiment_group"
            )

    return SFT1V2Config(
        schema=SFT1_V2_CONFIG_SCHEMA, state=state, objective=objective, freeze=freeze,
        selection=selection, source=source, data=data, teacher=teacher, cache=cache,
        optimizer=optimizer, runtime=runtime, checkpoint=checkpoint,
        validation=validation, output=output,
    )


def load_sft1_v2_config(path: Path) -> SFT1V2Config:
    return parse_sft1_v2_config(load_yaml_config(path))


__all__ = [
    "APPROVED_COUNTS", "EARLY4_STEPS", "SFT1V2CacheConfig", "SFT1V2CheckpointConfig",
    "SFT1V2Config", "SFT1V2DataConfig", "SFT1V2FreezeConfig",
    "SFT1V2ObjectiveConfig", "SFT1V2OptimizerConfig", "SFT1V2OutputConfig",
    "SFT1V2RuntimeConfig", "SFT1V2SelectionConfig", "SFT1V2SourceConfig", "SFT1V2StateConfig",
    "SFT1V2TeacherConfig", "SFT1V2ValidationConfig", "SFT1_V2_CONFIG_SCHEMA",
    "STATE_INTERFACE_OBJECTIVE_VERSION", "load_sft1_v2_config",
    "parse_sft1_v2_config",
]
