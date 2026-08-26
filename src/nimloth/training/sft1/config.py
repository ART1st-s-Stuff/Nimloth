"""Legacy SFT1 defaults and the strict state-interface-v2 canary schema."""

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


# Kept solely for the historical CLI adapter below.
_SFT1_YAML_TO_ARG: dict[tuple[str, str], str] = {
    ("data", "train_jsonl"): "train_jsonl",
    ("data", "val_jsonl"): "val_jsonl",
    ("latent", "token_count"): "latent_token_count",
    ("latent", "query_mode"): "latent_query_mode",
    ("latent", "mask_query_labels"): "mask_latent_query_labels",
    ("tuning", "lora"): "lora",
    ("tuning", "lora_r"): "lora_r",
    ("tuning", "lora_alpha"): "lora_alpha",
    ("train", "epochs"): "epochs",
    ("train", "batch_size"): "batch_size",
    ("train", "grad_accum"): "grad_accum",
    ("train", "lr"): "lr",
    ("train", "embedding_lr"): "embedding_lr",
    ("train", "max_length"): "max_length",
    ("train", "max_pixels"): "max_pixels",
}


def sft1_yaml_defaults(path: Path) -> dict[str, Any]:
    """Read historical permissive CLI defaults (not valid for v2)."""

    cfg = load_yaml_config(path)
    defaults: dict[str, Any] = {}
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            dest = _SFT1_YAML_TO_ARG.get((section, key))
            if dest is not None and value is not None:
                defaults[dest] = value
    return defaults


SFT1_V2_CONFIG_SCHEMA = "nimloth_sft1_state_v2_config_v1"
STATE_INTERFACE_OBJECTIVE_VERSION = "nimloth_state_interface_v2_canary"


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
    freeze_vision_tower: bool
    freeze_dino_teacher: bool
    freeze_instruction_action_teacher: bool
    train_query_adapter: bool
    train_fresh_projector: bool
    train_readouts: bool


@dataclass(frozen=True)
class SFT1V2OptimizerConfig:
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class SFT1V2Config:
    schema: str
    state: SFT1V2StateConfig
    objective: SFT1V2ObjectiveConfig
    freeze: SFT1V2FreezeConfig
    optimizer: SFT1V2OptimizerConfig

    @property
    def identity(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _strict_section(
    raw: Mapping[str, Any],
    name: str,
    fields: set[str],
) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"SFT1-v2 config section {name!r} is required")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"unknown SFT1-v2 config field: {name}.{unknown[0]}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"missing SFT1-v2 config field: {name}.{missing[0]}")
    if any(value[field] is None for field in fields):
        field = next(field for field in sorted(fields) if value[field] is None)
        raise ValueError(f"SFT1-v2 config field may not be null: {name}.{field}")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    if (positive and result <= 0.0) or (not positive and result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def parse_sft1_v2_config(raw: Mapping[str, Any]) -> SFT1V2Config:
    """Parse the fail-closed v2 canary schema with no inferred semantics."""

    if not isinstance(raw, Mapping):
        raise ValueError("SFT1-v2 config must be a mapping")
    top_fields = {"schema", "state", "objective", "freeze", "optimizer"}
    unknown = sorted(set(raw) - top_fields)
    if unknown:
        raise ValueError(f"unknown SFT1-v2 config section: {unknown[0]}")
    missing = sorted(top_fields - set(raw))
    if missing:
        raise ValueError(f"missing SFT1-v2 config section: {missing[0]}")
    schema = raw["schema"]
    if schema != SFT1_V2_CONFIG_SCHEMA:
        raise ValueError(
            f"unsupported SFT1-v2 config schema {schema!r}; legacy state config cannot resume"
        )

    state_raw = _strict_section(
        raw,
        "state",
        {
            "objective_version",
            "latent_query_mode",
            "query_tune",
            "grid_tokens",
            "qwen_hidden_dim",
            "state_dim",
            "projector_hidden_dim",
            "instruction_teacher_dim",
            "action_dim",
            "movement_action_indices",
        },
    )
    movement_raw = state_raw["movement_action_indices"]
    if not isinstance(movement_raw, (list, tuple)):
        raise ValueError("state.movement_action_indices must be a sequence")
    state = SFT1V2StateConfig(
        objective_version=str(state_raw["objective_version"]),
        latent_query_mode=str(state_raw["latent_query_mode"]),
        query_tune=str(state_raw["query_tune"]),
        grid_tokens=_integer(state_raw["grid_tokens"], "state.grid_tokens"),
        qwen_hidden_dim=_integer(
            state_raw["qwen_hidden_dim"], "state.qwen_hidden_dim"
        ),
        state_dim=_integer(state_raw["state_dim"], "state.state_dim"),
        projector_hidden_dim=_integer(
            state_raw["projector_hidden_dim"], "state.projector_hidden_dim"
        ),
        instruction_teacher_dim=_integer(
            state_raw["instruction_teacher_dim"],
            "state.instruction_teacher_dim",
        ),
        action_dim=_integer(state_raw["action_dim"], "state.action_dim"),
        movement_action_indices=tuple(movement_raw),  # type: ignore[arg-type]
    )
    if state.objective_version != STATE_INTERFACE_OBJECTIVE_VERSION:
        raise ValueError("unsupported or legacy state-interface objective version")
    if state.latent_query_mode != "inject" or state.query_tune != "adapter":
        raise ValueError("v2 canary requires injected queries with additive adapter tuning")
    if (state.grid_tokens, state.state_dim, state.action_dim) != (16, 1024, 8):
        raise ValueError("v2 deployed contract is exactly K16 x 1024 with eight actions")
    if (state.qwen_hidden_dim, state.instruction_teacher_dim) != (2048, 2048):
        raise ValueError(
            "v2 ID176 interface requires 2048-dimensional Qwen hidden and "
            "exact-instruction teacher targets"
        )
    movement = state.movement_action_indices
    if movement != OBSERVED_MOVEMENT_ACTION_INDICES:
        raise ValueError(
            "state.movement_action_indices must equal the identity-bound "
            f"move_forward/move_right/move_left mapping {OBSERVED_MOVEMENT_ACTION_INDICES}"
        )

    objective_raw = _strict_section(
        raw,
        "objective",
        {
            "visual_weight",
            "visual_relation_coefficient",
            "instruction_weight",
            "instruction_contrastive_coefficient",
            "observed_feasibility_weight",
            "actor_preservation_weight",
            "state_policy_weight",
            "policy_temperature",
            "contrastive_temperature",
        },
    )
    objective = SFT1V2ObjectiveConfig(
        weights=SFT1V2LossWeights(
            visual=_number(objective_raw["visual_weight"], "objective.visual_weight"),
            visual_relation_coefficient=_number(
                objective_raw["visual_relation_coefficient"],
                "objective.visual_relation_coefficient",
            ),
            instruction=_number(
                objective_raw["instruction_weight"], "objective.instruction_weight"
            ),
            instruction_contrastive_coefficient=_number(
                objective_raw["instruction_contrastive_coefficient"],
                "objective.instruction_contrastive_coefficient",
            ),
            observed_feasibility=_number(
                objective_raw["observed_feasibility_weight"],
                "objective.observed_feasibility_weight",
            ),
            actor_preservation=_number(
                objective_raw["actor_preservation_weight"],
                "objective.actor_preservation_weight",
            ),
            state_policy=_number(
                objective_raw["state_policy_weight"],
                "objective.state_policy_weight",
            ),
        ),
        policy_temperature=_number(
            objective_raw["policy_temperature"],
            "objective.policy_temperature",
            positive=True,
        ),
        contrastive_temperature=_number(
            objective_raw["contrastive_temperature"],
            "objective.contrastive_temperature",
            positive=True,
        ),
    )

    freeze_raw = _strict_section(
        raw,
        "freeze",
        {
            "freeze_qwen_language_body",
            "freeze_vision_tower",
            "freeze_dino_teacher",
            "freeze_instruction_action_teacher",
            "train_query_adapter",
            "train_fresh_projector",
            "train_readouts",
        },
    )
    freeze = SFT1V2FreezeConfig(
        **{
            field: _boolean(freeze_raw[field], f"freeze.{field}")
            for field in freeze_raw
        }
    )
    if not all(
        (
            freeze.freeze_qwen_language_body,
            freeze.freeze_vision_tower,
            freeze.freeze_dino_teacher,
            freeze.freeze_instruction_action_teacher,
            freeze.train_query_adapter,
            freeze.train_fresh_projector,
            freeze.train_readouts,
        )
    ):
        raise ValueError(
            "v2 canary freeze contract permits only query adapter, fresh projector, and readouts"
        )

    optimizer_raw = _strict_section(
        raw, "optimizer", {"learning_rate", "weight_decay"}
    )
    optimizer = SFT1V2OptimizerConfig(
        learning_rate=_number(
            optimizer_raw["learning_rate"], "optimizer.learning_rate", positive=True
        ),
        weight_decay=_number(
            optimizer_raw["weight_decay"], "optimizer.weight_decay"
        ),
    )
    return SFT1V2Config(
        schema=str(schema),
        state=state,
        objective=objective,
        freeze=freeze,
        optimizer=optimizer,
    )


def load_sft1_v2_config(path: Path) -> SFT1V2Config:
    return parse_sft1_v2_config(load_yaml_config(path))


__all__ = [
    "SFT1V2Config",
    "SFT1V2FreezeConfig",
    "SFT1V2ObjectiveConfig",
    "SFT1V2OptimizerConfig",
    "SFT1V2StateConfig",
    "SFT1_V2_CONFIG_SCHEMA",
    "STATE_INTERFACE_OBJECTIVE_VERSION",
    "load_sft1_v2_config",
    "parse_sft1_v2_config",
    "sft1_yaml_defaults",
]
