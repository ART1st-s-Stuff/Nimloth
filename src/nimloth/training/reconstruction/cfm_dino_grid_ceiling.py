"""Decoder-only CFM training for direct-DINO grid8/grid16 ceiling cells.

The condition path is restricted to the immutable multigrid cache built from
original archived observations. Importing this module does not authorize an
experiment, GPU, or W&B launch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nimloth.recon.cfm import (
    CFMConfig,
    SpatialConditionedFlowUNet,
    conditional_flow_matching_loss,
)
from nimloth.training.reconstruction.cfm_forensic_query_state import _load_image_uint8
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_SHUFFLE_ALGORITHM,
    make_global_shuffle_mapping,
)
from nimloth.training.reconstruction.dino_grid_ceiling_cache import (
    DINO_GRID_CEILING_CACHE_SCHEMA,
    DINO_GRID_CEILING_FEATURE_IDENTITIES,
    DinoGridCeilingCacheDataset,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_SELECTION_ALL_TRAIN,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION,
)

DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA = (
    "nimloth_dino_grid_reconstruction_ceiling_cfm_checkpoint_v1"
)
DINO_GRID_CEILING_TRAIN_METADATA_SCHEMA = (
    "nimloth_dino_grid_reconstruction_ceiling_train_v1"
)
DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA = (
    "nimloth_dino_grid_reconstruction_ceiling_train_summary_v1"
)
CELLS: dict[str, dict[str, Any]] = {
    "spatial_dino8": {
        "decoder_family": "spatial_grid_v1",
        "condition_family": "direct_dino",
        "grid_size": 8,
    },
    "spatial_dino16": {
        "decoder_family": "spatial_grid_v1",
        "condition_family": "direct_dino",
        "grid_size": 16,
    },
}
IMAGE_SIZE = 128
TOKEN_DIM = 1024
BASE_CHANNELS = 64
CONDITION_DIM = 256
TIME_DIM = 512
TRAIN_ITEMS = 12_836
EXTERNAL_ITEMS = 1_413
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
FINAL_STEP = 4_000
EVAL_INTERVAL = 1_000
SAVE_INTERVAL = 1_000
TRAIN_SEED = 20260921
NOISE_SEEDS = (20260931, 20260932, 20260933)
FIXED_TIMES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95)
SAMPLE_ROWS = 256
SAMPLE_ODE_STEPS = 50
SAMPLE_SELECTION_ALGORITHM = "sha256_oracle_ladder_v1"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class LoadedDinoGridCeilingSplit:
    """One direct-DINO view paired with exact Stage B RGB rows."""

    conditions: torch.Tensor
    images_uint8: torch.Tensor
    rows: tuple[dict[str, Any], ...]
    grid_size: int
    cache_schema: str
    cache_fingerprint: str
    feature_identity: str
    embedded_state_cache_fingerprint: str
    source_grid4_cache_fingerprint: str
    selection_identity: str
    split_name: str
    row_set_identity: str
    image_preprocessing: Mapping[str, Any]

    def __len__(self) -> int:
        return int(self.conditions.shape[0])


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


def _is_git_commit(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and not (set(value) - _HEX)


def _current_source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[4],
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("DINO-grid trainer cannot authenticate source commit") from error
    commit = result.stdout.strip()
    if not _is_git_commit(commit):
        raise RuntimeError("DINO-grid trainer source commit is invalid")
    return commit


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _reject_protected_output(output: Path, protected: Sequence[Path]) -> None:
    if not output.is_absolute() or _has_symlink_component(output):
        raise ValueError("DINO-grid output must be an absolute non-symlink path")
    candidate = output.parent.resolve() / output.name
    for supplied in protected:
        if not supplied.is_absolute() or _has_symlink_component(supplied):
            raise ValueError("DINO-grid input must be absolute and non-symlinked")
        root = supplied.resolve()
        if candidate == root or root in candidate.parents:
            raise ValueError("DINO-grid output must not be inside an immutable input")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"DINO-grid checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"DINO-grid evidence already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _validate_split(split: LoadedDinoGridCeilingSplit, *, role: str) -> None:
    expected_count = TRAIN_ITEMS if role == FORENSIC_SELECTION_ALL_TRAIN else EXTERNAL_ITEMS
    tokens = split.grid_size**2 if isinstance(split, LoadedDinoGridCeilingSplit) else -1
    if (
        not isinstance(split, LoadedDinoGridCeilingSplit)
        or split.split_name != role
        or split.grid_size not in {8, 16}
        or split.cache_schema != DINO_GRID_CEILING_CACHE_SCHEMA
        or len(split) != expected_count
        or tuple(split.conditions.shape) != (expected_count, tokens, TOKEN_DIM)
        or split.conditions.dtype != torch.float32
        or not split.conditions.is_contiguous()
        or not bool(torch.isfinite(split.conditions).all())
        or tuple(split.images_uint8.shape)
        != (expected_count, 3, IMAGE_SIZE, IMAGE_SIZE)
        or split.images_uint8.dtype != torch.uint8
        or len(split.rows) != expected_count
        or not all(
            _is_sha256(value)
            for value in (
                split.cache_fingerprint,
                split.feature_identity,
                split.embedded_state_cache_fingerprint,
                split.source_grid4_cache_fingerprint,
                split.selection_identity,
                split.row_set_identity,
            )
        )
        or split.image_preprocessing
        != {
            "size": IMAGE_SIZE,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        }
    ):
        raise ValueError("DINO-grid split shape/cache/grid/role contract is invalid")
    identities: set[str] = set()
    for row in split.rows:
        if (
            not isinstance(row, Mapping)
            or row.get("selection_role") != role
            or not isinstance(row.get("row_identity"), str)
            or not row["row_identity"]
            or not _is_sha256(row.get("original_image_sha256"))
            or row["row_identity"] in identities
        ):
            raise ValueError("DINO-grid split row identity is invalid")
        identities.add(str(row["row_identity"]))


def validate_dino_grid_ceiling_split_pair(
    train: LoadedDinoGridCeilingSplit,
    validation: LoadedDinoGridCeilingSplit,
) -> None:
    _validate_split(train, role=FORENSIC_SELECTION_ALL_TRAIN)
    _validate_split(validation, role=FORENSIC_SELECTION_EXTERNAL_VALIDATION)
    if (
        train.grid_size != validation.grid_size
        or train.feature_identity != validation.feature_identity
        or train.embedded_state_cache_fingerprint
        != validation.embedded_state_cache_fingerprint
        or train.source_grid4_cache_fingerprint
        != validation.source_grid4_cache_fingerprint
        or train.selection_identity != validation.selection_identity
        or train.row_set_identity == validation.row_set_identity
        or train.image_preprocessing != validation.image_preprocessing
    ):
        raise ValueError("DINO-grid train/external matched split identity is invalid")
    train_rows = {str(row["row_identity"]) for row in train.rows}
    validation_rows = {str(row["row_identity"]) for row in validation.rows}
    train_images = {str(row["original_image_sha256"]) for row in train.rows}
    validation_images = {str(row["original_image_sha256"]) for row in validation.rows}
    if train_rows & validation_rows or train_images & validation_images:
        raise ValueError("DINO-grid train/external row or image overlap is forbidden")


def _split_identity(
    *, cache_fingerprint: str, selection_identity: str, role: str,
    rows: Sequence[Mapping[str, Any]],
) -> str:
    return _identity(
        {
            "cache_fingerprint": cache_fingerprint,
            "selection_identity": selection_identity,
            "role": role,
            "rows": [
                {
                    "selection_ordinal": row["selection_ordinal"],
                    "row_identity": row["row_identity"],
                    "original_image_sha256": row["original_image_sha256"],
                }
                for row in rows
            ],
        }
    )


def load_dino_grid_ceiling_splits(
    *, multigrid_cache: str | Path, grid_size: int, image_size: int,
) -> tuple[LoadedDinoGridCeilingSplit, LoadedDinoGridCeilingSplit]:
    """Load one direct view without opening or returning any SFT1 state tensor."""

    if grid_size not in {8, 16} or image_size != IMAGE_SIZE:
        raise ValueError("DINO-grid split requires explicit grid8/grid16 and RGB128")
    multigrid_root = Path(multigrid_cache)
    if (
        not multigrid_root.is_absolute()
        or multigrid_root.is_symlink()
        or not multigrid_root.is_dir()
    ):
        raise ValueError("DINO-grid cache input must be an absolute regular directory")
    dataset = DinoGridCeilingCacheDataset(multigrid_root, grid_size=grid_size)
    manifest = dataset.manifest
    source_grid4 = manifest["source_grid4_cache"]
    grouped: dict[str, list[dict[str, Any]]] = {
        FORENSIC_SELECTION_ALL_TRAIN: [],
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: [],
    }
    for index in range(len(dataset)):
        item = dataset[index]
        role = item.get("selection_role")
        if role not in grouped:
            raise ValueError("DINO-grid cache contains a cross-stage role")
        grouped[str(role)].append(item)
    if {role: len(rows) for role, rows in grouped.items()} != {
        FORENSIC_SELECTION_ALL_TRAIN: TRAIN_ITEMS,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: EXTERNAL_ITEMS,
    }:
        raise ValueError("DINO-grid cache Stage B role counts are not exact")
    preprocessing = {
        "size": IMAGE_SIZE,
        "resample": "bicubic",
        "range": [-1, 1],
        "color_space": "sRGB",
    }
    result: list[LoadedDinoGridCeilingSplit] = []
    for role in (FORENSIC_SELECTION_ALL_TRAIN, FORENSIC_SELECTION_EXTERNAL_VALIDATION):
        items = grouped[role]
        conditions = torch.stack([item.pop("condition") for item in items]).contiguous()
        rows = tuple(dict(item) for item in items)
        images = torch.stack(
            [_load_image_uint8(str(row["original_image_path"]), IMAGE_SIZE) for row in rows]
        )
        result.append(
            LoadedDinoGridCeilingSplit(
                conditions=conditions,
                images_uint8=images,
                rows=rows,
                grid_size=grid_size,
                cache_schema=DINO_GRID_CEILING_CACHE_SCHEMA,
                cache_fingerprint=dataset.cache_fingerprint,
                feature_identity=dataset.feature_identity,
                embedded_state_cache_fingerprint=str(
                    source_grid4["embedded_state_cache_fingerprint"]
                ),
                source_grid4_cache_fingerprint=str(source_grid4["cache_fingerprint"]),
                selection_identity=str(manifest["selection"]["identity"]),
                split_name=role,
                row_set_identity=_split_identity(
                    cache_fingerprint=dataset.cache_fingerprint,
                    selection_identity=str(manifest["selection"]["identity"]),
                    role=role,
                    rows=rows,
                ),
                image_preprocessing=preprocessing,
            )
        )
    validate_dino_grid_ceiling_split_pair(result[0], result[1])
    return result[0], result[1]


def _flatten_condition(condition: torch.Tensor, *, grid_size: int | None = None) -> torch.Tensor:
    if condition.ndim != 3 or condition.shape[1] not in {16, 64, 256}:
        raise ValueError("DINO-grid condition must be [B,16|64|256,token_dim]")
    inferred = math.isqrt(int(condition.shape[1]))
    if (
        inferred**2 != condition.shape[1]
        or (grid_size is not None and inferred != grid_size)
        or condition.shape[2] != TOKEN_DIM
        or not condition.is_floating_point()
        or not bool(torch.isfinite(condition).all())
    ):
        raise ValueError("DINO-grid condition shape/grid/dtype is invalid")
    return condition.reshape(condition.shape[0], -1)


def build_dino_grid_ceiling_model(config: CFMConfig) -> SpatialConditionedFlowUNet:
    _validate_model_config(config)
    return SpatialConditionedFlowUNet(config)


def build_decoder_optimizer(
    model: nn.Module, *, learning_rate: float, weight_decay: float,
) -> torch.optim.AdamW:
    if not isinstance(model, SpatialConditionedFlowUNet):
        raise TypeError("DINO-grid optimizer requires the spatial decoder family")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("DINO-grid learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("DINO-grid weight_decay must be finite and non-negative")
    parameters = tuple(model.parameters())
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("every DINO-grid decoder parameter must be trainable")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _require_decoder_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_parameters = tuple(model.parameters())
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    if (
        len(model_parameters) != len(optimizer_parameters)
        or {id(value) for value in model_parameters}
        != {id(value) for value in optimizer_parameters}
    ):
        raise ValueError("DINO-grid optimizer/checkpoint must be decoder-only")


def _validate_optimizer_invariants(
    optimizer: torch.optim.Optimizer,
    invariants: Mapping[str, Any],
) -> None:
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 1:
        raise ValueError("DINO-grid checkpoint requires exactly one AdamW param group")
    group = optimizer.param_groups[0]
    if (
        float(group.get("lr", math.nan)) != float(invariants["learning_rate"])
        or float(group.get("weight_decay", math.nan))
        != float(invariants["weight_decay"])
        or tuple(group.get("betas", ())) != (0.9, 0.999)
        or float(group.get("eps", math.nan)) != 1e-8
        or group.get("amsgrad") is not False
        or group.get("maximize") is not False
        or group.get("foreach") is not None
        or group.get("capturable") is not False
        or group.get("differentiable") is not False
        or group.get("fused") is not None
        or group.get("decoupled_weight_decay") is not True
    ):
        raise ValueError("DINO-grid AdamW checkpoint hyperparameters mismatch invariants")


def _validate_model_config(config: CFMConfig, *, grid_size: int | None = None) -> None:
    expected_grid = math.isqrt(config.token_count) if isinstance(config, CFMConfig) else -1
    if (
        not isinstance(config, CFMConfig)
        or expected_grid not in {8, 16}
        or expected_grid**2 != config.token_count
        or (grid_size is not None and expected_grid != grid_size)
        or config.image_size != IMAGE_SIZE
        or config.token_dim != TOKEN_DIM
        or config.base_channels != BASE_CHANNELS
        or config.condition_dim != CONDITION_DIM
        or config.time_dim != TIME_DIM
        or config.input_channels != 3
        or config.output_channels != 3
    ):
        raise ValueError("DINO-grid CFM architecture/grid contract is not exact")


def _decoder_parameter_count(config: CFMConfig) -> int:
    rng = torch.get_rng_state()
    try:
        return sum(parameter.numel() for parameter in build_dino_grid_ceiling_model(config).parameters())
    finally:
        torch.set_rng_state(rng)


def build_dino_grid_ceiling_invariants(
    *, cell: str, config: CFMConfig, train: LoadedDinoGridCeilingSplit,
    validation: LoadedDinoGridCeilingSplit, batch_size: int,
    learning_rate: float, weight_decay: float, gradient_clip: float,
    final_step: int, evaluation_interval: int, save_interval: int, seed: int,
    noise_seeds: Sequence[int], fixed_times: Sequence[float], sample_rows: int,
    sample_ode_steps: int, source_commit: str, multigrid_cache_path: str,
    output_dir: str, wandb_project: str,
    wandb_mode: str, wandb_run_id: str, wandb_run_name: str,
) -> dict[str, Any]:
    owner = CELLS.get(cell)
    if owner is None:
        raise ValueError("DINO-grid training cell must be spatial_dino8 or spatial_dino16")
    grid_size = int(owner["grid_size"])
    _validate_model_config(config, grid_size=grid_size)
    validate_dino_grid_ceiling_split_pair(train, validation)
    if train.grid_size != grid_size:
        raise ValueError("DINO-grid cell and direct condition view mismatch")
    if not (
        batch_size == BATCH_SIZE
        and learning_rate == LEARNING_RATE
        and weight_decay == WEIGHT_DECAY
        and gradient_clip == GRADIENT_CLIP
        and final_step == FINAL_STEP
        and evaluation_interval == EVAL_INTERVAL
        and save_interval == SAVE_INTERVAL
        and seed == TRAIN_SEED
        and tuple(noise_seeds) == NOISE_SEEDS
        and tuple(float(value) for value in fixed_times) == FIXED_TIMES
        and sample_rows == SAMPLE_ROWS
        and sample_ode_steps == SAMPLE_ODE_STEPS
        and _is_git_commit(source_commit)
        and all(
            Path(value).is_absolute()
            for value in (multigrid_cache_path, output_dir)
        )
        and wandb_project == "nimloth-recon"
        and wandb_mode == "online"
        and bool(wandb_run_id)
        and bool(wandb_run_name)
    ):
        raise ValueError("DINO-grid matched training/evaluation contract drift")
    return {
        "schema": "nimloth_dino_grid_reconstruction_ceiling_invariants_v1",
        "cell": cell,
        "decoder_family": "spatial_grid_v1",
        "condition_family": "direct_dino",
        "condition_owner": "direct_original_observation_dino",
        "trainable_owner": "decoder_only",
        "grid_size": grid_size,
        "token_count": grid_size**2,
        "source_commit": source_commit,
        "multigrid_cache_path": multigrid_cache_path,
        "output_dir": output_dir,
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "forensic_only": True,
        "deployable": False,
        "sft1_quality_established": False,
        "sft2_ready": False,
        "cfm_config": config.to_metadata(),
        "decoder_parameter_count": _decoder_parameter_count(config),
        "cache_schema": train.cache_schema,
        "train_cache_fingerprint": train.cache_fingerprint,
        "validation_cache_fingerprint": validation.cache_fingerprint,
        "feature_identity": train.feature_identity,
        "train_row_set_identity": train.row_set_identity,
        "validation_row_set_identity": validation.row_set_identity,
        "embedded_state_cache_fingerprint": train.embedded_state_cache_fingerprint,
        "source_grid4_cache_fingerprint": train.source_grid4_cache_fingerprint,
        "selection_identity": train.selection_identity,
        "train_items": len(train),
        "validation_items": len(validation),
        "image_preprocessing": dict(train.image_preprocessing),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "final_step": final_step,
        "evaluation_interval": evaluation_interval,
        "save_interval": save_interval,
        "seed": seed,
        "sampler": "torch_randint_global_rng_v1",
        "scheduler": "none",
        "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
        "noise_seeds": list(noise_seeds),
        "fixed_times": [float(value) for value in fixed_times],
        "sample_rows": sample_rows,
        "sample_ode_steps": sample_ode_steps,
        "sample_selection_algorithm": SAMPLE_SELECTION_ALGORITHM,
        "checkpoint_selection": "final_step4000_only_no_best_selection",
        "cross_grid_resume_allowed": False,
        "additional_steps_or_fallback_allowed": False,
    }


def validate_dino_grid_ceiling_invariants(
    invariants: Mapping[str, Any], config: CFMConfig,
) -> Mapping[str, Any]:
    cell = invariants.get("cell") if isinstance(invariants, Mapping) else None
    owner = CELLS.get(str(cell))
    required = {
        "schema", "cell", "decoder_family", "condition_family", "condition_owner",
        "trainable_owner", "grid_size", "token_count", "source_commit",
        "multigrid_cache_path", "output_dir", "wandb_project",
        "wandb_mode", "wandb_run_id", "wandb_run_name", "forensic_only",
        "deployable", "sft1_quality_established", "sft2_ready", "cfm_config",
        "decoder_parameter_count", "cache_schema", "train_cache_fingerprint",
        "validation_cache_fingerprint", "feature_identity", "train_row_set_identity",
        "validation_row_set_identity", "embedded_state_cache_fingerprint",
        "source_grid4_cache_fingerprint", "selection_identity", "train_items",
        "validation_items", "image_preprocessing", "batch_size", "learning_rate",
        "weight_decay", "gradient_clip", "final_step", "evaluation_interval",
        "save_interval", "seed", "sampler", "scheduler", "shuffle_algorithm", "noise_seeds",
        "fixed_times", "sample_rows", "sample_ode_steps",
        "sample_selection_algorithm", "checkpoint_selection",
        "cross_grid_resume_allowed", "additional_steps_or_fallback_allowed",
    }
    if (
        not isinstance(invariants, Mapping)
        or set(invariants) != required
        or owner is None
        or invariants.get("schema")
        != "nimloth_dino_grid_reconstruction_ceiling_invariants_v1"
        or invariants.get("decoder_family") != "spatial_grid_v1"
        or invariants.get("condition_family") != "direct_dino"
        or invariants.get("condition_owner") != "direct_original_observation_dino"
        or invariants.get("trainable_owner") != "decoder_only"
        or invariants.get("grid_size") != owner["grid_size"]
        or invariants.get("token_count") != int(owner["grid_size"]) ** 2
        or not _is_git_commit(invariants.get("source_commit"))
        or any(
            not isinstance(invariants.get(field), str)
            or not Path(str(invariants[field])).is_absolute()
            for field in ("multigrid_cache_path", "output_dir")
        )
        or invariants.get("wandb_project") != "nimloth-recon"
        or invariants.get("wandb_mode") != "online"
        or not invariants.get("wandb_run_id")
        or not invariants.get("wandb_run_name")
        or invariants.get("forensic_only") is not True
        or invariants.get("deployable") is not False
        or invariants.get("sft1_quality_established") is not False
        or invariants.get("sft2_ready") is not False
        or invariants.get("cfm_config") != config.to_metadata()
        or invariants.get("cache_schema") != DINO_GRID_CEILING_CACHE_SCHEMA
        or invariants.get("train_cache_fingerprint")
        != invariants.get("validation_cache_fingerprint")
        or invariants.get("feature_identity")
        != DINO_GRID_CEILING_FEATURE_IDENTITIES.get(int(owner["grid_size"]))
        or invariants.get("image_preprocessing")
        != {
            "size": IMAGE_SIZE,
            "resample": "bicubic",
            "range": [-1, 1],
            "color_space": "sRGB",
        }
        or invariants.get("train_items") != TRAIN_ITEMS
        or invariants.get("validation_items") != EXTERNAL_ITEMS
        or invariants.get("batch_size") != BATCH_SIZE
        or invariants.get("learning_rate") != LEARNING_RATE
        or invariants.get("weight_decay") != WEIGHT_DECAY
        or invariants.get("gradient_clip") != GRADIENT_CLIP
        or invariants.get("final_step") != FINAL_STEP
        or invariants.get("evaluation_interval") != EVAL_INTERVAL
        or invariants.get("save_interval") != SAVE_INTERVAL
        or invariants.get("seed") != TRAIN_SEED
        or invariants.get("sampler") != "torch_randint_global_rng_v1"
        or invariants.get("scheduler") != "none"
        or invariants.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
        or invariants.get("noise_seeds") != list(NOISE_SEEDS)
        or invariants.get("fixed_times") != list(FIXED_TIMES)
        or invariants.get("sample_rows") != SAMPLE_ROWS
        or invariants.get("sample_ode_steps") != SAMPLE_ODE_STEPS
        or invariants.get("sample_selection_algorithm") != SAMPLE_SELECTION_ALGORITHM
        or invariants.get("checkpoint_selection")
        != "final_step4000_only_no_best_selection"
        or invariants.get("cross_grid_resume_allowed") is not False
        or invariants.get("additional_steps_or_fallback_allowed") is not False
        or not all(
            _is_sha256(invariants.get(field))
            for field in (
                "train_cache_fingerprint", "validation_cache_fingerprint",
                "feature_identity", "train_row_set_identity",
                "validation_row_set_identity", "embedded_state_cache_fingerprint",
                "source_grid4_cache_fingerprint", "selection_identity",
            )
        )
    ):
        raise ValueError("DINO-grid cell/grid/checkpoint invariants are invalid")
    _validate_model_config(config, grid_size=int(owner["grid_size"]))
    if invariants.get("decoder_parameter_count") != _decoder_parameter_count(config):
        raise ValueError("DINO-grid decoder parameter-count invariant mismatch")
    return invariants


def save_dino_grid_ceiling_checkpoint(
    path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
    step: int, invariants: Mapping[str, Any],
) -> None:
    validate_dino_grid_ceiling_invariants(invariants, model.config)
    if getattr(model, "decoder_family", None) != "spatial_grid_v1":
        raise ValueError("DINO-grid checkpoint decoder family mismatch")
    checkpoint_path = Path(path)
    if (
        not checkpoint_path.is_absolute()
        or _has_symlink_component(checkpoint_path)
        or checkpoint_path.resolve().parent != Path(invariants["output_dir"]).resolve()
    ):
        raise ValueError("DINO-grid checkpoint output path identity mismatch")
    _require_decoder_optimizer(model, optimizer)
    _validate_optimizer_invariants(optimizer, invariants)
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 < step <= int(invariants["final_step"])
        or (
            step != int(invariants["final_step"])
            and step % int(invariants["save_interval"]) != 0
        )
    ):
        raise ValueError("DINO-grid checkpoint step is invalid")
    _atomic_torch_save(
        {
            "schema": DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "invariants": dict(invariants),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        checkpoint_path,
    )


def load_dino_grid_ceiling_checkpoint(
    path: str | Path, *, model: nn.Module, optimizer: torch.optim.Optimizer,
    expected_invariants: Mapping[str, Any], device: torch.device,
) -> int:
    validate_dino_grid_ceiling_invariants(expected_invariants, model.config)
    checkpoint_path = Path(path)
    if (
        not checkpoint_path.is_absolute()
        or _has_symlink_component(checkpoint_path)
        or not checkpoint_path.is_file()
        or checkpoint_path.resolve().parent
        != Path(expected_invariants["output_dir"]).resolve()
    ):
        raise ValueError("DINO-grid resume checkpoint/output identity mismatch")
    _require_decoder_optimizer(model, optimizer)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as error:
        raise ValueError("DINO-grid checkpoint is unreadable") from error
    required = {
        "schema", "model", "optimizer", "step", "invariants",
        "torch_rng_state", "cuda_rng_state_all",
    }
    step = payload.get("step") if isinstance(payload, Mapping) else None
    cpu_rng = payload.get("torch_rng_state") if isinstance(payload, Mapping) else None
    cuda_rng = payload.get("cuda_rng_state_all") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA
        or payload.get("invariants") != dict(expected_invariants)
        or isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 < step <= int(expected_invariants["final_step"])
        or (
            step != int(expected_invariants["final_step"])
            and step % int(expected_invariants["save_interval"]) != 0
        )
        or not isinstance(cpu_rng, torch.Tensor)
        or cpu_rng.ndim != 1
        or cpu_rng.dtype != torch.uint8
        or (
            cuda_rng is not None
            and (
                not isinstance(cuda_rng, list)
                or any(
                    not isinstance(state, torch.Tensor)
                    or state.ndim != 1
                    or state.dtype != torch.uint8
                    for state in cuda_rng
                )
            )
        )
    ):
        raise ValueError("DINO-grid cell/grid/invariant resume mismatch")
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    except Exception as error:
        raise ValueError("DINO-grid decoder/optimizer resume state is invalid") from error
    _require_decoder_optimizer(model, optimizer)
    _validate_optimizer_invariants(optimizer, expected_invariants)
    torch.set_rng_state(cpu_rng.cpu())
    if torch.cuda.is_available() and cuda_rng is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    return step


@torch.inference_mode()
def evaluate_random_time_sensitivity(
    model: nn.Module, split: LoadedDinoGridCeilingSplit, device: torch.device,
    *, batch_size: int, seeds: Sequence[int],
) -> dict[str, Any]:
    _validate_split(split, role=FORENSIC_SELECTION_EXTERNAL_VALIDATION)
    per_seed: list[dict[str, Any]] = []
    was_training = model.training
    model.eval()
    try:
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            mapping = make_global_shuffle_mapping(item_count=len(split), seed=seed)
            correct_sum = shuffled_sum = 0.0
            count = 0
            for start in range(0, len(split), batch_size):
                stop = min(start + batch_size, len(split))
                target = split.images_uint8[start:stop].to(device).float().div(127.5).sub(1)
                noise = torch.randn(target.shape, device=device, generator=generator)
                flow_time = torch.rand((stop - start,), device=device, generator=generator)
                interpolated = (
                    (1 - flow_time[:, None, None, None]) * noise
                    + flow_time[:, None, None, None] * target
                )
                velocity = target - noise
                correct = _flatten_condition(
                    split.conditions[start:stop].to(device), grid_size=split.grid_size
                )
                shuffled = _flatten_condition(
                    split.conditions[mapping[start:stop]].to(device), grid_size=split.grid_size
                )
                correct_loss = torch.nn.functional.mse_loss(
                    model(interpolated, flow_time, correct), velocity, reduction="none"
                ).flatten(1).mean(1)
                shuffled_loss = torch.nn.functional.mse_loss(
                    model(interpolated, flow_time, shuffled), velocity, reduction="none"
                ).flatten(1).mean(1)
                correct_sum += float(correct_loss.sum().cpu())
                shuffled_sum += float(shuffled_loss.sum().cpu())
                count += stop - start
            correct_mean = correct_sum / count
            shuffled_mean = shuffled_sum / count
            per_seed.append(
                {
                    "seed": seed,
                    "correct_flow_mse": correct_mean,
                    "shuffled_flow_mse": shuffled_mean,
                    "shuffled_minus_correct": shuffled_mean - correct_mean,
                    "num_items": count,
                }
            )
    finally:
        model.train(was_training)
    report: dict[str, Any] = {
        "schema": "nimloth_dino_grid_ceiling_random_time_v1",
        "seeds": list(seeds),
        "per_seed": per_seed,
    }
    report["identity"] = _identity(report)
    return report


def _training_config(grid_size: int) -> CFMConfig:
    return CFMConfig(
        image_size=IMAGE_SIZE,
        token_count=grid_size**2,
        token_dim=TOKEN_DIM,
        base_channels=BASE_CHANNELS,
        condition_dim=CONDITION_DIM,
        time_dim=TIME_DIM,
        input_channels=3,
        output_channels=3,
    )


def _sample_selection_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"oracle-ladder-v1:{row['row_identity']}".encode()
        ).hexdigest(),
    )[:SAMPLE_ROWS]
    if len(selected) != SAMPLE_ROWS:
        raise ValueError("DINO-grid sample plan lacks 256 external rows")
    return _identity(
        {
            "algorithm": SAMPLE_SELECTION_ALGORITHM,
            "row_identities": [row["row_identity"] for row in selected],
            "image_sha256": [row["original_image_sha256"] for row in selected],
        }
    )


def _select_auto_resume_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("--resume requested but no committed checkpoint exists")
    if any(_has_symlink_component(path) for path in checkpoints):
        raise ValueError("DINO-grid automatic resume rejects checkpoint symlinks")
    return checkpoints[-1]


def _validate_completed_training_output(root: Path) -> Mapping[str, Any]:
    metadata_path = root / "metadata.json"
    summary_path = root / "summary.json"
    if any(not path.is_file() or path.is_symlink() for path in (metadata_path, summary_path)):
        raise ValueError("DINO-grid completed output metadata/summary is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = Path(str(summary.get("final_checkpoint", "")))
    if (
        summary.get("schema") != DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA
        or summary.get("status") != "completed_final_step4000"
        or summary.get("final_step") != FINAL_STEP
        or summary.get("additional_steps_or_fallback_allowed") is not False
        or summary.get("identity")
        != _identity({key: value for key, value in summary.items() if key != "identity"})
        or metadata.get("schema") != DINO_GRID_CEILING_TRAIN_METADATA_SCHEMA
        or metadata.get("invariants") is None
        or summary.get("metadata_identity") != _identity(metadata)
        or checkpoint.parent.resolve() != root.resolve()
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or summary.get("final_checkpoint_sha256") != _sha256_file(checkpoint)
    ):
        raise ValueError("DINO-grid completed output identity is invalid")
    return summary


def train_dino_grid_ceiling_cfm(args: argparse.Namespace) -> int:
    """Train one exact grid8/grid16 cell to controlling final step 4000."""

    owner = CELLS.get(args.cell)
    if owner is None:
        raise ValueError("DINO-grid trainer accepts only grid8/grid16 cells")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _reject_protected_output(args.output_dir, (args.multigrid_cache,))
    if (
        args.wandb_project != "nimloth-recon"
        or args.wandb_mode != "online"
        or not args.wandb_run_id
        or not args.wandb_run_name
    ):
        raise ValueError("DINO-grid training requires exact online W&B identity")
    source_commit = _current_source_commit()
    resuming = args.resume or args.resume_checkpoint is not None
    if args.output_dir.exists() and not resuming:
        raise FileExistsError("DINO-grid output exists; exact resume is required")
    if resuming and not args.output_dir.is_dir():
        raise FileNotFoundError("DINO-grid resume requires its existing output")
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError("use either --resume or --resume-checkpoint")
    if (
        args.resume_checkpoint is not None
        and args.resume_checkpoint.resolve().parent != args.output_dir.resolve()
    ):
        raise ValueError("DINO-grid resume checkpoint must belong to the same output")

    torch.manual_seed(TRAIN_SEED)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    grid_size = int(owner["grid_size"])
    train, validation = load_dino_grid_ceiling_splits(
        multigrid_cache=args.multigrid_cache,
        grid_size=grid_size,
        image_size=IMAGE_SIZE,
    )
    config = _training_config(grid_size)
    model = build_dino_grid_ceiling_model(config).to(args.device)
    optimizer = build_decoder_optimizer(
        model, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    invariants = build_dino_grid_ceiling_invariants(
        cell=args.cell,
        config=config,
        train=train,
        validation=validation,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        gradient_clip=GRADIENT_CLIP,
        final_step=FINAL_STEP,
        evaluation_interval=EVAL_INTERVAL,
        save_interval=SAVE_INTERVAL,
        seed=TRAIN_SEED,
        noise_seeds=NOISE_SEEDS,
        fixed_times=FIXED_TIMES,
        sample_rows=SAMPLE_ROWS,
        sample_ode_steps=SAMPLE_ODE_STEPS,
        source_commit=source_commit,
        multigrid_cache_path=str(args.multigrid_cache.resolve()),
        output_dir=str(args.output_dir.resolve()),
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_run_id=args.wandb_run_id,
        wandb_run_name=args.wandb_run_name,
    )
    producer: dict[str, Any] = {
        "module": "nimloth.training.reconstruction.cfm_dino_grid_ceiling",
        "source_commit": source_commit,
        "resolved_config": {
            "cell": args.cell,
            "multigrid_cache": str(args.multigrid_cache.resolve()),
            "output": str(args.output_dir.resolve()),
            "device": str(args.device),
            "wandb_project": args.wandb_project,
            "wandb_mode": args.wandb_mode,
            "wandb_run_id": args.wandb_run_id,
            "wandb_run_name": args.wandb_run_name,
            "invariants_identity": _identity(invariants),
        },
    }
    producer["identity"] = _identity(producer)
    metadata = {
        "schema": DINO_GRID_CEILING_TRAIN_METADATA_SCHEMA,
        "cell": args.cell,
        "invariants": invariants,
        "producer": producer,
        "tracking": {
            "enabled": True,
            "project": args.wandb_project,
            "mode": args.wandb_mode,
            "run_id": args.wandb_run_id,
            "run_name": args.wandb_run_name,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("DINO-grid resume metadata/W&B identity mismatch")
    elif resuming:
        raise ValueError("DINO-grid resume requires exact committed metadata")
    else:
        _atomic_json(metadata, metadata_path)

    resume_path = args.resume_checkpoint
    if args.resume:
        resume_path = _select_auto_resume_checkpoint(args.output_dir)
    start_step = 0
    if resume_path is not None:
        start_step = load_dino_grid_ceiling_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_invariants=invariants,
            device=args.device,
        )
    if start_step >= FINAL_STEP:
        raise ValueError("final step4000 checkpoint cannot resume further training")

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            config=metadata,
            dir=str(args.output_dir),
            resume="must" if resume_path is not None else "never",
        )
    except Exception as error:
        raise RuntimeError("DINO-grid W&B initialization failed") from error
    finally:
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if (
        wandb_run.id != args.wandb_run_id
        or wandb_run.name != args.wandb_run_name
        or wandb_run.project != args.wandb_project
    ):
        wandb_run.finish()
        raise RuntimeError("DINO-grid W&B returned a different run identity")

    log_path = args.output_dir / "train_step_log.csv"
    if not log_path.exists():
        with log_path.open("x", newline="") as stream:
            csv.writer(stream).writerow(["time", "step", "train_flow_mse", "external_report_identity"])
    last_report: Mapping[str, Any] | None = None
    try:
        for step in range(start_step + 1, FINAL_STEP + 1):
            indices = torch.randint(len(train), (BATCH_SIZE,))
            condition = train.conditions[indices].to(args.device)
            target = train.images_uint8[indices].to(args.device).float().div(127.5).sub(1)
            model.train()
            loss = conditional_flow_matching_loss(
                model, target, _flatten_condition(condition, grid_size=grid_size)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if step % EVAL_INTERVAL == 0 or step == FINAL_STEP:
                last_report = evaluate_random_time_sensitivity(
                    model,
                    validation,
                    args.device,
                    batch_size=BATCH_SIZE,
                    seeds=NOISE_SEEDS,
                )
                with log_path.open("a", newline="") as stream:
                    csv.writer(stream).writerow(
                        [time.time(), step, float(loss.detach().cpu()), last_report["identity"]]
                    )
                wandb_run.log(
                    {
                        "cfm/train_flow_mse": float(loss.detach().cpu()),
                        "cfm/external_correct_flow_mse": sum(
                            item["correct_flow_mse"] for item in last_report["per_seed"]
                        ) / len(NOISE_SEEDS),
                        "cfm/external_shuffled_flow_mse": sum(
                            item["shuffled_flow_mse"] for item in last_report["per_seed"]
                        ) / len(NOISE_SEEDS),
                    },
                    step=step,
                )
            if step % SAVE_INTERVAL == 0 and step != FINAL_STEP:
                save_dino_grid_ceiling_checkpoint(
                    args.output_dir / f"checkpoint_{step:09d}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    invariants=invariants,
                )
        final_path = args.output_dir / f"checkpoint_{FINAL_STEP:09d}.pt"
        save_dino_grid_ceiling_checkpoint(
            final_path,
            model=model,
            optimizer=optimizer,
            step=FINAL_STEP,
            invariants=invariants,
        )
        core = {
            "schema": "nimloth_dino_grid_ceiling_train_core_completion_v1",
            "status": "core_completed_final_step4000",
            "cell": args.cell,
            "final_checkpoint": str(final_path.resolve()),
            "final_checkpoint_sha256": _sha256_file(final_path),
            "invariants_identity": _identity(invariants),
        }
        core["identity"] = _identity(core)
        _atomic_json(core, args.output_dir / "core_completion.json")
        summary = {
            "schema": DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA,
            "status": "completed_final_step4000",
            "cell": args.cell,
            "final_step": FINAL_STEP,
            "final_checkpoint": str(final_path.resolve()),
            "final_checkpoint_sha256": _sha256_file(final_path),
            "random_time_report": last_report,
            "sample_selection_identity": _sample_selection_identity(validation.rows),
            "invariants_identity": _identity(invariants),
            "metadata_identity": _identity(metadata),
            "producer_identity": producer["identity"],
            "evaluation_pending": True,
            "additional_steps_or_fallback_allowed": False,
        }
        summary["identity"] = _identity(summary)
        _atomic_json(summary, args.output_dir / "summary.json")
        try:
            _validate_completed_training_output(args.output_dir)
        except Exception as error:
            failure = {
                "schema": "nimloth_dino_grid_ceiling_post_validation_failure_v1",
                "status": "core_completed_post_validation_failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "core_completion_identity": core["identity"],
            }
            failure["identity"] = _identity(failure)
            _atomic_json(failure, args.output_dir / "post_validation_failure.json")
            raise
    finally:
        wandb_run.finish()
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one direct-original-observation-DINO grid8/grid16 CFM decoder"
    )
    parser.add_argument("--cell", required=True, choices=tuple(CELLS))
    parser.add_argument("--multigrid-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=torch.device)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-mode", required=True, choices=("online",))
    parser.add_argument("--wandb-run-id", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return train_dino_grid_ceiling_cfm(build_cli_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELLS",
    "DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA",
    "DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA",
    "LoadedDinoGridCeilingSplit",
    "build_cli_parser",
    "build_decoder_optimizer",
    "build_dino_grid_ceiling_invariants",
    "build_dino_grid_ceiling_model",
    "evaluate_random_time_sensitivity",
    "load_dino_grid_ceiling_checkpoint",
    "load_dino_grid_ceiling_splits",
    "save_dino_grid_ceiling_checkpoint",
    "train_dino_grid_ceiling_cfm",
    "validate_dino_grid_ceiling_invariants",
    "validate_dino_grid_ceiling_split_pair",
]
