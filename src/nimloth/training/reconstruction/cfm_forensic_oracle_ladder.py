"""Matched decoder-only CFM training for the Formal38 oracle ladder.

This owner trains exactly one of the three preregistered fresh cells.  The
immutable ``token_state`` baseline is evaluated separately and is never resumed
or rewritten here.  Implementing this module does not authorize a GPU launch.
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
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
)
from nimloth.training.reconstruction.cfm_forensic_query_state import (
    load_forensic_image_splits,
)
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_SHUFFLE_ALGORITHM,
    evaluate_query_state_multi_noise_sensitivity,
    make_global_shuffle_mapping,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_SELECTION_ALL_TRAIN,
    FORENSIC_SELECTION_EXTERNAL_VALIDATION,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    FORENSIC_DINO_ORACLE_CACHE_SCHEMA,
    ForensicDinoOracleCacheDataset,
)

ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA = (
    "nimloth_query_state_forensic_oracle_ladder_cfm_checkpoint_v1"
)
ORACLE_LADDER_METADATA_SCHEMA = (
    "nimloth_query_state_forensic_oracle_ladder_train_v1"
)
ORACLE_LADDER_SUMMARY_SCHEMA = (
    "nimloth_query_state_forensic_oracle_ladder_train_summary_v1"
)
CELLS: dict[str, dict[str, Any]] = {
    "token_state": {
        "decoder_family": "token_set_v1",
        "condition_family": "state",
        "train": False,
    },
    "token_oracle": {
        "decoder_family": "token_set_v1",
        "condition_family": "dino_oracle",
        "train": True,
    },
    "spatial_state": {
        "decoder_family": "spatial_grid_v1",
        "condition_family": "state",
        "train": True,
    },
    "spatial_oracle": {
        "decoder_family": "spatial_grid_v1",
        "condition_family": "dino_oracle",
        "train": True,
    },
}
TRAINABLE_CELLS = tuple(cell for cell, owner in CELLS.items() if owner["train"])
IMAGE_SIZE = 128
TOKEN_COUNT = 16
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
class LoadedOracleLadderSplit:
    """One condition family paired with the exact Stage B RGB rows."""

    conditions: torch.Tensor
    images_uint8: torch.Tensor
    rows: tuple[dict[str, Any], ...]
    condition_family: str
    cache_schema: str
    cache_fingerprint: str
    source_state_cache_fingerprint: str
    selection_identity: str
    split_name: str
    row_set_identity: str
    image_preprocessing: Mapping[str, Any]

    def __len__(self) -> int:
        return int(self.conditions.shape[0])


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
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
        raise RuntimeError("oracle-ladder cannot authenticate implementation commit") from error
    commit = result.stdout.strip()
    if not _is_git_commit(commit):
        raise RuntimeError("oracle-ladder implementation commit is invalid")
    return commit


def _has_symlink_component(path: Path) -> bool:
    absolute = path.absolute()
    return any(
        candidate.is_symlink()
        for candidate in (absolute, *absolute.parents)
    )


def _reject_protected_output(output: Path, protected: Sequence[Path]) -> None:
    if not output.is_absolute() or _has_symlink_component(output):
        raise ValueError("oracle-ladder output must be an absolute non-symlink path")
    candidate = output.parent.resolve() / output.name
    for raw_root in protected:
        if not raw_root.is_absolute() or _has_symlink_component(raw_root):
            raise ValueError("oracle-ladder immutable input must be absolute and non-symlinked")
        root = raw_root.resolve()
        if candidate == root or root in candidate.parents:
            raise ValueError("oracle-ladder output must not be inside an immutable input root")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _random_time_provenance(
    *,
    item_count: int,
    image_shape: Sequence[int],
    batch_size: int,
    seeds: Sequence[int],
    device: torch.device,
) -> Mapping[str, Any]:
    per_seed: list[Mapping[str, Any]] = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise_digest = hashlib.sha256()
        time_digest = hashlib.sha256()
        for start in range(0, item_count, batch_size):
            count = min(batch_size, item_count - start)
            noise = torch.randn(
                (count, *image_shape),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            flow_time = torch.rand(
                (count,),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
            time_digest.update(flow_time.detach().cpu().contiguous().numpy().tobytes())
        mapping = make_global_shuffle_mapping(item_count=item_count, seed=seed)
        item = {
            "seed": seed,
            "noise_sha256": noise_digest.hexdigest(),
            "time_sha256": time_digest.hexdigest(),
            "shuffle_indices_sha256": _tensor_sha256(mapping),
            "generator": f"torch_generator_{device.type}_manual_seed_v1",
        }
        per_seed.append({**item, "identity": _identity(item)})
    result: dict[str, Any] = {
        "item_count": item_count,
        "image_shape": list(image_shape),
        "batch_size": batch_size,
        "per_seed": per_seed,
    }
    result["identity"] = _identity(result)
    return result


def _flatten_condition(condition: torch.Tensor) -> torch.Tensor:
    if (
        condition.ndim != 3
        or tuple(condition.shape[1:]) != (TOKEN_COUNT, TOKEN_DIM)
        or not condition.is_floating_point()
        or not bool(torch.isfinite(condition).all())
    ):
        raise ValueError("oracle-ladder condition must be finite [B,K16,token_dim]")
    return condition.reshape(condition.shape[0], TOKEN_COUNT * TOKEN_DIM)


def _decoder_parameter_count(decoder_family: str, config: CFMConfig) -> int:
    rng = torch.get_rng_state()
    try:
        model = build_oracle_ladder_model(
            decoder_family=decoder_family,
            config=config,
        )
        return sum(parameter.numel() for parameter in model.parameters())
    finally:
        torch.set_rng_state(rng)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"oracle-ladder checkpoint already exists: {path}")
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


def _validate_split(split: LoadedOracleLadderSplit, *, role: str) -> None:
    expected_count = TRAIN_ITEMS if role == FORENSIC_SELECTION_ALL_TRAIN else EXTERNAL_ITEMS
    expected_schema = (
        FORENSIC_QUERY_STATE_CACHE_SCHEMA
        if split.condition_family == "state"
        else FORENSIC_DINO_ORACLE_CACHE_SCHEMA
    )
    if (
        not isinstance(split, LoadedOracleLadderSplit)
        or split.split_name != role
        or split.condition_family not in {"state", "dino_oracle"}
        or split.cache_schema != expected_schema
        or len(split) != expected_count
        or split.conditions.shape != (expected_count, TOKEN_COUNT, TOKEN_DIM)
        or split.conditions.dtype != torch.float32
        or not split.conditions.is_contiguous()
        or not bool(torch.isfinite(split.conditions).all())
        or split.images_uint8.shape
        != (expected_count, 3, IMAGE_SIZE, IMAGE_SIZE)
        or split.images_uint8.dtype != torch.uint8
        or len(split.rows) != expected_count
        or not all(
            _is_sha256(value)
            for value in (
                split.cache_fingerprint,
                split.source_state_cache_fingerprint,
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
        raise ValueError("oracle-ladder split shape/cache/role contract is invalid")
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
            raise ValueError("oracle-ladder split row identity is invalid")
        identities.add(str(row["row_identity"]))


def validate_oracle_ladder_split_pair(
    train: LoadedOracleLadderSplit,
    validation: LoadedOracleLadderSplit,
) -> None:
    _validate_split(train, role=FORENSIC_SELECTION_ALL_TRAIN)
    _validate_split(validation, role=FORENSIC_SELECTION_EXTERNAL_VALIDATION)
    if (
        train.condition_family != validation.condition_family
        or train.source_state_cache_fingerprint
        != validation.source_state_cache_fingerprint
        or train.selection_identity != validation.selection_identity
        or train.row_set_identity == validation.row_set_identity
        or train.image_preprocessing != validation.image_preprocessing
    ):
        raise ValueError("oracle-ladder train/external matched condition identity is invalid")
    train_rows = {str(row["row_identity"]) for row in train.rows}
    external_rows = {str(row["row_identity"]) for row in validation.rows}
    train_images = {str(row["original_image_sha256"]) for row in train.rows}
    external_images = {str(row["original_image_sha256"]) for row in validation.rows}
    if train_rows & external_rows or train_images & external_images:
        raise ValueError("oracle-ladder train/external row or image overlap is forbidden")


def _split_identity(
    *,
    cache_fingerprint: str,
    selection_identity: str,
    role: str,
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


def _oracle_rows_by_role(
    oracle_cache: str | Path,
) -> tuple[ForensicDinoOracleCacheDataset, dict[str, list[dict[str, Any]]]]:
    dataset = ForensicDinoOracleCacheDataset(oracle_cache)
    grouped: dict[str, list[dict[str, Any]]] = {
        FORENSIC_SELECTION_ALL_TRAIN: [],
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: [],
    }
    for index in range(len(dataset)):
        item = dataset[index]
        role = str(item.get("selection_role"))
        if role not in grouped:
            raise ValueError("oracle cache contains a cross-stage role")
        grouped[role].append(item)
    if {role: len(rows) for role, rows in grouped.items()} != {
        FORENSIC_SELECTION_ALL_TRAIN: TRAIN_ITEMS,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: EXTERNAL_ITEMS,
    }:
        raise ValueError("oracle cache Stage B role counts are not exact")
    return dataset, grouped


def load_oracle_ladder_splits(
    *,
    state_cache: str | Path,
    oracle_cache: str | Path | None,
    condition_family: str,
    image_size: int,
) -> tuple[LoadedOracleLadderSplit, LoadedOracleLadderSplit]:
    """Load one matched condition family with the same source RGB rows."""

    if image_size != IMAGE_SIZE or condition_family not in {"state", "dino_oracle"}:
        raise ValueError("oracle-ladder condition/image contract is not exact")
    state_train, state_external, _source = load_forensic_image_splits(
        state_cache,
        image_size=image_size,
        experiment_stage="stage_b_diagnostic",
    )
    state_manifest = json.loads(
        (Path(state_cache) / "manifest.json").read_text(encoding="utf-8")
    )
    selection_identity = str(state_manifest["selection"]["identity"])
    if condition_family == "state":
        values = {
            FORENSIC_SELECTION_ALL_TRAIN: state_train.states,
            FORENSIC_SELECTION_EXTERNAL_VALIDATION: state_external.states,
        }
        cache_schema = FORENSIC_QUERY_STATE_CACHE_SCHEMA
        cache_fingerprint = state_train.cache_fingerprint
    else:
        if oracle_cache is None:
            raise ValueError("dino_oracle condition requires its exact typed cache")
        oracle_dataset, grouped = _oracle_rows_by_role(oracle_cache)
        oracle_manifest = oracle_dataset.manifest
        if (
            oracle_manifest["source_state_cache"]["cache_fingerprint"]
            != state_train.cache_fingerprint
            or oracle_manifest["selection"]["identity"] != selection_identity
        ):
            raise ValueError("oracle/state cache source or selection identity mismatch")
        values = {
            role: torch.stack([item["condition"] for item in rows]).contiguous()
            for role, rows in grouped.items()
        }
        cache_schema = FORENSIC_DINO_ORACLE_CACHE_SCHEMA
        cache_fingerprint = oracle_dataset.cache_fingerprint

    state_splits = {
        FORENSIC_SELECTION_ALL_TRAIN: state_train,
        FORENSIC_SELECTION_EXTERNAL_VALIDATION: state_external,
    }
    result: list[LoadedOracleLadderSplit] = []
    for role in (FORENSIC_SELECTION_ALL_TRAIN, FORENSIC_SELECTION_EXTERNAL_VALIDATION):
        state_split = state_splits[role]
        rows = tuple(dict(row) for row in state_split.rows)
        if condition_family == "dino_oracle":
            oracle_rows = grouped[role]
            for state_row, oracle_row in zip(rows, oracle_rows, strict=True):
                comparable = {key: value for key, value in oracle_row.items() if key != "condition"}
                if comparable != state_row:
                    raise ValueError("oracle/state cache ordered row identity mismatch")
        result.append(
            LoadedOracleLadderSplit(
                conditions=values[role].detach().cpu().float().contiguous(),
                images_uint8=state_split.images_uint8,
                rows=rows,
                condition_family=condition_family,
                cache_schema=cache_schema,
                cache_fingerprint=cache_fingerprint,
                source_state_cache_fingerprint=state_train.cache_fingerprint,
                selection_identity=selection_identity,
                split_name=role,
                row_set_identity=_split_identity(
                    cache_fingerprint=cache_fingerprint,
                    selection_identity=selection_identity,
                    role=role,
                    rows=rows,
                ),
                image_preprocessing=state_split.image_preprocessing,
            )
        )
    validate_oracle_ladder_split_pair(result[0], result[1])
    return result[0], result[1]


def build_oracle_ladder_model(
    *,
    decoder_family: str,
    config: CFMConfig,
) -> TokenConditionedFlowUNet | SpatialConditionedFlowUNet:
    if decoder_family == "token_set_v1":
        return TokenConditionedFlowUNet(config)
    if decoder_family == "spatial_grid_v1":
        return SpatialConditionedFlowUNet(config)
    raise ValueError(f"unsupported oracle-ladder decoder family: {decoder_family}")


def build_decoder_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    if not isinstance(model, (TokenConditionedFlowUNet, SpatialConditionedFlowUNet)):
        raise TypeError("oracle-ladder optimizer requires a registered decoder family")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("oracle-ladder learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("oracle-ladder weight_decay must be finite and non-negative")
    parameters = tuple(model.parameters())
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("every oracle-ladder decoder parameter must be trainable")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _require_decoder_optimizer(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    model_parameters = tuple(model.parameters())
    optimizer_parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", ())
    )
    if (
        len(model_parameters) != len(optimizer_parameters)
        or {id(parameter) for parameter in model_parameters}
        != {id(parameter) for parameter in optimizer_parameters}
    ):
        raise ValueError("oracle-ladder optimizer/checkpoint must be decoder-only")


def _validate_model_config(config: CFMConfig) -> None:
    if (
        not isinstance(config, CFMConfig)
        or config.image_size != IMAGE_SIZE
        or config.token_count != TOKEN_COUNT
        or config.token_dim != TOKEN_DIM
        or config.base_channels != BASE_CHANNELS
        or config.condition_dim != CONDITION_DIM
        or config.time_dim != TIME_DIM
        or config.input_channels != 3
        or config.output_channels != 3
    ):
        raise ValueError("oracle-ladder CFM architecture contract is not exact")


def build_oracle_ladder_invariants(
    *,
    cell: str,
    config: CFMConfig,
    train: LoadedOracleLadderSplit,
    validation: LoadedOracleLadderSplit,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip: float,
    final_step: int,
    evaluation_interval: int,
    save_interval: int,
    seed: int,
    noise_seeds: Sequence[int],
    fixed_times: Sequence[float],
    sample_rows: int,
    sample_ode_steps: int,
    sample_selection_identity: str,
    source_commit: str,
    state_cache_path: str,
    oracle_cache_path: str,
    output_dir: str,
    wandb_project: str,
    wandb_mode: str,
    wandb_run_id: str,
    wandb_run_name: str,
) -> dict[str, Any]:
    if cell not in TRAINABLE_CELLS:
        raise ValueError("oracle-ladder training cell must be one of three fresh cells")
    owner = CELLS[cell]
    _validate_model_config(config)
    validate_oracle_ladder_split_pair(train, validation)
    if (
        train.condition_family != owner["condition_family"]
        or validation.condition_family != owner["condition_family"]
    ):
        raise ValueError("oracle-ladder cell/condition family mismatch")
    expected = (
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
        and _is_sha256(sample_selection_identity)
        and _is_git_commit(source_commit)
        and Path(state_cache_path).is_absolute()
        and Path(oracle_cache_path).is_absolute()
        and Path(output_dir).is_absolute()
        and wandb_project == "nimloth-recon"
        and wandb_mode == "online"
        and bool(wandb_run_id)
        and bool(wandb_run_name)
    )
    if not expected:
        raise ValueError("oracle-ladder matched training/evaluation contract drift")
    parameter_count = _decoder_parameter_count(
        str(owner["decoder_family"]), config
    )
    return {
        "schema": "nimloth_query_state_forensic_oracle_ladder_invariants_v1",
        "cell": cell,
        "decoder_family": owner["decoder_family"],
        "condition_family": owner["condition_family"],
        "trainable_owner": "decoder_only",
        "source_commit": source_commit,
        "state_cache_path": state_cache_path,
        "oracle_cache_path": oracle_cache_path,
        "output_dir": output_dir,
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "forensic_only": True,
        "unsafe_actor_checkpoint": True,
        "deployable": False,
        "sft2_ready": False,
        "cfm_config": config.to_metadata(),
        "decoder_parameter_count": parameter_count,
        "train_cache_schema": train.cache_schema,
        "train_cache_fingerprint": train.cache_fingerprint,
        "train_row_set_identity": train.row_set_identity,
        "validation_cache_schema": validation.cache_schema,
        "validation_cache_fingerprint": validation.cache_fingerprint,
        "validation_row_set_identity": validation.row_set_identity,
        "source_state_cache_fingerprint": train.source_state_cache_fingerprint,
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
        "sample_selection_identity": sample_selection_identity,
        "checkpoint_selection": "final_step4000_only_no_best_selection",
        "additional_steps_or_fallback_allowed": False,
    }


_INVARIANT_FIELDS = {
    "schema",
    "cell",
    "decoder_family",
    "condition_family",
    "trainable_owner",
    "source_commit",
    "state_cache_path",
    "oracle_cache_path",
    "output_dir",
    "wandb_project",
    "wandb_mode",
    "wandb_run_id",
    "wandb_run_name",
    "forensic_only",
    "unsafe_actor_checkpoint",
    "deployable",
    "sft2_ready",
    "cfm_config",
    "decoder_parameter_count",
    "train_cache_schema",
    "train_cache_fingerprint",
    "train_row_set_identity",
    "validation_cache_schema",
    "validation_cache_fingerprint",
    "validation_row_set_identity",
    "source_state_cache_fingerprint",
    "selection_identity",
    "train_items",
    "validation_items",
    "image_preprocessing",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "final_step",
    "evaluation_interval",
    "save_interval",
    "seed",
    "sampler",
    "scheduler",
    "shuffle_algorithm",
    "noise_seeds",
    "fixed_times",
    "sample_rows",
    "sample_ode_steps",
    "sample_selection_algorithm",
    "sample_selection_identity",
    "checkpoint_selection",
    "additional_steps_or_fallback_allowed",
}


def validate_oracle_ladder_invariants(
    invariants: Mapping[str, Any],
    config: CFMConfig,
) -> Mapping[str, Any]:
    cell = invariants.get("cell") if isinstance(invariants, Mapping) else None
    owner = CELLS.get(str(cell))
    if (
        not isinstance(invariants, Mapping)
        or set(invariants) != _INVARIANT_FIELDS
        or owner is None
        or owner["train"] is not True
        or invariants.get("schema")
        != "nimloth_query_state_forensic_oracle_ladder_invariants_v1"
        or invariants.get("decoder_family") != owner["decoder_family"]
        or invariants.get("condition_family") != owner["condition_family"]
        or invariants.get("trainable_owner") != "decoder_only"
        or not _is_git_commit(invariants.get("source_commit"))
        or not isinstance(invariants.get("state_cache_path"), str)
        or not Path(invariants["state_cache_path"]).is_absolute()
        or not isinstance(invariants.get("oracle_cache_path"), str)
        or not Path(invariants["oracle_cache_path"]).is_absolute()
        or not isinstance(invariants.get("output_dir"), str)
        or not Path(invariants["output_dir"]).is_absolute()
        or invariants.get("wandb_project") != "nimloth-recon"
        or invariants.get("wandb_mode") != "online"
        or not isinstance(invariants.get("wandb_run_id"), str)
        or not invariants["wandb_run_id"]
        or not isinstance(invariants.get("wandb_run_name"), str)
        or not invariants["wandb_run_name"]
        or invariants.get("forensic_only") is not True
        or invariants.get("unsafe_actor_checkpoint") is not True
        or invariants.get("deployable") is not False
        or invariants.get("sft2_ready") is not False
        or invariants.get("cfm_config") != config.to_metadata()
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
        or invariants.get("sample_selection_algorithm")
        != SAMPLE_SELECTION_ALGORITHM
        or invariants.get("checkpoint_selection")
        != "final_step4000_only_no_best_selection"
        or invariants.get("additional_steps_or_fallback_allowed") is not False
        or not all(
            _is_sha256(invariants.get(field))
            for field in (
                "train_cache_fingerprint",
                "train_row_set_identity",
                "validation_cache_fingerprint",
                "validation_row_set_identity",
                "source_state_cache_fingerprint",
                "selection_identity",
                "sample_selection_identity",
            )
        )
    ):
        raise ValueError("oracle-ladder cell/decoder/checkpoint invariants are invalid")
    _validate_model_config(config)
    if invariants.get("decoder_parameter_count") != _decoder_parameter_count(
        str(owner["decoder_family"]), config
    ):
        raise ValueError("oracle-ladder decoder parameter-count invariant mismatch")
    return invariants


def save_oracle_ladder_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    invariants: Mapping[str, Any],
) -> None:
    validate_oracle_ladder_invariants(invariants, model.config)
    if getattr(model, "decoder_family", None) != invariants["decoder_family"]:
        raise ValueError("oracle-ladder checkpoint decoder family mismatch")
    checkpoint_path = Path(path)
    if (
        not checkpoint_path.is_absolute()
        or _has_symlink_component(checkpoint_path)
    ):
        raise ValueError("oracle-ladder checkpoint path must be absolute and non-symlinked")
    if checkpoint_path.resolve().parent != Path(invariants["output_dir"]).resolve():
        raise ValueError("oracle-ladder checkpoint output identity mismatch")
    _require_decoder_optimizer(model, optimizer)
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 < step <= int(invariants["final_step"])
        or (
            step != int(invariants["final_step"])
            and step % int(invariants["save_interval"]) != 0
        )
    ):
        raise ValueError("oracle-ladder checkpoint step is invalid")
    _atomic_torch_save(
        {
            "schema": ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA,
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


def load_oracle_ladder_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_invariants: Mapping[str, Any],
    device: torch.device,
) -> int:
    validate_oracle_ladder_invariants(expected_invariants, model.config)
    if getattr(model, "decoder_family", None) != expected_invariants["decoder_family"]:
        raise ValueError("oracle-ladder resume decoder family mismatch")
    checkpoint_path = Path(path)
    if (
        not checkpoint_path.is_absolute()
        or _has_symlink_component(checkpoint_path)
        or not checkpoint_path.is_file()
    ):
        raise ValueError("oracle-ladder resume checkpoint must be an absolute regular file")
    if checkpoint_path.resolve().parent != Path(
        expected_invariants["output_dir"]
    ).resolve():
        raise ValueError("oracle-ladder resume output/W&B identity mismatch")
    _require_decoder_optimizer(model, optimizer)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as error:
        raise ValueError("oracle-ladder checkpoint is unreadable") from error
    required = {
        "schema",
        "model",
        "optimizer",
        "step",
        "invariants",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != required
        or payload.get("schema") != ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA
        or payload.get("invariants") != dict(expected_invariants)
    ):
        raise ValueError("oracle-ladder cell/decoder/invariants resume mismatch")
    step = payload.get("step")
    cpu_rng = payload.get("torch_rng_state")
    cuda_rng = payload.get("cuda_rng_state_all")
    if (
        isinstance(step, bool)
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
        raise ValueError("oracle-ladder checkpoint step/RNG evidence is invalid")
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    except Exception as error:
        raise ValueError("oracle-ladder decoder/optimizer state is invalid") from error
    _require_decoder_optimizer(model, optimizer)
    torch.set_rng_state(cpu_rng.cpu())
    if torch.cuda.is_available() and cuda_rng is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    return step


@torch.inference_mode()
def evaluate_fixed_time_sensitivity(
    model: nn.Module,
    conditions: torch.Tensor,
    images_uint8: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    seeds: Sequence[int],
    fixed_times: Sequence[float],
) -> dict[str, Any]:
    """Compare correct/shuffled conditions at fixed target-path time values."""

    if (
        conditions.ndim != 3
        or tuple(conditions.shape[1:]) != (TOKEN_COUNT, TOKEN_DIM)
        or conditions.dtype != torch.float32
        or images_uint8.shape
        != (conditions.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE)
        or images_uint8.dtype != torch.uint8
        or conditions.shape[0] < 2
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("fixed-time sensitivity input shape/dtype is invalid")
    seed_values = tuple(seeds)
    time_values = tuple(float(value) for value in fixed_times)
    if (
        len(seed_values) < 1
        or len(set(seed_values)) != len(seed_values)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values)
        or not time_values
        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in time_values)
        or len(set(time_values)) != len(time_values)
    ):
        raise ValueError("fixed-time sensitivity seeds/times are invalid")
    model.eval()
    items: list[dict[str, Any]] = []
    for seed in seed_values:
        mapping = make_global_shuffle_mapping(item_count=len(conditions), seed=seed)
        for fixed_time in time_values:
            generator = torch.Generator(device=device).manual_seed(seed)
            correct_sum = 0.0
            shuffled_sum = 0.0
            noise_digest = hashlib.sha256()
            time_digest = hashlib.sha256()
            count = 0
            for start in range(0, len(conditions), batch_size):
                stop = min(start + batch_size, len(conditions))
                target = images_uint8[start:stop].to(device=device).float().div(127.5).sub(1)
                noise = torch.randn(
                    target.shape,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                noise_digest.update(
                    noise.detach().cpu().contiguous().numpy().tobytes()
                )
                time_value = torch.full(
                    (stop - start,), fixed_time, device=device, dtype=torch.float32
                )
                time_digest.update(
                    time_value.detach().cpu().contiguous().numpy().tobytes()
                )
                interpolated = (1.0 - fixed_time) * noise + fixed_time * target
                velocity = target - noise
                correct_condition = _flatten_condition(
                    conditions[start:stop].to(device=device)
                )
                shuffled_condition = _flatten_condition(
                    conditions[mapping[start:stop]].to(device=device)
                )
                correct = torch.nn.functional.mse_loss(
                    model(interpolated, time_value, correct_condition),
                    velocity,
                    reduction="none",
                ).flatten(1).mean(1)
                shuffled = torch.nn.functional.mse_loss(
                    model(interpolated, time_value, shuffled_condition),
                    velocity,
                    reduction="none",
                ).flatten(1).mean(1)
                correct_sum += float(correct.sum().cpu())
                shuffled_sum += float(shuffled.sum().cpu())
                count += stop - start
            correct_mean = correct_sum / count
            shuffled_mean = shuffled_sum / count
            items.append(
                {
                    "seed": seed,
                    "time": fixed_time,
                    "target_rgb_fraction_in_model_input": fixed_time,
                    "matched_noise": True,
                    "noise_sha256": noise_digest.hexdigest(),
                    "time_sha256": time_digest.hexdigest(),
                    "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
                    "shuffle_identity": _identity(
                        {
                            "seed": seed,
                            "mapping": mapping.tolist(),
                            "algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
                        }
                    ),
                    "num_items": count,
                    "correct_flow_mse": correct_mean,
                    "shuffled_flow_mse": shuffled_mean,
                    "shuffled_minus_correct": shuffled_mean - correct_mean,
                    "shuffled_over_correct": shuffled_mean / max(correct_mean, 1e-12),
                }
            )
    report: dict[str, Any] = {
        "schema": "nimloth_query_state_oracle_ladder_fixed_time_v1",
        "metric_unit": "mean velocity MSE per normalized RGB element",
        "num_items": len(conditions),
        "seeds": list(seed_values),
        "fixed_times": list(time_values),
        "per_seed_time": items,
    }
    report["identity"] = _identity(report)
    return report


def _sample_selection_identity(rows: Sequence[Mapping[str, Any]], *, count: int) -> str:
    if count > len(rows):
        raise ValueError("oracle-ladder sample selection count exceeds rows")
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"oracle-ladder-v1:{row['row_identity']}".encode()
        ).hexdigest(),
    )[:count]
    return _identity(
        {
            "algorithm": SAMPLE_SELECTION_ALGORITHM,
            "row_identities": [row["row_identity"] for row in selected],
            "image_sha256": [row["original_image_sha256"] for row in selected],
        }
    )


def _training_config() -> CFMConfig:
    return CFMConfig(
        image_size=IMAGE_SIZE,
        token_count=TOKEN_COUNT,
        token_dim=TOKEN_DIM,
        base_channels=BASE_CHANNELS,
        condition_dim=CONDITION_DIM,
        time_dim=TIME_DIM,
        input_channels=3,
        output_channels=3,
    )


def _select_auto_resume_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("--resume requested but no committed checkpoint exists")
    if any(_has_symlink_component(path) for path in checkpoints):
        raise ValueError("oracle-ladder automatic resume rejects checkpoint symlinks")
    return checkpoints[-1]


def train_oracle_ladder_cfm(args: argparse.Namespace) -> int:
    """Train exactly one fresh matrix cell under the matched Stage B budget."""

    if args.cell not in TRAINABLE_CELLS:
        raise ValueError("oracle-ladder trainer accepts only the three fresh cells")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _reject_protected_output(
        args.output_dir,
        (args.state_cache, args.oracle_cache),
    )
    if (
        args.wandb_project != "nimloth-recon"
        or args.wandb_mode != "online"
        or not args.wandb_run_id
        or not args.wandb_run_name
    ):
        raise ValueError(
            "oracle-ladder training requires nimloth-recon and exact W&B ID/name"
        )
    source_commit = _current_source_commit()
    resuming = args.resume or args.resume_checkpoint is not None
    if args.output_dir.exists() and not resuming:
        raise FileExistsError("oracle-ladder output exists; explicit exact resume required")
    if resuming and not args.output_dir.is_dir():
        raise FileNotFoundError("oracle-ladder resume requires its existing output directory")
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError("use either --resume or --resume-checkpoint")
    if (
        args.resume_checkpoint is not None
        and args.resume_checkpoint.resolve().parent != args.output_dir.resolve()
    ):
        raise ValueError("oracle-ladder resume checkpoint must belong to the same output")
    torch.manual_seed(TRAIN_SEED)
    if args.device.type == "cuda":
        torch.cuda.manual_seed_all(TRAIN_SEED)
    owner = CELLS[args.cell]
    train, validation = load_oracle_ladder_splits(
        state_cache=args.state_cache,
        oracle_cache=args.oracle_cache,
        condition_family=str(owner["condition_family"]),
        image_size=IMAGE_SIZE,
    )
    config = _training_config()
    model = build_oracle_ladder_model(
        decoder_family=str(owner["decoder_family"]), config=config
    ).to(args.device)
    optimizer = build_decoder_optimizer(
        model,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    sample_identity = _sample_selection_identity(
        validation.rows, count=SAMPLE_ROWS
    )
    invariants = build_oracle_ladder_invariants(
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
        sample_selection_identity=sample_identity,
        source_commit=source_commit,
        state_cache_path=str(args.state_cache.resolve()),
        oracle_cache_path=str(args.oracle_cache.resolve()),
        output_dir=str(args.output_dir.resolve()),
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        wandb_run_id=args.wandb_run_id,
        wandb_run_name=args.wandb_run_name,
    )
    resolved_config = {
        "cell": args.cell,
        "state_cache": str(args.state_cache.resolve()),
        "oracle_cache": str(args.oracle_cache.resolve()),
        "output": str(args.output_dir.resolve()),
        "device": str(args.device),
        "wandb_project": args.wandb_project,
        "wandb_mode": args.wandb_mode,
        "wandb_run_id": args.wandb_run_id,
        "wandb_run_name": args.wandb_run_name,
        "invariants_identity": _identity(invariants),
    }
    producer = {
        "module": (
            "nimloth.training.reconstruction.cfm_forensic_oracle_ladder"
        ),
        "source_commit": source_commit,
        "resolved_config": resolved_config,
    }
    producer["identity"] = _identity(producer)
    metadata = {
        "schema": ORACLE_LADDER_METADATA_SCHEMA,
        "cell": args.cell,
        "invariants": invariants,
        "state_cache": str(args.state_cache.resolve()),
        "oracle_cache": str(args.oracle_cache.resolve()),
        "output": str(args.output_dir.resolve()),
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
            raise ValueError("oracle-ladder resume metadata identity mismatch")
    elif resuming:
        raise ValueError("oracle-ladder resume requires its exact output/W&B metadata")
    else:
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    start_step = 0
    resume_path = args.resume_checkpoint
    if args.resume:
        resume_path = _select_auto_resume_checkpoint(args.output_dir)
    if resume_path is not None:
        start_step = load_oracle_ladder_checkpoint(
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
        raise RuntimeError("oracle-ladder W&B initialization failed") from error
    finally:
        torch.set_rng_state(cpu_rng)
        if torch.cuda.is_available() and cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    if (
        wandb_run.id != args.wandb_run_id
        or wandb_run.name != args.wandb_run_name
        or wandb_run.project != args.wandb_project
    ):
        raise RuntimeError("oracle-ladder W&B returned run identity mismatch")

    log_path = args.output_dir / "train_step_log.csv"
    if not log_path.exists():
        with log_path.open("x", newline="") as stream:
            csv.writer(stream).writerow(
                ["time", "step", "train_flow_mse", "random_time_report_identity"]
            )
    random_time_provenance = _random_time_provenance(
        item_count=len(validation),
        image_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        seeds=NOISE_SEEDS,
        device=args.device,
    )
    last_report: Mapping[str, Any] | None = None
    for step in range(start_step + 1, FINAL_STEP + 1):
        indices = torch.randint(len(train), (BATCH_SIZE,))
        condition = train.conditions[indices].to(args.device)
        target = train.images_uint8[indices].to(args.device).float().div(127.5).sub(1)
        model.train()
        loss = conditional_flow_matching_loss(
            model,
            target,
            _flatten_condition(condition),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if step % EVAL_INTERVAL == 0 or step == FINAL_STEP:
            measured = evaluate_query_state_multi_noise_sensitivity(
                model,
                validation.conditions,
                validation.images_uint8,
                args.device,
                batch_size=BATCH_SIZE,
                seeds=NOISE_SEEDS,
            )
            last_report = {
                **measured,
                "noise_time_provenance": random_time_provenance,
            }
            last_report["owner_identity"] = _identity(last_report)
            with log_path.open("a", newline="") as stream:
                csv.writer(stream).writerow(
                    [
                        time.time(),
                        step,
                        float(loss.detach().cpu()),
                        last_report["owner_identity"],
                    ]
                )
            wandb_run.log(
                {
                    "cfm/train_flow_mse": float(loss.detach().cpu()),
                    "cfm/external_correct_flow_mse": last_report["aggregate"]
                    ["correct_flow_mse"]["mean"],
                    "cfm/external_shuffled_flow_mse": last_report["aggregate"]
                    ["shuffled_flow_mse"]["mean"],
                },
                step=step,
            )
        if step % SAVE_INTERVAL == 0 and step != FINAL_STEP:
            save_oracle_ladder_checkpoint(
                args.output_dir / f"checkpoint_{step:09d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                invariants=invariants,
            )
    final_path = args.output_dir / f"checkpoint_{FINAL_STEP:09d}.pt"
    save_oracle_ladder_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        step=FINAL_STEP,
        invariants=invariants,
    )
    summary = {
        "schema": ORACLE_LADDER_SUMMARY_SCHEMA,
        "cell": args.cell,
        "status": "completed_final_step4000",
        "final_step": FINAL_STEP,
        "final_checkpoint": str(final_path.resolve()),
        "final_checkpoint_sha256": _sha256_file(final_path),
        "random_time_report": last_report,
        "invariants_identity": _identity(invariants),
        "metadata_identity": _identity(metadata),
        "producer_identity": producer["identity"],
        "evaluation_pending": True,
        "additional_steps_or_fallback_allowed": False,
    }
    summary["identity"] = _identity(summary)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    wandb_run.finish()
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one exact Formal38 oracle-ladder CFM cell"
    )
    parser.add_argument("--cell", required=True, choices=TRAINABLE_CELLS)
    parser.add_argument("--state-cache", required=True, type=Path)
    parser.add_argument("--oracle-cache", required=True, type=Path)
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
    return train_oracle_ladder_cfm(build_cli_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CELLS",
    "FIXED_TIMES",
    "NOISE_SEEDS",
    "ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA",
    "ORACLE_LADDER_SUMMARY_SCHEMA",
    "TRAINABLE_CELLS",
    "LoadedOracleLadderSplit",
    "build_cli_parser",
    "build_decoder_optimizer",
    "build_oracle_ladder_invariants",
    "build_oracle_ladder_model",
    "evaluate_fixed_time_sensitivity",
    "load_oracle_ladder_checkpoint",
    "load_oracle_ladder_splits",
    "save_oracle_ladder_checkpoint",
    "train_oracle_ladder_cfm",
    "validate_oracle_ladder_invariants",
]
