"""Train a decoder-only CFM probe from canonical direct Query-State caches.

This module is intentionally separate from the legacy SFT2 reconstruction
entrypoint.  It accepts only schema-v1 Query-State caches, preserves their
``[N,16,1024]`` state axis, and flattens state only at the generic CFM model
boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from nimloth.recon.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
)
from nimloth.training.reconstruction.query_state_cache import (
    QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
    QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
    QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA,
    QueryStateReconstructionCacheDataset,
)

QUERY_STATE_CFM_CHECKPOINT_SCHEMA = "nimloth_query_state_cfm_checkpoint_v1"
QUERY_STATE_CFM_RGB_ARTIFACT_SCHEMA = "nimloth_query_state_cfm_rgb_artifact_v1"
QUERY_STATE_SHUFFLE_ALGORITHM = "global_cyclic_shift_v1"
_QUERY_STATE_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")


class QueryStatePublicationGateFailure(ValueError):
    """Valid recomputed evidence failed a preregistered scientific threshold."""


@dataclass(frozen=True)
class LoadedQueryStateImageSplit:
    """Fully validated split used by the decoder runtime.

    ``states`` deliberately remains three-dimensional.  ``images_uint8`` is
    the corresponding original observation after the declared RGB resize.
    """

    states: torch.Tensor
    images_uint8: torch.Tensor
    rows: tuple[dict[str, Any], ...]
    cache_schema: str
    cache_fingerprint: str
    bundle_fingerprint: str
    source_manifest_identity: str
    template_identity: str
    checkpoint_identity: str
    split_name: str
    split_identity: str
    row_set_identity: str
    image_preprocessing: Mapping[str, Any]

    def __len__(self) -> int:
        return int(self.states.shape[0])


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_image_uint8(path: str | Path, image_size: int) -> torch.Tensor:
    if image_size < 8 or image_size % 8:
        raise ValueError("Query-State CFM image_size must be >= 8 and divisible by 8")
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    try:
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (image_size, image_size),
                resample=resample,
            )
            array = np.asarray(image, dtype=np.uint8).copy()
    except (OSError, ValueError) as error:
        raise ValueError(f"Query-State original observation is unreadable: {path}") from error
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def flatten_query_state_condition(state: torch.Tensor) -> torch.Tensor:
    """Expose the sole ``[B,16,1024] -> [B,16384]`` decoder boundary."""

    if (
        not isinstance(state, torch.Tensor)
        or state.ndim != 3
        or tuple(state.shape[1:]) != _QUERY_STATE_SHAPE
    ):
        shape = tuple(state.shape) if isinstance(state, torch.Tensor) else type(state).__name__
        raise ValueError(
            "canonical Query-State model input must have shape [B,16,1024] "
            f"(K16), got {shape}"
        )
    if not state.is_floating_point() or not torch.isfinite(state).all():
        raise ValueError("canonical Query-State must be finite floating point")
    # reshape remains a view for the cache's validated contiguous K16 tensors.
    flattened = state.reshape(state.shape[0], 16 * 1024)
    if flattened.untyped_storage().data_ptr() != state.untyped_storage().data_ptr():
        raise ValueError("canonical Query-State must be contiguous; flatten may not copy")
    return flattened


def build_query_state_cfm_model(
    *,
    image_size: int,
    base_channels: int,
    condition_dim: int,
    time_dim: int,
) -> TokenConditionedFlowUNet:
    """Build the exact K16 direct-state decoder; no second state encoder exists."""

    config = CFMConfig(
        image_size=image_size,
        token_count=16,
        token_dim=1024,
        base_channels=base_channels,
        condition_dim=condition_dim,
        time_dim=time_dim,
        input_channels=3,
        output_channels=3,
    )
    return TokenConditionedFlowUNet(config)


def load_query_state_image_split(
    cache_dir: str | Path,
    *,
    expected_role: str,
    image_size: int,
    max_items: int = -1,
) -> LoadedQueryStateImageSplit:
    """Strictly preload one Query-State cache and its original observations."""

    root = Path(cache_dir)
    manifest_path = root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Query-State cache manifest is missing or invalid") from error
    if not isinstance(raw, dict) or raw.get("schema") != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA:
        raise ValueError(
            "unsupported legacy cache schema; direct Query-State cache is required"
        )

    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or max_items == 0
        or max_items < -1
    ):
        raise ValueError("max_items must be -1 or a positive integer")
    if expected_role not in {
        QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
        QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
    }:
        raise ValueError(
            "Query-State CFM cache role must be all_train or external_validation"
        )
    dataset = QueryStateReconstructionCacheDataset(root)
    manifest = dataset.manifest
    if manifest.selection["role"] != expected_role:
        raise ValueError(
            f"Query-State CFM cache does not match required {expected_role} selection"
        )
    source_role = (
        "train"
        if expected_role == QUERY_STATE_CACHE_SELECTION_ALL_TRAIN
        else "validation"
    )
    declared_split = manifest.source_jsonl[source_role]["split"]
    other_role = "validation" if source_role == "train" else "train"
    other_split = manifest.source_jsonl[other_role]["split"]
    if declared_split == other_split or manifest.split["name"] != declared_split:
        raise ValueError(
            f"Query-State CFM cache does not match declared {expected_role} split role"
        )
    count = len(dataset) if max_items < 0 else min(max_items, len(dataset))
    if count < 2:
        raise ValueError(
            f"Query-State CFM split requires at least two rows for global shuffle: {count}"
        )

    states: list[torch.Tensor] = []
    images = torch.empty((count, 3, image_size, image_size), dtype=torch.uint8)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        item = dataset[index]
        state = item.pop("state")
        if state.shape != _QUERY_STATE_SHAPE or not state.is_contiguous():
            raise ValueError("Query-State cache reader must preserve contiguous [16,1024]")
        states.append(state.detach().cpu())
        images[index].copy_(_load_image_uint8(item["original_image_path"], image_size))
        rows.append(dict(item))
    stacked = torch.stack(states, dim=0).contiguous()
    if stacked.shape != (count, *_QUERY_STATE_SHAPE):
        raise ValueError("Query-State split preload changed the canonical K16 state axis")
    # The strict dataset has already bound the manifest row-set identity to the
    # ordered per-shard metadata hashes and validates each loaded shard metadata.
    loaded_row_set_identity = _sha256_mapping({"rows": rows})

    preprocessing = {
        "size": image_size,
        "resample": "bicubic",
        "range": [-1, 1],
        "color_space": "sRGB",
    }
    return LoadedQueryStateImageSplit(
        states=stacked,
        images_uint8=images,
        rows=tuple(rows),
        cache_schema=manifest.schema,
        cache_fingerprint=manifest.cache_fingerprint,
        bundle_fingerprint=_sha256_mapping(dict(manifest.bundle)),
        source_manifest_identity=str(manifest.source_jsonl["source_manifest_identity"]),
        template_identity=str(manifest.bundle["template_identity"]),
        checkpoint_identity=str(manifest.bundle["checkpoint_identity"]),
        split_name=expected_role,
        split_identity=str(manifest.split["identity"]),
        row_set_identity=loaded_row_set_identity,
        image_preprocessing=preprocessing,
    )


def validate_query_state_split_pair(
    train: LoadedQueryStateImageSplit,
    validation: LoadedQueryStateImageSplit,
) -> None:
    """Fail closed on role, producer identity, or row/image overlap."""

    if (
        train.split_name != QUERY_STATE_CACHE_SELECTION_ALL_TRAIN
        or validation.split_name != QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION
    ):
        raise ValueError(
            "Query-State CFM requires all_train and external_validation cache selections"
        )
    if train.split_identity == validation.split_identity or train.row_set_identity == validation.row_set_identity:
        raise ValueError("train and validation Query-State split/row-set identities must differ")
    identity_fields = (
        "bundle_fingerprint",
        "source_manifest_identity",
        "template_identity",
        "checkpoint_identity",
    )
    for field in identity_fields:
        left = getattr(train, field)
        right = getattr(validation, field)
        if not _is_sha256(left) or left != right:
            raise ValueError(f"train/validation Query-State {field} mismatch")
    train_rows = {row.get("row_identity") for row in train.rows}
    validation_rows = {row.get("row_identity") for row in validation.rows}
    train_images = {row.get("original_image_sha256") for row in train.rows}
    validation_images = {row.get("original_image_sha256") for row in validation.rows}
    if None in train_rows | validation_rows or None in train_images | validation_images:
        raise ValueError("Query-State split rows require row and original-image identities")
    if train_rows & validation_rows:
        raise ValueError("train/validation Query-State row overlap is forbidden")
    if train_images & validation_images:
        raise ValueError("train/validation original-image overlap is forbidden")
    if train.image_preprocessing != validation.image_preprocessing:
        raise ValueError("train/validation image preprocessing identity mismatch")


def build_decoder_optimizer(
    model: TokenConditionedFlowUNet,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Construct an optimizer whose complete and only owner is the CFM decoder."""

    if not isinstance(model, TokenConditionedFlowUNet):
        raise TypeError("Query-State CFM optimizer requires TokenConditionedFlowUNet")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    parameters = tuple(model.parameters())
    if not parameters or not all(parameter.requires_grad for parameter in parameters):
        raise ValueError("every Query-State CFM decoder parameter must be trainable")
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def _require_decoder_only_optimizer(
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
) -> None:
    model_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    ]
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if (
        optimizer_ids != model_ids
        or len(optimizer_parameters) != len(model_ids)
        or not all(isinstance(parameter, nn.Parameter) for parameter in optimizer_parameters)
    ):
        raise ValueError(
            "Query-State CFM checkpoint optimizer must be decoder-only; "
            "producer/Qwen/projector parameters are forbidden"
        )


def _validate_cfm_config(config: CFMConfig) -> None:
    if (
        not isinstance(config, CFMConfig)
        or config.token_count != 16
        or config.token_dim != 1024
        or config.input_channels != 3
        or config.output_channels != 3
    ):
        raise ValueError("Query-State CFM config must use 16x1024 tokens and RGB I/O")


def build_checkpoint_invariants(
    *,
    config: CFMConfig,
    train_cache_schema: str,
    train_cache_fingerprint: str,
    train_split_identity: str,
    train_row_set_identity: str,
    validation_cache_schema: str,
    validation_cache_fingerprint: str,
    validation_split_identity: str,
    validation_row_set_identity: str,
    bundle_fingerprint: str,
    source_manifest_identity: str,
    template_identity: str,
    checkpoint_identity: str,
    train_items: int,
    validation_items: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    shuffle_algorithm: str,
    validation_noise_seeds: Sequence[int],
    publication_noise_seeds: Sequence[int],
    publication_min_shuffled_minus_correct: float,
    publication_sample_items: int,
    publication_ode_steps: int,
    publication_noise_seed: int,
    publication_sample_batch_size: int,
    image_preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind resume to the complete cache/model/data/optimizer identity."""

    _validate_cfm_config(config)
    if (
        train_cache_schema != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA
        or validation_cache_schema != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA
    ):
        raise ValueError("legacy or cross-schema cache cannot enter Query-State CFM")
    identities = (
        train_cache_fingerprint,
        train_split_identity,
        train_row_set_identity,
        validation_cache_fingerprint,
        validation_split_identity,
        validation_row_set_identity,
        bundle_fingerprint,
        source_manifest_identity,
        template_identity,
        checkpoint_identity,
    )
    if not all(_is_sha256(value) for value in identities):
        raise ValueError("Query-State CFM cache/split fingerprint must be SHA256")
    if train_cache_fingerprint == validation_cache_fingerprint:
        raise ValueError("train and validation Query-State cache fingerprints must differ")
    if train_split_identity == validation_split_identity:
        raise ValueError("train and validation Query-State split identities must differ")
    if train_row_set_identity == validation_row_set_identity:
        raise ValueError("train and validation Query-State row-set identities must differ")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (
        train_items,
        validation_items,
        batch_size,
    )):
        raise ValueError("Query-State CFM item counts and batch_size must be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Query-State CFM seed must be an integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Query-State CFM learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Query-State CFM weight_decay must be finite and non-negative")
    if shuffle_algorithm != QUERY_STATE_SHUFFLE_ALGORITHM:
        raise ValueError("unsupported Query-State global shuffle algorithm")
    seeds = tuple(validation_noise_seeds)
    publication_seeds = tuple(publication_noise_seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seeds
    ):
        raise ValueError("preregistered validation_noise_seeds need at least two unique integers")
    if (
        len(publication_seeds) < 2
        or len(set(publication_seeds)) != len(publication_seeds)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in publication_seeds)
        or not set(seeds).issubset(publication_seeds)
    ):
        raise ValueError("publication_noise_seeds must be unique and include validation seeds")
    if (
        isinstance(publication_min_shuffled_minus_correct, bool)
        or not isinstance(publication_min_shuffled_minus_correct, (int, float))
        or not math.isfinite(float(publication_min_shuffled_minus_correct))
        or float(publication_min_shuffled_minus_correct) <= 0
    ):
        raise ValueError(
            "publication sensitivity threshold must be finite and positive"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (
            publication_sample_items,
            publication_ode_steps,
            publication_sample_batch_size,
        )
    ) or isinstance(publication_noise_seed, bool) or not isinstance(publication_noise_seed, int):
        raise ValueError("Query-State CFM publication sampling invariants are invalid")
    if publication_sample_items > validation_items:
        raise ValueError("publication sample count exceeds the validation split")
    if not isinstance(image_preprocessing, Mapping) or not image_preprocessing:
        raise ValueError("Query-State CFM image preprocessing identity is required")
    # Round-trip rejects non-JSON checkpoint invariants before serialization.
    preprocessing = json.loads(json.dumps(dict(image_preprocessing), sort_keys=True))
    return {
        "cfm_config": config.to_metadata(),
        "state_shape": list(_QUERY_STATE_SHAPE),
        "train_cache_schema": train_cache_schema,
        "train_cache_fingerprint": train_cache_fingerprint,
        "train_split_identity": train_split_identity,
        "train_row_set_identity": train_row_set_identity,
        "validation_cache_schema": validation_cache_schema,
        "validation_cache_fingerprint": validation_cache_fingerprint,
        "validation_split_identity": validation_split_identity,
        "validation_row_set_identity": validation_row_set_identity,
        "bundle_fingerprint": bundle_fingerprint,
        "source_manifest_identity": source_manifest_identity,
        "template_identity": template_identity,
        "checkpoint_identity": checkpoint_identity,
        "train_items": train_items,
        "validation_items": validation_items,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
        "shuffle_algorithm": shuffle_algorithm,
        "validation_noise_seeds": list(seeds),
        "publication_noise_seeds": list(publication_seeds),
        "publication_min_shuffled_minus_correct": float(
            publication_min_shuffled_minus_correct
        ),
        "publication_sample_items": publication_sample_items,
        "publication_ode_steps": publication_ode_steps,
        "publication_noise_seed": publication_noise_seed,
        "publication_sample_batch_size": publication_sample_batch_size,
        "image_preprocessing": preprocessing,
    }


_INVARIANT_FIELDS = {
    "cfm_config",
    "state_shape",
    "train_cache_schema",
    "train_cache_fingerprint",
    "train_split_identity",
    "train_row_set_identity",
    "validation_cache_schema",
    "validation_cache_fingerprint",
    "validation_split_identity",
    "validation_row_set_identity",
    "bundle_fingerprint",
    "source_manifest_identity",
    "template_identity",
    "checkpoint_identity",
    "train_items",
    "validation_items",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "shuffle_algorithm",
    "validation_noise_seeds",
    "publication_noise_seeds",
    "publication_min_shuffled_minus_correct",
    "publication_sample_items",
    "publication_ode_steps",
    "publication_noise_seed",
    "publication_sample_batch_size",
    "image_preprocessing",
}


def _validate_checkpoint_invariants(
    invariants: Mapping[str, Any],
    *,
    config: CFMConfig,
) -> None:
    if not isinstance(invariants, Mapping) or set(invariants) != _INVARIANT_FIELDS:
        raise ValueError("Query-State CFM checkpoint invariants schema is invalid")
    if invariants.get("cfm_config") != config.to_metadata():
        raise ValueError("Query-State CFM checkpoint config invariants mismatch")
    if invariants.get("state_shape") != list(_QUERY_STATE_SHAPE):
        raise ValueError("Query-State CFM checkpoint K16 state shape mismatch")
    if (
        invariants.get("train_cache_schema")
        != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA
        or invariants.get("validation_cache_schema")
        != QUERY_STATE_RECONSTRUCTION_CACHE_SCHEMA
    ):
        raise ValueError("legacy or cross-schema cache invariants are forbidden")
    for field in (
        "train_cache_fingerprint",
        "train_split_identity",
        "train_row_set_identity",
        "validation_cache_fingerprint",
        "validation_split_identity",
        "validation_row_set_identity",
        "bundle_fingerprint",
        "source_manifest_identity",
        "template_identity",
        "checkpoint_identity",
    ):
        if not _is_sha256(invariants.get(field)):
            raise ValueError(f"Query-State CFM checkpoint {field} is invalid")
    if invariants["train_cache_fingerprint"] == invariants["validation_cache_fingerprint"]:
        raise ValueError("Query-State CFM checkpoint train/validation cache overlap")
    if invariants["train_split_identity"] == invariants["validation_split_identity"]:
        raise ValueError("Query-State CFM checkpoint train/validation split overlap")
    if invariants["train_row_set_identity"] == invariants["validation_row_set_identity"]:
        raise ValueError("Query-State CFM checkpoint train/validation row-set overlap")
    if invariants.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM:
        raise ValueError("Query-State CFM checkpoint shuffle invariants mismatch")
    seeds = invariants.get("validation_noise_seeds")
    publication_seeds = invariants.get("publication_noise_seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seeds
    ):
        raise ValueError("Query-State CFM checkpoint validation noise seeds are invalid")
    if (
        not isinstance(publication_seeds, list)
        or len(publication_seeds) < 2
        or len(set(publication_seeds)) != len(publication_seeds)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in publication_seeds)
        or not set(seeds).issubset(publication_seeds)
    ):
        raise ValueError(
            "Query-State CFM checkpoint publication noise seed invariants are invalid"
        )
    if any(
        isinstance(invariants.get(field), bool)
        or not isinstance(invariants.get(field), int)
        or invariants[field] < 1
        for field in (
            "train_items",
            "validation_items",
            "batch_size",
            "publication_sample_items",
            "publication_ode_steps",
            "publication_sample_batch_size",
        )
    ) or invariants["publication_sample_items"] > invariants["validation_items"]:
        raise ValueError("Query-State CFM checkpoint item/batch invariants are invalid")
    publication_threshold = invariants.get("publication_min_shuffled_minus_correct")
    if (
        isinstance(publication_threshold, bool)
        or not isinstance(publication_threshold, (int, float))
        or not math.isfinite(float(publication_threshold))
        or float(publication_threshold) <= 0
    ):
        raise ValueError(
            "Query-State CFM checkpoint publication sensitivity threshold is invalid"
        )
    if (
        isinstance(invariants.get("seed"), bool)
        or not isinstance(invariants.get("seed"), int)
        or isinstance(invariants.get("publication_noise_seed"), bool)
        or not isinstance(invariants.get("publication_noise_seed"), int)
        or not isinstance(invariants.get("learning_rate"), (int, float))
        or not math.isfinite(float(invariants["learning_rate"]))
        or float(invariants["learning_rate"]) <= 0
        or not isinstance(invariants.get("weight_decay"), (int, float))
        or not math.isfinite(float(invariants["weight_decay"]))
        or float(invariants["weight_decay"]) < 0
        or not isinstance(invariants.get("image_preprocessing"), Mapping)
        or not invariants["image_preprocessing"]
    ):
        raise ValueError("Query-State CFM checkpoint optimizer/preprocessing invariants are invalid")


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary checkpoint already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def save_query_state_cfm_checkpoint(
    path: str | Path,
    *,
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_validation_mse: float,
    invariants: Mapping[str, Any],
) -> None:
    """Atomically serialize decoder/optimizer/RNG state and no producer owner."""

    _validate_cfm_config(model.config)
    _require_decoder_only_optimizer(model, optimizer)
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Query-State CFM checkpoint step must be non-negative")
    if math.isnan(best_validation_mse):
        raise ValueError("best_validation_mse must not be NaN")
    _validate_checkpoint_invariants(invariants, config=model.config)
    payload = {
        "schema": QUERY_STATE_CFM_CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_validation_mse": float(best_validation_mse),
        "invariants": dict(invariants),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    _atomic_torch_save(payload, Path(path))


def load_query_state_cfm_checkpoint(
    path: str | Path,
    *,
    model: TokenConditionedFlowUNet,
    optimizer: torch.optim.Optimizer,
    expected_invariants: Mapping[str, Any],
    device: torch.device,
) -> tuple[int, float]:
    """Restore only an exact-schema, exact-cache decoder checkpoint."""

    _validate_cfm_config(model.config)
    _require_decoder_only_optimizer(model, optimizer)
    _validate_checkpoint_invariants(expected_invariants, config=model.config)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as error:
        raise ValueError("Query-State CFM checkpoint is unreadable") from error
    required = {
        "schema",
        "model",
        "optimizer",
        "step",
        "best_validation_mse",
        "invariants",
        "torch_rng_state",
        "cuda_rng_state_all",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Query-State CFM checkpoint schema/owners are invalid")
    if payload["schema"] != QUERY_STATE_CFM_CHECKPOINT_SCHEMA:
        raise ValueError("legacy or cross-schema Query-State CFM checkpoint rejected")
    if payload["invariants"] != dict(expected_invariants):
        raise ValueError(
            "Query-State CFM resume invariants/cache fingerprint mismatch:\n"
            + json.dumps(
                {
                    "checkpoint": payload.get("invariants"),
                    "expected": dict(expected_invariants),
                },
                indent=2,
                sort_keys=True,
            )
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    _require_decoder_only_optimizer(model, optimizer)
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_states = payload["cuda_rng_state_all"]
    if torch.cuda.is_available() and cuda_states is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    return int(payload["step"]), float(payload["best_validation_mse"])


def make_global_shuffle_mapping(*, item_count: int, seed: int) -> torch.Tensor:
    """Create a deterministic full-split derangement before batching."""

    if isinstance(item_count, bool) or not isinstance(item_count, int) or item_count < 2:
        raise ValueError("global Query-State shuffle requires at least two items")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("global Query-State shuffle seed must be an integer")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shift = int(torch.randint(1, item_count, (1,), generator=generator).item())
    mapping = (torch.arange(item_count, dtype=torch.long) + shift).remainder(item_count)
    if torch.any(mapping == torch.arange(item_count)):
        raise RuntimeError("global Query-State shuffle unexpectedly contains fixed points")
    return mapping


@torch.no_grad()
def evaluate_query_state_condition_sensitivity(
    model: nn.Module,
    states: torch.Tensor,
    images_uint8: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate correct/global-shuffled conditions with shared noise and time."""

    if (
        states.ndim != 3
        or tuple(states.shape[1:]) != _QUERY_STATE_SHAPE
        or not states.is_floating_point()
        or not torch.isfinite(states).all()
    ):
        raise ValueError("condition sensitivity requires finite [N,16,1024] state")
    if (
        images_uint8.ndim != 4
        or images_uint8.shape[0] != states.shape[0]
        or images_uint8.shape[1] != 3
        or images_uint8.dtype != torch.uint8
    ):
        raise ValueError("condition sensitivity requires matching uint8 RGB images")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("condition sensitivity batch_size must be positive")
    count = int(states.shape[0])
    mapping = make_global_shuffle_mapping(item_count=count, seed=seed)
    mapping_identity = _sha256_mapping(
        {
            "algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            "seed": seed,
            "indices": mapping.tolist(),
        }
    )
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0
    correct_sum = 0.0
    shuffled_sum = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        correct_state = states[start:stop].to(device=device, dtype=torch.float32)
        shuffled_state = states[mapping[start:stop]].to(
            device=device,
            dtype=torch.float32,
        )
        target = images_uint8[start:stop].to(
            device=device,
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        batch = stop - start
        noise = torch.randn(
            target.shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        flow_time = torch.rand(
            (batch,),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        interpolation = (
            (1.0 - flow_time[:, None, None, None]) * noise
            + flow_time[:, None, None, None] * target
        )
        target_velocity = target - noise
        correct_velocity = model(
            interpolation,
            flow_time,
            flatten_query_state_condition(correct_state),
        )
        shuffled_velocity = model(
            interpolation,
            flow_time,
            flatten_query_state_condition(shuffled_state),
        )
        correct_loss = torch.nn.functional.mse_loss(
            correct_velocity,
            target_velocity,
            reduction="none",
        ).flatten(1).mean(1)
        shuffled_loss = torch.nn.functional.mse_loss(
            shuffled_velocity,
            target_velocity,
            reduction="none",
        ).flatten(1).mean(1)
        correct_sum += float(correct_loss.sum().cpu())
        shuffled_sum += float(shuffled_loss.sum().cpu())
        total += batch
    model.train(was_training)
    correct = correct_sum / total
    shuffled = shuffled_sum / total
    return {
        "correct_flow_mse": correct,
        "shuffled_flow_mse": shuffled,
        "shuffled_minus_correct": shuffled - correct,
        "shuffled_over_correct": shuffled / max(correct, 1e-12),
        "num_items": total,
        "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
        "shuffle_indices": mapping.tolist(),
        "shuffle_identity": mapping_identity,
        "noise_time_seed": seed,
    }


def evaluate_query_state_multi_noise_sensitivity(
    model: nn.Module,
    states: torch.Tensor,
    images_uint8: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Run preregistered full-split sensitivity and retain every seed result."""

    registered = tuple(seeds)
    if len(registered) < 2 or len(set(registered)) != len(registered) or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in registered
    ):
        raise ValueError("multi-noise condition sensitivity needs at least two unique seeds")
    per_seed = [
        evaluate_query_state_condition_sensitivity(
            model,
            states,
            images_uint8,
            device,
            batch_size=batch_size,
            seed=seed,
        )
        for seed in registered
    ]
    metric_names = (
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
    )
    aggregate = {
        name: {
            "mean": sum(float(result[name]) for result in per_seed) / len(per_seed),
            "min": min(float(result[name]) for result in per_seed),
            "max": max(float(result[name]) for result in per_seed),
        }
        for name in metric_names
    }
    identity_payload = {
        "schema": "nimloth_query_state_cfm_multi_noise_sensitivity_v1",
        "seeds": list(registered),
        "num_items": int(states.shape[0]),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    return {**identity_payload, "identity": _sha256_mapping(identity_payload)}


def _tensor_to_srgb(tensor: torch.Tensor) -> Image.Image:
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("CFM reconstruction must have exactly three RGB channels")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError("CFM reconstruction tensor must be finite floating point")
    array = (
        tensor.detach()
        .cpu()
        .float()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _labelled_strip(
    original: Image.Image,
    reconstruction: Image.Image,
    *,
    row_identity: str,
) -> Image.Image:
    label_height = 18
    width = original.width + reconstruction.width
    output = Image.new("RGB", (width, max(original.height, reconstruction.height) + label_height), "white")
    output.paste(original, (0, label_height))
    output.paste(reconstruction, (original.width, label_height))
    draw = ImageDraw.Draw(output)
    draw.text((2, 2), f"original {row_identity}", fill=(0, 0, 0))
    draw.text((original.width + 2, 2), "CFM reconstruction", fill=(0, 0, 0))
    return output


def _vertical_contact_sheet(strips: Sequence[Image.Image]) -> Image.Image:
    if not strips:
        raise ValueError("RGB contact sheet requires at least one strip")
    width = max(strip.width for strip in strips)
    height = sum(strip.height for strip in strips)
    output = Image.new("RGB", (width, height), "white")
    offset = 0
    for strip in strips:
        output.paste(strip, (0, offset))
        offset += strip.height
    return output


def _validate_multi_noise_publication_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
    item_count: int,
    min_shuffled_minus_correct: float,
) -> dict[str, Any]:
    outer_fields = {"schema", "seeds", "num_items", "per_seed", "aggregate", "identity"}
    per_seed_fields = {
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
        "num_items",
        "shuffle_algorithm",
        "shuffle_indices",
        "shuffle_identity",
        "noise_time_seed",
    }
    metric_names = (
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
    )
    seeds = list(expected_seeds)
    if (
        isinstance(min_shuffled_minus_correct, bool)
        or not isinstance(min_shuffled_minus_correct, (int, float))
        or not math.isfinite(float(min_shuffled_minus_correct))
        or float(min_shuffled_minus_correct) <= 0
    ):
        raise ValueError("publication sensitivity threshold must be finite and positive")
    threshold = float(min_shuffled_minus_correct)
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != outer_fields
        or evidence.get("schema") != "nimloth_query_state_cfm_multi_noise_sensitivity_v1"
        or evidence.get("seeds") != seeds
        or len(seeds) < 2
        or len(set(seeds)) != len(seeds)
        or evidence.get("num_items") != item_count
        or not isinstance(evidence.get("per_seed"), list)
        or len(evidence["per_seed"]) != len(seeds)
    ):
        raise ValueError("complete preregistered multi-noise evidence is required")
    for seed, result in zip(seeds, evidence["per_seed"], strict=True):
        if not isinstance(result, Mapping) or set(result) != per_seed_fields:
            raise ValueError("multi-noise per-seed evidence schema is incomplete")
        mapping = result.get("shuffle_indices")
        expected_mapping = make_global_shuffle_mapping(
            item_count=item_count,
            seed=seed,
        ).tolist()
        if (
            result.get("noise_time_seed") != seed
            or result.get("num_items") != item_count
            or result.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or mapping != expected_mapping
        ):
            raise ValueError(
                "multi-noise per-seed mapping must exactly match the registered shuffle"
            )
        expected_mapping_identity = _sha256_mapping({
            "algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            "seed": seed,
            "indices": mapping,
        })
        if result.get("shuffle_identity") != expected_mapping_identity:
            raise ValueError("multi-noise per-seed shuffle identity mismatch")
        values = {name: result.get(name) for name in metric_names}
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("multi-noise per-seed metrics must be finite")
        correct = float(values["correct_flow_mse"])
        shuffled = float(values["shuffled_flow_mse"])
        if not math.isclose(
            float(values["shuffled_minus_correct"]), shuffled - correct,
            rel_tol=1e-9, abs_tol=1e-12,
        ) or not math.isclose(
            float(values["shuffled_over_correct"]), shuffled / max(correct, 1e-12),
            rel_tol=1e-9, abs_tol=1e-12,
        ):
            raise ValueError("multi-noise per-seed derived metrics mismatch")
    aggregate = evidence.get("aggregate")
    if not isinstance(aggregate, Mapping) or set(aggregate) != set(metric_names):
        raise ValueError("multi-noise aggregate evidence schema is incomplete")
    for name in metric_names:
        values = [float(item[name]) for item in evidence["per_seed"]]
        expected = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
        actual = aggregate.get(name)
        if not isinstance(actual, Mapping) or set(actual) != set(expected) or any(
            not isinstance(actual[key], (int, float))
            or isinstance(actual[key], bool)
            or not math.isclose(float(actual[key]), value, rel_tol=1e-9, abs_tol=1e-12)
            for key, value in expected.items()
        ):
            raise ValueError("multi-noise recomputed aggregate metrics mismatch")
    identity_payload = {key: value for key, value in evidence.items() if key != "identity"}
    if evidence.get("identity") != _sha256_mapping(identity_payload):
        raise ValueError("multi-noise evidence identity mismatch")

    per_seed_verdict = [
        {
            "seed": seed,
            "shuffled_minus_correct": float(result["shuffled_minus_correct"]),
            "passed": float(result["shuffled_minus_correct"]) >= threshold,
        }
        for seed, result in zip(seeds, evidence["per_seed"], strict=True)
    ]
    failed = [result for result in per_seed_verdict if not result["passed"]]
    if failed:
        raise QueryStatePublicationGateFailure(
            "RGB publication sensitivity gate failed: every registered seed must meet "
            f"shuffled_minus_correct >= {threshold}; failures={failed}"
        )
    return {
        "metric": "shuffled_minus_correct",
        "comparison": "greater_than_or_equal_per_registered_seed_v1",
        "publication_min_shuffled_minus_correct": threshold,
        "passed": True,
        "minimum_observed_shuffled_minus_correct": min(
            result["shuffled_minus_correct"] for result in per_seed_verdict
        ),
        "per_seed": per_seed_verdict,
    }


def _load_rgb_publication_checkpoint(
    checkpoint_path: Path,
    *,
    validation_cache: str | Path,
    device: torch.device,
) -> tuple[TokenConditionedFlowUNet, LoadedQueryStateImageSplit, Mapping[str, Any], str]:
    """Load the sole authoritative decoder, optimizer, and validation source."""

    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError("RGB artifact decoder checkpoint path is invalid")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("RGB artifact decoder checkpoint is unreadable") from error
    required = {
        "schema", "model", "optimizer", "step", "best_validation_mse",
        "invariants", "torch_rng_state", "cuda_rng_state_all",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("schema") != QUERY_STATE_CFM_CHECKPOINT_SCHEMA
        or not isinstance(payload.get("model"), Mapping)
        or not isinstance(payload.get("optimizer"), Mapping)
        or isinstance(payload.get("step"), bool)
        or not isinstance(payload.get("step"), int)
        or payload["step"] < 0
        or not isinstance(payload.get("best_validation_mse"), (int, float))
        or math.isnan(float(payload["best_validation_mse"]))
        or not isinstance(payload.get("torch_rng_state"), torch.Tensor)
        or not (
            payload.get("cuda_rng_state_all") is None
            or isinstance(payload.get("cuda_rng_state_all"), list)
        )
    ):
        raise ValueError("RGB artifact decoder checkpoint schema/owners are invalid")
    invariants = payload.get("invariants")
    if not isinstance(invariants, Mapping) or not isinstance(
        invariants.get("cfm_config"), Mapping
    ):
        raise ValueError("RGB artifact checkpoint invariants/config are invalid")
    try:
        config = CFMConfig(**dict(invariants["cfm_config"]))
    except (TypeError, ValueError) as error:
        raise ValueError("RGB artifact CFM config schema is invalid") from error
    _validate_checkpoint_invariants(invariants, config=config)

    validation_split = load_query_state_image_split(
        validation_cache,
        expected_role=QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
        image_size=config.image_size,
        max_items=-1,
    )
    expected = {
        "validation_cache_schema": validation_split.cache_schema,
        "validation_cache_fingerprint": validation_split.cache_fingerprint,
        "validation_split_identity": validation_split.split_identity,
        "validation_row_set_identity": validation_split.row_set_identity,
        "validation_items": len(validation_split),
        "image_preprocessing": dict(validation_split.image_preprocessing),
    }
    if any(invariants.get(key) != value for key, value in expected.items()):
        raise ValueError("RGB artifact validation cache/split/row-set/item invariants mismatch")

    decoder = TokenConditionedFlowUNet(config).to(device)
    try:
        decoder.load_state_dict(payload["model"], strict=True)
        optimizer = build_decoder_optimizer(
            decoder,
            learning_rate=float(invariants["learning_rate"]),
            weight_decay=float(invariants["weight_decay"]),
        )
        optimizer.load_state_dict(payload["optimizer"])
        _require_decoder_only_optimizer(decoder, optimizer)
    except (RuntimeError, TypeError, ValueError, KeyError) as error:
        raise ValueError("RGB artifact decoder checkpoint model/optimizer state is invalid") from error
    if len(optimizer.param_groups) != 1 or any(
        float(group.get("lr", float("nan"))) != float(invariants["learning_rate"])
        or float(group.get("weight_decay", float("nan"))) != float(invariants["weight_decay"])
        for group in optimizer.param_groups
    ):
        raise ValueError("RGB artifact decoder checkpoint optimizer invariants mismatch")
    decoder.eval().requires_grad_(False)
    return decoder, validation_split, invariants, checkpoint_sha256


def save_rgb_reconstruction_artifacts(
    *,
    output_dir: str | Path,
    decoder_checkpoint: str | Path,
    validation_cache: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Authoritatively evaluate and sample one checkpoint for RGB publication."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"RGB reconstruction artifact already exists: {destination}")
    checkpoint_path = Path(decoder_checkpoint)
    decoder, validation_split, invariants, decoder_checkpoint_sha256 = (
        _load_rgb_publication_checkpoint(
            checkpoint_path,
            validation_cache=validation_cache,
            device=device,
        )
    )
    condition_sensitivity = evaluate_query_state_multi_noise_sensitivity(
        decoder,
        validation_split.states,
        validation_split.images_uint8,
        device,
        batch_size=int(invariants["batch_size"]),
        seeds=invariants["publication_noise_seeds"],
    )
    publication_sensitivity_gate = _validate_multi_noise_publication_evidence(
        condition_sensitivity,
        expected_seeds=invariants["publication_noise_seeds"],
        item_count=len(validation_split),
        min_shuffled_minus_correct=float(
            invariants["publication_min_shuffled_minus_correct"]
        ),
    )

    count = int(invariants["publication_sample_items"])
    ode_steps = int(invariants["publication_ode_steps"])
    noise_seed = int(invariants["publication_noise_seed"])
    sample_batch_size = int(invariants["publication_sample_batch_size"])
    image_preprocessing = dict(validation_split.image_preprocessing)
    selection_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    selected_indices = torch.randperm(
        len(validation_split), generator=selection_generator
    )[:count]
    selected_states = validation_split.states[selected_indices].to(dtype=torch.float32)
    rows = tuple(validation_split.rows[index] for index in selected_indices.tolist())
    noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    initial_noise = torch.randn(
        (count, 3, decoder.config.image_size, decoder.config.image_size),
        generator=noise_generator,
    )
    reconstructions = sample_euler(
        decoder,
        flatten_query_state_condition(selected_states),
        initial_noise,
        steps=ode_steps,
        device=device,
        chunk_size=sample_batch_size,
    )
    original_images = [
        _preprocessed_pil(validation_split.images_uint8[index])
        for index in selected_indices.tolist()
    ]

    destination.mkdir(parents=True)
    reconstruction_dir = destination / "reconstructions"
    original_dir = destination / "originals"
    strip_dir = destination / "strips"
    reconstruction_dir.mkdir()
    original_dir.mkdir()
    strip_dir.mkdir()
    strips: list[Image.Image] = []
    strip_paths: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    try:
        for index, (source, row) in enumerate(zip(original_images, rows, strict=True)):
            row_identity = row.get("row_identity")
            if not isinstance(row_identity, str) or not row_identity:
                raise ValueError("RGB artifact row identity is required")
            original_path_value = row.get("original_image_path")
            original_sha256 = row.get("original_image_sha256")
            source_path = Path(original_path_value) if isinstance(original_path_value, str) else Path()
            if (
                not source_path.is_absolute()
                or not source_path.is_file()
                or not _is_sha256(original_sha256)
                or _sha256_file(source_path) != original_sha256
            ):
                raise ValueError("RGB artifact exact original-image SHA256 identity mismatch")
            image_size = image_preprocessing.get("size")
            if isinstance(image_size, bool) or not isinstance(image_size, int):
                raise ValueError("RGB artifact preprocessing size identity is invalid")
            expected_original = _load_image_uint8(source_path, image_size)
            original = source.convert("RGB")
            provided_original = torch.from_numpy(
                np.asarray(original, dtype=np.uint8).copy()
            ).permute(2, 0, 1)
            if not torch.equal(provided_original, expected_original):
                raise ValueError(
                    "RGB artifact provided original does not match the exact preprocessed source image"
                )
            reconstruction = _tensor_to_srgb(reconstructions[index])
            if original.size != reconstruction.size:
                raise ValueError("original/reconstruction sizes must match preprocessing")
            stem = f"row_{index:05d}_{hashlib.sha256(row_identity.encode()).hexdigest()[:12]}"
            original_path = original_dir / f"{stem}.png"
            reconstruction_path = reconstruction_dir / f"{stem}.png"
            strip_path = strip_dir / f"{stem}_strip.png"
            original.save(original_path, format="PNG")
            reconstruction.save(reconstruction_path, format="PNG")
            strip = _labelled_strip(
                original,
                reconstruction,
                row_identity=row_identity,
            )
            strip.save(strip_path, format="PNG")
            strips.append(strip)
            strip_paths.append(str(strip_path))
            metadata_rows.append(
                {
                    **dict(row),
                    "original_path": str(original_path),
                    "original_png_sha256": _sha256_file(original_path),
                    "reconstruction_path": str(reconstruction_path),
                    "reconstruction_png_sha256": _sha256_file(reconstruction_path),
                    "strip_path": str(strip_path),
                    "strip_png_sha256": _sha256_file(strip_path),
                }
            )
            reconstruction.close()
            original.close()
        contact = _vertical_contact_sheet(strips)
        contact_path = destination / "contact_sheet.png"
        contact.save(contact_path, format="PNG")
        contact.close()
        metadata = {
            "schema": QUERY_STATE_CFM_RGB_ARTIFACT_SCHEMA,
            "diagnostic_role": "secondary_post_hoc_rgb_probe",
            "automatic_sft1_selection": False,
            "automatic_sft2_authorization": False,
            "color_space": "sRGB",
            "channels": 3,
            "tensor_range": [-1, 1],
            "conversion": "clamp_-1_1_then_round_255_srgb_v1",
            "ode_solver": "midpoint_euler",
            "ode_steps": ode_steps,
            "noise_seed": noise_seed,
            "image_preprocessing": dict(image_preprocessing),
            "decoder_checkpoint": str(checkpoint_path.resolve()),
            "decoder_checkpoint_sha256": decoder_checkpoint_sha256,
            "cfm_config": decoder.config.to_metadata(),
            "validation_cache_fingerprint": validation_split.cache_fingerprint,
            "validation_split_identity": validation_split.split_identity,
            "validation_row_set_identity": validation_split.row_set_identity,
            "validation_item_count": len(validation_split),
            "checkpoint_invariants_sha256": _sha256_mapping(dict(invariants)),
            "condition_sensitivity": dict(condition_sensitivity),
            "publication_sensitivity_gate": publication_sensitivity_gate,
            "sample_selection": {
                "algorithm": "torch_randperm_cpu_v1",
                "seed": noise_seed,
                "indices": selected_indices.tolist(),
            },
            "contact_sheet_path": str(contact_path),
            "contact_sheet_png_sha256": _sha256_file(contact_path),
            "rows": metadata_rows,
        }
        metadata["artifact_identity"] = _sha256_mapping(metadata)
        metadata_path = destination / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "metadata_path": str(metadata_path),
            "contact_sheet_path": str(contact_path),
            "strip_paths": strip_paths,
            "artifact_identity": metadata["artifact_identity"],
            "condition_sensitivity": condition_sensitivity,
            "publication_sensitivity_gate": publication_sensitivity_gate,
        }
    except Exception:
        # Non-overwrite output makes a partial artifact unusable; remove only
        # files created inside the newly-owned destination.
        import shutil

        shutil.rmtree(destination)
        raise
    finally:
        for strip in strips:
            strip.close()
        for image in original_images:
            image.close()


def _preprocessed_pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(image.permute(1, 2, 0).contiguous().numpy(), mode="RGB")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def train_query_state_cfm(args: argparse.Namespace) -> int:
    """Run decoder-only train/eval/sample orchestration for an approved experiment."""

    if args.resume and args.resume_checkpoint is not None:
        raise ValueError("use either --resume or --resume-checkpoint, not both")
    positive_integer_fields = (
        "batch_size",
        "max_steps",
        "evaluation_interval",
        "sample_items",
        "sample_ode_steps",
        "sample_batch_size",
    )
    for field in positive_integer_fields:
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if args.save_interval < 0:
        raise ValueError("--save-interval must be non-negative")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be finite and positive")
    if (
        not math.isfinite(args.publication_min_shuffled_minus_correct)
        or args.publication_min_shuffled_minus_correct <= 0
    ):
        raise ValueError(
            "--publication-min-shuffled-minus-correct must be finite and positive"
        )
    output_dir = args.output_dir
    if output_dir.exists() and not (args.resume or args.resume_checkpoint is not None):
        raise FileExistsError("Query-State CFM output exists; explicit resume is required")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)

    train_split = load_query_state_image_split(
        args.train_cache,
        expected_role=QUERY_STATE_CACHE_SELECTION_ALL_TRAIN,
        image_size=args.image_size,
        max_items=args.max_train_items,
    )
    validation_split = load_query_state_image_split(
        args.validation_cache,
        expected_role=QUERY_STATE_CACHE_SELECTION_EXTERNAL_VALIDATION,
        image_size=args.image_size,
        max_items=-1,
    )
    validate_query_state_split_pair(train_split, validation_split)

    model = build_query_state_cfm_model(
        image_size=args.image_size,
        base_channels=args.base_channels,
        condition_dim=args.condition_dim,
        time_dim=args.time_dim,
    ).to(device)
    optimizer = build_decoder_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    publication_noise_seeds = tuple(dict.fromkeys([
        *args.validation_noise_seeds,
        *args.final_robustness_noise_seeds,
    ]))
    invariants = build_checkpoint_invariants(
        config=model.config,
        train_cache_schema=train_split.cache_schema,
        train_cache_fingerprint=train_split.cache_fingerprint,
        train_split_identity=train_split.split_identity,
        train_row_set_identity=train_split.row_set_identity,
        validation_cache_schema=validation_split.cache_schema,
        validation_cache_fingerprint=validation_split.cache_fingerprint,
        validation_split_identity=validation_split.split_identity,
        validation_row_set_identity=validation_split.row_set_identity,
        bundle_fingerprint=train_split.bundle_fingerprint,
        source_manifest_identity=train_split.source_manifest_identity,
        template_identity=train_split.template_identity,
        checkpoint_identity=train_split.checkpoint_identity,
        train_items=len(train_split),
        validation_items=len(validation_split),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        shuffle_algorithm=QUERY_STATE_SHUFFLE_ALGORITHM,
        validation_noise_seeds=args.validation_noise_seeds,
        publication_noise_seeds=publication_noise_seeds,
        publication_min_shuffled_minus_correct=(
            args.publication_min_shuffled_minus_correct
        ),
        publication_sample_items=min(args.sample_items, len(validation_split)),
        publication_ode_steps=args.sample_ode_steps,
        publication_noise_seed=args.sample_noise_seed,
        publication_sample_batch_size=args.sample_batch_size,
        image_preprocessing=train_split.image_preprocessing,
    )
    metadata = {
        "task": "direct_query_state_cfm_rgb_reconstruction",
        "diagnostic_role": "secondary_post_hoc_rgb_probe",
        "trainable_owner": "TokenConditionedFlowUNet only",
        "state_source": "human-gated direct Query-State reconstruction cache",
        "target": "same-row original observation",
        "automatic_sft1_selection": False,
        "automatic_sft2_authorization": False,
        "train_cache": str(args.train_cache.resolve()),
        "validation_cache": str(args.validation_cache.resolve()),
        "invariants": invariants,
        "tracking": {
            "wandb_project": args.wandb_project,
            "wandb_run_name": args.wandb_run_name,
            "enabled": not args.no_wandb,
        },
    }
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing != metadata:
            raise ValueError("Query-State CFM output metadata/resume identity mismatch")

    start_step = 0
    best_validation_mse = float("inf")
    resume_path = args.resume_checkpoint
    if args.resume:
        resume_path = _latest_checkpoint(output_dir)
        if resume_path is None:
            raise FileNotFoundError("--resume requested but no checkpoint_*.pt exists")
    if resume_path is not None:
        start_step, best_validation_mse = load_query_state_cfm_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_invariants=invariants,
            device=device,
        )
        if start_step > args.max_steps:
            raise ValueError("resume checkpoint step exceeds --max-steps")

    wandb_run = None
    if not args.no_wandb:
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=metadata,
                dir=str(output_dir),
                resume="allow" if resume_path is not None else None,
            )
        except Exception as error:
            raise RuntimeError("W&B tracking was requested but initialization failed") from error
        finally:
            # Tracking must not alter decoder data/noise streams or resume semantics.
            torch.set_rng_state(cpu_rng_state)
            if torch.cuda.is_available() and cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    log_path = output_dir / "train_step_log.csv"
    if not log_path.exists():
        with log_path.open("x", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    "time",
                    "step",
                    "train_flow_mse",
                    "validation_correct_flow_mse",
                    "validation_shuffled_flow_mse",
                    "validation_shuffle_identity",
                ]
            )
    last_metrics: dict[str, Any] | None = None
    for step in range(start_step + 1, args.max_steps + 1):
        # Use checkpointed global torch RNG so resume continues the exact data stream.
        indices = torch.randint(len(train_split), (args.batch_size,))
        states = train_split.states[indices].to(device=device, dtype=torch.float32)
        targets = train_split.images_uint8[indices].to(
            device=device,
            dtype=torch.float32,
        ).div(127.5).sub(1.0)
        model.train()
        loss = conditional_flow_matching_loss(
            model,
            targets,
            flatten_query_state_condition(states),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        loss_value = float(loss.detach().cpu())

        evaluate = step == 1 or step % args.evaluation_interval == 0 or step == args.max_steps
        if evaluate:
            last_metrics = evaluate_query_state_multi_noise_sensitivity(
                model,
                validation_split.states,
                validation_split.images_uint8,
                device,
                batch_size=args.batch_size,
                seeds=args.validation_noise_seeds,
            )
            comparable_mse = last_metrics["aggregate"]["correct_flow_mse"]["mean"]
            if comparable_mse < best_validation_mse:
                best_validation_mse = comparable_mse
                save_query_state_cfm_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_validation_mse=best_validation_mse,
                    invariants=invariants,
                )
            with log_path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(
                    [
                        time.time(),
                        step,
                        loss_value,
                        last_metrics["aggregate"]["correct_flow_mse"]["mean"],
                        last_metrics["aggregate"]["shuffled_flow_mse"]["mean"],
                        last_metrics["identity"],
                    ]
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "cfm/train_flow_mse": loss_value,
                        "cfm/validation_correct_flow_mse": last_metrics["aggregate"]["correct_flow_mse"]["mean"],
                        "cfm/validation_shuffled_flow_mse": last_metrics["aggregate"]["shuffled_flow_mse"]["mean"],
                    },
                    step=step,
                )
        if args.save_interval > 0 and step % args.save_interval == 0:
            save_query_state_cfm_checkpoint(
                output_dir / f"checkpoint_{step:09d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                best_validation_mse=best_validation_mse,
                invariants=invariants,
            )

    final_path = output_dir / f"checkpoint_{args.max_steps:09d}.pt"
    save_query_state_cfm_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        step=args.max_steps,
        best_validation_mse=best_validation_mse,
        invariants=invariants,
    )
    sample_artifact = save_rgb_reconstruction_artifacts(
        output_dir=output_dir / "rgb_samples",
        decoder_checkpoint=final_path,
        validation_cache=args.validation_cache,
        device=device,
    )
    full_validation = sample_artifact["condition_sensitivity"]

    summary = {
        "status": "completed",
        "final_checkpoint": str(final_path),
        "best_validation_mse": best_validation_mse,
        "last_validation": last_metrics,
        "full_split_validation": full_validation,
        "rgb_sample_artifact": sample_artifact,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if wandb_run is not None:
        wandb_run.log(
            {
                "cfm/full_validation_correct_flow_mse": full_validation["aggregate"]["correct_flow_mse"]["mean"],
                "cfm/full_validation_shuffled_flow_mse": full_validation["aggregate"]["shuffled_flow_mse"]["mean"],
            },
            step=args.max_steps + 1,
        )
        wandb_run.finish()
    return 0


def _positive_publication_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "publication sensitivity threshold must be a number"
        ) from error
    if not math.isfinite(threshold) or threshold <= 0:
        raise argparse.ArgumentTypeError(
            "publication sensitivity threshold must be finite and positive"
        )
    return threshold


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the Query-State-only CLI; legacy checkpoint fields are absent."""

    parser = argparse.ArgumentParser(
        description=(
            "Train a post-hoc RGB CFM decoder from human-gated direct "
            "Query-State caches (experiment approval required)"
        )
    )
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--time-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--max-train-items", type=int, default=-1)
    parser.add_argument("--sample-items", type=int, default=8)
    parser.add_argument("--sample-ode-steps", type=int, default=50)
    parser.add_argument("--sample-noise-seed", type=int, default=20260831)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument(
        "--validation-noise-seeds",
        type=int,
        nargs="+",
        required=True,
        help="preregistered fixed seeds reused unchanged at every checkpoint",
    )
    parser.add_argument(
        "--final-robustness-noise-seeds",
        type=int,
        nargs="*",
        default=(),
        help="optional extra seeds used only for the final full-split gate",
    )
    parser.add_argument(
        "--publication-min-shuffled-minus-correct",
        type=_positive_publication_threshold,
        required=True,
        help=(
            "preregistered positive per-seed minimum shuffled-minus-correct "
            "required before RGB artifact sampling/publication"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_query_state_cfm(build_cli_parser().parse_args(argv))


__all__ = [
    "QUERY_STATE_CFM_CHECKPOINT_SCHEMA",
    "QUERY_STATE_CFM_RGB_ARTIFACT_SCHEMA",
    "QUERY_STATE_SHUFFLE_ALGORITHM",
    "LoadedQueryStateImageSplit",
    "QueryStatePublicationGateFailure",
    "build_checkpoint_invariants",
    "build_cli_parser",
    "build_decoder_optimizer",
    "build_query_state_cfm_model",
    "evaluate_query_state_condition_sensitivity",
    "evaluate_query_state_multi_noise_sensitivity",
    "flatten_query_state_condition",
    "load_query_state_cfm_checkpoint",
    "load_query_state_image_split",
    "make_global_shuffle_mapping",
    "save_query_state_cfm_checkpoint",
    "save_rgb_reconstruction_artifacts",
    "train_query_state_cfm",
    "validate_query_state_split_pair",
]


if __name__ == "__main__":
    raise SystemExit(main())
