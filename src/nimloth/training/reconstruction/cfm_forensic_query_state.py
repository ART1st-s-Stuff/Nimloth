"""Stage-A CFM mechanics probe for the Formal38 unsafe forensic Query-State.

This entry point is deliberately incompatible with the deployable Query-State CFM
owner.  It accepts one exact forensic cache, trains only the RGB decoder, and
allows RGB publication only from the final step-10000 checkpoint after the
mechanics-train condition-sensitivity gate passes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from nimloth.recon.cfm import (
    CFMConfig,
    TokenConditionedFlowUNet,
    conditional_flow_matching_loss,
    sample_euler,
)
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_SHUFFLE_ALGORITHM,
    LoadedQueryStateImageSplit,
    _load_image_uint8,
    _validate_multi_noise_publication_evidence,
    build_decoder_optimizer,
    build_query_state_cfm_model,
    evaluate_query_state_multi_noise_sensitivity,
    flatten_query_state_condition,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    FORENSIC_QUERY_STATE_CACHE_SCHEMA,
    FORENSIC_QUERY_STATE_OWNER_ROLE,
    FORENSIC_SELECTION_MECHANICS_TRAIN,
    FORENSIC_SELECTION_MECHANICS_VALIDATION,
    ForensicQueryStateCacheDataset,
)

FORMAL38_SOURCE_COMMIT = "4838e5fdb469dffb78909e307cf11a808cb2d29e"
FORMAL38_UNSAFE_CONTROL_SHA256 = "414daefe2b501a22805691aa101d76fcc0f5b28447a1332d81b19b3434e838af"
FORMAL38_FAILURE_MANIFEST_SHA256 = "1b9c74ed400da5e3180f04a4402ff36773f6329c7b8fcbc4e4feaeee6bc71340"
FORMAL38_RUN_IDENTITY = "0f82a37c9e191e543d29f8e66857ca1d12a1e2941c2962fc24203666c4f5bcf1"
FORENSIC_CFM_CHECKPOINT_SCHEMA = "nimloth_query_state_forensic_cfm_checkpoint_v1"
FORENSIC_CFM_RGB_ARTIFACT_SCHEMA = "nimloth_query_state_forensic_cfm_rgb_artifact_v1"
FORENSIC_EXPERIMENT_STAGE = "mechanics_only"
FORENSIC_FINAL_STEP = 10_000
FORENSIC_PASS_MIN_DELTA = 1e-4
FORENSIC_PASS_MIN_AGGREGATE_RATIO = 1.25
FORENSIC_WATERMARKS = (
    "mechanics_only",
    "unsafe_actor_checkpoint",
    "not_heldout",
    "not_deployable",
)
_STATE_SHAPE = (16, 1024)
_HEX = frozenset("0123456789abcdef")
_SOURCE_FIELDS = {
    "source_commit", "control_sha256", "failure_manifest_sha256",
    "config_identity", "run_identity",
}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_forensic_stage_a_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject every deployable/legacy/stage-B/wrong-Formal38 cache owner."""

    checkpoint = manifest.get("checkpoint") if isinstance(manifest, Mapping) else None
    selection = manifest.get("selection") if isinstance(manifest, Mapping) else None
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != FORENSIC_QUERY_STATE_CACHE_SCHEMA
        or manifest.get("owner_role") != FORENSIC_QUERY_STATE_OWNER_ROLE
        or manifest.get("forensic_only") is not True
        or any(manifest.get(field) is not False for field in ("authoritative", "terminal_primary", "deployable", "sft2_ready"))
        or not _is_sha256(manifest.get("cache_fingerprint"))
    ):
        raise ValueError("forensic Stage A cache schema/owner/watermarks are invalid")
    if (
        not isinstance(selection, Mapping)
        or selection.get("stage") != FORENSIC_EXPERIMENT_STAGE
        or selection.get("roles") != {
            FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
            FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
        }
    ):
        raise ValueError("forensic CFM accepts only the explicit mechanics_only Stage A roles")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("source_commit") != FORMAL38_SOURCE_COMMIT
        or checkpoint.get("control_sha256") != FORMAL38_UNSAFE_CONTROL_SHA256
        or checkpoint.get("failure_manifest_sha256") != FORMAL38_FAILURE_MANIFEST_SHA256
        or checkpoint.get("world_size") != 8
        or Path(str(checkpoint.get("checkpoint_path", ""))).name != "unsafe_update_00001605"
        or not isinstance(checkpoint.get("actor_failure"), Mapping)
        or checkpoint["actor_failure"].get("passed") is not False
        or checkpoint.get("config_identity") != FORMAL38_RUN_IDENTITY
        or checkpoint.get("run_identity") != FORMAL38_RUN_IDENTITY
    ):
        raise ValueError("forensic CFM requires the exact Formal38 update1605 unsafe source")
    return manifest


def _source_identity(manifest: Mapping[str, Any]) -> dict[str, str]:
    checkpoint = manifest["checkpoint"]
    return {field: str(checkpoint[field]) for field in sorted(_SOURCE_FIELDS)}


def load_forensic_image_splits(
    cache_dir: str | Path, *, image_size: int
) -> tuple[LoadedQueryStateImageSplit, LoadedQueryStateImageSplit, Mapping[str, str]]:
    """Load both fixed roles from one strict live forensic cache."""

    root = Path(cache_dir)
    try:
        raw = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("forensic cache manifest is missing or invalid") from error
    manifest = validate_forensic_stage_a_manifest(raw)
    dataset = ForensicQueryStateCacheDataset(root)
    role_indices: dict[str, list[int]] = {
        FORENSIC_SELECTION_MECHANICS_TRAIN: [],
        FORENSIC_SELECTION_MECHANICS_VALIDATION: [],
    }
    items: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        role = item.get("selection_role")
        if role not in role_indices:
            raise ValueError("forensic cache contains a non-Stage-A selection role")
        role_indices[role].append(index)
        items.append(item)
    if {role: len(indices) for role, indices in role_indices.items()} != {
        FORENSIC_SELECTION_MECHANICS_TRAIN: 48,
        FORENSIC_SELECTION_MECHANICS_VALIDATION: 16,
    }:
        raise ValueError("forensic Stage A role counts must be exactly 48/16")
    preprocessing = {"size": image_size, "resample": "bicubic", "range": [-1, 1], "color_space": "sRGB"}
    source = _source_identity(manifest)
    splits: list[LoadedQueryStateImageSplit] = []
    for role in (FORENSIC_SELECTION_MECHANICS_TRAIN, FORENSIC_SELECTION_MECHANICS_VALIDATION):
        selected = [items[index] for index in role_indices[role]]
        states = torch.stack([item["state"] for item in selected]).detach().cpu().float().contiguous()
        if states.shape != (len(selected), *_STATE_SHAPE) or not torch.isfinite(states).all():
            raise ValueError("forensic cache must preserve finite [N,16,1024] state")
        images = torch.stack([_load_image_uint8(item["original_image_path"], image_size) for item in selected])
        rows = tuple({key: value for key, value in item.items() if key != "state"} for item in selected)
        split_identity = _sha256_mapping({"cache_fingerprint": manifest["cache_fingerprint"], "role": role, "selection_identity": manifest["selection"]["identity"]})
        row_set_identity = _sha256_mapping({"rows": rows})
        splits.append(LoadedQueryStateImageSplit(
            states=states, images_uint8=images, rows=rows,
            cache_schema=FORENSIC_QUERY_STATE_CACHE_SCHEMA,
            cache_fingerprint=str(manifest["cache_fingerprint"]),
            bundle_fingerprint=_sha256_mapping(source),
            source_manifest_identity=str(manifest["source_jsonl"]["source_manifest_identity"]),
            template_identity=str(manifest["checkpoint"]["model_data_identities"]["template_identity"]),
            checkpoint_identity=FORMAL38_UNSAFE_CONTROL_SHA256,
            split_name=role, split_identity=split_identity,
            row_set_identity=row_set_identity, image_preprocessing=preprocessing,
        ))
    validate_forensic_split_pair(splits[0], splits[1])
    return splits[0], splits[1], source


def validate_forensic_split_pair(train: LoadedQueryStateImageSplit, validation: LoadedQueryStateImageSplit) -> None:
    if train.split_name != FORENSIC_SELECTION_MECHANICS_TRAIN or validation.split_name != FORENSIC_SELECTION_MECHANICS_VALIDATION:
        raise ValueError("forensic Stage A requires mechanics_train and mechanics_validation roles")
    if train.cache_schema != FORENSIC_QUERY_STATE_CACHE_SCHEMA or validation.cache_schema != FORENSIC_QUERY_STATE_CACHE_SCHEMA:
        raise ValueError("deployable or legacy cache cannot enter forensic CFM")
    if train.cache_fingerprint != validation.cache_fingerprint or not _is_sha256(train.cache_fingerprint):
        raise ValueError("forensic Stage A roles must come from the same exact cache")
    if train.split_identity == validation.split_identity or train.row_set_identity == validation.row_set_identity:
        raise ValueError("forensic Stage A role identities must differ")
    if train.image_preprocessing != validation.image_preprocessing:
        raise ValueError("forensic Stage A image preprocessing mismatch")
    train_rows = {row.get("row_identity") for row in train.rows}
    validation_rows = {row.get("row_identity") for row in validation.rows}
    train_images = {row.get("original_image_sha256") for row in train.rows}
    validation_images = {row.get("original_image_sha256") for row in validation.rows}
    if None in train_rows | validation_rows | train_images | validation_images:
        raise ValueError("forensic Stage A row/image identities are required")
    if train_rows & validation_rows:
        raise ValueError("forensic Stage A row overlap is forbidden")
    if train_images & validation_images:
        raise ValueError("forensic Stage A image overlap is forbidden")


def _validate_source_identity(source: Mapping[str, Any]) -> dict[str, str]:
    if (
        set(source) != _SOURCE_FIELDS
        or source.get("source_commit") != FORMAL38_SOURCE_COMMIT
        or source.get("control_sha256") != FORMAL38_UNSAFE_CONTROL_SHA256
        or source.get("failure_manifest_sha256") != FORMAL38_FAILURE_MANIFEST_SHA256
        or source.get("config_identity") != FORMAL38_RUN_IDENTITY
        or source.get("run_identity") != FORMAL38_RUN_IDENTITY
    ):
        raise ValueError("forensic CFM source identity is not exact Formal38 update1605")
    return {field: str(source[field]) for field in sorted(source)}


def build_forensic_checkpoint_invariants(
    *, config: CFMConfig, cache_fingerprint: str,
    train_split_identity: str, train_row_set_identity: str,
    validation_split_identity: str, validation_row_set_identity: str,
    source_identity: Mapping[str, Any], train_items: int, validation_items: int,
    batch_size: int, learning_rate: float, weight_decay: float, gradient_clip: float,
    evaluation_interval: int, save_interval: int, seed: int,
    noise_seeds: Sequence[int], sample_items: int, sample_ode_steps: int,
    sample_noise_seed: int, sample_batch_size: int,
    image_preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    if config.token_count != 16 or config.token_dim != 1024 or config.input_channels != 3 or config.output_channels != 3:
        raise ValueError("forensic CFM requires exact CFMConfig(16,1024) and RGB I/O")
    identities = (cache_fingerprint, train_split_identity, train_row_set_identity, validation_split_identity, validation_row_set_identity)
    if not all(_is_sha256(value) for value in identities) or train_split_identity == validation_split_identity or train_row_set_identity == validation_row_set_identity:
        raise ValueError("forensic CFM cache/role identities are invalid")
    registered = tuple(noise_seeds)
    if len(registered) < 3 or len(set(registered)) != len(registered) or any(isinstance(value, bool) or not isinstance(value, int) for value in registered):
        raise ValueError("forensic Stage A requires at least three unique preregistered noise seeds")
    if (train_items, validation_items) != (48, 16):
        raise ValueError("forensic Stage A item counts must be exactly 48/16")
    integers = (
        batch_size, evaluation_interval, sample_items, sample_ode_steps,
        sample_batch_size,
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integers)
        or isinstance(save_interval, bool)
        or not isinstance(save_interval, int)
        or save_interval < 0
        or sample_items < 8
        or sample_items > train_items
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(sample_noise_seed, bool)
        or not isinstance(sample_noise_seed, int)
    ):
        raise ValueError("forensic Stage A cadence/batch/sample/seed invariants are invalid")
    if (
        not math.isfinite(learning_rate)
        or learning_rate <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
        or not math.isfinite(gradient_clip)
        or gradient_clip <= 0
    ):
        raise ValueError("forensic Stage A optimizer invariants are invalid")
    if not isinstance(image_preprocessing, Mapping) or not image_preprocessing:
        raise ValueError("forensic Stage A image preprocessing identity is required")
    return {
        "experiment_stage": FORENSIC_EXPERIMENT_STAGE,
        "watermarks": list(FORENSIC_WATERMARKS),
        "cache_schema": FORENSIC_QUERY_STATE_CACHE_SCHEMA,
        "cache_fingerprint": cache_fingerprint,
        "train_role": FORENSIC_SELECTION_MECHANICS_TRAIN,
        "train_split_identity": train_split_identity,
        "train_row_set_identity": train_row_set_identity,
        "validation_role": FORENSIC_SELECTION_MECHANICS_VALIDATION,
        "validation_split_identity": validation_split_identity,
        "validation_row_set_identity": validation_row_set_identity,
        "source_identity": _validate_source_identity(source_identity),
        "state_shape": list(_STATE_SHAPE), "cfm_config": config.to_metadata(),
        "train_items": train_items, "validation_items": validation_items,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "weight_decay": weight_decay, "gradient_clip": gradient_clip,
        "evaluation_interval": evaluation_interval, "save_interval": save_interval,
        "seed": seed,
        "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
        "noise_seeds": list(registered), "max_steps": FORENSIC_FINAL_STEP,
        "pass_min_delta": FORENSIC_PASS_MIN_DELTA,
        "pass_min_aggregate_ratio": FORENSIC_PASS_MIN_AGGREGATE_RATIO,
        "sample_items": sample_items, "sample_ode_steps": sample_ode_steps,
        "sample_noise_seed": sample_noise_seed, "sample_batch_size": sample_batch_size,
        "image_preprocessing": json.loads(json.dumps(dict(image_preprocessing), sort_keys=True)),
    }


_INVARIANT_FIELDS = {
    "experiment_stage", "watermarks", "cache_schema", "cache_fingerprint", "train_role",
    "train_split_identity", "train_row_set_identity", "validation_role",
    "validation_split_identity", "validation_row_set_identity", "source_identity",
    "state_shape", "cfm_config", "train_items", "validation_items", "batch_size",
    "learning_rate", "weight_decay", "gradient_clip", "evaluation_interval",
    "save_interval", "seed", "shuffle_algorithm", "noise_seeds",
    "max_steps", "pass_min_delta", "pass_min_aggregate_ratio", "sample_items",
    "sample_ode_steps", "sample_noise_seed", "sample_batch_size", "image_preprocessing",
}


def _validate_invariants(invariants: Mapping[str, Any], config: CFMConfig) -> None:
    if not isinstance(invariants, Mapping) or set(invariants) != _INVARIANT_FIELDS:
        raise ValueError("forensic CFM checkpoint invariants schema is invalid")
    rebuilt = build_forensic_checkpoint_invariants(
        config=config, cache_fingerprint=invariants["cache_fingerprint"],
        train_split_identity=invariants["train_split_identity"], train_row_set_identity=invariants["train_row_set_identity"],
        validation_split_identity=invariants["validation_split_identity"], validation_row_set_identity=invariants["validation_row_set_identity"],
        source_identity=invariants["source_identity"], train_items=invariants["train_items"], validation_items=invariants["validation_items"],
        batch_size=invariants["batch_size"], learning_rate=invariants["learning_rate"],
        weight_decay=invariants["weight_decay"], gradient_clip=invariants["gradient_clip"],
        evaluation_interval=invariants["evaluation_interval"], save_interval=invariants["save_interval"],
        seed=invariants["seed"],
        noise_seeds=invariants["noise_seeds"], sample_items=invariants["sample_items"], sample_ode_steps=invariants["sample_ode_steps"],
        sample_noise_seed=invariants["sample_noise_seed"], sample_batch_size=invariants["sample_batch_size"], image_preprocessing=invariants["image_preprocessing"],
    )
    if dict(invariants) != rebuilt or invariants["cfm_config"] != config.to_metadata():
        raise ValueError("forensic CFM checkpoint invariants/resume identity mismatch")


def _require_decoder_optimizer(model: TokenConditionedFlowUNet, optimizer: torch.optim.Optimizer) -> None:
    model_ids = {id(parameter) for parameter in model.parameters()}
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if {id(parameter) for parameter in parameters} != model_ids or len(parameters) != len(model_ids):
        raise ValueError("forensic CFM optimizer/checkpoint must be decoder-only")


def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"forensic CFM checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        if temporary.exists(): temporary.unlink()
        raise


def save_forensic_cfm_checkpoint(path: str | Path, *, model: TokenConditionedFlowUNet, optimizer: torch.optim.Optimizer, step: int, invariants: Mapping[str, Any]) -> None:
    _validate_invariants(invariants, model.config)
    _require_decoder_optimizer(model, optimizer)
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= FORENSIC_FINAL_STEP:
        raise ValueError("forensic CFM checkpoint step is invalid")
    _atomic_save({
        "schema": FORENSIC_CFM_CHECKPOINT_SCHEMA, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "step": step, "invariants": dict(invariants),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, Path(path))


def load_forensic_cfm_checkpoint(path: str | Path, *, model: TokenConditionedFlowUNet, optimizer: torch.optim.Optimizer, expected_invariants: Mapping[str, Any], device: torch.device) -> int:
    _validate_invariants(expected_invariants, model.config); _require_decoder_optimizer(model, optimizer)
    try: payload = torch.load(path, map_location=device, weights_only=False)
    except Exception as error: raise ValueError("forensic CFM checkpoint is unreadable") from error
    required = {"schema", "model", "optimizer", "step", "invariants", "torch_rng_state", "cuda_rng_state_all"}
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema") != FORENSIC_CFM_CHECKPOINT_SCHEMA:
        raise ValueError("deployable, legacy, or cross-schema forensic CFM checkpoint rejected")
    if payload["invariants"] != dict(expected_invariants):
        raise ValueError("forensic CFM resume cache/invariants mismatch")
    step = payload["step"]
    cpu_rng = payload["torch_rng_state"]
    cuda_rng = payload["cuda_rng_state_all"]
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 0 <= step <= FORENSIC_FINAL_STEP
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
        raise ValueError("forensic CFM checkpoint step/RNG state is invalid")
    try:
        model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
    except Exception as error: raise ValueError("forensic CFM decoder/optimizer checkpoint state is invalid") from error
    _require_decoder_optimizer(model, optimizer)
    torch.set_rng_state(cpu_rng.cpu())
    if torch.cuda.is_available() and cuda_rng is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng])
    return step


def evaluate_stage_a_pass(evidence: Mapping[str, Any]) -> dict[str, Any]:
    seeds = evidence.get("seeds") if isinstance(evidence, Mapping) else None
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("Stage A pass requires complete evidence for at least three preregistered seeds")
    _validate_multi_noise_publication_evidence(
        evidence,
        expected_seeds=seeds,
        item_count=48,
        min_shuffled_minus_correct=FORENSIC_PASS_MIN_DELTA,
    )
    per_seed = evidence["per_seed"]
    deltas: list[float] = []
    correct: list[float] = []
    shuffled: list[float] = []
    for seed, item in zip(seeds, per_seed, strict=True):
        if not isinstance(item, Mapping) or item.get("noise_time_seed") != seed or item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM:
            raise ValueError("Stage A pass per-seed shared-noise/global-shuffle evidence is invalid")
        values = (item.get("correct_flow_mse"), item.get("shuffled_flow_mse"), item.get("shuffled_minus_correct"))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise ValueError("Stage A pass metrics must be finite")
        c, s, delta = map(float, values)
        if not math.isclose(delta, s - c, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Stage A pass per-seed delta is inconsistent")
        correct.append(c); shuffled.append(s); deltas.append(delta)
    failed = [{"seed": seed, "delta": delta} for seed, delta in zip(seeds, deltas, strict=True) if delta < FORENSIC_PASS_MIN_DELTA]
    if failed:
        raise ValueError(f"Stage A per-seed delta gate failed: {failed}")
    aggregate_ratio = (sum(shuffled) / len(shuffled)) / max(sum(correct) / len(correct), 1e-12)
    if not math.isfinite(aggregate_ratio) or aggregate_ratio < FORENSIC_PASS_MIN_AGGREGATE_RATIO:
        raise ValueError(f"Stage A aggregate shuffled/correct ratio gate failed: {aggregate_ratio}")
    return {
        "passed": True, "source_role": FORENSIC_SELECTION_MECHANICS_TRAIN,
        "checkpoint_step": FORENSIC_FINAL_STEP,
        "metric_unit": "mean conditional-flow velocity MSE per normalized [-1,1] RGB image element",
        "aggregation": "per seed: mean over rows and RGB elements; aggregate ratio: mean shuffled MSE / mean correct MSE across seeds",
        "per_seed_min_delta": FORENSIC_PASS_MIN_DELTA,
        "minimum_observed_delta": min(deltas),
        "aggregate_shuffled_over_correct": aggregate_ratio,
        "aggregate_min_ratio": FORENSIC_PASS_MIN_AGGREGATE_RATIO,
    }


def _tensor_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.detach().cpu().float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _uint8_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(tensor.permute(1, 2, 0).contiguous().numpy(), mode="RGB")


def _strip(original: Image.Image, reconstruction: Image.Image, row_identity: str) -> Image.Image:
    result = Image.new("RGB", (original.width + reconstruction.width, original.height + 18), "white")
    result.paste(original, (0, 18)); result.paste(reconstruction, (original.width, 18))
    draw = ImageDraw.Draw(result); draw.text((2, 2), f"original {row_identity}", fill="black"); draw.text((original.width + 2, 2), "CFM", fill="black")
    return result


def publish_forensic_rgb_artifacts(*, output_dir: str | Path, decoder_checkpoint: str | Path, cache_dir: str | Path, device: torch.device) -> dict[str, Any]:
    """Evaluate both roles, gate on mechanics-train, then publish final RGB only."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink(): raise FileExistsError(f"forensic RGB output exists: {destination}")
    checkpoint_path = Path(decoder_checkpoint)
    checkpoint_sha = _sha256_file(checkpoint_path)
    try: payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error: raise ValueError("forensic RGB decoder checkpoint is unreadable") from error
    if not isinstance(payload, dict) or payload.get("schema") != FORENSIC_CFM_CHECKPOINT_SCHEMA or payload.get("step") != FORENSIC_FINAL_STEP:
        raise ValueError("forensic RGB publication requires only the final step10000 checkpoint")
    invariants = payload.get("invariants")
    if not isinstance(invariants, Mapping) or not isinstance(invariants.get("cfm_config"), Mapping): raise ValueError("forensic RGB checkpoint invariants are invalid")
    try: config = CFMConfig(**dict(invariants["cfm_config"]))
    except Exception as error: raise ValueError("forensic RGB CFM config is invalid") from error
    _validate_invariants(invariants, config)
    train, validation, source = load_forensic_image_splits(cache_dir, image_size=config.image_size)
    validate_forensic_split_pair(train, validation)
    expected = {
        "cache_fingerprint": train.cache_fingerprint, "train_split_identity": train.split_identity,
        "train_row_set_identity": train.row_set_identity, "validation_split_identity": validation.split_identity,
        "validation_row_set_identity": validation.row_set_identity, "source_identity": dict(source),
        "train_items": len(train), "validation_items": len(validation),
        "image_preprocessing": dict(train.image_preprocessing),
    }
    if any(invariants.get(key) != value for key, value in expected.items()): raise ValueError("forensic RGB cache/source/checkpoint invariants mismatch")
    decoder = TokenConditionedFlowUNet(config).to(device)
    optimizer = build_decoder_optimizer(decoder, learning_rate=float(invariants["learning_rate"]), weight_decay=float(invariants["weight_decay"]))
    decoder.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"]); _require_decoder_optimizer(decoder, optimizer)
    decoder.eval().requires_grad_(False)
    reports = {
        FORENSIC_SELECTION_MECHANICS_TRAIN: evaluate_query_state_multi_noise_sensitivity(decoder, train.states, train.images_uint8, device, batch_size=int(invariants["batch_size"]), seeds=invariants["noise_seeds"]),
        FORENSIC_SELECTION_MECHANICS_VALIDATION: evaluate_query_state_multi_noise_sensitivity(decoder, validation.states, validation.images_uint8, device, batch_size=int(invariants["batch_size"]), seeds=invariants["noise_seeds"]),
    }
    gate = evaluate_stage_a_pass(reports[FORENSIC_SELECTION_MECHANICS_TRAIN])
    count = int(invariants["sample_items"]); seed = int(invariants["sample_noise_seed"])
    indices = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))[:count]
    states = train.states[indices].float(); noise = torch.randn((count, 3, config.image_size, config.image_size), generator=torch.Generator().manual_seed(seed))
    reconstructions = sample_euler(decoder, flatten_query_state_condition(states), noise, steps=int(invariants["sample_ode_steps"]), device=device, chunk_size=int(invariants["sample_batch_size"]))
    destination.mkdir(parents=True)
    strips_dir = destination / "strips"; strips_dir.mkdir()
    originals_dir = destination / "originals"; originals_dir.mkdir()
    reconstructions_dir = destination / "reconstructions"; reconstructions_dir.mkdir()
    strips: list[Image.Image] = []; strip_paths: list[str] = []; rows: list[dict[str, Any]] = []
    try:
        for ordinal, selected_index in enumerate(indices.tolist()):
            original = _uint8_image(train.images_uint8[selected_index]); reconstruction = _tensor_image(reconstructions[ordinal])
            source_row = train.rows[selected_index]
            row_identity = str(source_row["row_identity"]); strip = _strip(original, reconstruction, row_identity)
            stem = f"row_{ordinal:05d}_{hashlib.sha256(row_identity.encode()).hexdigest()[:12]}"
            original_path = originals_dir / f"{stem}.png"
            reconstruction_path = reconstructions_dir / f"{stem}.png"
            strip_path = strips_dir / f"{stem}_strip.png"
            original.save(original_path, format="PNG")
            reconstruction.save(reconstruction_path, format="PNG")
            strip.save(strip_path, format="PNG")
            strips.append(strip); strip_paths.append(str(strip_path)); rows.append({
                "row_identity": row_identity,
                "selection_index": selected_index,
                "source_original_image_sha256": source_row["original_image_sha256"],
                "original_path": str(original_path),
                "original_png_sha256": _sha256_file(original_path),
                "reconstruction_path": str(reconstruction_path),
                "reconstruction_png_sha256": _sha256_file(reconstruction_path),
                "strip_path": str(strip_path),
                "strip_png_sha256": _sha256_file(strip_path),
            })
            original.close(); reconstruction.close()
        contact = Image.new("RGB", (max(image.width for image in strips), sum(image.height for image in strips)), "white")
        offset = 0
        for image in strips: contact.paste(image, (0, offset)); offset += image.height
        contact_path = destination / "contact_sheet.png"; contact.save(contact_path, format="PNG"); contact.close()
        metric_contract = {"unit": gate["metric_unit"], "aggregation": gate["aggregation"], "correct_and_shuffled_share_noise_and_time": True, "global_nonidentity_shuffle": QUERY_STATE_SHUFFLE_ALGORITHM}
        metadata = {
            "schema": FORENSIC_CFM_RGB_ARTIFACT_SCHEMA, "watermarks": list(FORENSIC_WATERMARKS),
            "forensic_only": True, "unsafe_actor_checkpoint": True, "heldout": False, "deployable": False,
            "diagnostic_role": "secondary_post_hoc_rgb_probe", "primary_direct_dino_metrics_required_separately": True,
            "formal38_actor_failure_remains_valid": True, "experiment_stage": FORENSIC_EXPERIMENT_STAGE,
            "decoder_checkpoint": str(checkpoint_path.resolve()), "decoder_checkpoint_sha256": checkpoint_sha,
            "decoder_checkpoint_step": FORENSIC_FINAL_STEP, "checkpoint_invariants_sha256": _sha256_mapping(dict(invariants)),
            "cache_fingerprint": train.cache_fingerprint, "source_identity": dict(source), "cfm_config": config.to_metadata(),
            "color_space": "sRGB", "channels": 3, "tensor_range": [-1, 1], "metric_contract": metric_contract,
            "split_reports": reports, "pass_source_role": FORENSIC_SELECTION_MECHANICS_TRAIN,
            "mechanics_validation_controls_pass": False, "checkpoint_selection": "final_step10000_only_no_best_selection",
            "stage_a_pass": gate, "sample_selection": {"role": FORENSIC_SELECTION_MECHANICS_TRAIN, "algorithm": "torch_randperm_cpu_v1", "seed": seed, "indices": indices.tolist()},
            "rows": rows, "contact_sheet_path": str(contact_path), "contact_sheet_sha256": _sha256_file(contact_path),
        }
        metadata["artifact_identity"] = _sha256_mapping(metadata)
        metadata_path = destination / "metadata.json"; metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"metadata_path": str(metadata_path), "contact_sheet_path": str(contact_path), "strip_paths": strip_paths, "stage_a_pass": gate, "artifact_identity": metadata["artifact_identity"]}
    except BaseException:
        shutil.rmtree(destination); raise
    finally:
        for image in strips: image.close()


def _latest_checkpoint(output: Path) -> Path | None:
    paths = sorted(output.glob("checkpoint_*.pt")); return paths[-1] if paths else None


def train_forensic_query_state_cfm(args: argparse.Namespace) -> int:
    if args.experiment_stage != FORENSIC_EXPERIMENT_STAGE:
        raise ValueError("only mechanics_only Stage A is implemented")
    if args.max_steps != FORENSIC_FINAL_STEP:
        raise ValueError("Stage A max_steps is locked to 10000")
    if len(args.noise_seeds) < 3 or len(set(args.noise_seeds)) != len(args.noise_seeds):
        raise ValueError("at least three unique preregistered noise seeds are required")
    for field in ("batch_size", "evaluation_interval", "sample_items", "sample_ode_steps", "sample_batch_size"):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if isinstance(args.save_interval, bool) or not isinstance(args.save_interval, int) or args.save_interval < 0:
        raise ValueError("--save-interval must be non-negative")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("--gradient-clip must be finite and positive")
    if args.resume and args.resume_checkpoint is not None:
        raise ValueError("use either --resume or --resume-checkpoint")
    if not args.no_wandb and (
        not isinstance(args.wandb_run_id, str)
        or not args.wandb_run_id.strip()
        or not isinstance(args.wandb_run_name, str)
        or not args.wandb_run_name.strip()
    ):
        raise ValueError("tracked forensic CFM requires explicit W&B run ID and run name")
    output = args.output_dir
    if output.exists() and not (args.resume or args.resume_checkpoint):
        raise FileExistsError("forensic CFM output exists; explicit resume required")
    device = torch.device(args.device); torch.manual_seed(args.seed)
    if device.type == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.manual_seed_all(args.seed)
    train, validation, source = load_forensic_image_splits(args.cache, image_size=args.image_size)
    output.mkdir(parents=True, exist_ok=True)
    model = build_query_state_cfm_model(image_size=args.image_size, base_channels=args.base_channels, condition_dim=args.condition_dim, time_dim=args.time_dim).to(device)
    optimizer = build_decoder_optimizer(model, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    invariants = build_forensic_checkpoint_invariants(
        config=model.config, cache_fingerprint=train.cache_fingerprint,
        train_split_identity=train.split_identity, train_row_set_identity=train.row_set_identity,
        validation_split_identity=validation.split_identity, validation_row_set_identity=validation.row_set_identity,
        source_identity=source, train_items=len(train), validation_items=len(validation), batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip, evaluation_interval=args.evaluation_interval,
        save_interval=args.save_interval, seed=args.seed, noise_seeds=args.noise_seeds,
        sample_items=args.sample_items, sample_ode_steps=args.sample_ode_steps, sample_noise_seed=args.sample_noise_seed,
        sample_batch_size=args.sample_batch_size, image_preprocessing=train.image_preprocessing,
    )
    metadata = {
        "task": "formal38_unsafe_forensic_query_state_cfm_stage_a",
        "watermarks": list(FORENSIC_WATERMARKS),
        "trainable_owner": "TokenConditionedFlowUNet only",
        "target": "matching original observation",
        "invariants": invariants,
        "tracking": {
            "enabled": not args.no_wandb,
            "wandb_project": args.wandb_project,
            "wandb_run_id": args.wandb_run_id,
            "wandb_run_name": args.wandb_run_name,
        },
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata: raise ValueError("forensic CFM output resume identity mismatch")
    else: metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    start = 0; resume = args.resume_checkpoint
    if args.resume:
        resume = _latest_checkpoint(output)
        if resume is None: raise FileNotFoundError("--resume requested but no checkpoint exists")
    if resume is not None:
        start = load_forensic_cfm_checkpoint(resume, model=model, optimizer=optimizer, expected_invariants=invariants, device=device)
    if start >= FORENSIC_FINAL_STEP:
        raise ValueError("final step10000 checkpoint cannot be resumed for further training")
    wandb_run = None
    if not args.no_wandb:
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                id=args.wandb_run_id,
                name=args.wandb_run_name,
                config=metadata,
                dir=str(output),
                resume="must" if resume is not None else "never",
            )
        except Exception as error:
            raise RuntimeError("W&B tracking was requested but initialization failed") from error
        finally:
            torch.set_rng_state(cpu_rng)
            if torch.cuda.is_available() and cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
    log_path = output / "train_step_log.csv"
    if not log_path.exists():
        with log_path.open("x", newline="") as stream: csv.writer(stream).writerow(["time", "step", "train_flow_mse", "mechanics_train_report_identity", "mechanics_validation_report_identity"])
    for step in range(start + 1, FORENSIC_FINAL_STEP + 1):
        indices = torch.randint(len(train), (args.batch_size,)); states = train.states[indices].to(device).float(); targets = train.images_uint8[indices].to(device).float().div(127.5).sub(1)
        model.train(); loss = conditional_flow_matching_loss(model, targets, flatten_query_state_condition(states)); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip); optimizer.step()
        if step == 1 or step % args.evaluation_interval == 0 or step == FORENSIC_FINAL_STEP:
            reports = [evaluate_query_state_multi_noise_sensitivity(model, split.states, split.images_uint8, device, batch_size=args.batch_size, seeds=args.noise_seeds) for split in (train, validation)]
            with log_path.open("a", newline="") as stream:
                csv.writer(stream).writerow([time.time(), step, float(loss.detach().cpu()), reports[0]["identity"], reports[1]["identity"]])
            if wandb_run is not None:
                wandb_run.log({
                    "cfm/train_flow_mse": float(loss.detach().cpu()),
                    "cfm/mechanics_train_correct_flow_mse": reports[0]["aggregate"]["correct_flow_mse"]["mean"],
                    "cfm/mechanics_train_shuffled_flow_mse": reports[0]["aggregate"]["shuffled_flow_mse"]["mean"],
                    "cfm/mechanics_validation_correct_flow_mse_report_only": reports[1]["aggregate"]["correct_flow_mse"]["mean"],
                    "cfm/mechanics_validation_shuffled_flow_mse_report_only": reports[1]["aggregate"]["shuffled_flow_mse"]["mean"],
                }, step=step)
        if args.save_interval > 0 and step % args.save_interval == 0 and step != FORENSIC_FINAL_STEP:
            save_forensic_cfm_checkpoint(output / f"checkpoint_{step:09d}.pt", model=model, optimizer=optimizer, step=step, invariants=invariants)
    final = output / f"checkpoint_{FORENSIC_FINAL_STEP:09d}.pt"; save_forensic_cfm_checkpoint(final, model=model, optimizer=optimizer, step=FORENSIC_FINAL_STEP, invariants=invariants)
    artifact = publish_forensic_rgb_artifacts(output_dir=output / "rgb_samples", decoder_checkpoint=final, cache_dir=args.cache, device=device)
    summary = {"status": "completed", "watermarks": list(FORENSIC_WATERMARKS), "final_checkpoint": str(final), "rgb_artifact": artifact}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({
            "cfm/stage_a_pass": 1,
            "cfm/stage_a_aggregate_shuffled_over_correct": artifact["stage_a_pass"]["aggregate_shuffled_over_correct"],
        }, step=FORENSIC_FINAL_STEP + 1)
        wandb_run.finish()
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Stage-A decoder-only CFM mechanics probe for exact Formal38 unsafe update1605 (launch approval required)")
    parser.add_argument("--experiment-stage", choices=(FORENSIC_EXPERIMENT_STAGE,), required=True)
    parser.add_argument("--cache", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=64); parser.add_argument("--base-channels", type=int, default=64); parser.add_argument("--condition-dim", type=int, default=256); parser.add_argument("--time-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--learning-rate", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4); parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=FORENSIC_FINAL_STEP); parser.add_argument("--evaluation-interval", type=int, default=500); parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--noise-seeds", type=int, nargs="+", required=True); parser.add_argument("--sample-items", type=int, default=8); parser.add_argument("--sample-ode-steps", type=int, default=50); parser.add_argument("--sample-noise-seed", type=int, default=20260901); parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901); parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda"); parser.add_argument("--resume", action="store_true"); parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--wandb-project", default="nimloth-recon"); parser.add_argument("--wandb-run-id"); parser.add_argument("--wandb-run-name"); parser.add_argument("--no-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return train_forensic_query_state_cfm(build_cli_parser().parse_args(argv))


__all__ = [
    "FORENSIC_CFM_CHECKPOINT_SCHEMA",
    "FORENSIC_CFM_RGB_ARTIFACT_SCHEMA",
    "FORENSIC_WATERMARKS",
    "FORMAL38_FAILURE_MANIFEST_SHA256",
    "FORMAL38_SOURCE_COMMIT",
    "FORMAL38_UNSAFE_CONTROL_SHA256",
    "build_cli_parser",
    "build_decoder_optimizer",
    "build_forensic_checkpoint_invariants",
    "evaluate_stage_a_pass",
    "load_forensic_cfm_checkpoint",
    "load_forensic_image_splits",
    "publish_forensic_rgb_artifacts",
    "save_forensic_cfm_checkpoint",
    "train_forensic_query_state_cfm",
    "validate_forensic_split_pair",
    "validate_forensic_stage_a_manifest",
]

if __name__ == "__main__": raise SystemExit(main())
