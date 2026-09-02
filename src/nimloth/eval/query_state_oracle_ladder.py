"""Strict four-cell evaluation for the Formal38 reconstruction oracle ladder.

The evaluator is read-only.  It compares one immutable token/state baseline and
three final-step fresh decoder checkpoints, then publishes fixed-time and
pure-noise evidence without selecting a best checkpoint or noise seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, FrozenDINOGridTargets
from nimloth.recon.cfm import CFMConfig, TokenConditionedFlowUNet, sample_euler
from nimloth.training.reconstruction.cfm_forensic_oracle_ladder import (
    BATCH_SIZE,
    CELLS,
    EXTERNAL_ITEMS,
    FINAL_STEP,
    FIXED_TIMES,
    IMAGE_SIZE,
    NOISE_SEEDS,
    ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA,
    ORACLE_LADDER_SUMMARY_SCHEMA,
    SAMPLE_ODE_STEPS,
    SAMPLE_ROWS,
    LoadedOracleLadderSplit,
    _current_source_commit,
    _flatten_condition,
    _has_symlink_component,
    _identity,
    _is_git_commit,
    _is_sha256,
    _reject_protected_output,
    _sha256_file,
    build_decoder_optimizer,
    build_oracle_ladder_model,
    evaluate_fixed_time_sensitivity,
    load_oracle_ladder_splits,
    validate_oracle_ladder_invariants,
)
from nimloth.training.reconstruction.cfm_forensic_posthoc_inspection import (
    FINAL_CHECKPOINT_SHA256,
    STAGE_B_INITIAL_NOISE_SHA256,
    STAGE_B_SAMPLE_INDICES,
    STAGE_B_SAMPLE_INDICES_SHA256,
)
from nimloth.training.reconstruction.cfm_forensic_query_state import (
    FORENSIC_CFM_CHECKPOINT_SCHEMA,
    FORENSIC_STAGE_B_EXPERIMENT_STAGE,
    FORENSIC_STAGE_B_FINAL_STEP,
    _sha256_tensor_bytes,
)
from nimloth.training.reconstruction.cfm_forensic_query_state import (
    _validate_invariants as validate_baseline_invariants,
)
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_SHUFFLE_ALGORITHM,
    evaluate_query_state_multi_noise_sensitivity,
    make_global_shuffle_mapping,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    ForensicQueryStateCacheDataset,
    _publish_noreplace,
)
from nimloth.training.reconstruction.forensic_query_state_oracle_cache import (
    EXACT_DINO_FEATURE_IDENTITY,
    ForensicDinoOracleCacheDataset,
    _validate_teacher,
)

ORACLE_LADDER_REPORT_SCHEMA = (
    "nimloth_query_state_forensic_oracle_ladder_report_v1"
)
SAMPLE_SELECTION_ALGORITHM = "sha256_oracle_ladder_v1"
CELL_ORDER = (
    "token_state",
    "token_oracle",
    "spatial_state",
    "spatial_oracle",
)
BOOTSTRAP_SEED = 20260951
BOOTSTRAP_RESAMPLES = 10_000
CONTACT_SAMPLE_SEED = 20260921
ID198_SUMMARY_SHA256 = "a2f30b1c67556e6b37b36ae115d1430be5fa61a218c52c7be4b8f551a579c717"
ID198_EXTERNAL_REPORT_SHA256 = "ed07caa67115576fe036aee4c8a0f1d379243670632c9c7bba82d4506e435de0"
ID198_ROW_IMAGE_PAIR_IDENTITY = "b8eaddac0c595f26525cf60e6de98312d41bd32271e243deb1a4c61aabe40c86"


@dataclass(frozen=True)
class OracleLadderSamplePlan:
    indices: tuple[int, ...]
    row_identities: tuple[str, ...]
    image_sha256: tuple[str, ...]
    algorithm: str
    identity: str


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _mapping_identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    metadata = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(metadata + tensor.numpy().tobytes()).hexdigest()


def build_sample_plan(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> OracleLadderSamplePlan:
    """Choose rows by identity only, independently of model outputs."""

    values = tuple(rows)
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or count > len(values)
    ):
        raise ValueError("oracle-ladder sample count is invalid")
    by_identity: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, row in enumerate(values):
        identity = row.get("row_identity") if isinstance(row, Mapping) else None
        image_sha = row.get("original_image_sha256") if isinstance(row, Mapping) else None
        if (
            not isinstance(identity, str)
            or not identity
            or identity in by_identity
            or not _is_sha256(image_sha)
            or row.get("selection_role") != "external_validation"
        ):
            raise ValueError("oracle-ladder sample rows require unique external row identity")
        by_identity[identity] = (index, row)
    selected = sorted(
        by_identity.values(),
        key=lambda item: hashlib.sha256(
            f"oracle-ladder-v1:{item[1]['row_identity']}".encode()
        ).hexdigest(),
    )[:count]
    row_identities = tuple(str(item[1]["row_identity"]) for item in selected)
    image_sha256 = tuple(str(item[1]["original_image_sha256"]) for item in selected)
    identity = _mapping_identity(
        {
            "algorithm": SAMPLE_SELECTION_ALGORITHM,
            "row_identities": row_identities,
            "image_sha256": image_sha256,
        }
    )
    return OracleLadderSamplePlan(
        indices=tuple(item[0] for item in selected),
        row_identities=row_identities,
        image_sha256=image_sha256,
        algorithm=SAMPLE_SELECTION_ALGORITHM,
        identity=identity,
    )


def paired_row_bootstrap(
    differences: torch.Tensor,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Bootstrap already-paired, seed-averaged row effects."""

    if (
        not isinstance(differences, torch.Tensor)
        or differences.ndim != 1
        or differences.numel() < 1
        or not differences.is_floating_point()
        or not bool(torch.isfinite(differences).all())
    ):
        raise ValueError("paired bootstrap requires one-dimensional finite row effects")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 1
    ):
        raise ValueError("paired bootstrap seed/resamples are invalid")
    values = differences.detach().cpu().to(torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        values.numel(),
        (resamples, values.numel()),
        generator=generator,
    )
    means = values[indices].mean(dim=1)
    quantiles = torch.quantile(
        means,
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    )
    return {
        "mean": float(values.mean()),
        "lower_95": float(quantiles[0]),
        "upper_95": float(quantiles[1]),
        "row_count": values.numel(),
        "seed": seed,
        "resamples": resamples,
        "statistical_unit": "external_validation_row",
    }


def _validate_sample_tensors(
    *,
    row_count: int,
    seed_count: int,
    source_rgb: torch.Tensor,
    source_dino: torch.Tensor,
    correct_rgb: torch.Tensor,
    shuffled_rgb: torch.Tensor,
    correct_dino: torch.Tensor,
    shuffled_dino: torch.Tensor,
) -> None:
    if (
        source_rgb.ndim != 4
        or source_rgb.shape[0] != row_count
        or source_rgb.shape[1] != 3
        or source_dino.ndim != 3
        or source_dino.shape[0] != row_count
        or correct_rgb.shape != (row_count, seed_count, *source_rgb.shape[1:])
        or shuffled_rgb.shape != correct_rgb.shape
        or correct_dino.shape != (row_count, seed_count, *source_dino.shape[1:])
        or shuffled_dino.shape != correct_dino.shape
        or any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in (
                source_rgb,
                source_dino,
                correct_rgb,
                shuffled_rgb,
                correct_dino,
                shuffled_dino,
            )
        )
    ):
        raise ValueError("oracle-ladder sample metric tensors are invalid")


def sample_pair_metrics(
    *,
    rows: Sequence[Mapping[str, Any]],
    noise_seeds: Sequence[int],
    source_rgb: torch.Tensor,
    source_dino: torch.Tensor,
    correct_rgb: torch.Tensor,
    shuffled_rgb: torch.Tensor,
    correct_dino: torch.Tensor,
    shuffled_dino: torch.Tensor,
) -> dict[str, Any]:
    """Average fixed noise seeds inside each row before any row statistic."""

    row_values = tuple(rows)
    seeds = tuple(noise_seeds)
    if (
        not row_values
        or len(seeds) < 1
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("oracle-ladder sample rows/noise seeds are invalid")
    _validate_sample_tensors(
        row_count=len(row_values),
        seed_count=len(seeds),
        source_rgb=source_rgb,
        source_dino=source_dino,
        correct_rgb=correct_rgb,
        shuffled_rgb=shuffled_rgb,
        correct_dino=correct_dino,
        shuffled_dino=shuffled_dino,
    )
    source_rgb_expanded = source_rgb[:, None]
    source_dino_expanded = source_dino[:, None]
    correct_rgb_l1 = (correct_rgb - source_rgb_expanded).abs().flatten(2).mean(2)
    shuffled_rgb_l1 = (shuffled_rgb - source_rgb_expanded).abs().flatten(2).mean(2)
    correct_rgb_rmse = (
        (correct_rgb - source_rgb_expanded).square().flatten(2).mean(2).sqrt()
    )
    shuffled_rgb_rmse = (
        (shuffled_rgb - source_rgb_expanded).square().flatten(2).mean(2).sqrt()
    )
    correct_dino_mse = (
        (correct_dino - source_dino_expanded).square().flatten(2).mean(2)
    )
    shuffled_dino_mse = (
        (shuffled_dino - source_dino_expanded).square().flatten(2).mean(2)
    )
    source_flat = source_dino_expanded.expand_as(correct_dino).flatten(2)
    correct_cosine = torch.nn.functional.cosine_similarity(
        correct_dino.flatten(2), source_flat, dim=2, eps=1e-12
    )
    shuffled_cosine = torch.nn.functional.cosine_similarity(
        shuffled_dino.flatten(2), source_flat, dim=2, eps=1e-12
    )
    metrics = {
        "correct_rgb_l1": correct_rgb_l1,
        "shuffled_rgb_l1": shuffled_rgb_l1,
        "correct_rgb_rmse": correct_rgb_rmse,
        "shuffled_rgb_rmse": shuffled_rgb_rmse,
        "correct_dino_mse": correct_dino_mse,
        "shuffled_dino_mse": shuffled_dino_mse,
        "correct_dino_cosine": correct_cosine,
        "shuffled_dino_cosine": shuffled_cosine,
    }
    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(row_values):
        if not isinstance(row.get("row_identity"), str):
            raise TypeError("oracle-ladder sample metric row identity is invalid")
        item = {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["original_image_sha256"],
            "per_seed": [
                {
                    "seed": seed,
                    **{
                        name: float(value[index, seed_index])
                        for name, value in metrics.items()
                    },
                }
                for seed_index, seed in enumerate(seeds)
            ],
            **{
                name: float(value[index].mean())
                for name, value in metrics.items()
            },
        }
        item["rgb_l1_shuffled_minus_correct"] = (
            item["shuffled_rgb_l1"] - item["correct_rgb_l1"]
        )
        item["rgb_rmse_shuffled_minus_correct"] = (
            item["shuffled_rgb_rmse"] - item["correct_rgb_rmse"]
        )
        item["dino_mse_shuffled_minus_correct"] = (
            item["shuffled_dino_mse"] - item["correct_dino_mse"]
        )
        item["dino_cosine_correct_minus_shuffled"] = (
            item["correct_dino_cosine"] - item["shuffled_dino_cosine"]
        )
        result_rows.append(item)
    paired_effects = {
        "rgb_l1_shuffled_minus_correct": paired_row_bootstrap(
            torch.tensor(
                [row["rgb_l1_shuffled_minus_correct"] for row in result_rows],
                dtype=torch.float64,
            ),
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "rgb_rmse_shuffled_minus_correct": paired_row_bootstrap(
            torch.tensor(
                [row["rgb_rmse_shuffled_minus_correct"] for row in result_rows],
                dtype=torch.float64,
            ),
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "dino_mse_shuffled_minus_correct": paired_row_bootstrap(
            torch.tensor(
                [row["dino_mse_shuffled_minus_correct"] for row in result_rows],
                dtype=torch.float64,
            ),
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
        "dino_cosine_correct_minus_shuffled": paired_row_bootstrap(
            torch.tensor(
                [row["dino_cosine_correct_minus_shuffled"] for row in result_rows],
                dtype=torch.float64,
            ),
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        ),
    }
    report: dict[str, Any] = {
        "schema": "nimloth_query_state_oracle_ladder_sample_metrics_v1",
        "statistical_unit": "external_validation_row",
        "seed_aggregation": "mean_within_row_before_row_bootstrap",
        "noise_seeds": list(seeds),
        "source_dino_input_owner": "canonical_original_archived_observation",
        "generated_dino_input_owner": "decoder_generated_128px_srgb",
        "generated_dino_is_canonical_teacher_target": False,
        "paired_row_bootstrap": paired_effects,
        "rows": result_rows,
    }
    report["identity"] = _mapping_identity(report)
    return report


def _metric_effect(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    higher_is_better: bool,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> Mapping[str, Any]:
    left_values = torch.tensor([float(row[metric]) for row in left], dtype=torch.float64)
    right_values = torch.tensor([float(row[metric]) for row in right], dtype=torch.float64)
    difference = right_values - left_values if higher_is_better else left_values - right_values
    result = paired_row_bootstrap(
        difference,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    return {
        **result,
        "direction": "positive_means_right_is_better",
        "left": "baseline",
        "right": "comparison",
    }


def build_four_cell_comparison(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Build preregistered teacher and decoder effects from matched rows."""

    if set(reports) != set(CELL_ORDER):
        raise ValueError("oracle-ladder comparison requires exactly four cells")
    row_sets = {str(report.get("row_set_identity")) for report in reports.values()}
    selections = {
        str(report.get("sample_selection_identity")) for report in reports.values()
    }
    seed_sets = {tuple(report.get("noise_seeds", ())) for report in reports.values()}
    row_identities = {
        tuple(str(row.get("row_identity")) for row in report.get("rows", ()))
        for report in reports.values()
    }
    if (
        len(row_sets) != 1
        or not _is_sha256(next(iter(row_sets)))
        or len(selections) != 1
        or not _is_sha256(next(iter(selections)))
        or len(seed_sets) != 1
        or len(row_identities) != 1
        or not next(iter(row_identities))
    ):
        raise ValueError("oracle-ladder cells require matched row/selection/seed evidence")
    for cell in CELL_ORDER:
        report = reports[cell]
        owner = CELLS[cell]
        if (
            report.get("cell") != cell
            or report.get("decoder_family") != owner["decoder_family"]
            or report.get("condition_family") != owner["condition_family"]
        ):
            raise ValueError("oracle-ladder cell owner metadata mismatch")
    pairs = {
        "teacher_at_token": ("token_state", "token_oracle"),
        "decoder_at_state": ("token_state", "spatial_state"),
        "teacher_at_spatial": ("spatial_state", "spatial_oracle"),
        "decoder_at_oracle": ("token_oracle", "spatial_oracle"),
    }
    metrics = {
        "correct_dino_cosine": True,
        "correct_dino_mse": False,
        "correct_rgb_l1": False,
        "correct_rgb_rmse": False,
        "rgb_l1_shuffled_minus_correct": True,
        "rgb_rmse_shuffled_minus_correct": True,
        "dino_mse_shuffled_minus_correct": True,
        "dino_cosine_correct_minus_shuffled": True,
    }
    effects: dict[str, Any] = {}
    for effect, (left_cell, right_cell) in pairs.items():
        left = reports[left_cell]["rows"]
        right = reports[right_cell]["rows"]
        effects[effect] = {
            metric: _metric_effect(
                left,
                right,
                metric=metric,
                higher_is_better=higher,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            for metric, higher in metrics.items()
        }
        effects[effect]["left_cell"] = left_cell
        effects[effect]["right_cell"] = right_cell
    comparison: dict[str, Any] = {
        "schema": "nimloth_query_state_oracle_ladder_comparison_v1",
        "cells": list(CELL_ORDER),
        "effects": effects,
        "matched_row_set_identity": next(iter(row_sets)),
        "sample_selection_identity": next(iter(selections)),
        "noise_seeds": list(next(iter(seed_sets))),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "claim_scope": "forensic_representation_decodability_only",
        "actor_safety_or_deployability_established": False,
    }
    comparison["identity"] = _mapping_identity(comparison)
    return comparison


def save_four_cell_contact_sheet(
    output: str | Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    originals: Sequence[Image.Image],
    correct_samples: Mapping[str, Sequence[Image.Image]],
    sample_seed: int,
    sample_indices: Sequence[int],
    sample_indices_sha256: str,
    initial_noise_sha256: str,
) -> dict[str, Any]:
    """Save original plus four correct-condition cells for human inspection."""

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"oracle-ladder contact sheet exists: {destination}")
    row_values = tuple(rows)
    original_values = tuple(originals)
    index_values = tuple(sample_indices)
    if (
        not row_values
        or len(row_values) != len(original_values)
        or set(correct_samples) != set(CELL_ORDER)
        or any(len(correct_samples[cell]) != len(row_values) for cell in CELL_ORDER)
        or any(
            not isinstance(row.get("row_identity"), str)
            or not row["row_identity"]
            or not _is_sha256(row.get("original_image_sha256"))
            for row in row_values
        )
        or len(index_values) != len(row_values)
        or any(isinstance(index, bool) or not isinstance(index, int) for index in index_values)
        or isinstance(sample_seed, bool)
        or not isinstance(sample_seed, int)
        or not _is_sha256(sample_indices_sha256)
        or not _is_sha256(initial_noise_sha256)
    ):
        raise ValueError("oracle-ladder contact-sheet row/cell/seed contract is invalid")
    all_images = [*original_values]
    for cell in CELL_ORDER:
        all_images.extend(correct_samples[cell])
    size = original_values[0].size
    if any(image.mode != "RGB" or image.size != size for image in all_images):
        raise ValueError("oracle-ladder contact-sheet images must share RGB size")
    header = 18
    sheet = Image.new(
        "RGB",
        (size[0] * (1 + len(CELL_ORDER)), (size[1] + header) * len(row_values)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    columns = ("original", *CELL_ORDER)
    for row_index, (row, original) in enumerate(
        zip(row_values, original_values, strict=True)
    ):
        y = row_index * (size[1] + header)
        for column_index, column in enumerate(columns):
            x = column_index * size[0]
            draw.text((x + 1, y + 1), column, fill="black")
            image = original if column == "original" else correct_samples[column][row_index]
            sheet.paste(image, (x, y + header))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            sheet.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        sheet.close()
    row_identities = [str(row["row_identity"]) for row in row_values]
    image_sha256 = [str(row["original_image_sha256"]) for row in row_values]
    pair_identity = _mapping_identity(
        {
            "indices_sha256": sample_indices_sha256,
            "row_identities": row_identities,
            "image_sha256": image_sha256,
        }
    )
    metadata = {
        "columns": list(columns),
        "indices": list(index_values),
        "indices_sha256": sample_indices_sha256,
        "row_identities": row_identities,
        "image_sha256": image_sha256,
        "row_image_pair_identity": pair_identity,
        "sample_seed": sample_seed,
        "initial_noise_sha256": initial_noise_sha256,
        "ode_solver": "midpoint_euler",
        "ode_steps": SAMPLE_ODE_STEPS,
        "human_inspection_only": True,
        "controls_scientific_result": False,
        "png_sha256": _sha256_file(destination),
    }
    metadata["identity"] = _mapping_identity(metadata)
    return metadata


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach()
        .cpu()
        .float()
        .clamp(-1, 1)
        .add(1)
        .mul(127.5)
        .round()
        .byte()
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def _dino_generated_features(
    teacher: FrozenDINOGridTargets,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        pil_images = [_tensor_to_pil(image) for image in images[start : start + batch_size]]
        try:
            output = teacher.load_images(pil_images, device=device).detach().cpu().float()
        finally:
            for image in pil_images:
                image.close()
        if output.shape != (len(pil_images), 16, 1024) or not bool(
            torch.isfinite(output).all()
        ):
            raise ValueError("generated-image DINO feature output is invalid")
        outputs.append(output)
    return torch.cat(outputs, dim=0).contiguous()


def _load_new_cell_checkpoint(
    path: Path,
    *,
    cell: str,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any], str]:
    if _has_symlink_component(path) or not path.is_absolute() or not path.is_file():
        raise ValueError("oracle-ladder new cell checkpoint must be an absolute regular file")
    checkpoint_sha = _sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError(f"oracle-ladder {cell} checkpoint is unreadable") from error
    invariants = payload.get("invariants") if isinstance(payload, Mapping) else None
    cpu_rng = payload.get("torch_rng_state") if isinstance(payload, Mapping) else None
    cuda_rng = payload.get("cuda_rng_state_all") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {
            "schema",
            "model",
            "optimizer",
            "step",
            "invariants",
            "torch_rng_state",
            "cuda_rng_state_all",
        }
        or payload.get("schema") != ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA
        or payload.get("step") != FINAL_STEP
        or not isinstance(invariants, Mapping)
        or invariants.get("cell") != cell
        or not isinstance(invariants.get("cfm_config"), Mapping)
        or not isinstance(cpu_rng, torch.Tensor)
        or cpu_rng.ndim != 1
        or cpu_rng.dtype != torch.uint8
        or (
            cuda_rng is not None
            and (
                not isinstance(cuda_rng, list)
                or any(
                    not isinstance(value, torch.Tensor)
                    or value.ndim != 1
                    or value.dtype != torch.uint8
                    for value in cuda_rng
                )
            )
        )
    ):
        raise ValueError("oracle-ladder new cell final-checkpoint identity is invalid")
    try:
        config = CFMConfig(**dict(invariants["cfm_config"]))
    except (TypeError, ValueError) as error:
        raise ValueError("oracle-ladder new cell CFM config is invalid") from error
    validate_oracle_ladder_invariants(invariants, config)
    if invariants.get("output_dir") != str(path.parent.resolve()):
        raise ValueError("oracle-ladder new cell checkpoint output identity mismatch")
    model = build_oracle_ladder_model(
        decoder_family=str(invariants["decoder_family"]), config=config
    ).to(device)
    optimizer = build_decoder_optimizer(
        model,
        learning_rate=float(invariants["learning_rate"]),
        weight_decay=float(invariants["weight_decay"]),
    )
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
    except Exception as error:
        raise ValueError("oracle-ladder new cell model/optimizer state is invalid") from error
    if any(
        float(group.get("lr", float("nan"))) != float(invariants["learning_rate"])
        or float(group.get("weight_decay", float("nan")))
        != float(invariants["weight_decay"])
        for group in optimizer.param_groups
    ):
        raise ValueError("oracle-ladder new cell optimizer invariants mismatch")
    summary_path = path.parent / "summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise ValueError("oracle-ladder new cell requires completed training summary")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-ladder new cell training summary is unreadable") from error
    random_report = summary.get("random_time_report") if isinstance(summary, Mapping) else None
    random_plan = (
        random_report.get("noise_time_provenance")
        if isinstance(random_report, Mapping)
        else None
    )
    metadata_path = path.parent / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-ladder new cell metadata is unreadable") from error
    metadata_producer = (
        metadata.get("producer") if isinstance(metadata, Mapping) else None
    )
    metadata_config = (
        metadata_producer.get("resolved_config")
        if isinstance(metadata_producer, Mapping)
        else None
    )
    tracking = metadata.get("tracking") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(summary, Mapping)
        or set(summary)
        != {
            "schema",
            "cell",
            "status",
            "final_step",
            "final_checkpoint",
            "final_checkpoint_sha256",
            "random_time_report",
            "invariants_identity",
            "metadata_identity",
            "producer_identity",
            "evaluation_pending",
            "additional_steps_or_fallback_allowed",
            "identity",
        }
        or summary.get("schema") != ORACLE_LADDER_SUMMARY_SCHEMA
        or summary.get("cell") != cell
        or summary.get("status") != "completed_final_step4000"
        or summary.get("final_step") != FINAL_STEP
        or summary.get("final_checkpoint") != str(path.resolve())
        or summary.get("final_checkpoint_sha256") != checkpoint_sha
        or summary.get("invariants_identity") != _identity(invariants)
        or summary.get("metadata_identity") != _identity(metadata)
        or not _is_sha256(summary.get("producer_identity"))
        or not _is_sha256(summary.get("identity"))
        or summary["identity"]
        != _identity({key: value for key, value in summary.items() if key != "identity"})
        or not isinstance(metadata, Mapping)
        or set(metadata)
        != {
            "schema",
            "cell",
            "invariants",
            "state_cache",
            "oracle_cache",
            "output",
            "producer",
            "tracking",
        }
        or metadata.get("schema")
        != "nimloth_query_state_forensic_oracle_ladder_train_v1"
        or metadata.get("cell") != cell
        or metadata.get("invariants") != invariants
        or metadata.get("output") != str(path.parent.resolve())
        or metadata.get("output") != invariants.get("output_dir")
        or not isinstance(metadata.get("state_cache"), str)
        or not Path(metadata["state_cache"]).is_absolute()
        or not isinstance(metadata.get("oracle_cache"), str)
        or not Path(metadata["oracle_cache"]).is_absolute()
        or metadata.get("state_cache") != invariants.get("state_cache_path")
        or metadata.get("oracle_cache") != invariants.get("oracle_cache_path")
        or not _has_valid_identity(metadata_producer)
        or metadata_producer.get("identity") != summary.get("producer_identity")
        or metadata_producer.get("module")
        != "nimloth.training.reconstruction.cfm_forensic_oracle_ladder"
        or metadata_producer.get("source_commit") != invariants.get("source_commit")
        or not isinstance(metadata_config, Mapping)
        or set(metadata_config)
        != {
            "cell",
            "state_cache",
            "oracle_cache",
            "output",
            "device",
            "wandb_project",
            "wandb_mode",
            "wandb_run_id",
            "wandb_run_name",
            "invariants_identity",
        }
        or metadata_config.get("cell") != cell
        or metadata_config.get("state_cache") != metadata.get("state_cache")
        or metadata_config.get("oracle_cache") != metadata.get("oracle_cache")
        or metadata_config.get("output") != metadata.get("output")
        or metadata_config.get("wandb_project") != invariants.get("wandb_project")
        or metadata_config.get("wandb_mode") != invariants.get("wandb_mode")
        or metadata_config.get("wandb_run_id") != invariants.get("wandb_run_id")
        or metadata_config.get("wandb_run_name") != invariants.get("wandb_run_name")
        or metadata_config.get("invariants_identity") != _identity(invariants)
        or not isinstance(tracking, Mapping)
        or tracking
        != {
            "enabled": True,
            "project": invariants.get("wandb_project"),
            "mode": invariants.get("wandb_mode"),
            "run_id": invariants.get("wandb_run_id"),
            "run_name": invariants.get("wandb_run_name"),
        }
        or summary.get("evaluation_pending") is not True
        or summary.get("additional_steps_or_fallback_allowed") is not False
        or not isinstance(random_report, Mapping)
        or set(random_report)
        != {
            "schema",
            "seeds",
            "num_items",
            "per_seed",
            "aggregate",
            "identity",
            "noise_time_provenance",
            "owner_identity",
        }
        or random_report.get("owner_identity")
        != _identity(
            {
                key: value
                for key, value in random_report.items()
                if key != "owner_identity"
            }
        )
        or random_report.get("identity")
        != _identity(
            {
                key: value
                for key, value in random_report.items()
                if key not in {"identity", "noise_time_provenance", "owner_identity"}
            }
        )
        or not isinstance(random_plan, Mapping)
        or random_plan.get("identity")
        != _identity(
            {key: value for key, value in random_plan.items() if key != "identity"}
        )
    ):
        raise ValueError("oracle-ladder new cell training summary identity is invalid")
    model.eval().requires_grad_(False)
    return model, invariants, checkpoint_sha


def _load_baseline_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any], str]:
    if _has_symlink_component(path) or not path.is_absolute() or not path.is_file():
        raise ValueError("oracle-ladder token_state baseline must be an absolute regular file")
    checkpoint_sha = _sha256_file(path)
    if checkpoint_sha != FINAL_CHECKPOINT_SHA256:
        raise ValueError("oracle-ladder token_state baseline SHA256 mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("oracle-ladder token_state baseline is unreadable") from error
    invariants = payload.get("invariants") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != FORENSIC_CFM_CHECKPOINT_SCHEMA
        or payload.get("step") != FORENSIC_STAGE_B_FINAL_STEP
        or not isinstance(invariants, Mapping)
        or invariants.get("experiment_stage") != FORENSIC_STAGE_B_EXPERIMENT_STAGE
        or not isinstance(invariants.get("cfm_config"), Mapping)
    ):
        raise ValueError("oracle-ladder token_state baseline owner is invalid")
    config = CFMConfig(**dict(invariants["cfm_config"]))
    validate_baseline_invariants(invariants, config)
    model = TokenConditionedFlowUNet(config).to(device)
    try:
        model.load_state_dict(payload["model"], strict=True)
    except Exception as error:
        raise ValueError("oracle-ladder token_state baseline model state is invalid") from error
    model.eval().requires_grad_(False)
    return model, invariants, checkpoint_sha


def _evaluation_row_set_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return _mapping_identity(
        {
            "rows": [
                {
                    "row_identity": row["row_identity"],
                    "original_image_sha256": row["original_image_sha256"],
                }
                for row in rows
            ]
        }
    )


def _random_time_plan_evidence(
    *,
    item_count: int,
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
            count = min(start + batch_size, item_count) - start
            noise = torch.randn(
                (count, 3, IMAGE_SIZE, IMAGE_SIZE),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            times = torch.rand(
                (count,),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
            time_digest.update(times.detach().cpu().contiguous().numpy().tobytes())
        mapping = make_global_shuffle_mapping(item_count=item_count, seed=seed)
        evidence = {
            "seed": seed,
            "noise_sha256": noise_digest.hexdigest(),
            "time_sha256": time_digest.hexdigest(),
            "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            "shuffle_mapping_sha256": _tensor_sha256(mapping),
        }
        per_seed.append({**evidence, "identity": _mapping_identity(evidence)})
    result: dict[str, Any] = {
        "item_count": item_count,
        "batch_size": batch_size,
        "image_shape": [3, IMAGE_SIZE, IMAGE_SIZE],
        "per_seed": per_seed,
    }
    result["identity"] = _mapping_identity(result)
    return result


def _fixed_time_plan_hashes(
    *,
    item_count: int,
    batch_size: int,
    seeds: Sequence[int],
    fixed_times: Sequence[float],
    device: torch.device,
) -> Mapping[tuple[int, float], tuple[str, str]]:
    result: dict[tuple[int, float], tuple[str, str]] = {}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        noise_digest = hashlib.sha256()
        for start in range(0, item_count, batch_size):
            count = min(start + batch_size, item_count) - start
            noise = torch.randn(
                (count, 3, IMAGE_SIZE, IMAGE_SIZE),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            noise_digest.update(noise.detach().cpu().contiguous().numpy().tobytes())
        noise_sha256 = noise_digest.hexdigest()
        for fixed_time in fixed_times:
            time_digest = hashlib.sha256()
            for start in range(0, item_count, batch_size):
                count = min(start + batch_size, item_count) - start
                values = torch.full((count,), fixed_time, dtype=torch.float32)
                time_digest.update(values.numpy().tobytes())
            result[(seed, float(fixed_time))] = (
                noise_sha256,
                time_digest.hexdigest(),
            )
    return result


def _sample_cell(
    *,
    model: nn.Module,
    split: LoadedOracleLadderSplit,
    source_dino: torch.Tensor,
    plan: OracleLadderSamplePlan,
    teacher: FrozenDINOGridTargets,
    device: torch.device,
    sample_chunk_size: int,
    dino_batch_size: int,
) -> Mapping[str, Any]:
    selected = torch.tensor(plan.indices, dtype=torch.long)
    conditions = split.conditions[selected]
    source_rgb = split.images_uint8[selected].float().div(127.5).sub(1)
    source_features = source_dino[selected]
    rows = tuple(split.rows[index] for index in plan.indices)
    sampling_evidence: list[Mapping[str, Any]] = []
    correct_rgb_values: list[torch.Tensor] = []
    shuffled_rgb_values: list[torch.Tensor] = []
    correct_dino_values: list[torch.Tensor] = []
    shuffled_dino_values: list[torch.Tensor] = []
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (len(plan.indices), 3, IMAGE_SIZE, IMAGE_SIZE),
            generator=generator,
        )
        mapping = make_global_shuffle_mapping(item_count=len(plan.indices), seed=seed)
        donor_rows = [rows[index]["row_identity"] for index in mapping.tolist()]
        evidence = {
            "seed": seed,
            "initial_noise_sha256": _tensor_sha256(noise),
            "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            "shuffle_mapping": mapping.tolist(),
            "shuffle_donor_row_identities": donor_rows,
            "ode_solver": "midpoint_euler",
            "ode_steps": SAMPLE_ODE_STEPS,
        }
        sampling_evidence.append(
            {**evidence, "identity": _mapping_identity(evidence)}
        )
        correct = sample_euler(
            model,
            _flatten_condition(conditions),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=sample_chunk_size,
        )
        shuffled = sample_euler(
            model,
            _flatten_condition(conditions[mapping]),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=sample_chunk_size,
        )
        correct_rgb_values.append(correct)
        shuffled_rgb_values.append(shuffled)
        correct_dino_values.append(
            _dino_generated_features(
                teacher,
                correct,
                device=device,
                batch_size=dino_batch_size,
            )
        )
        shuffled_dino_values.append(
            _dino_generated_features(
                teacher,
                shuffled,
                device=device,
                batch_size=dino_batch_size,
            )
        )
    metrics = sample_pair_metrics(
        rows=rows,
        noise_seeds=NOISE_SEEDS,
        source_rgb=source_rgb,
        source_dino=source_features,
        correct_rgb=torch.stack(correct_rgb_values, dim=1),
        shuffled_rgb=torch.stack(shuffled_rgb_values, dim=1),
        correct_dino=torch.stack(correct_dino_values, dim=1),
        shuffled_dino=torch.stack(shuffled_dino_values, dim=1),
    )
    report = {
        **{key: value for key, value in metrics.items() if key != "identity"},
        "sampling_evidence": sampling_evidence,
    }
    report["identity"] = _mapping_identity(report)
    return report


def _load_id198_visual_reference(
    *,
    summary_path: Path,
    external_report_path: Path,
) -> Mapping[str, Any]:
    for path, expected_sha in (
        (summary_path, ID198_SUMMARY_SHA256),
        (external_report_path, ID198_EXTERNAL_REPORT_SHA256),
    ):
        if (
            not path.is_absolute()
            or _has_symlink_component(path)
            or not path.is_file()
            or _sha256_file(path) != expected_sha
        ):
            raise ValueError("oracle-ladder immutable ID198 artifact hash mismatch")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        external = json.loads(external_report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-ladder immutable ID198 artifact is unreadable") from error
    roles = summary.get("roles") if isinstance(summary, Mapping) else None
    visual = external.get("visual_selection") if isinstance(external, Mapping) else None
    rows = external.get("rows") if isinstance(external, Mapping) else None
    if (
        summary.get("schema") != "nimloth_forensic_query_state_feature_report_v1"
        or summary.get("experiment_stage") != "stage_b_diagnostic"
        or summary.get("cache_fingerprint")
        != "9bd942267140aede839087b09bb0f755bd023ec69d216a8ad77ee845fc120899"
        or not isinstance(roles, Mapping)
        or roles.get("external_validation", {}).get("visual_count") != 16
        or not isinstance(visual, Mapping)
        or visual.get("indices") != list(STAGE_B_SAMPLE_INDICES)
        or visual.get("seed") != CONTACT_SAMPLE_SEED
        or not isinstance(rows, list)
        or len(rows) != 16
        or visual.get("row_identities")
        != [row.get("row_identity") for row in rows]
    ):
        raise ValueError("oracle-ladder immutable ID198 visual contract is invalid")
    pairs = [
        {
            "row_identity": row.get("row_identity"),
            "original_image_sha256": row.get("image_sha256"),
        }
        for row in rows
    ]
    pair_identity = _mapping_identity({"pairs": pairs})
    if (
        pair_identity != ID198_ROW_IMAGE_PAIR_IDENTITY
        or any(
            not _is_sha256(pair["row_identity"])
            or not _is_sha256(pair["original_image_sha256"])
            for pair in pairs
        )
    ):
        raise ValueError("oracle-ladder immutable ID198 row/image pair identity drift")
    return {
        "summary_path": str(summary_path),
        "summary_sha256": ID198_SUMMARY_SHA256,
        "external_report_path": str(external_report_path),
        "external_report_sha256": ID198_EXTERNAL_REPORT_SHA256,
        "row_image_pairs": pairs,
        "row_image_pair_identity": pair_identity,
    }


def _contact_samples(
    *,
    models: Mapping[str, nn.Module],
    splits: Mapping[str, LoadedOracleLadderSplit],
    device: torch.device,
    chunk_size: int,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    list[Image.Image],
    dict[str, list[Image.Image]],
    str,
]:
    indices = torch.tensor(STAGE_B_SAMPLE_INDICES, dtype=torch.long)
    if _sha256_tensor_bytes(indices) != STAGE_B_SAMPLE_INDICES_SHA256:
        raise ValueError("oracle-ladder contact-sheet index contract drift")
    generator = torch.Generator(device="cpu").manual_seed(CONTACT_SAMPLE_SEED)
    noise = torch.randn((16, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    rows = tuple(splits["state"].rows[index] for index in STAGE_B_SAMPLE_INDICES)
    originals = [
        Image.fromarray(
            splits["state"].images_uint8[index].permute(1, 2, 0).numpy(),
            mode="RGB",
        )
        for index in STAGE_B_SAMPLE_INDICES
    ]
    samples: dict[str, list[Image.Image]] = {}
    for cell in CELL_ORDER:
        family = str(CELLS[cell]["condition_family"])
        condition = splits[family].conditions[indices]
        output = sample_euler(
            models[cell],
            _flatten_condition(condition),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=chunk_size,
        )
        samples[cell] = [_tensor_to_pil(image) for image in output]
    noise_sha256 = _sha256_tensor_bytes(noise)
    if noise_sha256 != STAGE_B_INITIAL_NOISE_SHA256:
        raise ValueError("oracle-ladder contact-sheet initial noise contract drift")
    return rows, originals, samples, noise_sha256


def evaluate_oracle_ladder(args: argparse.Namespace) -> Mapping[str, Any]:
    """Evaluate all four final cells and publish one manifest-last report."""

    destination = args.output_dir
    checkpoint_paths = (
        args.token_state_checkpoint,
        args.token_oracle_checkpoint,
        args.spatial_state_checkpoint,
        args.spatial_oracle_checkpoint,
    )
    _reject_protected_output(
        destination,
        (
            args.state_cache,
            args.oracle_cache,
            args.id198_summary.parent,
            args.id198_external_report.parent,
            *(path.parent for path in checkpoint_paths),
        ),
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("oracle-ladder evaluation output already exists")
    if (
        args.dino_dtype not in {torch.float32, torch.float16, torch.bfloat16}
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (
                args.dino_batch_size,
                args.flow_batch_size,
                args.sample_chunk_size,
            )
        )
        or args.flow_batch_size != BATCH_SIZE
        or args.sample_chunk_size != 8
    ):
        raise ValueError("oracle-ladder evaluation dtype/batch/chunk config is invalid")
    source_commit = _current_source_commit()
    id198_reference = _load_id198_visual_reference(
        summary_path=args.id198_summary,
        external_report_path=args.id198_external_report,
    )
    state_train, state_external = load_oracle_ladder_splits(
        state_cache=args.state_cache,
        oracle_cache=args.oracle_cache,
        condition_family="state",
        image_size=IMAGE_SIZE,
    )
    oracle_train, oracle_external = load_oracle_ladder_splits(
        state_cache=args.state_cache,
        oracle_cache=args.oracle_cache,
        condition_family="dino_oracle",
        image_size=IMAGE_SIZE,
    )
    if tuple(row["row_identity"] for row in state_external.rows) != tuple(
        row["row_identity"] for row in oracle_external.rows
    ):
        raise ValueError("oracle-ladder state/oracle external row order mismatch")
    if args.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    teacher = FrozenDINOGridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=args.device,
        dtype=args.dino_dtype,
        grid_size=4,
        batch_size=args.dino_batch_size,
    )
    dino_owner = _validate_teacher(teacher)
    try:
        oracle_manifest = json.loads(
            (args.oracle_cache / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-ladder oracle-cache manifest is unreadable") from error
    if (
        not isinstance(oracle_manifest, Mapping)
        or oracle_manifest.get("dino") != dino_owner
    ):
        raise ValueError("oracle-ladder oracle-cache/evaluation DINO owner mismatch")
    if EXACT_DINO_FEATURE_IDENTITY != _mapping_identity(
        {
            "source": teacher.identity.source,
            "revision": teacher.identity.revision,
            "processor_fingerprint": teacher.identity.processor_fingerprint,
            "hidden_size": teacher.identity.hidden_size,
            "grid_size": teacher.grid_size,
        }
    ):
        raise ValueError("oracle-ladder DINO feature identity drift")
    models: dict[str, nn.Module] = {}
    checkpoint_meta: dict[str, Any] = {}
    baseline, baseline_invariants, baseline_sha = _load_baseline_checkpoint(
        args.token_state_checkpoint,
        device=args.device,
    )
    models["token_state"] = baseline
    checkpoint_meta["token_state"] = {
        "path": str(args.token_state_checkpoint.resolve()),
        "sha256": baseline_sha,
        "invariants": baseline_invariants,
    }
    for cell, path in (
        ("token_oracle", args.token_oracle_checkpoint),
        ("spatial_state", args.spatial_state_checkpoint),
        ("spatial_oracle", args.spatial_oracle_checkpoint),
    ):
        model, invariants, checkpoint_sha = _load_new_cell_checkpoint(
            path,
            cell=cell,
            device=args.device,
        )
        models[cell] = model
        checkpoint_meta[cell] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha,
            "invariants": invariants,
        }
    split_pairs = {
        "state": (state_train, state_external),
        "dino_oracle": (oracle_train, oracle_external),
    }
    splits = {family: pair[1] for family, pair in split_pairs.items()}
    plan = build_sample_plan(state_external.rows, count=SAMPLE_ROWS)
    matched_row_identity = _evaluation_row_set_identity(state_external.rows)
    random_time_plan = _random_time_plan_evidence(
        item_count=len(state_external),
        batch_size=args.flow_batch_size,
        seeds=NOISE_SEEDS,
        device=args.device,
    )
    baseline_expected = {
        "cache_fingerprint": state_train.cache_fingerprint,
        "train_items": len(state_train),
        "validation_items": len(state_external),
        "experiment_stage": FORENSIC_STAGE_B_EXPERIMENT_STAGE,
        "max_steps": FORENSIC_STAGE_B_FINAL_STEP,
    }
    if any(
        baseline_invariants.get(field) != value
        for field, value in baseline_expected.items()
    ):
        raise ValueError("oracle-ladder token_state baseline/cache identity mismatch")
    for cell in CELL_ORDER[1:]:
        invariants = checkpoint_meta[cell]["invariants"]
        condition_family = str(CELLS[cell]["condition_family"])
        train_split, external_split = split_pairs[condition_family]
        expected = {
            "cell": cell,
            "decoder_family": CELLS[cell]["decoder_family"],
            "condition_family": condition_family,
            "train_cache_schema": train_split.cache_schema,
            "train_cache_fingerprint": train_split.cache_fingerprint,
            "train_row_set_identity": train_split.row_set_identity,
            "validation_cache_schema": external_split.cache_schema,
            "validation_cache_fingerprint": external_split.cache_fingerprint,
            "validation_row_set_identity": external_split.row_set_identity,
            "source_state_cache_fingerprint": (
                external_split.source_state_cache_fingerprint
            ),
            "selection_identity": external_split.selection_identity,
            "sample_selection_identity": plan.identity,
            "source_commit": source_commit,
            "state_cache_path": str(args.state_cache.resolve()),
            "oracle_cache_path": str(args.oracle_cache.resolve()),
            "final_step": 4_000,
        }
        if any(invariants.get(field) != value for field, value in expected.items()):
            raise ValueError(
                f"oracle-ladder {cell} checkpoint/live cache identity mismatch"
            )
    cell_reports: dict[str, Any] = {}
    for cell in CELL_ORDER:
        condition_family = str(CELLS[cell]["condition_family"])
        split = splits[condition_family]
        random_time = evaluate_query_state_multi_noise_sensitivity(
            models[cell],
            split.conditions,
            split.images_uint8,
            args.device,
            batch_size=args.flow_batch_size,
            seeds=NOISE_SEEDS,
        )
        fixed_time = evaluate_fixed_time_sensitivity(
            models[cell],
            split.conditions,
            split.images_uint8,
            args.device,
            batch_size=args.flow_batch_size,
            seeds=NOISE_SEEDS,
            fixed_times=FIXED_TIMES,
        )
        sample = _sample_cell(
            model=models[cell],
            split=split,
            source_dino=oracle_external.conditions,
            plan=plan,
            teacher=teacher,
            device=args.device,
            sample_chunk_size=args.sample_chunk_size,
            dino_batch_size=args.dino_batch_size,
        )
        cell_reports[cell] = {
            "cell": cell,
            "decoder_family": CELLS[cell]["decoder_family"],
            "condition_family": condition_family,
            "checkpoint": checkpoint_meta[cell],
            "row_set_identity": matched_row_identity,
            "sample_selection_identity": plan.identity,
            "noise_seeds": list(NOISE_SEEDS),
            "random_time_plan": random_time_plan,
            "random_time_flow": random_time,
            "fixed_time_flow": fixed_time,
            "sample_metrics": sample,
            "sample_metrics_identity": sample["identity"],
            "rows": sample["rows"],
        }
    comparison = build_four_cell_comparison(
        cell_reports,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    rows, originals, contact_samples, contact_noise_sha256 = _contact_samples(
        models=models,
        splits=splits,
        device=args.device,
        chunk_size=args.sample_chunk_size,
    )
    resolved_contact_pairs = [
        {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["original_image_sha256"],
        }
        for row in rows
    ]
    if resolved_contact_pairs != id198_reference["row_image_pairs"]:
        raise ValueError("oracle-ladder contact rows differ from immutable ID198 pairs")
    state_manifest_path = args.state_cache / "manifest.json"
    oracle_manifest_path = args.oracle_cache / "manifest.json"
    if (
        not state_manifest_path.is_file()
        or state_manifest_path.is_symlink()
        or not oracle_manifest_path.is_file()
        or oracle_manifest_path.is_symlink()
    ):
        raise ValueError("oracle-ladder input cache manifests are incomplete")
    resolved_config = {
        "state_cache": str(args.state_cache.resolve()),
        "oracle_cache": str(args.oracle_cache.resolve()),
        "token_state_checkpoint": str(args.token_state_checkpoint.resolve()),
        "token_oracle_checkpoint": str(args.token_oracle_checkpoint.resolve()),
        "spatial_state_checkpoint": str(args.spatial_state_checkpoint.resolve()),
        "spatial_oracle_checkpoint": str(args.spatial_oracle_checkpoint.resolve()),
        "id198_summary": str(args.id198_summary.resolve()),
        "id198_external_report": str(args.id198_external_report.resolve()),
        "output": str(destination.resolve()),
        "device": str(args.device),
        "dino_dtype": str(args.dino_dtype).removeprefix("torch."),
        "dino_batch_size": args.dino_batch_size,
        "flow_batch_size": args.flow_batch_size,
        "sample_chunk_size": args.sample_chunk_size,
        "noise_seeds": list(NOISE_SEEDS),
        "fixed_times": list(FIXED_TIMES),
        "sample_rows": SAMPLE_ROWS,
        "sample_ode_steps": SAMPLE_ODE_STEPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "contact_sample_seed": CONTACT_SAMPLE_SEED,
    }
    producer = {
        "module": "nimloth.eval.query_state_oracle_ladder",
        "source_commit": source_commit,
        "resolved_config": resolved_config,
    }
    producer["identity"] = _mapping_identity(producer)
    input_artifacts = {
        "state_cache": {
            "path": str(args.state_cache.resolve()),
            "manifest_sha256": _sha256_file(state_manifest_path),
            "cache_fingerprint": state_external.cache_fingerprint,
        },
        "oracle_cache": {
            "path": str(args.oracle_cache.resolve()),
            "manifest_sha256": _sha256_file(oracle_manifest_path),
            "cache_fingerprint": oracle_external.cache_fingerprint,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent)
    )
    try:
        contact_metadata = save_four_cell_contact_sheet(
            staging / "contact_sheet.png",
            rows=rows,
            originals=originals,
            correct_samples=contact_samples,
            sample_seed=CONTACT_SAMPLE_SEED,
            sample_indices=STAGE_B_SAMPLE_INDICES,
            sample_indices_sha256=STAGE_B_SAMPLE_INDICES_SHA256,
            initial_noise_sha256=contact_noise_sha256,
        )
        report: dict[str, Any] = {
            "schema": ORACLE_LADDER_REPORT_SCHEMA,
            "status": "completed_four_cell_forensic_evaluation",
            "forensic_only": True,
            "unsafe_actor_checkpoint": True,
            "not_deployable": True,
            "not_sft1_pass": True,
            "not_sft2_ready": True,
            "dino_feature_identity": EXACT_DINO_FEATURE_IDENTITY,
            "dino_owner": dino_owner,
            "producer": producer,
            "input_artifacts": input_artifacts,
            "id198_visual_reference": id198_reference,
            "source_state_cache_fingerprint": state_external.source_state_cache_fingerprint,
            "oracle_cache_fingerprint": oracle_external.cache_fingerprint,
            "selection_identity": state_external.selection_identity,
            "external_items": EXTERNAL_ITEMS,
            "evaluation_row_set_identity": matched_row_identity,
            "sample_plan": {
                "algorithm": plan.algorithm,
                "identity": plan.identity,
                "count": len(plan.indices),
                "row_identities": list(plan.row_identities),
                "image_sha256": list(plan.image_sha256),
            },
            "cell_reports": cell_reports,
            "comparison": comparison,
            "contact_sheet": contact_metadata,
            "claim_scope": "forensic_representation_decodability_only",
            "actor_failure_verdict_unchanged": True,
        }
        report["artifact_identity"] = _mapping_identity(report)
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_noreplace(staging, destination)
        return validate_oracle_ladder_report(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    finally:
        for image in originals:
            image.close()
        for images in contact_samples.values():
            for image in images:
                image.close()


def _has_valid_identity(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_sha256(value.get("identity"))
        and _mapping_identity(
            {key: item for key, item in value.items() if key != "identity"}
        )
        == value.get("identity")
    )


def _finite_metric(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_random_time_evidence(
    flow: object,
    plan: object,
) -> None:
    if (
        not _has_valid_identity(flow)
        or flow.get("schema")
        != "nimloth_query_state_cfm_multi_noise_sensitivity_v1"
        or flow.get("seeds") != list(NOISE_SEEDS)
        or flow.get("num_items") != EXTERNAL_ITEMS
        or not isinstance(flow.get("per_seed"), list)
        or len(flow["per_seed"]) != len(NOISE_SEEDS)
        or not isinstance(flow.get("aggregate"), Mapping)
        or not _has_valid_identity(plan)
        or plan.get("item_count") != EXTERNAL_ITEMS
        or plan.get("batch_size") != BATCH_SIZE
        or plan.get("image_shape") != [3, IMAGE_SIZE, IMAGE_SIZE]
        or not isinstance(plan.get("per_seed"), list)
        or len(plan["per_seed"]) != len(NOISE_SEEDS)
    ):
        raise ValueError("oracle-ladder random-time evidence is invalid")
    metrics = (
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
    )
    for seed, flow_item, plan_item in zip(
        NOISE_SEEDS,
        flow["per_seed"],
        plan["per_seed"],
        strict=True,
    ):
        expected_mapping = make_global_shuffle_mapping(
            item_count=EXTERNAL_ITEMS,
            seed=seed,
        ).tolist()
        expected_shuffle_identity = _mapping_identity(
            {
                "algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
                "seed": seed,
                "indices": expected_mapping,
            }
        )
        if (
            not isinstance(flow_item, Mapping)
            or flow_item.get("noise_time_seed") != seed
            or flow_item.get("num_items") != EXTERNAL_ITEMS
            or flow_item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or flow_item.get("shuffle_indices") != expected_mapping
            or flow_item.get("shuffle_identity") != expected_shuffle_identity
            or any(not _finite_metric(flow_item.get(metric)) for metric in metrics)
            or not math.isclose(
                float(flow_item["shuffled_minus_correct"]),
                float(flow_item["shuffled_flow_mse"])
                - float(flow_item["correct_flow_mse"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(flow_item["shuffled_over_correct"]),
                float(flow_item["shuffled_flow_mse"])
                / max(float(flow_item["correct_flow_mse"]), 1e-12),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not _has_valid_identity(plan_item)
            or plan_item.get("seed") != seed
            or plan_item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or not _is_sha256(plan_item.get("noise_sha256"))
            or not _is_sha256(plan_item.get("time_sha256"))
            or not _is_sha256(plan_item.get("shuffle_mapping_sha256"))
        ):
            raise ValueError("oracle-ladder random-time per-seed evidence is invalid")
    for metric in metrics:
        aggregate = flow["aggregate"].get(metric)
        values = [float(item[metric]) for item in flow["per_seed"]]
        if aggregate != {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }:
            raise ValueError("oracle-ladder random-time aggregate is inconsistent")


def _validate_fixed_time_evidence(value: object) -> None:
    if (
        not _has_valid_identity(value)
        or value.get("schema") != "nimloth_query_state_oracle_ladder_fixed_time_v1"
        or value.get("num_items") != EXTERNAL_ITEMS
        or value.get("seeds") != list(NOISE_SEEDS)
        or value.get("fixed_times") != list(FIXED_TIMES)
        or not isinstance(value.get("per_seed_time"), list)
        or len(value["per_seed_time"]) != len(NOISE_SEEDS) * len(FIXED_TIMES)
    ):
        raise ValueError("oracle-ladder fixed-time evidence is invalid")
    expected_pairs = [
        (seed, fixed_time)
        for seed in NOISE_SEEDS
        for fixed_time in FIXED_TIMES
    ]
    metric_names = (
        "correct_flow_mse",
        "shuffled_flow_mse",
        "shuffled_minus_correct",
        "shuffled_over_correct",
    )
    for (seed, fixed_time), item in zip(
        expected_pairs,
        value["per_seed_time"],
        strict=True,
    ):
        mapping = make_global_shuffle_mapping(item_count=EXTERNAL_ITEMS, seed=seed)
        expected_shuffle_identity = _mapping_identity(
            {
                "seed": seed,
                "mapping": mapping.tolist(),
                "algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            }
        )
        if (
            not isinstance(item, Mapping)
            or item.get("seed") != seed
            or item.get("time") != fixed_time
            or item.get("target_rgb_fraction_in_model_input") != fixed_time
            or item.get("matched_noise") is not True
            or item.get("num_items") != EXTERNAL_ITEMS
            or item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or item.get("shuffle_identity") != expected_shuffle_identity
            or not _is_sha256(item.get("noise_sha256"))
            or not _is_sha256(item.get("time_sha256"))
            or any(not _finite_metric(item.get(name)) for name in metric_names)
            or not math.isclose(
                float(item["shuffled_minus_correct"]),
                float(item["shuffled_flow_mse"])
                - float(item["correct_flow_mse"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(item["shuffled_over_correct"]),
                float(item["shuffled_flow_mse"])
                / max(float(item["correct_flow_mse"]), 1e-12),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("oracle-ladder fixed-time item evidence is invalid")


def _validate_sample_metric_evidence(value: object) -> None:
    if (
        not _has_valid_identity(value)
        or value.get("schema") != "nimloth_query_state_oracle_ladder_sample_metrics_v1"
        or value.get("statistical_unit") != "external_validation_row"
        or value.get("seed_aggregation")
        != "mean_within_row_before_row_bootstrap"
        or value.get("noise_seeds") != list(NOISE_SEEDS)
        or value.get("source_dino_input_owner")
        != "canonical_original_archived_observation"
        or value.get("generated_dino_input_owner")
        != "decoder_generated_128px_srgb"
        or value.get("generated_dino_is_canonical_teacher_target") is not False
        or not isinstance(value.get("rows"), list)
        or len(value["rows"]) != SAMPLE_ROWS
        or not isinstance(value.get("paired_row_bootstrap"), Mapping)
        or not isinstance(value.get("sampling_evidence"), list)
        or len(value["sampling_evidence"]) != len(NOISE_SEEDS)
    ):
        raise ValueError("oracle-ladder sample metric evidence is invalid")
    metric_names = (
        "correct_rgb_l1",
        "shuffled_rgb_l1",
        "correct_rgb_rmse",
        "shuffled_rgb_rmse",
        "correct_dino_mse",
        "shuffled_dino_mse",
        "correct_dino_cosine",
        "shuffled_dino_cosine",
    )
    delta_names = (
        "rgb_l1_shuffled_minus_correct",
        "rgb_rmse_shuffled_minus_correct",
        "dino_mse_shuffled_minus_correct",
        "dino_cosine_correct_minus_shuffled",
    )
    row_ids: set[str] = set()
    for row in value["rows"]:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("row_identity"), str)
            or row["row_identity"] in row_ids
            or not _is_sha256(row.get("original_image_sha256"))
            or not isinstance(row.get("per_seed"), list)
            or len(row["per_seed"]) != len(NOISE_SEEDS)
            or any(not _finite_metric(row.get(name)) for name in (*metric_names, *delta_names))
        ):
            raise ValueError("oracle-ladder sample metric row is invalid")
        row_ids.add(row["row_identity"])
        for seed, per_seed in zip(NOISE_SEEDS, row["per_seed"], strict=True):
            if (
                not isinstance(per_seed, Mapping)
                or per_seed.get("seed") != seed
                or any(not _finite_metric(per_seed.get(name)) for name in metric_names)
            ):
                raise ValueError("oracle-ladder sample per-seed metric is invalid")
        for name in metric_names:
            expected_mean = sum(
                float(per_seed[name]) for per_seed in row["per_seed"]
            ) / len(NOISE_SEEDS)
            if not math.isclose(
                float(row[name]), expected_mean, rel_tol=1e-6, abs_tol=1e-7
            ):
                raise ValueError("oracle-ladder sample seed aggregation is inconsistent")
        expected_deltas = {
            "rgb_l1_shuffled_minus_correct": (
                row["shuffled_rgb_l1"] - row["correct_rgb_l1"]
            ),
            "rgb_rmse_shuffled_minus_correct": (
                row["shuffled_rgb_rmse"] - row["correct_rgb_rmse"]
            ),
            "dino_mse_shuffled_minus_correct": (
                row["shuffled_dino_mse"] - row["correct_dino_mse"]
            ),
            "dino_cosine_correct_minus_shuffled": (
                row["correct_dino_cosine"] - row["shuffled_dino_cosine"]
            ),
        }
        if any(
            not math.isclose(
                float(row[name]), float(expected), rel_tol=1e-9, abs_tol=1e-9
            )
            for name, expected in expected_deltas.items()
        ):
            raise ValueError("oracle-ladder sample paired metric delta is inconsistent")
    for name in delta_names:
        expected = paired_row_bootstrap(
            torch.tensor(
                [float(row[name]) for row in value["rows"]],
                dtype=torch.float64,
            ),
            seed=BOOTSTRAP_SEED,
            resamples=BOOTSTRAP_RESAMPLES,
        )
        if value["paired_row_bootstrap"].get(name) != expected:
            raise ValueError("oracle-ladder sample paired bootstrap is inconsistent")
    row_identities = [row["row_identity"] for row in value["rows"]]
    for seed, evidence in zip(NOISE_SEEDS, value["sampling_evidence"], strict=True):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_noise = torch.randn(
            (SAMPLE_ROWS, 3, IMAGE_SIZE, IMAGE_SIZE),
            generator=generator,
        )
        expected_mapping = make_global_shuffle_mapping(
            item_count=SAMPLE_ROWS,
            seed=seed,
        ).tolist()
        expected_donors = [row_identities[index] for index in expected_mapping]
        if (
            not _has_valid_identity(evidence)
            or evidence.get("seed") != seed
            or evidence.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or evidence.get("ode_solver") != "midpoint_euler"
            or evidence.get("ode_steps") != SAMPLE_ODE_STEPS
            or evidence.get("initial_noise_sha256") != _tensor_sha256(expected_noise)
            or evidence.get("shuffle_mapping") != expected_mapping
            or evidence.get("shuffle_donor_row_identities") != expected_donors
        ):
            raise ValueError("oracle-ladder pure-noise sampling evidence is invalid")


def _require_identity(payload: object, *, field: str = "identity") -> Mapping[str, Any]:
    if (
        not isinstance(payload, Mapping)
        or not _is_sha256(payload.get(field))
        or payload[field]
        != _mapping_identity({key: value for key, value in payload.items() if key != field})
    ):
        raise ValueError("oracle-ladder nested evidence identity is invalid")
    return payload


def _validate_random_and_fixed_evidence(
    cell_report: Mapping[str, Any],
) -> None:
    _validate_random_time_evidence(
        cell_report.get("random_time_flow"),
        cell_report.get("random_time_plan"),
    )
    _validate_fixed_time_evidence(cell_report.get("fixed_time_flow"))
    random_plan = _require_identity(cell_report.get("random_time_plan"))
    random_flow = _require_identity(cell_report.get("random_time_flow"))
    fixed_flow = _require_identity(cell_report.get("fixed_time_flow"))
    if (
        random_plan.get("item_count") != EXTERNAL_ITEMS
        or random_plan.get("image_shape") != [3, IMAGE_SIZE, IMAGE_SIZE]
        or not isinstance(random_plan.get("batch_size"), int)
        or random_plan["batch_size"] < 1
        or not isinstance(random_plan.get("per_seed"), list)
        or [item.get("seed") for item in random_plan["per_seed"]]
        != list(NOISE_SEEDS)
        or any(_require_identity(item) is None for item in random_plan["per_seed"])
        or random_flow.get("schema")
        != "nimloth_query_state_cfm_multi_noise_sensitivity_v1"
        or random_flow.get("seeds") != list(NOISE_SEEDS)
        or random_flow.get("num_items") != EXTERNAL_ITEMS
        or not isinstance(random_flow.get("per_seed"), list)
        or len(random_flow["per_seed"]) != len(NOISE_SEEDS)
        or fixed_flow.get("schema")
        != "nimloth_query_state_oracle_ladder_fixed_time_v1"
        or fixed_flow.get("seeds") != list(NOISE_SEEDS)
        or fixed_flow.get("fixed_times") != list(FIXED_TIMES)
        or fixed_flow.get("num_items") != EXTERNAL_ITEMS
        or not isinstance(fixed_flow.get("per_seed_time"), list)
        or len(fixed_flow["per_seed_time"]) != len(NOISE_SEEDS) * len(FIXED_TIMES)
    ):
        raise ValueError("oracle-ladder random/fixed-time evidence contract is invalid")
    for seed, item in zip(NOISE_SEEDS, random_plan["per_seed"], strict=True):
        expected_mapping = make_global_shuffle_mapping(
            item_count=EXTERNAL_ITEMS,
            seed=seed,
        )
        if (
            item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or item.get("shuffle_mapping_sha256") != _tensor_sha256(expected_mapping)
            or not _is_sha256(item.get("noise_sha256"))
            or not _is_sha256(item.get("time_sha256"))
        ):
            raise ValueError("oracle-ladder random-time plan evidence is invalid")
    for seed, item in zip(NOISE_SEEDS, random_flow["per_seed"], strict=True):
        expected_mapping = make_global_shuffle_mapping(
            item_count=EXTERNAL_ITEMS,
            seed=seed,
        ).tolist()
        if (
            not isinstance(item, Mapping)
            or item.get("noise_time_seed") != seed
            or item.get("num_items") != EXTERNAL_ITEMS
            or item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or item.get("shuffle_indices") != expected_mapping
            or not _is_sha256(item.get("shuffle_identity"))
        ):
            raise ValueError("oracle-ladder random-time per-seed evidence is invalid")
    expected_pairs = [(seed, value) for seed in NOISE_SEEDS for value in FIXED_TIMES]
    for item, (seed, fixed_time) in zip(
        fixed_flow["per_seed_time"], expected_pairs, strict=True
    ):
        if (
            not isinstance(item, Mapping)
            or item.get("seed") != seed
            or item.get("time") != fixed_time
            or item.get("target_rgb_fraction_in_model_input") != fixed_time
            or item.get("matched_noise") is not True
            or item.get("num_items") != EXTERNAL_ITEMS
            or not _is_sha256(item.get("noise_sha256"))
            or not _is_sha256(item.get("shuffle_identity"))
            or any(
                not isinstance(item.get(metric), (int, float))
                or not math.isfinite(float(item[metric]))
                for metric in (
                    "correct_flow_mse",
                    "shuffled_flow_mse",
                    "shuffled_minus_correct",
                    "shuffled_over_correct",
                )
            )
        ):
            raise ValueError("oracle-ladder fixed-time item evidence is invalid")


def _validate_sample_evidence(
    cell_report: Mapping[str, Any],
    sample_plan: Mapping[str, Any],
) -> None:
    sample = _require_identity(cell_report.get("sample_metrics"))
    _validate_sample_metric_evidence(sample)
    rows = sample.get("rows")
    sampling = sample.get("sampling_evidence")
    if (
        sample.get("schema") != "nimloth_query_state_oracle_ladder_sample_metrics_v1"
        or sample.get("statistical_unit") != "external_validation_row"
        or sample.get("seed_aggregation")
        != "mean_within_row_before_row_bootstrap"
        or sample.get("noise_seeds") != list(NOISE_SEEDS)
        or sample.get("source_dino_input_owner")
        != "canonical_original_archived_observation"
        or sample.get("generated_dino_input_owner") != "decoder_generated_128px_srgb"
        or sample.get("generated_dino_is_canonical_teacher_target") is not False
        or not isinstance(rows, list)
        or len(rows) != SAMPLE_ROWS
        or rows != cell_report.get("rows")
        or [row.get("row_identity") for row in rows]
        != sample_plan.get("row_identities")
        or [row.get("original_image_sha256") for row in rows]
        != sample_plan.get("image_sha256")
        or cell_report.get("sample_metrics_identity") != sample.get("identity")
        or not isinstance(sampling, list)
        or len(sampling) != len(NOISE_SEEDS)
        or [item.get("seed") for item in sampling] != list(NOISE_SEEDS)
    ):
        raise ValueError("oracle-ladder sample metric evidence contract is invalid")
    metric_names = (
        "correct_rgb_l1",
        "shuffled_rgb_l1",
        "correct_rgb_rmse",
        "shuffled_rgb_rmse",
        "correct_dino_mse",
        "shuffled_dino_mse",
        "correct_dino_cosine",
        "shuffled_dino_cosine",
    )
    for row in rows:
        per_seed = row.get("per_seed") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or not _is_sha256(row.get("original_image_sha256"))
            or not isinstance(per_seed, list)
            or len(per_seed) != len(NOISE_SEEDS)
            or [value.get("seed") for value in per_seed] != list(NOISE_SEEDS)
            or any(
                not isinstance(row.get(metric), (int, float))
                or not math.isfinite(float(row[metric]))
                for metric in metric_names
            )
        ):
            raise ValueError("oracle-ladder per-row sample metric evidence is invalid")
        if (
            not math.isclose(
                row["rgb_l1_shuffled_minus_correct"],
                row["shuffled_rgb_l1"] - row["correct_rgb_l1"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                row["dino_mse_shuffled_minus_correct"],
                row["shuffled_dino_mse"] - row["correct_dino_mse"],
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("oracle-ladder per-row paired metric delta is inconsistent")
    for seed, item in zip(NOISE_SEEDS, sampling, strict=True):
        _require_identity(item)
        expected_mapping = make_global_shuffle_mapping(
            item_count=SAMPLE_ROWS,
            seed=seed,
        ).tolist()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_noise = torch.randn(
            (SAMPLE_ROWS, 3, IMAGE_SIZE, IMAGE_SIZE),
            generator=generator,
        )
        expected_donors = [
            sample_plan["row_identities"][index] for index in expected_mapping
        ]
        if (
            item.get("ode_solver") != "midpoint_euler"
            or item.get("ode_steps") != SAMPLE_ODE_STEPS
            or item.get("initial_noise_sha256") != _tensor_sha256(expected_noise)
            or item.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or item.get("shuffle_mapping") != expected_mapping
            or item.get("shuffle_donor_row_identities") != expected_donors
        ):
            raise ValueError("oracle-ladder pure-noise sampling evidence is invalid")
    paired = sample.get("paired_row_bootstrap")
    if not isinstance(paired, Mapping) or set(paired) != {
        "rgb_l1_shuffled_minus_correct",
        "rgb_rmse_shuffled_minus_correct",
        "dino_mse_shuffled_minus_correct",
        "dino_cosine_correct_minus_shuffled",
    }:
        raise ValueError("oracle-ladder paired-row bootstrap evidence is incomplete")
    for value in paired.values():
        if (
            not isinstance(value, Mapping)
            or value.get("row_count") != SAMPLE_ROWS
            or value.get("seed") != BOOTSTRAP_SEED
            or value.get("resamples") != BOOTSTRAP_RESAMPLES
            or value.get("statistical_unit") != "external_validation_row"
        ):
            raise ValueError("oracle-ladder paired-row bootstrap contract is invalid")


def validate_oracle_ladder_report(root: str | Path) -> Mapping[str, Any]:
    """Strictly revalidate the committed four-cell report and live artifacts."""

    supplied = Path(root)
    if _has_symlink_component(supplied):
        raise ValueError("oracle-ladder report root must not traverse a symlink")
    report_root = supplied.resolve()
    manifest_path = report_root / "manifest.json"
    contact_path = report_root / "contact_sheet.png"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not contact_path.is_file()
        or contact_path.is_symlink()
    ):
        raise ValueError("oracle-ladder report manifest/contact sheet is incomplete")
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("oracle-ladder report manifest is unreadable") from error
    required = {
        "schema",
        "status",
        "forensic_only",
        "unsafe_actor_checkpoint",
        "not_deployable",
        "not_sft1_pass",
        "not_sft2_ready",
        "dino_feature_identity",
        "dino_owner",
        "producer",
        "input_artifacts",
        "id198_visual_reference",
        "source_state_cache_fingerprint",
        "oracle_cache_fingerprint",
        "selection_identity",
        "external_items",
        "evaluation_row_set_identity",
        "sample_plan",
        "cell_reports",
        "comparison",
        "contact_sheet",
        "claim_scope",
        "actor_failure_verdict_unchanged",
        "artifact_identity",
    }
    if (
        not isinstance(report, Mapping)
        or set(report) != required
        or report.get("schema") != ORACLE_LADDER_REPORT_SCHEMA
        or report.get("status") != "completed_four_cell_forensic_evaluation"
        or report.get("forensic_only") is not True
        or report.get("unsafe_actor_checkpoint") is not True
        or report.get("not_deployable") is not True
        or report.get("not_sft1_pass") is not True
        or report.get("not_sft2_ready") is not True
        or report.get("actor_failure_verdict_unchanged") is not True
        or report.get("claim_scope") != "forensic_representation_decodability_only"
        or report.get("dino_feature_identity") != EXACT_DINO_FEATURE_IDENTITY
        or report.get("external_items") != EXTERNAL_ITEMS
        or not _is_sha256(report.get("source_state_cache_fingerprint"))
        or not _is_sha256(report.get("oracle_cache_fingerprint"))
        or not _is_sha256(report.get("selection_identity"))
        or not _is_sha256(report.get("evaluation_row_set_identity"))
        or not _is_sha256(report.get("artifact_identity"))
        or report["artifact_identity"]
        != _mapping_identity(
            {key: value for key, value in report.items() if key != "artifact_identity"}
        )
    ):
        raise ValueError("oracle-ladder report identity/watermark is invalid")
    producer = _require_identity(report["producer"])
    config = producer.get("resolved_config")
    if (
        producer.get("module") != "nimloth.eval.query_state_oracle_ladder"
        or not _is_git_commit(producer.get("source_commit"))
        or not isinstance(config, Mapping)
        or set(config)
        != {
            "state_cache",
            "oracle_cache",
            "token_state_checkpoint",
            "token_oracle_checkpoint",
            "spatial_state_checkpoint",
            "spatial_oracle_checkpoint",
            "id198_summary",
            "id198_external_report",
            "output",
            "device",
            "dino_dtype",
            "dino_batch_size",
            "flow_batch_size",
            "sample_chunk_size",
            "noise_seeds",
            "fixed_times",
            "sample_rows",
            "sample_ode_steps",
            "bootstrap_seed",
            "bootstrap_resamples",
            "contact_sample_seed",
        }
        or config.get("output") != str(report_root)
        or config.get("noise_seeds") != list(NOISE_SEEDS)
        or config.get("fixed_times") != list(FIXED_TIMES)
        or config.get("sample_rows") != SAMPLE_ROWS
        or config.get("sample_ode_steps") != SAMPLE_ODE_STEPS
        or config.get("bootstrap_seed") != BOOTSTRAP_SEED
        or config.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or config.get("contact_sample_seed") != CONTACT_SAMPLE_SEED
        or config.get("flow_batch_size") != BATCH_SIZE
        or config.get("sample_chunk_size") != 8
        or config.get("dino_dtype") not in {"float32", "float16", "bfloat16"}
        or any(
            isinstance(config.get(field), bool)
            or not isinstance(config.get(field), int)
            or config[field] < 1
            for field in ("dino_batch_size", "flow_batch_size", "sample_chunk_size")
        )
    ):
        raise ValueError("oracle-ladder report producer/config identity is invalid")
    dino_owner = report.get("dino_owner")
    if (
        not isinstance(dino_owner, Mapping)
        or dino_owner.get("source") != DINOV2_LARGE_IDENTITY.source
        or dino_owner.get("revision") != DINOV2_LARGE_IDENTITY.revision
        or dino_owner.get("processor_fingerprint")
        != DINOV2_LARGE_IDENTITY.processor_fingerprint
        or dino_owner.get("hidden_size") != DINOV2_LARGE_IDENTITY.hidden_size
        or dino_owner.get("grid_size") != 4
        or dino_owner.get("feature_identity") != EXACT_DINO_FEATURE_IDENTITY
        or dino_owner.get("input_owner") != "original_archived_observation"
        or dino_owner.get("resize_before_processor") is not False
        or dino_owner.get("pooling")
        != "final_patch_tokens_adaptive_avg_pool2d_4x4_row_major"
        or dino_owner.get("model_dtype") != config["dino_dtype"]
        or dino_owner.get("output_dtype") != "float32"
        or dino_owner.get("batch_size") != config["dino_batch_size"]
    ):
        raise ValueError("oracle-ladder report DINO owner identity is invalid")
    inputs = report.get("input_artifacts")
    if not isinstance(inputs, Mapping) or set(inputs) != {"state_cache", "oracle_cache"}:
        raise ValueError("oracle-ladder report cache inputs are incomplete")
    for name, expected_fingerprint in (
        ("state_cache", report["source_state_cache_fingerprint"]),
        ("oracle_cache", report["oracle_cache_fingerprint"]),
    ):
        artifact = inputs[name]
        path = Path(str(artifact.get("path", "")))
        live_manifest = path / "manifest.json"
        try:
            manifest_value = json.loads(live_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("oracle-ladder live cache manifest is unreadable") from error
        if (
            not isinstance(artifact, Mapping)
            or set(artifact) != {"path", "manifest_sha256", "cache_fingerprint"}
            or not path.is_absolute()
            or _has_symlink_component(path)
            or not live_manifest.is_file()
            or live_manifest.is_symlink()
            or artifact.get("manifest_sha256") != _sha256_file(live_manifest)
            or artifact.get("cache_fingerprint") != expected_fingerprint
            or not isinstance(manifest_value, Mapping)
            or manifest_value.get("cache_fingerprint") != expected_fingerprint
        ):
            raise ValueError("oracle-ladder report live cache identity drift")
    if (
        config["state_cache"] != inputs["state_cache"]["path"]
        or config["oracle_cache"] != inputs["oracle_cache"]["path"]
    ):
        raise ValueError("oracle-ladder report producer/cache path mismatch")
    state_reader = ForensicQueryStateCacheDataset(Path(config["state_cache"]))
    oracle_reader = ForensicDinoOracleCacheDataset(Path(config["oracle_cache"]))
    if (
        state_reader.cache_fingerprint != report["source_state_cache_fingerprint"]
        or oracle_reader.cache_fingerprint != report["oracle_cache_fingerprint"]
        or oracle_reader.manifest["selection"]["identity"]
        != report["selection_identity"]
        or oracle_reader.manifest.get("dino") != dino_owner
        or oracle_reader.manifest["producer"]["source_commit"]
        != producer["source_commit"]
    ):
        raise ValueError("oracle-ladder strict live cache reader identity drift")
    del state_reader, oracle_reader
    id198_reference = report.get("id198_visual_reference")
    expected_id198 = _load_id198_visual_reference(
        summary_path=Path(str(config["id198_summary"])),
        external_report_path=Path(str(config["id198_external_report"])),
    )
    if id198_reference != expected_id198:
        raise ValueError("oracle-ladder report immutable ID198 reference drift")
    sample_plan = report.get("sample_plan")
    if (
        not isinstance(sample_plan, Mapping)
        or set(sample_plan)
        != {"algorithm", "identity", "count", "row_identities", "image_sha256"}
        or sample_plan.get("algorithm") != SAMPLE_SELECTION_ALGORITHM
        or sample_plan.get("count") != SAMPLE_ROWS
        or len(sample_plan.get("row_identities", ())) != SAMPLE_ROWS
        or len(sample_plan.get("image_sha256", ())) != SAMPLE_ROWS
        or sample_plan.get("identity")
        != _mapping_identity(
            {
                "algorithm": SAMPLE_SELECTION_ALGORITHM,
                "row_identities": sample_plan.get("row_identities"),
                "image_sha256": sample_plan.get("image_sha256"),
            }
        )
    ):
        raise ValueError("oracle-ladder report sample plan identity is invalid")
    evidence_device = torch.device(str(config["device"]))
    if evidence_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "oracle-ladder strict random-plan validation requires its CUDA device"
        )
    expected_random_plan = _random_time_plan_evidence(
        item_count=EXTERNAL_ITEMS,
        batch_size=BATCH_SIZE,
        seeds=NOISE_SEEDS,
        device=evidence_device,
    )
    expected_fixed_hashes = _fixed_time_plan_hashes(
        item_count=EXTERNAL_ITEMS,
        batch_size=BATCH_SIZE,
        seeds=NOISE_SEEDS,
        fixed_times=FIXED_TIMES,
        device=evidence_device,
    )
    cell_reports = report.get("cell_reports")
    if not isinstance(cell_reports, Mapping) or set(cell_reports) != set(CELL_ORDER):
        raise ValueError("oracle-ladder report requires exactly four cells")
    for cell in CELL_ORDER:
        cell_report = cell_reports[cell]
        checkpoint = cell_report.get("checkpoint") if isinstance(cell_report, Mapping) else None
        if (
            not isinstance(cell_report, Mapping)
            or set(cell_report)
            != {
                "cell",
                "decoder_family",
                "condition_family",
                "checkpoint",
                "row_set_identity",
                "sample_selection_identity",
                "noise_seeds",
                "random_time_plan",
                "random_time_flow",
                "fixed_time_flow",
                "sample_metrics",
                "sample_metrics_identity",
                "rows",
            }
            or cell_report.get("cell") != cell
            or cell_report.get("decoder_family") != CELLS[cell]["decoder_family"]
            or cell_report.get("condition_family") != CELLS[cell]["condition_family"]
            or cell_report.get("row_set_identity")
            != report["evaluation_row_set_identity"]
            or cell_report.get("sample_selection_identity") != sample_plan["identity"]
            or cell_report.get("noise_seeds") != list(NOISE_SEEDS)
            or cell_report.get("random_time_plan") != expected_random_plan
            or not isinstance(checkpoint, Mapping)
            or config[f"{cell}_checkpoint"] != checkpoint.get("path")
            or set(checkpoint) != {"path", "sha256", "invariants"}
            or not _is_sha256(checkpoint.get("sha256"))
        ):
            raise ValueError("oracle-ladder report cell/checkpoint identity is invalid")
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if (
            not checkpoint_path.is_absolute()
            or _has_symlink_component(checkpoint_path)
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != checkpoint["sha256"]
        ):
            raise ValueError("oracle-ladder report live checkpoint hash drift")
        if cell == "token_state":
            if checkpoint["sha256"] != FINAL_CHECKPOINT_SHA256:
                raise ValueError("oracle-ladder immutable baseline SHA256 drift")
            validated_model, validated_invariants, validated_sha = (
                _load_baseline_checkpoint(checkpoint_path, device=torch.device("cpu"))
            )
        else:
            invariants = checkpoint["invariants"]
            if not isinstance(invariants, Mapping) or not isinstance(
                invariants.get("cfm_config"), Mapping
            ):
                raise ValueError("oracle-ladder report checkpoint invariants are invalid")
            config_value = CFMConfig(**dict(invariants["cfm_config"]))
            validate_oracle_ladder_invariants(invariants, config_value)
            if (
                invariants.get("cell") != cell
                or invariants.get("source_commit") != producer["source_commit"]
                or invariants.get("state_cache_path") != config["state_cache"]
                or invariants.get("oracle_cache_path") != config["oracle_cache"]
                or invariants.get("output_dir")
                != str(checkpoint_path.parent.resolve())
            ):
                raise ValueError(
                    "oracle-ladder report checkpoint cell/source/output mismatch"
                )
            validated_model, validated_invariants, validated_sha = (
                _load_new_cell_checkpoint(
                    checkpoint_path,
                    cell=cell,
                    device=torch.device("cpu"),
                )
            )
        if (
            validated_sha != checkpoint["sha256"]
            or dict(validated_invariants) != checkpoint["invariants"]
        ):
            raise ValueError("oracle-ladder report checkpoint payload drift")
        del validated_model
        _validate_random_and_fixed_evidence(cell_report)
        for item in cell_report["fixed_time_flow"]["per_seed_time"]:
            expected_hashes = expected_fixed_hashes[(item["seed"], item["time"])]
            if (item["noise_sha256"], item["time_sha256"]) != expected_hashes:
                raise ValueError("oracle-ladder fixed-time noise/time hash mismatch")
        _validate_sample_evidence(cell_report, sample_plan)
    matched_random_plan = cell_reports[CELL_ORDER[0]]["random_time_plan"]
    matched_fixed_plan = [
        {
            key: item[key]
            for key in (
                "seed",
                "time",
                "target_rgb_fraction_in_model_input",
                "noise_sha256",
                "time_sha256",
                "shuffle_algorithm",
                "shuffle_identity",
                "num_items",
            )
        }
        for item in cell_reports[CELL_ORDER[0]]["fixed_time_flow"]["per_seed_time"]
    ]
    for cell in CELL_ORDER[1:]:
        candidate_fixed = [
            {
                key: item[key]
                for key in (
                    "seed",
                    "time",
                    "target_rgb_fraction_in_model_input",
                    "noise_sha256",
                    "time_sha256",
                    "shuffle_algorithm",
                    "shuffle_identity",
                    "num_items",
                )
            }
            for item in cell_reports[cell]["fixed_time_flow"]["per_seed_time"]
        ]
        if (
            cell_reports[cell]["random_time_plan"] != matched_random_plan
            or candidate_fixed != matched_fixed_plan
        ):
            raise ValueError("oracle-ladder cross-cell noise/time/shuffle plan mismatch")
    comparison = _require_identity(report.get("comparison"))
    expected_comparison = build_four_cell_comparison(
        cell_reports,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    if comparison != expected_comparison:
        raise ValueError("oracle-ladder four-cell comparison evidence drift")
    contact = _require_identity(report.get("contact_sheet"))
    contact_rows = contact.get("row_identities")
    contact_images = contact.get("image_sha256")
    pair_identity = _mapping_identity(
        {
            "indices_sha256": contact.get("indices_sha256"),
            "row_identities": contact_rows,
            "image_sha256": contact_images,
        }
    )
    if (
        contact.get("human_inspection_only") is not True
        or contact.get("controls_scientific_result") is not False
        or contact.get("columns") != ["original", *CELL_ORDER]
        or contact.get("indices") != list(STAGE_B_SAMPLE_INDICES)
        or contact.get("indices_sha256") != STAGE_B_SAMPLE_INDICES_SHA256
        or contact.get("sample_seed") != CONTACT_SAMPLE_SEED
        or contact.get("initial_noise_sha256") != STAGE_B_INITIAL_NOISE_SHA256
        or contact.get("ode_solver") != "midpoint_euler"
        or contact.get("ode_steps") != SAMPLE_ODE_STEPS
        or len(contact_rows or ()) != 16
        or len(contact_images or ()) != 16
        or contact.get("row_image_pair_identity") != pair_identity
        or contact.get("png_sha256") != _sha256_file(contact_path)
    ):
        raise ValueError("oracle-ladder contact-sheet evidence contract is invalid")
    return report


def _dtype(value: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported DINO dtype: {value}") from error


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the exact Formal38 Query-State CFM oracle ladder"
    )
    parser.add_argument("--state-cache", required=True, type=Path)
    parser.add_argument("--oracle-cache", required=True, type=Path)
    parser.add_argument("--token-state-checkpoint", required=True, type=Path)
    parser.add_argument("--token-oracle-checkpoint", required=True, type=Path)
    parser.add_argument("--spatial-state-checkpoint", required=True, type=Path)
    parser.add_argument("--spatial-oracle-checkpoint", required=True, type=Path)
    parser.add_argument("--id198-summary", required=True, type=Path)
    parser.add_argument("--id198-external-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=torch.device)
    parser.add_argument(
        "--dino-dtype",
        required=True,
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--dino-batch-size", required=True, type=int)
    parser.add_argument("--flow-batch-size", required=True, type=int)
    parser.add_argument("--sample-chunk-size", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    args.dino_dtype = _dtype(args.dino_dtype)
    report = evaluate_oracle_ladder(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CELL_ORDER",
    "ORACLE_LADDER_REPORT_SCHEMA",
    "OracleLadderSamplePlan",
    "build_cli_parser",
    "build_four_cell_comparison",
    "build_sample_plan",
    "evaluate_oracle_ladder",
    "paired_row_bootstrap",
    "sample_pair_metrics",
    "save_four_cell_contact_sheet",
    "validate_oracle_ladder_report",
]
