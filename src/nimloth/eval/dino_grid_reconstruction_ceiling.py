"""Strict matched grid4/grid8/grid16 direct-DINO reconstruction evaluation.

This read-only experiment owner evaluates final-step decoders from one matched
pure-noise plan. It establishes forensic representation decodability only and
does not authorize an experiment launch or change actor/SFT readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn

from nimloth.backbone.dino_grid import DINOV2_LARGE_IDENTITY, FrozenDINOMultigridTargets
from nimloth.eval.query_state_oracle_ladder import (
    ID198_INITIAL_NOISE_SHA256,
    ID198_SAMPLE_INDICES,
    ID198_SAMPLE_INDICES_SHA256,
    _load_id198_visual_reference,
    build_sample_plan,
    paired_row_bootstrap,
)
from nimloth.eval.query_state_oracle_ladder import (
    _load_new_cell_checkpoint as _load_grid4_checkpoint,
)
from nimloth.recon.cfm import CFMConfig, sample_euler
from nimloth.training.reconstruction.cfm_dino_grid_ceiling import (
    DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA,
    DINO_GRID_CEILING_TRAIN_METADATA_SCHEMA,
    DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA,
    EXTERNAL_ITEMS,
    FINAL_STEP,
    IMAGE_SIZE,
    NOISE_SEEDS,
    SAMPLE_ODE_STEPS,
    SAMPLE_ROWS,
    TRAIN_ITEMS,
    _flatten_condition,
    _has_symlink_component,
    _identity,
    _is_git_commit,
    _is_sha256,
    _reject_protected_output,
    _sha256_file,
    _validate_optimizer_invariants,
    build_decoder_optimizer,
    build_dino_grid_ceiling_model,
    load_dino_grid_ceiling_splits,
    validate_dino_grid_ceiling_invariants,
)
from nimloth.training.reconstruction.cfm_dino_grid_ceiling import (
    _split_identity as _multigrid_split_identity,
)
from nimloth.training.reconstruction.cfm_forensic_oracle_ladder import (
    ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA,
)
from nimloth.training.reconstruction.cfm_forensic_oracle_ladder import (
    _split_identity as _grid4_split_identity,
)
from nimloth.training.reconstruction.cfm_forensic_query_state import (
    _load_image_uint8,
    _sha256_tensor_bytes,
)
from nimloth.training.reconstruction.cfm_query_state import (
    QUERY_STATE_SHUFFLE_ALGORITHM,
    make_global_shuffle_mapping,
)
from nimloth.training.reconstruction.dino_grid_ceiling_cache import (
    DinoGridCeilingCacheDataset,
    _MetadataOnlyGrid4CacheDataset,
    _validate_teacher,
)
from nimloth.training.reconstruction.forensic_query_state_cache import (
    _publish_noreplace,
)

DINO_GRID_CEILING_REPORT_SCHEMA = "nimloth_dino_grid_reconstruction_ceiling_report_v1"
CELL_ORDER = ("grid4", "grid8", "grid16")
BOOTSTRAP_SEED = 20260951
BOOTSTRAP_RESAMPLES = 10_000
CONTACT_SAMPLE_SEED = 20260921
SAMPLE_CHUNK_SIZE = 8
_METRIC_DIRECTIONS = {
    "correct_rgb128_l1": False,
    "correct_rgb128_rmse": False,
    "correct_rgb32_l1": False,
    "correct_rgb32_rmse": False,
    "correct_rgb32_gradient_l1": False,
    "correct_rgb16_l1": False,
    "correct_rgb16_rmse": False,
    "correct_rgb16_gradient_l1": False,
    "correct_dino4_mse": False,
    "correct_dino4_cosine": True,
    "correct_dino16_mse": False,
    "correct_dino16_cosine": True,
}
_ALL_SAMPLE_METRICS = {
    f"{prefix}_{name.removeprefix('correct_')}"
    for prefix in ("correct", "shuffled")
    for name in _METRIC_DIRECTIONS
}
_COMPARISON_DIRECTIONS = {
    f"{prefix}_{name.removeprefix('correct_')}": higher
    for prefix in ("correct", "shuffled")
    for name, higher in _METRIC_DIRECTIONS.items()
}


@dataclass(frozen=True)
class _Grid4EvaluationSplit:
    conditions: torch.Tensor
    images_uint8: torch.Tensor
    rows: tuple[dict[str, Any], ...]
    cache_fingerprint: str
    embedded_state_cache_fingerprint: str
    selection_identity: str
    split_name: str
    row_set_identity: str


def _load_grid4_evaluation_splits(
    grid4_cache: str | Path,
) -> tuple[_Grid4EvaluationSplit, _Grid4EvaluationSplit]:
    """Load grid4 train/external rows without opening the embedded state cache."""

    root = Path(grid4_cache)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("grid4 cache must be an absolute regular directory")
    dataset = _MetadataOnlyGrid4CacheDataset(root)
    grouped: dict[str, list[dict[str, Any]]] = {
        "all_train": [],
        "external_validation": [],
    }
    for index in range(len(dataset)):
        item = dataset[index]
        role = str(item.get("selection_role"))
        if role not in grouped:
            raise ValueError("grid4 cache contains a cross-stage role")
        grouped[role].append(item)
    if {role: len(items) for role, items in grouped.items()} != {
        "all_train": TRAIN_ITEMS,
        "external_validation": EXTERNAL_ITEMS,
    }:
        raise ValueError("grid4 cache train/external counts are not exact")
    cache_fingerprint = str(dataset.manifest["cache_fingerprint"])
    selection_identity = str(dataset.manifest["selection"]["identity"])
    embedded = str(
        dataset.manifest["source_state_cache"]["cache_fingerprint"]
    )
    result: list[_Grid4EvaluationSplit] = []
    for role in ("all_train", "external_validation"):
        items = grouped[role]
        if role == "external_validation":
            conditions = torch.stack(
                [item.pop("condition") for item in items]
            ).contiguous()
        else:
            for item in items:
                item.pop("condition")
            conditions = torch.empty(0, 16, 1024, dtype=torch.float32)
        rows = tuple(dict(item) for item in items)
        images = (
            torch.stack(
                [
                    _load_image_uint8(str(row["original_image_path"]), IMAGE_SIZE)
                    for row in rows
                ]
            )
            if role == "external_validation"
            else torch.empty(0, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.uint8)
        )
        result.append(
            _Grid4EvaluationSplit(
                conditions=conditions,
                images_uint8=images,
                rows=rows,
                cache_fingerprint=cache_fingerprint,
                embedded_state_cache_fingerprint=embedded,
                selection_identity=selection_identity,
                split_name=role,
                row_set_identity=_grid4_split_identity(
                    cache_fingerprint=cache_fingerprint,
                    selection_identity=selection_identity,
                    role=role,
                    rows=rows,
                ),
            )
        )
    return result[0], result[1]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _mapping_identity(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _current_source_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("DINO-grid evaluator cannot authenticate source commit") from error
    commit = result.stdout.strip()
    if not _is_git_commit(commit):
        raise RuntimeError("DINO-grid evaluator source commit is invalid")
    return commit


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(
        f"{tensor.dtype}:{tuple(tensor.shape)}:".encode() + tensor.numpy().tobytes()
    ).hexdigest()


def _validate_metric_tensors(
    *, rows: Sequence[Mapping[str, Any]], noise_seeds: Sequence[int],
    source_rgb: torch.Tensor, correct_rgb: torch.Tensor, shuffled_rgb: torch.Tensor,
    source_dino: Mapping[int, torch.Tensor],
    correct_dino: Mapping[int, torch.Tensor],
    shuffled_dino: Mapping[int, torch.Tensor],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[int, ...]]:
    row_values = tuple(rows)
    seeds = tuple(noise_seeds)
    row_count = len(row_values)
    seed_count = len(seeds)
    if (
        row_count < 1
        or seed_count < 1
        or len(set(seeds)) != seed_count
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or source_rgb.ndim != 4
        or tuple(source_rgb.shape[:2]) != (row_count, 3)
        or correct_rgb.shape != (row_count, seed_count, *source_rgb.shape[1:])
        or shuffled_rgb.shape != correct_rgb.shape
        or set(source_dino) != {4, 16}
        or set(correct_dino) != {4, 16}
        or set(shuffled_dino) != {4, 16}
        or any(
            not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())
            for tensor in (source_rgb, correct_rgb, shuffled_rgb)
        )
    ):
        raise ValueError("DINO-grid multiscale metric tensor contract is invalid")
    for grid_size in (4, 16):
        source = source_dino[grid_size]
        correct = correct_dino[grid_size]
        shuffled = shuffled_dino[grid_size]
        if (
            source.ndim != 3
            or source.shape[0] != row_count
            or source.shape[1] != grid_size**2
            or correct.shape != (row_count, seed_count, *source.shape[1:])
            or shuffled.shape != correct.shape
            or any(
                not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())
                for tensor in (source, correct, shuffled)
            )
        ):
            raise ValueError(f"DINO-grid grid{grid_size} metric owner shape is invalid")
    for row in row_values:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("row_identity"), str)
            or not row["row_identity"]
            or not _is_sha256(row.get("original_image_sha256"))
        ):
            raise ValueError("DINO-grid metric rows require image-bound identity")
    return row_values, seeds


def _rgb_scale_metrics(
    source: torch.Tensor, generated: torch.Tensor, *, size: int,
) -> dict[str, torch.Tensor]:
    rows, seeds = generated.shape[:2]
    source_scaled = torch.nn.functional.interpolate(
        source, size=(size, size), mode="area"
    )
    generated_scaled = torch.nn.functional.interpolate(
        generated.reshape(rows * seeds, *generated.shape[2:]),
        size=(size, size),
        mode="area",
    ).reshape(rows, seeds, 3, size, size)
    difference = generated_scaled - source_scaled[:, None]
    horizontal = (
        generated_scaled[..., :, 1:] - generated_scaled[..., :, :-1]
        - (source_scaled[..., :, 1:] - source_scaled[..., :, :-1])[:, None]
    )
    vertical = (
        generated_scaled[..., 1:, :] - generated_scaled[..., :-1, :]
        - (source_scaled[..., 1:, :] - source_scaled[..., :-1, :])[:, None]
    )
    return {
        "l1": difference.abs().flatten(2).mean(2),
        "rmse": difference.square().flatten(2).mean(2).sqrt(),
        "gradient_l1": 0.5
        * (
            horizontal.abs().flatten(2).mean(2)
            + vertical.abs().flatten(2).mean(2)
        ),
    }


def _dino_metrics(source: torch.Tensor, generated: torch.Tensor) -> dict[str, torch.Tensor]:
    expanded = source[:, None].expand_as(generated)
    return {
        "mse": (generated - expanded).square().flatten(2).mean(2),
        "cosine": torch.nn.functional.cosine_similarity(
            generated.flatten(2), expanded.flatten(2), dim=2, eps=1e-12
        ),
    }


def sample_multiscale_metrics(
    *, rows: Sequence[Mapping[str, Any]], noise_seeds: Sequence[int],
    source_rgb: torch.Tensor, correct_rgb: torch.Tensor, shuffled_rgb: torch.Tensor,
    source_dino: Mapping[int, torch.Tensor],
    correct_dino: Mapping[int, torch.Tensor],
    shuffled_dino: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    """Average all preregistered noise seeds within each external row."""

    row_values, seeds = _validate_metric_tensors(
        rows=rows,
        noise_seeds=noise_seeds,
        source_rgb=source_rgb,
        correct_rgb=correct_rgb,
        shuffled_rgb=shuffled_rgb,
        source_dino=source_dino,
        correct_dino=correct_dino,
        shuffled_dino=shuffled_dino,
    )
    values: dict[str, torch.Tensor] = {}
    for prefix, generated in (("correct", correct_rgb), ("shuffled", shuffled_rgb)):
        source_expanded = source_rgb[:, None]
        difference = generated - source_expanded
        values[f"{prefix}_rgb128_l1"] = difference.abs().flatten(2).mean(2)
        values[f"{prefix}_rgb128_rmse"] = difference.square().flatten(2).mean(2).sqrt()
        for size in (32, 16):
            metrics = _rgb_scale_metrics(source_rgb, generated, size=size)
            values[f"{prefix}_rgb{size}_l1"] = metrics["l1"]
            values[f"{prefix}_rgb{size}_rmse"] = metrics["rmse"]
            values[f"{prefix}_rgb{size}_gradient_l1"] = metrics["gradient_l1"]
    for prefix, generated_map in (("correct", correct_dino), ("shuffled", shuffled_dino)):
        for grid_size in (4, 16):
            metrics = _dino_metrics(source_dino[grid_size], generated_map[grid_size])
            values[f"{prefix}_dino{grid_size}_mse"] = metrics["mse"]
            values[f"{prefix}_dino{grid_size}_cosine"] = metrics["cosine"]

    result_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(row_values):
        item: dict[str, Any] = {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["original_image_sha256"],
            "per_seed": [
                {
                    "seed": seed,
                    **{
                        name: float(metric[row_index, seed_index])
                        for name, metric in values.items()
                    },
                }
                for seed_index, seed in enumerate(seeds)
            ],
            **{
                name: float(metric[row_index].mean())
                for name, metric in values.items()
            },
        }
        result_rows.append(item)
    report: dict[str, Any] = {
        "schema": "nimloth_dino_grid_ceiling_multiscale_metrics_v1",
        "statistical_unit": "external_validation_row",
        "seed_aggregation": "mean_within_row_before_row_bootstrap",
        "noise_seeds": list(seeds),
        "source_dino_input_owner": "original_archived_observation",
        "generated_dino_input_owner": "decoder_generated_128px_srgb",
        "generated_dino_is_canonical_teacher_target": False,
        "aggregate": {
            name: {"mean": float(metric.mean())}
            for name, metric in values.items()
        },
        "rows": result_rows,
    }
    report["identity"] = _mapping_identity(report)
    return report


def _metric_effect(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
    *, metric: str, higher_is_better: bool, bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    baseline_values = torch.tensor(
        [float(row[metric]) for row in baseline], dtype=torch.float64
    )
    candidate_values = torch.tensor(
        [float(row[metric]) for row in candidate], dtype=torch.float64
    )
    differences = (
        candidate_values - baseline_values
        if higher_is_better
        else baseline_values - candidate_values
    )
    result = paired_row_bootstrap(
        differences, seed=bootstrap_seed, resamples=bootstrap_resamples
    )
    return {**result, "direction": "positive_means_candidate_improves_over_grid4"}


def build_three_grid_comparison(
    reports: Mapping[str, Mapping[str, Any]], *, bootstrap_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Build all preregistered paired-row effects against immutable grid4."""

    if set(reports) != set(CELL_ORDER):
        raise ValueError("DINO-grid comparison requires exactly grid4/grid8/grid16")
    row_sets = {str(report.get("row_set_identity")) for report in reports.values()}
    selections = {
        str(report.get("sample_selection_identity")) for report in reports.values()
    }
    seed_sets = {tuple(report.get("noise_seeds", ())) for report in reports.values()}
    row_orders = {
        tuple(str(row.get("row_identity")) for row in report.get("rows", ()))
        for report in reports.values()
    }
    image_orders = {
        tuple(str(row.get("original_image_sha256")) for row in report.get("rows", ()))
        for report in reports.values()
    }
    if (
        len(row_sets) != 1
        or not _is_sha256(next(iter(row_sets)))
        or len(selections) != 1
        or not _is_sha256(next(iter(selections)))
        or len(seed_sets) != 1
        or len(row_orders) != 1
        or len(image_orders) != 1
        or not next(iter(row_orders))
        or any(reports[cell].get("cell") != cell for cell in CELL_ORDER)
    ):
        raise ValueError("DINO-grid cells require matched seed/selection/row evidence")
    baseline = reports["grid4"]["rows"]
    available_directions = {
        metric: higher
        for metric, higher in _COMPARISON_DIRECTIONS.items()
        if all(
            metric in row
            for report in reports.values()
            for row in report["rows"]
        )
    }
    if not set(_METRIC_DIRECTIONS).issubset(available_directions):
        raise ValueError("DINO-grid comparison lacks preregistered correct-condition metrics")
    comparisons: dict[str, Any] = {}
    for cell in ("grid8", "grid16"):
        effects = {
            metric.removeprefix("correct_") + "_improvement": _metric_effect(
                baseline,
                reports[cell]["rows"],
                metric=metric,
                higher_is_better=higher,
                bootstrap_seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            for metric, higher in available_directions.items()
        }
        effects["statistical_pass"] = all(
            effects[name]["lower_95"] > 0
            for name in (
                "rgb32_l1_improvement",
                "rgb32_gradient_l1_improvement",
            )
        )
        comparisons[f"{cell}_vs_grid4"] = effects
    result: dict[str, Any] = {
        "schema": "nimloth_dino_grid_ceiling_three_grid_comparison_v1",
        "cells": list(CELL_ORDER),
        "statistical_unit": "external_validation_row",
        "seed_aggregation": "mean_within_row_before_row_bootstrap",
        "matched_row_set_identity": next(iter(row_sets)),
        "sample_selection_identity": next(iter(selections)),
        "noise_seeds": list(next(iter(seed_sets))),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "comparisons": comparisons,
        "decision_contract": {
            "human_fixed_rows": 16,
            "human_required_recognizable": 12,
            "statistical_metrics": [
                "rgb32_l1_improvement",
                "rgb32_gradient_l1_improvement",
            ],
            "statistical_rule": "both_paired_95ci_lower_bounds_gt_zero",
            "recommendation": "smallest_grid_passing_human_and_statistical_rules",
        },
        "claim_scope": "forensic_representation_decodability_only",
    }
    result["identity"] = _mapping_identity(result)
    return result


def apply_human_structure_judgments(
    comparison: Mapping[str, Any], *, judgments: Mapping[str, Sequence[bool]],
) -> dict[str, Any]:
    """Apply the fixed 12/16 human gate without changing statistical evidence."""

    if (
        comparison.get("schema")
        != "nimloth_dino_grid_ceiling_three_grid_comparison_v1"
        or comparison.get("identity")
        != _mapping_identity(
            {key: value for key, value in comparison.items() if key != "identity"}
        )
        or comparison.get("decision_contract")
        != {
            "human_fixed_rows": 16,
            "human_required_recognizable": 12,
            "statistical_metrics": [
                "rgb32_l1_improvement",
                "rgb32_gradient_l1_improvement",
            ],
            "statistical_rule": "both_paired_95ci_lower_bounds_gt_zero",
            "recommendation": "smallest_grid_passing_human_and_statistical_rules",
        }
        or set(judgments) != {"grid8", "grid16"}
    ):
        raise ValueError("DINO-grid human judgment contract is invalid")
    grids: dict[str, Any] = {}
    for cell in ("grid8", "grid16"):
        values = tuple(judgments[cell])
        if len(values) != 16 or any(type(value) is not bool for value in values):
            raise ValueError("DINO-grid requires exactly 16 boolean judgments per candidate")
        recognizable = sum(values)
        statistical = comparison["comparisons"][f"{cell}_vs_grid4"]["statistical_pass"]
        grids[cell] = {
            "grid_size": int(cell.removeprefix("grid")),
            "human_recognizable": recognizable,
            "human_pass": recognizable >= 12,
            "statistical_pass": statistical,
            "passed": recognizable >= 12 and statistical,
            "judgments": list(values),
        }
    recommended = next(
        (grids[cell]["grid_size"] for cell in ("grid8", "grid16") if grids[cell]["passed"]),
        None,
    )
    result: dict[str, Any] = {
        "schema": "nimloth_dino_grid_ceiling_human_decision_v1",
        "grids": grids,
        "recommended_grid": recommended,
        "no_candidate_passed": recommended is None,
        "claim_scope": "forensic_representation_decodability_only",
    }
    result["identity"] = _mapping_identity(result)
    return result


def save_three_grid_contact_sheet(
    output: str | Path, *, rows: Sequence[Mapping[str, Any]],
    originals: Sequence[Image.Image],
    correct_samples: Mapping[str, Sequence[Image.Image]],
    sample_indices: Sequence[int], sample_seed: int,
    initial_noise_sha256: str,
) -> dict[str, Any]:
    """Publish fixed original/grid4/grid8/grid16 columns for human scoring."""

    destination = Path(output)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"DINO-grid contact sheet already exists: {destination}")
    row_values = tuple(rows)
    original_values = tuple(originals)
    indices = tuple(sample_indices)
    if (
        not row_values
        or len(row_values) != len(original_values)
        or len(indices) != len(row_values)
        or set(correct_samples) != set(CELL_ORDER)
        or any(len(correct_samples[cell]) != len(row_values) for cell in CELL_ORDER)
        or any(
            not isinstance(row.get("row_identity"), str)
            or not _is_sha256(row.get("original_image_sha256"))
            for row in row_values
        )
        or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
        or isinstance(sample_seed, bool)
        or not isinstance(sample_seed, int)
        or not _is_sha256(initial_noise_sha256)
    ):
        raise ValueError("DINO-grid contact sheet row/cell/seed contract is invalid")
    all_images = [*original_values]
    for cell in CELL_ORDER:
        all_images.extend(correct_samples[cell])
    size = original_values[0].size
    if any(image.mode != "RGB" or image.size != size for image in all_images):
        raise ValueError("DINO-grid contact sheet images must share RGB size")
    header = 18
    sheet = Image.new(
        "RGB", (size[0] * 4, header + size[1] * len(row_values)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for column_index, label in enumerate(("original", *CELL_ORDER)):
        draw.text((column_index * size[0] + 1, 1), label, fill="black")
    for row_index, original in enumerate(original_values):
        for column_index, image in enumerate(
            (original, *(correct_samples[cell][row_index] for cell in CELL_ORDER))
        ):
            sheet.paste(
                image, (column_index * size[0], header + row_index * size[1])
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            sheet.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        sheet.close()
    metadata: dict[str, Any] = {
        "columns": ["original", "grid4", "grid8", "grid16"],
        "indices": list(indices),
        "row_identities": [row["row_identity"] for row in row_values],
        "image_sha256": [row["original_image_sha256"] for row in row_values],
        "sample_seed": sample_seed,
        "initial_noise_sha256": initial_noise_sha256,
        "ode_solver": "midpoint_euler",
        "ode_steps": SAMPLE_ODE_STEPS,
        "human_judgment_rule": "at_least_12_of_fixed_16",
        "human_judgment_target": (
            "corresponding_wall_floor_boundary_door_window_or_"
            "large_furniture_placement"
        ),
        "controls_scientific_result": True,
        "png_sha256": _sha256_file(destination),
    }
    metadata["identity"] = _mapping_identity(metadata)
    return metadata


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    array = (
        value.detach().cpu().float().clamp(-1, 1).add(1).mul(127.5).round()
        .byte().permute(1, 2, 0).numpy()
    )
    return Image.fromarray(array, mode="RGB")


@torch.inference_mode()
def _generated_multigrid_features(
    teacher: FrozenDINOMultigridTargets, images: torch.Tensor, *,
    device: torch.device, batch_size: int,
) -> dict[int, torch.Tensor]:
    outputs: dict[int, list[torch.Tensor]] = {4: [], 16: []}
    for start in range(0, len(images), batch_size):
        pil_images = [_tensor_to_pil(image) for image in images[start : start + batch_size]]
        try:
            processed = teacher.image_processor(images=pil_images, return_tensors="pt")
            parameter = next(teacher.model.parameters())
            pixels = processed["pixel_values"].to(device=parameter.device, dtype=parameter.dtype)
            hidden = teacher.model(pixel_values=pixels).last_hidden_state
            patch_size = int(teacher.model.config.patch_size)
            height = int(pixels.shape[-2]) // patch_size
            width = int(pixels.shape[-1]) // patch_size
            if (height, width) != (37, 37):
                raise ValueError("generated-image DINO scoring requires native37 geometry")
            native = hidden[:, -(height * width) :, :].reshape(
                len(pil_images), height, width, DINOV2_LARGE_IDENTITY.hidden_size
            ).permute(0, 3, 1, 2).float()
            for grid_size in (4, 16):
                pooled = torch.nn.functional.adaptive_avg_pool2d(
                    native, (grid_size, grid_size)
                )
                outputs[grid_size].append(
                    pooled.permute(0, 2, 3, 1).reshape(
                        len(pil_images), grid_size**2, DINOV2_LARGE_IDENTITY.hidden_size
                    ).detach().cpu().contiguous()
                )
        finally:
            for image in pil_images:
                image.close()
    return {grid: torch.cat(values, dim=0) for grid, values in outputs.items()}


def _load_new_checkpoint(
    path: Path, *, cell: str, device: torch.device,
) -> tuple[nn.Module, Mapping[str, Any], str]:
    if not path.is_absolute() or _has_symlink_component(path) or not path.is_file():
        raise ValueError("DINO-grid final checkpoint must be an absolute regular file")
    sha256 = _sha256_file(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("DINO-grid final checkpoint is unreadable") from error
    invariants = payload.get("invariants") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or set(payload)
        != {"schema", "model", "optimizer", "step", "invariants", "torch_rng_state", "cuda_rng_state_all"}
        or payload.get("schema") != DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA
        or payload.get("step") != FINAL_STEP
        or not isinstance(invariants, Mapping)
        or invariants.get("cell") != cell
        or not isinstance(invariants.get("cfm_config"), Mapping)
    ):
        raise ValueError("DINO-grid final checkpoint cell/grid identity is invalid")
    config = CFMConfig(**dict(invariants["cfm_config"]))
    validate_dino_grid_ceiling_invariants(invariants, config)
    if invariants.get("output_dir") != str(path.parent.resolve()):
        raise ValueError("DINO-grid checkpoint output identity mismatch")
    model = build_dino_grid_ceiling_model(config).to(device)
    optimizer = build_decoder_optimizer(
        model,
        learning_rate=float(invariants["learning_rate"]),
        weight_decay=float(invariants["weight_decay"]),
    )
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        _validate_optimizer_invariants(optimizer, invariants)
    except Exception as error:
        raise ValueError("DINO-grid final decoder/optimizer state is invalid") from error
    metadata_path = path.parent / "metadata.json"
    summary_path = path.parent / "summary.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DINO-grid final checkpoint owner evidence is unreadable") from error
    producer = metadata.get("producer") if isinstance(metadata, Mapping) else None
    if (
        metadata_path.is_symlink()
        or summary_path.is_symlink()
        or metadata.get("schema") != DINO_GRID_CEILING_TRAIN_METADATA_SCHEMA
        or metadata.get("cell") != cell
        or metadata.get("invariants") != invariants
        or not isinstance(producer, Mapping)
        or producer.get("module")
        != "nimloth.training.reconstruction.cfm_dino_grid_ceiling"
        or producer.get("identity")
        != _identity({key: value for key, value in producer.items() if key != "identity"})
        or summary.get("schema") != DINO_GRID_CEILING_TRAIN_SUMMARY_SCHEMA
        or summary.get("status") != "completed_final_step4000"
        or summary.get("cell") != cell
        or summary.get("final_step") != FINAL_STEP
        or summary.get("final_checkpoint") != str(path.resolve())
        or summary.get("final_checkpoint_sha256") != sha256
        or summary.get("invariants_identity") != _identity(invariants)
        or summary.get("metadata_identity") != _identity(metadata)
        or summary.get("producer_identity") != producer.get("identity")
        or summary.get("evaluation_pending") is not True
        or summary.get("additional_steps_or_fallback_allowed") is not False
        or summary.get("identity")
        != _identity({key: value for key, value in summary.items() if key != "identity"})
    ):
        raise ValueError("DINO-grid final checkpoint summary/metadata identity is invalid")
    model.eval().requires_grad_(False)
    return model, invariants, sha256


def _sample_cell(
    *, model: nn.Module, conditions: torch.Tensor, source_rgb: torch.Tensor,
    source_dino: Mapping[int, torch.Tensor], rows: Sequence[Mapping[str, Any]],
    teacher: FrozenDINOMultigridTargets, device: torch.device,
    dino_batch_size: int, sample_chunk_size: int,
) -> dict[str, Any]:
    correct_values: list[torch.Tensor] = []
    shuffled_values: list[torch.Tensor] = []
    correct_dino: dict[int, list[torch.Tensor]] = {4: [], 16: []}
    shuffled_dino: dict[int, list[torch.Tensor]] = {4: [], 16: []}
    sampling: list[dict[str, Any]] = []
    grid_size = math.isqrt(int(conditions.shape[1]))
    for seed in NOISE_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn((len(rows), 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
        mapping = make_global_shuffle_mapping(item_count=len(rows), seed=seed)
        correct = sample_euler(
            model,
            _flatten_condition(conditions, grid_size=grid_size),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=sample_chunk_size,
        )
        shuffled = sample_euler(
            model,
            _flatten_condition(conditions[mapping], grid_size=grid_size),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=sample_chunk_size,
        )
        correct_values.append(correct)
        shuffled_values.append(shuffled)
        correct_features = _generated_multigrid_features(
            teacher, correct, device=device, batch_size=dino_batch_size
        )
        shuffled_features = _generated_multigrid_features(
            teacher, shuffled, device=device, batch_size=dino_batch_size
        )
        for grid in (4, 16):
            correct_dino[grid].append(correct_features[grid])
            shuffled_dino[grid].append(shuffled_features[grid])
        evidence = {
            "seed": seed,
            "initial_noise_sha256": _tensor_sha256(noise),
            "shuffle_algorithm": QUERY_STATE_SHUFFLE_ALGORITHM,
            "shuffle_mapping": mapping.tolist(),
            "shuffle_donor_row_identities": [rows[index]["row_identity"] for index in mapping],
            "ode_solver": "midpoint_euler",
            "ode_steps": SAMPLE_ODE_STEPS,
        }
        evidence["identity"] = _mapping_identity(evidence)
        sampling.append(evidence)
    metrics = sample_multiscale_metrics(
        rows=rows,
        noise_seeds=NOISE_SEEDS,
        source_rgb=source_rgb,
        correct_rgb=torch.stack(correct_values, dim=1),
        shuffled_rgb=torch.stack(shuffled_values, dim=1),
        source_dino=source_dino,
        correct_dino={grid: torch.stack(values, dim=1) for grid, values in correct_dino.items()},
        shuffled_dino={grid: torch.stack(values, dim=1) for grid, values in shuffled_dino.items()},
    )
    report = {
        **{key: value for key, value in metrics.items() if key != "identity"},
        "sampling_evidence": sampling,
    }
    report["identity"] = _mapping_identity(report)
    return report


def _contact_samples(
    *, models: Mapping[str, nn.Module], conditions: Mapping[str, torch.Tensor],
    source_images: torch.Tensor, rows: Sequence[Mapping[str, Any]],
    device: torch.device, chunk_size: int,
) -> tuple[tuple[Mapping[str, Any], ...], list[Image.Image], dict[str, list[Image.Image]], str]:
    indices = torch.tensor(ID198_SAMPLE_INDICES, dtype=torch.long)
    if _sha256_tensor_bytes(indices) != ID198_SAMPLE_INDICES_SHA256:
        raise ValueError("DINO-grid fixed ID198 index contract drift")
    generator = torch.Generator(device="cpu").manual_seed(CONTACT_SAMPLE_SEED)
    noise = torch.randn((16, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator)
    noise_sha256 = _sha256_tensor_bytes(noise)
    if noise_sha256 != ID198_INITIAL_NOISE_SHA256:
        raise ValueError("DINO-grid fixed ID198 initial-noise contract drift")
    selected_rows = tuple(rows[index] for index in ID198_SAMPLE_INDICES)
    originals = [
        Image.fromarray(source_images[index].permute(1, 2, 0).numpy(), mode="RGB")
        for index in ID198_SAMPLE_INDICES
    ]
    samples: dict[str, list[Image.Image]] = {}
    for cell in CELL_ORDER:
        value = conditions[cell][indices]
        grid_size = math.isqrt(int(value.shape[1]))
        output = sample_euler(
            models[cell],
            _flatten_condition(value, grid_size=grid_size),
            noise,
            steps=SAMPLE_ODE_STEPS,
            device=device,
            chunk_size=chunk_size,
        )
        samples[cell] = [_tensor_to_pil(image) for image in output]
    return selected_rows, originals, samples, noise_sha256


def evaluate_dino_grid_reconstruction_ceiling(args: argparse.Namespace) -> Mapping[str, Any]:
    """Run the exact read-only 256x3 three-grid pure-noise evaluation."""

    destination = args.output_dir
    checkpoints = {
        "grid4": args.grid4_checkpoint,
        "grid8": args.grid8_checkpoint,
        "grid16": args.grid16_checkpoint,
    }
    _reject_protected_output(
        destination,
        (
            args.grid4_cache,
            args.multigrid_cache,
            args.id198_summary.parent,
            args.id198_external_report.parent,
            *(path.parent for path in checkpoints.values()),
        ),
    )
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("DINO-grid evaluation output already exists")
    if (
        args.device.type == "cuda" and not torch.cuda.is_available()
        or args.dino_dtype not in {torch.float32, torch.float16, torch.bfloat16}
        or args.sample_chunk_size != SAMPLE_CHUNK_SIZE
        or isinstance(args.dino_batch_size, bool)
        or not isinstance(args.dino_batch_size, int)
        or args.dino_batch_size < 1
        or not all(
            _is_git_commit(value)
            for value in (
                args.grid4_train_source_commit,
                args.multigrid_cache_source_commit,
                args.multigrid_train_source_commit,
            )
        )
    ):
        raise ValueError("DINO-grid evaluation device/batch/source contract is invalid")
    source_commit = _current_source_commit()
    id198_reference = _load_id198_visual_reference(
        summary_path=args.id198_summary,
        external_report_path=args.id198_external_report,
    )
    grid4_train, grid4_split = _load_grid4_evaluation_splits(args.grid4_cache)
    grid8_train, grid8_split = load_dino_grid_ceiling_splits(
        multigrid_cache=args.multigrid_cache,
        grid_size=8,
        image_size=IMAGE_SIZE,
    )
    grid16_train, grid16_split = load_dino_grid_ceiling_splits(
        multigrid_cache=args.multigrid_cache,
        grid_size=16,
        image_size=IMAGE_SIZE,
    )
    splits = {"grid4": grid4_split, "grid8": grid8_split, "grid16": grid16_split}
    row_orders = {
        tuple((row["row_identity"], row["original_image_sha256"]) for row in split.rows)
        for split in splits.values()
    }
    if len(row_orders) != 1:
        raise ValueError("DINO-grid external row/image order is not matched")

    teacher = FrozenDINOMultigridTargets.from_pretrained(
        DINOV2_LARGE_IDENTITY,
        device=args.device,
        dtype=args.dino_dtype,
        batch_size=args.dino_batch_size,
    )
    dino_owner = _validate_teacher(teacher)
    try:
        manifest = json.loads(
            (args.multigrid_cache / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DINO-grid multigrid manifest is unreadable") from error
    if (
        manifest.get("dino") != dino_owner
        or manifest["producer"].get("source_commit")
        != args.multigrid_cache_source_commit
    ):
        raise ValueError("DINO-grid live cache/teacher/source identity drift")

    models: dict[str, nn.Module] = {}
    checkpoint_evidence: dict[str, Any] = {}
    grid4_model, grid4_invariants, grid4_sha = _load_grid4_checkpoint(
        args.grid4_checkpoint, cell="spatial_oracle", device=args.device
    )
    if (
        grid4_invariants.get("source_commit") != args.grid4_train_source_commit
        or grid4_invariants.get("oracle_cache_path") != str(args.grid4_cache.resolve())
        or grid4_invariants.get("source_state_cache_fingerprint")
        != grid4_split.embedded_state_cache_fingerprint
        or grid4_invariants.get("train_cache_fingerprint")
        != grid4_train.cache_fingerprint
        or grid4_invariants.get("validation_cache_fingerprint")
        != grid4_split.cache_fingerprint
        or grid4_invariants.get("train_row_set_identity")
        != grid4_train.row_set_identity
        or grid4_invariants.get("validation_row_set_identity")
        != grid4_split.row_set_identity
        or grid4_invariants.get("selection_identity")
        != grid4_split.selection_identity
    ):
        raise ValueError("DINO-grid immutable grid4 checkpoint/cache/source mismatch")
    models["grid4"] = grid4_model
    checkpoint_evidence["grid4"] = {
        "path": str(args.grid4_checkpoint.resolve()),
        "sha256": grid4_sha,
        "schema": ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA,
        "cell": "spatial_oracle",
        "invariants": grid4_invariants,
    }
    for cell, training_cell, train_split in (
        ("grid8", "spatial_dino8", grid8_train),
        ("grid16", "spatial_dino16", grid16_train),
    ):
        model, invariants, sha256 = _load_new_checkpoint(
            checkpoints[cell], cell=training_cell, device=args.device
        )
        if (
            invariants.get("source_commit") != args.multigrid_train_source_commit
            or invariants.get("multigrid_cache_path") != str(args.multigrid_cache.resolve())
            or invariants.get("source_grid4_cache_fingerprint")
            != grid4_split.cache_fingerprint
            or invariants.get("embedded_state_cache_fingerprint")
            != grid4_split.embedded_state_cache_fingerprint
            or invariants.get("train_cache_fingerprint")
            != train_split.cache_fingerprint
            or invariants.get("validation_cache_fingerprint")
            != splits[cell].cache_fingerprint
            or invariants.get("feature_identity") != train_split.feature_identity
            or invariants.get("train_row_set_identity")
            != train_split.row_set_identity
            or invariants.get("validation_row_set_identity")
            != splits[cell].row_set_identity
            or invariants.get("selection_identity")
            != train_split.selection_identity
        ):
            raise ValueError(f"DINO-grid {cell} checkpoint/live source identity drift")
        models[cell] = model
        checkpoint_evidence[cell] = {
            "path": str(checkpoints[cell].resolve()),
            "sha256": sha256,
            "schema": DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA,
            "cell": training_cell,
            "invariants": invariants,
        }
    invariant_values = {
        cell: evidence["invariants"] for cell, evidence in checkpoint_evidence.items()
    }
    matched_fields = (
        "decoder_family", "decoder_parameter_count", "image_preprocessing",
        "batch_size", "learning_rate", "weight_decay", "gradient_clip",
        "final_step", "evaluation_interval", "save_interval", "seed",
        "noise_seeds", "sample_rows", "sample_ode_steps",
    )
    if any(
        invariant_values[cell].get(field) != invariant_values["grid4"].get(field)
        for cell in ("grid8", "grid16")
        for field in matched_fields
    ):
        raise ValueError("DINO-grid checkpoints do not share architecture/training/eval budget")
    normalized_configs = []
    for value in invariant_values.values():
        metadata = dict(value["cfm_config"])
        metadata.pop("token_count")
        normalized_configs.append(metadata)
    if normalized_configs[1:] != normalized_configs[:1] * 2:
        raise ValueError("DINO-grid decoder capacity differs beyond token count")

    plan = build_sample_plan(grid4_split.rows, count=SAMPLE_ROWS)
    selected = torch.tensor(plan.indices, dtype=torch.long)
    source_rgb = grid4_split.images_uint8[selected].float().div(127.5).sub(1)
    source_dino = {
        4: grid4_split.conditions[selected],
        16: grid16_split.conditions[selected],
    }
    cell_reports: dict[str, Any] = {}
    for cell in CELL_ORDER:
        split = splits[cell]
        sample = _sample_cell(
            model=models[cell],
            conditions=split.conditions[selected],
            source_rgb=source_rgb,
            source_dino=source_dino,
            rows=tuple(split.rows[index] for index in plan.indices),
            teacher=teacher,
            device=args.device,
            dino_batch_size=args.dino_batch_size,
            sample_chunk_size=args.sample_chunk_size,
        )
        cell_reports[cell] = {
            "cell": cell,
            "grid_size": {"grid4": 4, "grid8": 8, "grid16": 16}[cell],
            "condition_owner": "direct_original_observation_dino",
            "checkpoint": checkpoint_evidence[cell],
            "row_set_identity": _mapping_identity(
                {
                    "rows": [
                        {
                            "row_identity": row["row_identity"],
                            "original_image_sha256": row["original_image_sha256"],
                        }
                        for row in grid4_split.rows
                    ]
                }
            ),
            "sample_selection_identity": plan.identity,
            "noise_seeds": list(NOISE_SEEDS),
            "sample_metrics": sample,
            "rows": sample["rows"],
        }
    comparison = build_three_grid_comparison(
        cell_reports,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    contact_rows, originals, contact_samples, contact_noise_sha = _contact_samples(
        models=models,
        conditions={cell: splits[cell].conditions for cell in CELL_ORDER},
        source_images=grid4_split.images_uint8,
        rows=grid4_split.rows,
        device=args.device,
        chunk_size=args.sample_chunk_size,
    )
    contact_pairs = [
        {
            "row_identity": row["row_identity"],
            "original_image_sha256": row["original_image_sha256"],
        }
        for row in contact_rows
    ]
    if contact_pairs != id198_reference["row_image_pairs"]:
        raise ValueError("DINO-grid fixed contact rows differ from immutable ID198")

    input_artifacts: dict[str, Any] = {}
    cache_fingerprints = {
        "grid4_cache": grid4_split.cache_fingerprint,
        "multigrid_cache": str(manifest["cache_fingerprint"]),
    }
    for name, path in (
        ("grid4_cache", args.grid4_cache),
        ("multigrid_cache", args.multigrid_cache),
    ):
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"DINO-grid {name} manifest is incomplete")
        input_artifacts[name] = {
            "path": str(path.resolve()),
            "manifest_sha256": _sha256_file(manifest_path),
            "cache_fingerprint": cache_fingerprints[name],
        }
    producer: dict[str, Any] = {
        "module": "nimloth.eval.dino_grid_reconstruction_ceiling",
        "source_commit": source_commit,
        "resolved_config": {
            "grid4_cache": str(args.grid4_cache.resolve()),
            "multigrid_cache": str(args.multigrid_cache.resolve()),
            "grid4_checkpoint": str(args.grid4_checkpoint.resolve()),
            "grid8_checkpoint": str(args.grid8_checkpoint.resolve()),
            "grid16_checkpoint": str(args.grid16_checkpoint.resolve()),
            "grid4_train_source_commit": args.grid4_train_source_commit,
            "multigrid_cache_source_commit": args.multigrid_cache_source_commit,
            "multigrid_train_source_commit": args.multigrid_train_source_commit,
            "id198_summary": str(args.id198_summary.resolve()),
            "id198_external_report": str(args.id198_external_report.resolve()),
            "output": str(destination.resolve()),
            "device": str(args.device),
            "dino_dtype": str(args.dino_dtype).removeprefix("torch."),
            "dino_batch_size": args.dino_batch_size,
            "sample_chunk_size": args.sample_chunk_size,
            "sample_rows": SAMPLE_ROWS,
            "noise_seeds": list(NOISE_SEEDS),
            "sample_ode_steps": SAMPLE_ODE_STEPS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "contact_sample_seed": CONTACT_SAMPLE_SEED,
        },
    }
    producer["identity"] = _mapping_identity(producer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    try:
        contact_metadata = save_three_grid_contact_sheet(
            staging / "contact_sheet.png",
            rows=contact_rows,
            originals=originals,
            correct_samples=contact_samples,
            sample_indices=ID198_SAMPLE_INDICES,
            sample_seed=CONTACT_SAMPLE_SEED,
            initial_noise_sha256=contact_noise_sha,
        )
        report: dict[str, Any] = {
            "schema": DINO_GRID_CEILING_REPORT_SCHEMA,
            "status": "core_completed_three_grid_evaluation_human_judgment_pending",
            "forensic_only": True,
            "not_deployable": True,
            "not_sft1_pass": True,
            "not_sft2_ready": True,
            "actor_failure_verdict_unchanged": True,
            "condition_owner": "direct_original_observation_dino",
            "dino_owner": dino_owner,
            "producer": producer,
            "input_artifacts": input_artifacts,
            "id198_visual_reference": id198_reference,
            "external_items": EXTERNAL_ITEMS,
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
            "human_decision": None,
            "claim_scope": "forensic_representation_decodability_only",
        }
        report["artifact_identity"] = _mapping_identity(report)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_noreplace(staging, destination)
        del models, teacher, source_dino, source_rgb, splits
        del grid4_train, grid4_split, grid8_train, grid8_split
        del grid16_train, grid16_split
        try:
            return validate_dino_grid_ceiling_report(destination)
        except Exception as error:
            failure_path = destination.parent / f"{destination.name}.post_validation_failure.json"
            failure = {
                "schema": "nimloth_dino_grid_ceiling_report_post_validation_failure_v1",
                "status": "core_completed_post_validation_failed",
                "report_root": str(destination.resolve()),
                "artifact_identity": report["artifact_identity"],
                "error_type": type(error).__name__,
                "error": str(error),
            }
            failure["identity"] = _mapping_identity(failure)
            with failure_path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            raise
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


def _validate_cell_sample_evidence(
    cell_report: Mapping[str, Any], sample_plan: Mapping[str, Any]
) -> None:
    sample = cell_report.get("sample_metrics")
    if (
        not isinstance(sample, Mapping)
        or set(sample)
        != {
            "schema", "statistical_unit", "seed_aggregation", "noise_seeds",
            "source_dino_input_owner", "generated_dino_input_owner",
            "generated_dino_is_canonical_teacher_target", "aggregate", "rows",
            "sampling_evidence", "identity",
        }
        or sample.get("schema")
        != "nimloth_dino_grid_ceiling_multiscale_metrics_v1"
        or sample.get("statistical_unit") != "external_validation_row"
        or sample.get("seed_aggregation")
        != "mean_within_row_before_row_bootstrap"
        or sample.get("noise_seeds") != list(NOISE_SEEDS)
        or sample.get("source_dino_input_owner")
        != "original_archived_observation"
        or sample.get("generated_dino_input_owner")
        != "decoder_generated_128px_srgb"
        or sample.get("generated_dino_is_canonical_teacher_target") is not False
        or sample.get("identity")
        != _mapping_identity({key: value for key, value in sample.items() if key != "identity"})
    ):
        raise ValueError("DINO-grid cell sample owner/identity evidence is invalid")
    rows = sample.get("rows")
    aggregate = sample.get("aggregate")
    if (
        not isinstance(rows, list)
        or len(rows) != SAMPLE_ROWS
        or rows != cell_report.get("rows")
        or [row.get("row_identity") for row in rows]
        != sample_plan.get("row_identities")
        or [row.get("original_image_sha256") for row in rows]
        != sample_plan.get("image_sha256")
        or not isinstance(aggregate, Mapping)
        or set(aggregate) != _ALL_SAMPLE_METRICS
    ):
        raise ValueError("DINO-grid cell sample rows/aggregate are incomplete")
    for row in rows:
        per_seed = row.get("per_seed") if isinstance(row, Mapping) else None
        if (
            not isinstance(row, Mapping)
            or set(row) != {
                "row_identity", "original_image_sha256", "per_seed",
                *_ALL_SAMPLE_METRICS,
            }
            or not _is_sha256(row.get("original_image_sha256"))
            or not isinstance(per_seed, list)
            or len(per_seed) != len(NOISE_SEEDS)
        ):
            raise ValueError("DINO-grid cell per-row metric evidence is invalid")
        for seed, per_seed_item in zip(NOISE_SEEDS, per_seed, strict=True):
            if (
                not isinstance(per_seed_item, Mapping)
                or set(per_seed_item) != {"seed", *_ALL_SAMPLE_METRICS}
                or per_seed_item.get("seed") != seed
                or any(
                    isinstance(per_seed_item.get(metric), bool)
                    or not isinstance(per_seed_item.get(metric), (int, float))
                    or not math.isfinite(float(per_seed_item[metric]))
                    for metric in _ALL_SAMPLE_METRICS
                )
            ):
                raise ValueError("DINO-grid cell per-seed metric evidence is invalid")
        for metric in _ALL_SAMPLE_METRICS:
            expected = sum(float(item[metric]) for item in per_seed) / len(NOISE_SEEDS)
            if not math.isclose(float(row[metric]), expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("DINO-grid metric seeds were not averaged inside row")
    for metric in _ALL_SAMPLE_METRICS:
        value = aggregate[metric]
        expected = sum(float(row[metric]) for row in rows) / len(rows)
        if (
            not isinstance(value, Mapping)
            or set(value) != {"mean"}
            or not math.isclose(float(value["mean"]), expected, rel_tol=1e-6, abs_tol=1e-7)
        ):
            raise ValueError("DINO-grid absolute metric aggregate is inconsistent")
    sampling = sample.get("sampling_evidence")
    if not isinstance(sampling, list) or len(sampling) != len(NOISE_SEEDS):
        raise ValueError("DINO-grid sampling evidence is incomplete")
    row_ids = sample_plan["row_identities"]
    for seed, evidence in zip(NOISE_SEEDS, sampling, strict=True):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        expected_noise = torch.randn(
            (SAMPLE_ROWS, 3, IMAGE_SIZE, IMAGE_SIZE), generator=generator
        )
        mapping = make_global_shuffle_mapping(item_count=SAMPLE_ROWS, seed=seed)
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("identity")
            != _mapping_identity(
                {key: value for key, value in evidence.items() if key != "identity"}
            )
            or evidence.get("seed") != seed
            or evidence.get("initial_noise_sha256") != _tensor_sha256(expected_noise)
            or evidence.get("shuffle_algorithm") != QUERY_STATE_SHUFFLE_ALGORITHM
            or evidence.get("shuffle_mapping") != mapping.tolist()
            or evidence.get("shuffle_donor_row_identities")
            != [row_ids[index] for index in mapping.tolist()]
            or evidence.get("ode_solver") != "midpoint_euler"
            or evidence.get("ode_steps") != SAMPLE_ODE_STEPS
        ):
            raise ValueError("DINO-grid matched pure-noise sampling evidence is invalid")


def _live_split_lineage(
    dataset: _MetadataOnlyGrid4CacheDataset | DinoGridCeilingCacheDataset,
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "all_train": [],
        "external_validation": [],
    }
    for index in range(len(dataset)):
        row = dataset.row_metadata(index)
        role = str(row.get("selection_role"))
        if role not in grouped:
            raise ValueError("DINO-grid live cache contains a cross-stage role")
        grouped[role].append(row)
    if {role: len(rows) for role, rows in grouped.items()} != {
        "all_train": TRAIN_ITEMS,
        "external_validation": EXTERNAL_ITEMS,
    }:
        raise ValueError("DINO-grid live cache split counts are not exact")
    cache_fingerprint = str(dataset.manifest["cache_fingerprint"])
    selection_identity = str(dataset.manifest["selection"]["identity"])
    return {
        "cache_fingerprint": cache_fingerprint,
        "selection_identity": selection_identity,
        "train_row_set_identity": _multigrid_split_identity(
            cache_fingerprint=cache_fingerprint,
            selection_identity=selection_identity,
            role="all_train",
            rows=grouped["all_train"],
        ),
        "validation_row_set_identity": _multigrid_split_identity(
            cache_fingerprint=cache_fingerprint,
            selection_identity=selection_identity,
            role="external_validation",
            rows=grouped["external_validation"],
        ),
    }


def validate_dino_grid_ceiling_report(root: str | Path) -> Mapping[str, Any]:
    """Strict-read a manifest-last report and recheck live artifact hashes."""

    supplied = Path(root)
    if _has_symlink_component(supplied):
        raise ValueError("DINO-grid report root must not traverse symlinks")
    report_root = supplied.resolve()
    manifest_path = report_root / "manifest.json"
    contact_path = report_root / "contact_sheet.png"
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not contact_path.is_file()
        or contact_path.is_symlink()
    ):
        raise ValueError("DINO-grid report manifest/contact is incomplete")
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DINO-grid report manifest is unreadable") from error
    required = {
        "schema", "status", "forensic_only", "not_deployable", "not_sft1_pass",
        "not_sft2_ready", "actor_failure_verdict_unchanged", "condition_owner",
        "dino_owner", "producer", "input_artifacts", "id198_visual_reference",
        "external_items", "sample_plan", "cell_reports", "comparison",
        "contact_sheet", "human_decision", "claim_scope", "artifact_identity",
    }
    if (
        not isinstance(report, Mapping)
        or set(report) != required
        or report.get("schema") != DINO_GRID_CEILING_REPORT_SCHEMA
        or report.get("status")
        != "core_completed_three_grid_evaluation_human_judgment_pending"
        or any(report.get(field) is not True for field in ("forensic_only", "not_deployable", "not_sft1_pass", "not_sft2_ready", "actor_failure_verdict_unchanged"))
        or report.get("condition_owner") != "direct_original_observation_dino"
        or report.get("external_items") != EXTERNAL_ITEMS
        or report.get("human_decision") is not None
        or report.get("claim_scope") != "forensic_representation_decodability_only"
        or report.get("artifact_identity")
        != _mapping_identity({key: value for key, value in report.items() if key != "artifact_identity"})
    ):
        raise ValueError("DINO-grid report schema/status/identity is invalid")
    producer = report.get("producer")
    config = producer.get("resolved_config") if isinstance(producer, Mapping) else None
    if (
        not isinstance(producer, Mapping)
        or producer.get("module") != "nimloth.eval.dino_grid_reconstruction_ceiling"
        or not _is_git_commit(producer.get("source_commit"))
        or producer.get("identity")
        != _mapping_identity({key: value for key, value in producer.items() if key != "identity"})
        or not isinstance(config, Mapping)
        or set(config)
        != {
            "grid4_cache", "multigrid_cache", "grid4_checkpoint",
            "grid8_checkpoint", "grid16_checkpoint", "grid4_train_source_commit",
            "multigrid_cache_source_commit", "multigrid_train_source_commit",
            "id198_summary", "id198_external_report", "output", "device",
            "dino_dtype", "dino_batch_size", "sample_chunk_size",
            "sample_rows", "noise_seeds",
            "sample_ode_steps", "bootstrap_seed", "bootstrap_resamples",
            "contact_sample_seed",
        }
        or config.get("output") != str(report_root)
        or config.get("sample_chunk_size") != SAMPLE_CHUNK_SIZE
        or config.get("sample_rows") != SAMPLE_ROWS
        or config.get("noise_seeds") != list(NOISE_SEEDS)
        or config.get("sample_ode_steps") != SAMPLE_ODE_STEPS
        or config.get("bootstrap_seed") != BOOTSTRAP_SEED
        or config.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
        or config.get("contact_sample_seed") != CONTACT_SAMPLE_SEED
        or config.get("dino_dtype") not in {"float32", "float16", "bfloat16"}
        or isinstance(config.get("dino_batch_size"), bool)
        or not isinstance(config.get("dino_batch_size"), int)
        or config["dino_batch_size"] < 1
        or not all(
            _is_git_commit(config.get(field))
            for field in (
                "grid4_train_source_commit", "multigrid_cache_source_commit",
                "multigrid_train_source_commit",
            )
        )
    ):
        raise ValueError("DINO-grid report producer/config identity is invalid")
    dino_owner = report.get("dino_owner")
    if (
        not isinstance(dino_owner, Mapping)
        or dino_owner.get("source") != DINOV2_LARGE_IDENTITY.source
        or dino_owner.get("revision") != DINOV2_LARGE_IDENTITY.revision
        or dino_owner.get("processor_fingerprint")
        != DINOV2_LARGE_IDENTITY.processor_fingerprint
        or dino_owner.get("hidden_size") != DINOV2_LARGE_IDENTITY.hidden_size
        or dino_owner.get("native_grid_size") != 37
        or dino_owner.get("native_tokens") != 1369
        or dino_owner.get("input_owner") != "original_archived_observation"
        or dino_owner.get("resize_before_processor") is not False
        or dino_owner.get("model_dtype") != config["dino_dtype"]
        or dino_owner.get("output_dtype") != "float32"
        or dino_owner.get("batch_size") != config["dino_batch_size"]
    ):
        raise ValueError("DINO-grid report DINO owner identity is invalid")
    inputs = report.get("input_artifacts")
    if not isinstance(inputs, Mapping) or set(inputs) != {"grid4_cache", "multigrid_cache"}:
        raise ValueError("DINO-grid report input evidence is incomplete")
    for name, artifact in inputs.items():
        path = Path(str(artifact.get("path", "")))
        live_manifest = path / "manifest.json"
        if (
            set(artifact) != {"path", "manifest_sha256", "cache_fingerprint"}
            or not _is_sha256(artifact.get("cache_fingerprint"))
            or not path.is_absolute()
            or _has_symlink_component(path)
            or not live_manifest.is_file()
            or live_manifest.is_symlink()
            or artifact.get("manifest_sha256") != _sha256_file(live_manifest)
            or config.get(name) != str(path)
        ):
            raise ValueError("DINO-grid report live cache identity drift")
    grid4_reader = _MetadataOnlyGrid4CacheDataset(Path(config["grid4_cache"]))
    if (
        grid4_reader.manifest["cache_fingerprint"]
        != inputs["grid4_cache"]["cache_fingerprint"]
    ):
        raise ValueError("DINO-grid report live grid4-cache fingerprint drift")
    embedded_state_cache_fingerprint = grid4_reader.manifest[
        "source_state_cache"
    ]["cache_fingerprint"]
    grid4_lineage = _live_split_lineage(grid4_reader)
    live_external_rows = tuple(
        grid4_reader.row_metadata(index)
        for index in range(len(grid4_reader))
        if grid4_reader.row_metadata(index).get("selection_role")
        == "external_validation"
    )
    expected_plan_value = build_sample_plan(live_external_rows, count=SAMPLE_ROWS)
    expected_sample_plan = {
        "algorithm": expected_plan_value.algorithm,
        "identity": expected_plan_value.identity,
        "count": len(expected_plan_value.indices),
        "row_identities": list(expected_plan_value.row_identities),
        "image_sha256": list(expected_plan_value.image_sha256),
    }
    live_external_row_set_identity = _mapping_identity(
        {
            "rows": [
                {
                    "row_identity": row["row_identity"],
                    "original_image_sha256": row["original_image_sha256"],
                }
                for row in live_external_rows
            ]
        }
    )
    del grid4_reader
    multigrid_reader = DinoGridCeilingCacheDataset(
        Path(config["multigrid_cache"]), grid_size=16
    )
    if (
        multigrid_reader.cache_fingerprint
        != inputs["multigrid_cache"]["cache_fingerprint"]
    ):
        raise ValueError("DINO-grid report live multigrid-cache fingerprint drift")
    if (
        multigrid_reader.manifest.get("dino") != dino_owner
        or multigrid_reader.manifest["producer"].get("source_commit")
        != config["multigrid_cache_source_commit"]
    ):
        raise ValueError("DINO-grid report live multigrid DINO/source drift")
    multigrid_lineage = _live_split_lineage(multigrid_reader)
    live_feature_identities = {
        grid_size: str(
            multigrid_reader.manifest["views"][f"grid{grid_size}"][
                "feature_identity"
            ]
        )
        for grid_size in (8, 16)
    }
    del multigrid_reader
    expected_id198 = _load_id198_visual_reference(
        summary_path=Path(config["id198_summary"]),
        external_report_path=Path(config["id198_external_report"]),
    )
    if report.get("id198_visual_reference") != expected_id198:
        raise ValueError("DINO-grid report immutable ID198 reference drift")
    sample_plan = report.get("sample_plan")
    if not isinstance(sample_plan, Mapping) or sample_plan != expected_sample_plan:
        raise ValueError("DINO-grid report sample-plan identity is invalid")
    cell_reports = report.get("cell_reports")
    if not isinstance(cell_reports, Mapping) or set(cell_reports) != set(CELL_ORDER):
        raise ValueError("DINO-grid report requires exactly three cells")
    for cell in CELL_ORDER:
        cell_report = cell_reports[cell]
        checkpoint = cell_report.get("checkpoint") if isinstance(cell_report, Mapping) else None
        if (
            not isinstance(cell_report, Mapping)
            or set(cell_report)
            != {
                "cell", "grid_size", "condition_owner", "checkpoint",
                "row_set_identity", "sample_selection_identity", "noise_seeds",
                "sample_metrics", "rows",
            }
            or cell_report.get("cell") != cell
            or cell_report.get("grid_size")
            != {"grid4": 4, "grid8": 8, "grid16": 16}[cell]
            or cell_report.get("condition_owner") != "direct_original_observation_dino"
            or cell_report.get("sample_selection_identity") != sample_plan["identity"]
            or cell_report.get("noise_seeds") != list(NOISE_SEEDS)
            or cell_report.get("rows") != cell_report.get("sample_metrics", {}).get("rows")
            or cell_report.get("row_set_identity")
            != live_external_row_set_identity
            or not isinstance(checkpoint, Mapping)
            or set(checkpoint) != {"path", "sha256", "schema", "cell", "invariants"}
            or config.get(f"{cell}_checkpoint") != checkpoint.get("path")
        ):
            raise ValueError("DINO-grid report cell/sample/checkpoint identity is invalid")
        checkpoint_path = Path(str(checkpoint.get("path", "")))
        if (
            not checkpoint_path.is_absolute()
            or _has_symlink_component(checkpoint_path)
            or not checkpoint_path.is_file()
            or checkpoint.get("sha256") != _sha256_file(checkpoint_path)
        ):
            raise ValueError("DINO-grid report live checkpoint hash drift")
        if cell == "grid4":
            model, invariants, checkpoint_sha = _load_grid4_checkpoint(
                checkpoint_path, cell="spatial_oracle", device=torch.device("cpu")
            )
            expected_schema = ORACLE_LADDER_CFM_CHECKPOINT_SCHEMA
            expected_cell = "spatial_oracle"
            valid_source = (
                invariants.get("source_commit") == config["grid4_train_source_commit"]
                and invariants.get("oracle_cache_path") == config["grid4_cache"]
                and invariants.get("source_state_cache_fingerprint")
                == embedded_state_cache_fingerprint
                and invariants.get("train_cache_fingerprint")
                == grid4_lineage["cache_fingerprint"]
                and invariants.get("validation_cache_fingerprint")
                == grid4_lineage["cache_fingerprint"]
                and invariants.get("train_row_set_identity")
                == grid4_lineage["train_row_set_identity"]
                and invariants.get("validation_row_set_identity")
                == grid4_lineage["validation_row_set_identity"]
                and invariants.get("selection_identity")
                == grid4_lineage["selection_identity"]
            )
        else:
            expected_cell = f"spatial_dino{cell.removeprefix('grid')}"
            model, invariants, checkpoint_sha = _load_new_checkpoint(
                checkpoint_path, cell=expected_cell, device=torch.device("cpu")
            )
            expected_schema = DINO_GRID_CEILING_CFM_CHECKPOINT_SCHEMA
            valid_source = (
                invariants.get("source_commit")
                == config["multigrid_train_source_commit"]
                and invariants.get("multigrid_cache_path")
                == config["multigrid_cache"]
                and invariants.get("source_grid4_cache_fingerprint")
                == grid4_lineage["cache_fingerprint"]
                and invariants.get("embedded_state_cache_fingerprint")
                == embedded_state_cache_fingerprint
                and invariants.get("train_cache_fingerprint")
                == multigrid_lineage["cache_fingerprint"]
                and invariants.get("validation_cache_fingerprint")
                == multigrid_lineage["cache_fingerprint"]
                and invariants.get("train_row_set_identity")
                == multigrid_lineage["train_row_set_identity"]
                and invariants.get("validation_row_set_identity")
                == multigrid_lineage["validation_row_set_identity"]
                and invariants.get("selection_identity")
                == multigrid_lineage["selection_identity"]
                and invariants.get("feature_identity")
                == live_feature_identities[int(cell.removeprefix("grid"))]
            )
        if (
            checkpoint.get("schema") != expected_schema
            or checkpoint.get("cell") != expected_cell
            or checkpoint.get("invariants") != invariants
            or checkpoint_sha != checkpoint.get("sha256")
            or not valid_source
        ):
            raise ValueError("DINO-grid report checkpoint owner/source identity drift")
        del model
        _validate_cell_sample_evidence(cell_report, sample_plan)
    sampling_plans = {
        _canonical_json({"sampling": cell_reports[cell]["sample_metrics"]["sampling_evidence"]})
        for cell in CELL_ORDER
    }
    if len(sampling_plans) != 1:
        raise ValueError("DINO-grid cross-cell pure-noise/shuffle plan mismatch")
    expected_comparison = build_three_grid_comparison(
        cell_reports,
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    )
    if report.get("comparison") != expected_comparison:
        raise ValueError("DINO-grid report paired comparison drift")
    contact = report.get("contact_sheet")
    if (
        not isinstance(contact, Mapping)
        or contact.get("identity")
        != _mapping_identity({key: value for key, value in contact.items() if key != "identity"})
        or contact.get("columns") != ["original", "grid4", "grid8", "grid16"]
        or contact.get("indices") != list(ID198_SAMPLE_INDICES)
        or contact.get("sample_seed") != CONTACT_SAMPLE_SEED
        or contact.get("initial_noise_sha256") != ID198_INITIAL_NOISE_SHA256
        or contact.get("human_judgment_rule") != "at_least_12_of_fixed_16"
        or contact.get("human_judgment_target")
        != "corresponding_wall_floor_boundary_door_window_or_large_furniture_placement"
        or contact.get("controls_scientific_result") is not True
        or len(contact.get("row_identities", ())) != 16
        or contact.get("row_identities")
        != [pair["row_identity"] for pair in expected_id198["row_image_pairs"]]
        or contact.get("image_sha256")
        != [pair["original_image_sha256"] for pair in expected_id198["row_image_pairs"]]
        or contact.get("png_sha256") != _sha256_file(contact_path)
    ):
        raise ValueError("DINO-grid report fixed contact-sheet identity is invalid")
    return report


def _dtype(value: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[value]
    except KeyError as error:
        raise ValueError(f"unsupported DINO dtype: {value}") from error


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate matched direct-DINO grid4/grid8/grid16 RGB reconstruction"
    )
    parser.add_argument("--grid4-cache", required=True, type=Path)
    parser.add_argument("--multigrid-cache", required=True, type=Path)
    parser.add_argument("--grid4-checkpoint", required=True, type=Path)
    parser.add_argument("--grid8-checkpoint", required=True, type=Path)
    parser.add_argument("--grid16-checkpoint", required=True, type=Path)
    parser.add_argument("--grid4-train-source-commit", required=True)
    parser.add_argument("--multigrid-cache-source-commit", required=True)
    parser.add_argument("--multigrid-train-source-commit", required=True)
    parser.add_argument("--id198-summary", required=True, type=Path)
    parser.add_argument("--id198-external-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, type=torch.device)
    parser.add_argument(
        "--dino-dtype", required=True, choices=("float32", "float16", "bfloat16")
    )
    parser.add_argument("--dino-batch-size", required=True, type=int)
    parser.add_argument("--sample-chunk-size", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_cli_parser().parse_args(argv)
    args.dino_dtype = _dtype(args.dino_dtype)
    report = evaluate_dino_grid_reconstruction_ceiling(args)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CELL_ORDER",
    "DINO_GRID_CEILING_REPORT_SCHEMA",
    "apply_human_structure_judgments",
    "build_cli_parser",
    "build_three_grid_comparison",
    "evaluate_dino_grid_reconstruction_ceiling",
    "sample_multiscale_metrics",
    "save_three_grid_contact_sheet",
    "validate_dino_grid_ceiling_report",
]
