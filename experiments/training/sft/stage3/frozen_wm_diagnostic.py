"""Immutable cache sealing and WM-only Stage3 capacity diagnostics.

The production eval-only path writes one ordered Stage2/DINO sequence per
trajectory.  This module seals those shards, then trains the production
direct or zero-initialized residual temporal-spatial predictor without updating Qwen, Query,
projector, ValueHead, or language-model parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from nimloth.wm.grid import (
    GridPredictorConfig,
    ResidualTemporalSpatialGridPredictor,
    TemporalSpatialGridPredictor,
)

CACHE_SCHEMA = "frozen_wm_cache_v1"
SHARD_SCHEMA = "frozen_wm_trajectory_shard_v1"
RUN_SCHEMA = "frozen_wm_diagnostic_v1"
MODES = ("stage2_state", "dino")
PREDICTOR_TYPES = {
    "direct": TemporalSpatialGridPredictor,
    "residual": ResidualTemporalSpatialGridPredictor,
}
Predictor = TemporalSpatialGridPredictor | ResidualTemporalSpatialGridPredictor


def predictor_type_from_run(run: dict) -> type:
    """Old identities without a kind describe direct prediction only."""
    kind = run["config"].get("predictor_kind", "direct")
    if kind not in PREDICTOR_TYPES:
        raise ValueError(f"unknown diagnostic predictor kind: {kind}")
    predictor_type = PREDICTOR_TYPES[kind]
    if run.get("trainable_modules") != [predictor_type.__name__]:
        raise ValueError("diagnostic predictor kind/module identity mismatch")
    return predictor_type


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def replace_run_progress_json(path: Path, value: object) -> None:
    """Atomically replace mutable run progress, never immutable checkpoint artifacts."""
    if path.name not in {"status.json", "convergence.json", "pointers.json"}:
        raise ValueError(f"not a mutable run progress file: {path}")
    if not (path.parent / "run.json").is_file():
        raise ValueError(f"mutable progress must belong to a run root: {path}")
    temporary = path.with_name(path.name + ".tmp")
    payload = _json_bytes(value)
    with temporary.open("xb") as stream:
        stream.write(payload)
    temporary.replace(path)


def _load_shard(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SHARD_SCHEMA:
        raise ValueError(f"unsupported frozen-WM shard schema: {path}")
    if payload.get("dtype") != "float32":
        raise ValueError(f"frozen-WM shard must record float32 grids: {path}")
    if not isinstance(payload.get("records"), list) or not payload["records"]:
        raise ValueError(f"frozen-WM shard contains no trajectories: {path}")
    return payload


def _validate_record(record: dict, *, horizon: int, path: Path) -> dict:
    trajectory = str(record.get("trajectory_id", ""))
    states, dino, actions = record.get("states"), record.get("dino"), record.get("actions")
    if not trajectory:
        raise ValueError(f"empty trajectory identity: {path}")
    if not all(isinstance(value, torch.Tensor) for value in (states, dino, actions)):
        raise ValueError(f"trajectory tensors missing: {path}:{trajectory}")
    if states.dtype != torch.float32 or dino.dtype != torch.float32:
        raise ValueError(f"trajectory grids must be float32: {path}:{trajectory}")
    if states.shape != dino.shape or states.ndim != 3:
        raise ValueError(f"state/DINO shape mismatch: {path}:{trajectory}")
    if actions.dtype != torch.int64 or actions.shape != (states.shape[0] - 1,):
        raise ValueError(f"action alignment mismatch: {path}:{trajectory}")
    if states.shape[0] <= horizon:
        raise ValueError(f"trajectory has no complete T={horizon} window: {path}:{trajectory}")
    if not torch.isfinite(states).all() or not torch.isfinite(dino).all():
        raise ValueError(f"non-finite trajectory grid: {path}:{trajectory}")
    if torch.any(actions < 0):
        raise ValueError(f"negative action index: {path}:{trajectory}")
    return {
        "trajectory_id": trajectory,
        "state_count": int(states.shape[0]),
        "action_count": int(actions.shape[0]),
        "window_count": int(states.shape[0] - horizon),
        "grid_tokens": int(states.shape[1]),
        "state_dim": int(states.shape[2]),
        "action_min": int(actions.min()),
        "action_max": int(actions.max()),
    }


def seal_cache(directory: Path, *, expected_ranks: int | None = None) -> dict:
    """Validate immutable export shards and publish their content manifest."""
    directory = directory.resolve()
    manifest_path, complete_path = directory / "manifest.json", directory / "COMPLETE"
    if manifest_path.exists() or complete_path.exists():
        raise FileExistsError(f"cache is already sealed: {directory}")
    paths = sorted(directory.glob("rank_*_batch_*.pt"))
    if not paths:
        raise ValueError(f"no frozen-WM shards found: {directory}")

    identities, horizons, ranks, trajectories = set(), set(), set(), set()
    records, shards = [], []
    grid_shape = None
    for path in paths:
        payload = _load_shard(path)
        identity_json = payload.get("identity_json")
        if not isinstance(identity_json, str):
            raise ValueError(f"missing canonical export identity: {path}")
        json.loads(identity_json)
        identities.add(identity_json)
        horizons.add(int(payload.get("prediction_horizon", 0)))
        ranks.add(int(payload.get("rank", -1)))
        shard_records = []
        for index, record in enumerate(payload["records"]):
            summary = _validate_record(
                record,
                horizon=int(payload["prediction_horizon"]),
                path=path,
            )
            if summary["trajectory_id"] in trajectories:
                raise ValueError(f"duplicate trajectory identity: {summary['trajectory_id']}")
            trajectories.add(summary["trajectory_id"])
            shape = (summary["grid_tokens"], summary["state_dim"])
            if grid_shape is not None and grid_shape != shape:
                raise ValueError(f"grid shape changed across cache: {grid_shape} != {shape}")
            grid_shape = shape
            summary.update(shard=path.name, record_index=index)
            records.append(summary)
            shard_records.append(summary["trajectory_id"])
        shards.append(
            {
                "path": path.name,
                "sha256": file_sha256(path),
                "rank": int(payload["rank"]),
                "batch_index": int(payload["batch_index"]),
                "trajectory_ids": shard_records,
            }
        )
    completion_paths = sorted(directory.glob("rank_*_COMPLETE.json"))
    completions = {}
    for path in completion_paths:
        completion = json.loads(path.read_text(encoding="utf-8"))
        if completion.get("schema") != "frozen_wm_rank_complete_v1":
            raise ValueError(f"unsupported frozen-WM rank completion marker: {path}")
        rank = int(completion.get("rank", -1))
        if rank in completions:
            raise ValueError(f"duplicate frozen-WM rank completion marker: {rank}")
        completions[rank] = completion
    if set(completions) != ranks:
        raise ValueError(
            f"completed cache ranks {sorted(completions)} != shard ranks {sorted(ranks)}"
        )
    for rank, completion in completions.items():
        rank_shards = [item for item in shards if item["rank"] == rank]
        expected = [
            {"path": item["path"], "sha256": item["sha256"]}
            for item in rank_shards
        ]
        if completion.get("identity_json") not in identities:
            raise ValueError(f"rank {rank} completion provenance differs from its shards")
        if completion.get("shards") != expected:
            raise ValueError(f"rank {rank} completion shard list is incomplete or changed")
        if int(completion.get("batch_count", -1)) != len(rank_shards):
            raise ValueError(f"rank {rank} completion batch count is incorrect")
        if int(completion.get("trajectory_count", -1)) != sum(
            len(item["trajectory_ids"]) for item in rank_shards
        ):
            raise ValueError(f"rank {rank} completion trajectory count is incorrect")
        if int(completion.get("window_count", -1)) != sum(
            record["window_count"]
            for item in rank_shards
            for record in records
            if record["shard"] == item["path"]
        ):
            raise ValueError(f"rank {rank} completion window count is incorrect")
    if len(identities) != 1 or len(horizons) != 1:
        raise ValueError("cache shards disagree on provenance or prediction horizon")
    horizon = next(iter(horizons))
    if horizon < 1:
        raise ValueError("cache prediction horizon must be positive")
    if expected_ranks is not None and ranks != set(range(expected_ranks)):
        raise ValueError(f"cache ranks {sorted(ranks)} != expected {list(range(expected_ranks))}")
    records.sort(key=lambda item: item["trajectory_id"])
    identity = json.loads(next(iter(identities)))
    identity_contract = {
        "prediction_horizon": horizon,
        "grid_tokens": grid_shape[0],
        "state_dim": grid_shape[1],
    }
    mismatches = {
        key: (identity.get(key), value)
        for key, value in identity_contract.items()
        if identity.get(key) != value
    }
    if mismatches:
        raise ValueError(f"export provenance disagrees with tensor contract: {mismatches}")
    count_mismatches = {
        "trajectory_count": (identity.get("trajectory_count"), len(records)),
        "window_count": (
            identity.get("window_count"),
            sum(item["window_count"] for item in records),
        ),
    }
    count_mismatches = {
        key: values for key, values in count_mismatches.items() if values[0] != values[1]
    }
    if count_mismatches:
        raise ValueError(f"export is incomplete for its source split: {count_mismatches}")
    manifest = {
        "schema": CACHE_SCHEMA,
        "dtype": "float32",
        "identity": identity,
        "prediction_horizon": horizon,
        "grid_tokens": grid_shape[0],
        "state_dim": grid_shape[1],
        "trajectory_count": len(records),
        "observation_count": sum(item["state_count"] for item in records),
        "action_count": sum(item["action_count"] for item in records),
        "window_count": sum(item["window_count"] for item in records),
        "action_min": min(item["action_min"] for item in records),
        "action_max": max(item["action_max"] for item in records),
        "ranks": sorted(ranks),
        "shards": shards,
        "records": records,
    }
    atomic_json(manifest_path, manifest)
    complete_path.write_text(file_sha256(manifest_path) + "\n", encoding="utf-8")
    return manifest


class FrozenTrajectoryCache:
    """Lazy, hash-verified access to a sealed trajectory cache."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        manifest_path, complete_path = self.directory / "manifest.json", self.directory / "COMPLETE"
        if not manifest_path.is_file() or not complete_path.is_file():
            raise FileNotFoundError(f"incomplete frozen-WM cache: {self.directory}")
        if complete_path.read_text(encoding="utf-8").strip() != file_sha256(manifest_path):
            raise ValueError(f"frozen-WM cache completion digest mismatch: {self.directory}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"unsupported frozen-WM cache: {self.directory}")
        self.manifest_sha256 = file_sha256(manifest_path)
        for shard in self.manifest["shards"]:
            path = self.directory / shard["path"]
            if not path.is_file() or file_sha256(path) != shard["sha256"]:
                raise ValueError(f"frozen-WM shard identity mismatch: {path}")
        self.records = tuple(self.manifest["records"])

    def load(self, index: int) -> dict:
        entry = self.records[index]
        payload = _load_shard(self.directory / entry["shard"])
        record = payload["records"][int(entry["record_index"])]
        summary = _validate_record(
            record,
            horizon=int(self.manifest["prediction_horizon"]),
            path=self.directory / entry["shard"],
        )
        if summary["trajectory_id"] != entry["trajectory_id"]:
            raise ValueError("cache index points to a different trajectory")
        return record


def _windows(record: dict, *, mode: str, horizon: int) -> dict[str, torch.Tensor]:
    if mode not in MODES:
        raise ValueError(f"unknown frozen-WM mode: {mode}")
    source = record["states"] if mode == "stage2_state" else record["dino"]
    count = source.shape[0] - horizon
    starts = torch.arange(count)
    future = starts[:, None] + torch.arange(1, horizon + 1)
    action_index = starts[:, None] + torch.arange(horizon)
    return {
        "current": source[starts],
        "target": source[future],
        "dino": record["dino"][future],
        "current_dino": record["dino"][starts],
        "actions": record["actions"][action_index],
    }


def _batch_records(
    cache: FrozenTrajectoryCache,
    indices: Iterable[int],
    *,
    mode: str,
    donor_perturbations: bool = True,
) -> dict[str, object]:
    rows = []
    horizon = int(cache.manifest["prediction_horizon"])
    for index in indices:
        record = cache.load(int(index))
        row = _windows(record, mode=mode, horizon=horizon)
        row["trajectory_id"] = str(record["trajectory_id"])
        rows.append(row)
    if not rows:
        raise ValueError("empty trajectory batch")
    current = torch.cat([row["current"] for row in rows])
    target = torch.cat([row["target"] for row in rows])
    dino = torch.cat([row["dino"] for row in rows])
    current_dino = torch.cat([row["current_dino"] for row in rows])
    actions = torch.cat([row["actions"] for row in rows])
    trajectory_ids = [row["trajectory_id"] for row in rows for _ in range(row["current"].shape[0])]

    result = {
        "current": current,
        "target": target,
        "dino": dino,
        "current_dino": current_dino,
        "actions": actions,
        "trajectory_ids": trajectory_ids,
    }
    if donor_perturbations:
        if len(rows) < 2:
            raise ValueError("cross-trajectory perturbations require at least two trajectories")
        donor_current, donor_actions = [], []
        for index, row in enumerate(rows):
            donor = rows[(index + 1) % len(rows)]
            starts = torch.arange(row["current"].shape[0]) % donor["current"].shape[0]
            donor_current.append(donor["current"][starts])
            donor_actions.append(donor["actions"][starts])
        result.update(
            donor_current=torch.cat(donor_current),
            donor_actions=torch.cat(donor_actions),
        )
    return result


def train_means(cache: FrozenTrajectoryCache) -> dict[str, torch.Tensor | list[int]]:
    """Per-horizon means with the same overlapping-window weighting as evaluation."""
    horizon = int(cache.manifest["prediction_horizon"])
    sums = {mode: [None] * horizon for mode in MODES}
    counts = [0] * horizon
    for index in range(len(cache.records)):
        record = cache.load(index)
        for mode in MODES:
            targets = _windows(record, mode=mode, horizon=horizon)["target"]
            for step in range(horizon):
                total = targets[:, step].double().sum(dim=0)
                sums[mode][step] = total if sums[mode][step] is None else sums[mode][step] + total
        for step in range(horizon):
            counts[step] += int(record["states"].shape[0] - horizon)
    if any(count < 1 for count in counts):
        raise ValueError("training cache has no complete prediction windows")
    return {
        mode: torch.stack([(value / counts[step]).float()
                           for step, value in enumerate(sums[mode])])
        for mode in MODES
    } | {"counts": counts}


def _bootstrap_ci(
    values: dict[str, tuple[float, int]],
    *,
    seed: int,
    draws: int = 1000,
) -> list[float]:
    ordered = sorted(values)
    if not ordered:
        return [float("nan"), float("nan")]
    sums = torch.tensor([values[key][0] for key in ordered], dtype=torch.float64)
    counts = torch.tensor([values[key][1] for key in ordered], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(draws):
        sample = torch.randint(len(ordered), (len(ordered),), generator=generator)
        estimates.append((sums[sample].sum() / counts[sample].sum()).item())
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


@torch.no_grad()
def evaluate_predictor(
    predictor: Predictor,
    cache: FrozenTrajectoryCache,
    *,
    mode: str,
    means: dict[str, torch.Tensor | list[int]],
    device: torch.device,
    trajectory_batch: int = 32,
    seed: int = 42,
) -> dict:
    predictor.eval()
    horizon = int(cache.manifest["prediction_horizon"])
    totals: dict[tuple[str, str, int], list[float]] = {}
    per_trajectory: dict[tuple[str, str, int], dict[str, tuple[float, int]]] = {}
    moments: dict[tuple[str, str, int], dict[str, torch.Tensor | float | int]] = {}

    indices = list(range(len(cache.records)))
    batches = [indices[begin : begin + trajectory_batch]
               for begin in range(0, len(indices), trajectory_batch)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-1].insert(0, batches[-2].pop())
    if any(len(selected) < 2 for selected in batches):
        raise ValueError("evaluation requires at least two trajectories")
    for selected in batches:
        batch = _batch_records(cache, selected, mode=mode)
        current = batch["current"].to(device)
        actions = batch["actions"].to(device)
        empty = torch.empty((current.shape[0], 0), dtype=torch.long, device=device)
        predictions = {
            "model": predictor.rollout_from_history(current[:, None], empty, actions),
            "input_copy": current[:, None].expand(-1, horizon, -1, -1),
            "current_dino_copy": batch["current_dino"].to(device)[:, None].expand(
                -1, horizon, -1, -1
            ),
            "train_mean": means[mode].to(device)[None].expand(current.shape[0], -1, -1, -1),
            "dino_train_mean": means["dino"].to(device)[None].expand(
                current.shape[0], -1, -1, -1
            ),
            "cross_trajectory_state_donor": predictor.rollout_from_history(
                batch["donor_current"].to(device)[:, None], empty, actions
            ),
            "cross_trajectory_action_donor": predictor.rollout_from_history(
                current[:, None], empty, batch["donor_actions"].to(device)
            ),
        }
        spaces = {
            "target_space": batch["target"].to(device),
            "dino_space": batch["dino"].to(device),
        }
        trajectory_ids = batch["trajectory_ids"]
        for name, prediction in predictions.items():
            for space, target in spaces.items():
                for step in range(horizon):
                    error = (prediction[:, step].float() - target[:, step].float()).square()
                    row_sums = error.flatten(1).sum(dim=1).cpu()
                    row_count = error[0].numel()
                    cosine = F.cosine_similarity(
                        prediction[:, step].float().flatten(1),
                        target[:, step].float().flatten(1),
                    ).cpu()
                    key = (name, space, step + 1)
                    aggregate = totals.setdefault(key, [0.0, 0.0, 0.0])
                    aggregate[0] += float(row_sums.sum())
                    aggregate[1] += row_count * len(row_sums)
                    aggregate[2] += float(cosine.sum())
                    by_trajectory = per_trajectory.setdefault(key, {})
                    for identity, value in zip(trajectory_ids, row_sums.tolist(), strict=True):
                        previous = by_trajectory.get(identity, (0.0, 0))
                        by_trajectory[identity] = (previous[0] + value, previous[1] + row_count)
                    value = prediction[:, step].double().cpu()
                    truth = target[:, step].double().cpu()
                    stat = moments.setdefault(key, {
                        "prediction_sum": torch.zeros_like(value[0]),
                        "target_sum": torch.zeros_like(truth[0]),
                        "prediction_square_sum": 0.0,
                        "target_square_sum": 0.0,
                        "cross_sum": 0.0,
                        "rows": 0,
                    })
                    stat["prediction_sum"] += value.sum(dim=0)
                    stat["target_sum"] += truth.sum(dim=0)
                    stat["prediction_square_sum"] += float(value.square().sum())
                    stat["target_square_sum"] += float(truth.square().sum())
                    stat["cross_sum"] += float((value * truth).sum())
                    stat["rows"] += value.shape[0]

    result = {
        "schema": "frozen_wm_metrics_v1",
        "mode": mode,
        "cache_manifest_sha256": cache.manifest_sha256,
        "trajectory_count": len(cache.records),
        "window_count": int(cache.manifest["window_count"]),
        "by_horizon": {},
    }
    for (name, space, step), (squared_sum, count, cosine_sum) in sorted(totals.items()):
        output = result["by_horizon"].setdefault(str(step), {}).setdefault(name, {})
        stat = moments[(name, space, step)]
        rows = int(stat["rows"])
        prediction_centered_ss = float(stat["prediction_square_sum"]) - float(
            stat["prediction_sum"].square().sum()
        ) / rows
        target_centered_ss = float(stat["target_square_sum"]) - float(
            stat["target_sum"].square().sum()
        ) / rows
        prediction_centered_ss = max(0.0, prediction_centered_ss)
        target_centered_ss = max(0.0, target_centered_ss)
        centered_cross = float(stat["cross_sum"]) - float(
            (stat["prediction_sum"] * stat["target_sum"]).sum()
        ) / rows
        output[space] = {
            "mse": squared_sum / count,
            "cosine": cosine_sum / (
                count / (cache.manifest["grid_tokens"] * cache.manifest["state_dim"])
            ),
            "trajectory_bootstrap_mse_95ci": _bootstrap_ci(
                per_trajectory[(name, space, step)], seed=seed + step
            ),
            "bootstrap_unit": "trajectory_id",
            "bootstrap_draws": 1000,
            "observation_variance_ratio": (
                prediction_centered_ss / target_centered_ss if target_centered_ss > 0 else None
            ),
            "global_centered_cosine": (
                centered_cross / math.sqrt(prediction_centered_ss * target_centered_ss)
                if prediction_centered_ss > 0 and target_centered_ss > 0 else None
            ),
        }
    result["perturbation_contract"] = {
        "coverage": 1.0,
        "donor_identity": "different_trajectory_id",
        "task_or_action_matched": False,
        "note": "cache lacks task metadata; donor controls are cross-trajectory, not task-matched",
    }
    return result


@dataclass(frozen=True)
class DiagnosticConfig:
    mode: str
    predictor_kind: str = "direct"
    steps: int = 46
    effective_batch: int = 64
    trajectory_microbatch: int = 8
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    wm_weight_start: float = 0.1
    wm_weight_end: float = 1.0
    wm_warmup_fraction: float = 0.3
    seed: int = 42
    checkpoint_steps: tuple[int, ...] = (1, 5, 10, 46)


def _schedule(cache: FrozenTrajectoryCache, config: DiagnosticConfig, step: int) -> list[int]:
    updates_per_epoch = math.ceil(len(cache.records) / config.effective_batch)
    epoch, offset = divmod(step, updates_per_epoch)
    permutation = list(range(len(cache.records)))
    random.Random(config.seed + epoch).shuffle(permutation)
    selected = permutation[
        offset * config.effective_batch : (offset + 1) * config.effective_batch
    ]
    if not selected:
        raise ValueError("optimizer batch must contain a real trajectory")
    return selected


def _checkpoint(
    output: Path,
    *,
    step: int,
    predictor: Predictor,
    optimizer: torch.optim.Optimizer,
    run_identity: dict,
) -> Path:
    destination = output / f"step_{step:06d}"
    temporary = output / f".step_{step:06d}.tmp"
    if destination.exists() or temporary.exists():
        raise FileExistsError(destination)
    temporary.mkdir()
    torch.save(predictor.state_dict(), temporary / "predictor.pt")
    torch.save(
        {
            "schema": RUN_SCHEMA,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "run_identity": run_identity,
        },
        temporary / "training_state.pt",
    )
    (temporary / "COMMITTED").write_text("complete\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def _load_checkpoint(
    checkpoint: Path,
    *,
    predictor: Predictor,
    optimizer: torch.optim.Optimizer,
    run_identity: dict,
) -> int:
    if not (checkpoint / "COMMITTED").is_file():
        raise ValueError(f"diagnostic checkpoint is not committed: {checkpoint}")
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("schema") != RUN_SCHEMA or state.get("run_identity") != run_identity:
        raise ValueError("diagnostic resume identity mismatch")
    predictor.load_state_dict(torch.load(checkpoint / "predictor.pt", weights_only=True))
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    return int(state["step"])


def convergence_update(history: dict, *, step: int, metrics: dict) -> dict:
    """Adjacent epoch improvement, including regressions as insufficient progress."""
    values = [row["model"]["target_space"]["mse"]
              for row in metrics["by_horizon"].values()]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("non-finite/negative or empty convergence metrics")
    metric = sum(values) / len(values)
    if not math.isfinite(metric) or metric < 0:
        raise ValueError("non-finite/negative convergence metric")
    previous = history["evaluations"][-1]["metric"] if history["evaluations"] else None
    improvement = ((previous - metric) / previous if previous else 0.0)
    insufficient = history["insufficient"] + 1 if previous is not None and improvement < .01 else 0
    evaluations = history["evaluations"] + [
        {"step": step, "metric": metric, "relative_improvement": improvement}
    ]
    best = min(evaluations, key=lambda row: row["metric"])
    return {"evaluations": evaluations, "insufficient": insufficient,
            "best_step": best["step"], "status": "converged" if insufficient >= 2 else "running"}


def continuation_source(checkpoint: Path, identity: dict) -> dict:
    """Verify completed source and pin every consumed restoration artifact."""
    run = json.loads((checkpoint.parent / "run.json").read_text())
    if run != identity or not (checkpoint / "COMMITTED").is_file():
        raise ValueError("continuation source identity mismatch")
    if (checkpoint.parent / "COMPLETE").read_text().strip() != checkpoint.name:
        raise ValueError("continuation requires the completed source checkpoint")
    metrics = json.loads((checkpoint / "metrics.json").read_text())
    if (metrics.get("cache_manifest_sha256") != identity["eval_manifest_sha256"]
            or metrics.get("mode") != identity["config"]["mode"]):
        raise ValueError("continuation source metrics identity mismatch")
    return {"checkpoint": str(checkpoint.resolve()), "sha256": {
        name: file_sha256(checkpoint / name)
        for name in ("predictor.pt", "training_state.pt", "metrics.json", "COMMITTED")
    }, "run_sha256": file_sha256(checkpoint.parent / "run.json")}


def train(
    train_cache: FrozenTrajectoryCache,
    eval_cache: FrozenTrajectoryCache,
    output: Path,
    *,
    config: DiagnosticConfig,
    device: torch.device,
    resume: Path | None = None,
    predictor_config: GridPredictorConfig | None = None,
    continue_from: Path | None = None,
    walltime_seconds: float = 3600,
) -> dict:
    if (config.mode not in MODES or config.predictor_kind not in PREDICTOR_TYPES
            or config.steps < 1 or config.effective_batch < 2
            or not 1 <= config.trajectory_microbatch <= config.effective_batch
            or not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0
            or not 0 <= config.wm_weight_start <= config.wm_weight_end
            or not 0 <= config.wm_warmup_fraction <= 1):
        raise ValueError("invalid frozen-WM diagnostic configuration")
    for field in ("prediction_horizon", "grid_tokens", "state_dim"):
        if train_cache.manifest[field] != eval_cache.manifest[field]:
            raise ValueError(f"train/eval cache {field} mismatch")
    if {row["trajectory_id"] for row in train_cache.records} & {
        row["trajectory_id"] for row in eval_cache.records
    }:
        raise ValueError("train/eval frozen-WM caches overlap by trajectory identity")
    provenance_fields = (
        "stage2_checkpoint",
        "stage2_policy_fingerprint",
        "stage2_config_sha256",
        "stage2_commit_marker_sha256",
        "stage2_grid_config_sha256",
        "stage2_projector_sha256",
        "dino_cache_fingerprint",
        "prediction_horizon",
        "grid_tokens",
        "state_dim",
        "action_dim",
        "source_commit",
    )
    provenance_mismatches = {
        field: (
            train_cache.manifest["identity"].get(field),
            eval_cache.manifest["identity"].get(field),
        )
        for field in provenance_fields
        if train_cache.manifest["identity"].get(field)
        != eval_cache.manifest["identity"].get(field)
    }
    if provenance_mismatches:
        raise ValueError(
            f"train/eval frozen-WM cache provenance mismatch: {provenance_mismatches}"
        )

    default_predictor = GridPredictorConfig(
        grid_tokens=int(train_cache.manifest["grid_tokens"]),
        emb_dim=int(train_cache.manifest["state_dim"]),
        action_dim=8,
        history_size=1,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
    )
    predictor_config = predictor_config or default_predictor
    if (predictor_config.grid_tokens != default_predictor.grid_tokens
            or predictor_config.emb_dim != default_predictor.emb_dim):
        raise ValueError("predictor grid contract does not match cache")
    if predictor_config.history_size != 1:
        raise ValueError("frozen-WM diagnostic preserves production H=1")
    if (int(train_cache.manifest["action_min"]) < 0
            or int(train_cache.manifest["action_max"]) >= predictor_config.action_dim
            or int(eval_cache.manifest["action_min"]) < 0
            or int(eval_cache.manifest["action_max"]) >= predictor_config.action_dim):
        raise ValueError("cached action index is outside the production predictor action space")
    config_identity = json.loads(json.dumps(asdict(config)))
    # Preserve exact identities of existing direct runs for faithful resume.
    if config.predictor_kind == "direct":
        config_identity.pop("predictor_kind")
    predictor_type = PREDICTOR_TYPES[config.predictor_kind]
    run_identity = {
        "schema": RUN_SCHEMA,
        "config": config_identity,
        "predictor_config": asdict(predictor_config),
        "train_manifest_sha256": train_cache.manifest_sha256,
        "eval_manifest_sha256": eval_cache.manifest_sha256,
        "trainable_modules": [predictor_type.__name__],
        "target_is_fixed": True,
        "dtype": "float32",
        "schedule": "production_python_shuffle_effective_batch_v1",
    }

    source_identity = run_identity
    convergence = None
    source_step = 0
    if continue_from is not None:
        if not math.isfinite(walltime_seconds) or walltime_seconds <= 0:
            raise ValueError("walltime must be positive and finite")
        provenance = continuation_source(continue_from, source_identity)
        source_step = int(config.steps)
        epoch_steps = math.ceil(len(train_cache.records) / config.effective_batch)
        if source_step % epoch_steps:
            raise ValueError("continuation source must end on an epoch boundary")
        run_identity = {**run_identity, "continuation": {
            "source": provenance, "epoch_steps": epoch_steps,
            "metric": "mean_horizon_model_target_space_mse",
            "relative_threshold": .01, "patience": 2,
            "warmup_steps": max(1, int(config.steps * config.wm_warmup_fraction)),
        }}
        convergence = convergence_update(
            {"evaluations": [], "insufficient": 0}, step=source_step,
            metrics=json.loads((continue_from / "metrics.json").read_text()),
        )
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "run.json", run_identity)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    else:
        if not output.is_dir() or json.loads((output / "run.json").read_text()) != run_identity:
            raise ValueError("resume output identity mismatch")
        if (output / "COMPLETE").exists():
            raise ValueError("cannot resume a completed frozen-WM diagnostic")
    predictor = predictor_type(predictor_config).to(
        device=device, dtype=torch.float32
    )
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=config.learning_rate)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimized != {id(parameter) for parameter in predictor.parameters()}:
        raise AssertionError("only and all WM predictor parameters must be optimized")
    restore = resume or continue_from
    start_step = 0 if restore is None else _load_checkpoint(
        restore, predictor=predictor, optimizer=optimizer,
        run_identity=run_identity if resume is not None else source_identity,
    )
    if continue_from is not None:
        if resume is None and start_step != source_step:
            raise ValueError("source step does not match original budget")
        if resume is not None:
            convergence = json.loads((resume / "convergence.json").read_text())
            if convergence["evaluations"][-1]["step"] != start_step:
                raise ValueError("convergence history/checkpoint mismatch")
            if convergence["status"] == "converged":
                raise ValueError("checkpoint already converged")
    if continue_from is None and start_step >= config.steps:
        raise ValueError("resume checkpoint already reached the requested update budget")
    means = train_means(train_cache)
    atomic_json(
        output / "train_means.json",
        {
            "window_count_by_horizon": means["counts"],
            "stage2_state_mean_sha256": hashlib.sha256(
                means["stage2_state"].numpy().tobytes()
            ).hexdigest(),
            "dino_mean_sha256": hashlib.sha256(means["dino"].numpy().tobytes()).hexdigest(),
        },
    ) if resume is None else None

    log_path = output / "train_steps.jsonl"
    if resume is not None:
        if not log_path.is_file():
            raise ValueError("resume requires the existing train-step log")
        logged_steps = [
            int(json.loads(line)["step"])
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if logged_steps != list(range(source_step + 1, start_step + 1)):
            raise ValueError(
                "resume train-step log must end exactly at the checkpoint step"
            )
        later_checkpoints = [
            path
            for path in output.glob("step_*")
            if path.is_dir() and int(path.name.removeprefix("step_")) > start_step
        ]
        if later_checkpoints:
            raise ValueError("resume output contains checkpoints newer than the selected boundary")
    checkpoints = set(config.checkpoint_steps) | {config.steps}
    started_at = time.monotonic()
    if continue_from is not None:
        replace_run_progress_json(output / "status.json", {
            "status": "running", "last_step": start_step, "converged": False,
        })
    step_indices = (itertools.count(start_step) if continue_from is not None
                    else range(start_step, config.steps))
    for step_index in step_indices:
        predictor.train()
        selected = _schedule(train_cache, config, step_index)
        optimizer.zero_grad(set_to_none=True)
        window_counts = [int(train_cache.records[index]["window_count"]) for index in selected]
        total_windows = sum(window_counts)
        weighted_loss = 0.0
        warmup_steps = max(1, int(config.steps * config.wm_warmup_fraction))
        if step_index >= warmup_steps:
            wm_weight = config.wm_weight_end
        else:
            progress = step_index / warmup_steps
            wm_weight = config.wm_weight_start + (
                config.wm_weight_end - config.wm_weight_start
            ) * 0.5 * (1.0 - math.cos(math.pi * progress))
        for begin in range(0, len(selected), config.trajectory_microbatch):
            micro_indices = selected[begin : begin + config.trajectory_microbatch]
            batch = _batch_records(
                train_cache, micro_indices, mode=config.mode, donor_perturbations=False
            )
            current = batch["current"].to(device)
            actions = batch["actions"].to(device)
            target = batch["target"].to(device)
            empty = torch.empty((current.shape[0], 0), dtype=torch.long, device=device)
            predicted = predictor.rollout_from_history(current[:, None], empty, actions)
            micro_loss = (predicted.float() - target.float()).square().mean()
            if not torch.isfinite(micro_loss):
                raise FloatingPointError(f"non-finite frozen-WM loss at step {step_index + 1}")
            scale = current.shape[0] / total_windows
            (micro_loss * scale * wm_weight).backward()
            weighted_loss += float(micro_loss.detach()) * scale
        if any(
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
            for parameter in predictor.parameters()
        ):
            raise FloatingPointError(f"missing/non-finite WM gradient at step {step_index + 1}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            predictor.parameters(), config.max_grad_norm
        )
        optimizer.step()
        completed = step_index + 1
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "step": completed,
                "loss": weighted_loss,
                "weighted_loss": weighted_loss * wm_weight,
                "lambda_wm": wm_weight,
                "trajectory_count": len(selected),
                "trajectory_microbatch": config.trajectory_microbatch,
                "window_count": total_windows,
                "gradient_norm_before_clip": float(gradient_norm),
                "learning_rate": config.learning_rate,
            }, allow_nan=False) + "\n")
        at_epoch = continue_from is not None and completed % epoch_steps == 0
        if at_epoch or (continue_from is None and completed in checkpoints):
            checkpoint = _checkpoint(
                output,
                step=completed,
                predictor=predictor,
                optimizer=optimizer,
                run_identity=run_identity,
            )
            metrics = evaluate_predictor(
                predictor,
                eval_cache,
                mode=config.mode,
                means=means,
                device=device,
                seed=config.seed,
            )
            atomic_json(checkpoint / "metrics.json", metrics)
            if continue_from is not None:
                convergence = convergence_update(convergence, step=completed, metrics=metrics)
                atomic_json(checkpoint / "convergence.json", convergence)
                replace_run_progress_json(output / "convergence.json", convergence)
                replace_run_progress_json(output / "pointers.json", {
                    "last": str(checkpoint.resolve()),
                    "best": str((continue_from if convergence["best_step"] == source_step
                                 else output / f"step_{convergence['best_step']:06d}").resolve()),
                })
                if convergence["status"] == "converged":
                    replace_run_progress_json(output / "status.json", {
                        "status": "converged", "last_step": completed, "converged": True,
                    })
                    (output / "COMPLETE").write_text(checkpoint.name + "\n")
                    return metrics
                low_disk = shutil.disk_usage(output).free < 10 * 1024 ** 3
                if low_disk or time.monotonic() - started_at >= walltime_seconds:
                    replace_run_progress_json(output / "status.json", {
                        "status": "disk_paused" if low_disk else "walltime_paused",
                        "last_step": completed,
                        "converged": False,
                    })
                    return metrics
    final = output / f"step_{config.steps:06d}"
    (output / "COMPLETE").write_text(final.name + "\n", encoding="utf-8")
    return json.loads((final / "metrics.json").read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-cache")
    seal.add_argument("--directory", type=Path, required=True)
    seal.add_argument("--expected-ranks", type=int)
    run = sub.add_parser("train")
    run.add_argument("--train-cache", type=Path, required=True)
    run.add_argument("--eval-cache", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--predictor-kind", choices=tuple(PREDICTOR_TYPES), default="direct")
    run.add_argument("--steps", type=int, default=46)
    run.add_argument("--effective-batch", type=int, default=64)
    run.add_argument("--trajectory-microbatch", type=int, default=8)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--wm-weight-start", type=float, default=0.1)
    run.add_argument("--wm-weight-end", type=float, default=1.0)
    run.add_argument("--wm-warmup-fraction", type=float, default=0.3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--checkpoint-steps", type=int, nargs="+", default=[1, 5, 10, 46])
    run.add_argument("--device", default="cuda")
    run.add_argument("--resume", type=Path)
    run.add_argument("--continue-from", type=Path)
    run.add_argument("--walltime-seconds", type=float, default=3600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "seal-cache":
        print(json.dumps(seal_cache(args.directory, expected_ranks=args.expected_ranks), indent=2))
        return 0
    config = DiagnosticConfig(
        mode=args.mode,
        predictor_kind=args.predictor_kind,
        steps=args.steps,
        effective_batch=args.effective_batch,
        trajectory_microbatch=args.trajectory_microbatch,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        wm_weight_start=args.wm_weight_start,
        wm_weight_end=args.wm_weight_end,
        wm_warmup_fraction=args.wm_warmup_fraction,
        seed=args.seed,
        checkpoint_steps=tuple(args.checkpoint_steps),
    )
    metrics = train(
        FrozenTrajectoryCache(args.train_cache),
        FrozenTrajectoryCache(args.eval_cache),
        args.output,
        config=config,
        device=torch.device(args.device),
        resume=args.resume,
        continue_from=args.continue_from,
        walltime_seconds=args.walltime_seconds,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
