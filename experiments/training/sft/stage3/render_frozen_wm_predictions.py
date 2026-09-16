"""Render matched frozen-WM predictions against real future DINO grids.

This is a run-owned diagnostic.  It reads sealed frozen-WM evaluation features
and two completed WM-only checkpoints; it never loads or updates the language
model, Query embeddings, projector, or training state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from experiments.training.sft.stage3.frozen_wm_diagnostic import (
    FrozenTrajectoryCache,
    Predictor,
    file_sha256,
    predictor_type_from_run,
)
from experiments.training.sft.stage3.render_dino_feature_comparison import (
    error_image,
    feature_image,
    pca_basis,
)
from nimloth.wm.grid import GridPredictorConfig

SCHEMA = "frozen_wm_visualization_v1"


def load_image_index(jsonl: Path) -> tuple[dict[str, dict], int]:
    """Load unique trajectory/image identities without accepting ambiguous joins."""
    records: dict[str, dict] = {}
    with jsonl.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            identity = str(record.get("id", ""))
            paths = record.get("image_paths")
            actions = record.get("action_indices")
            if (
                not identity
                or not isinstance(paths, list)
                or not paths
                or not isinstance(actions, list)
            ):
                raise ValueError(f"invalid image identity at {jsonl}:{line_number}")
            if identity in records:
                raise ValueError(f"duplicate eval trajectory identity: {identity}")
            resolved = tuple(
                Path(value) if Path(value).is_absolute() else (jsonl.parent / value).resolve()
                for value in paths
                if isinstance(value, str) and value
            )
            if len(resolved) != len(paths):
                raise ValueError(f"invalid image path at {jsonl}:{line_number}")
            records[identity] = {
                "image_paths": resolved,
                "action_count": len(actions),
            }
    if not records:
        raise ValueError(f"no eval records found: {jsonl}")
    return records, len(records)


def join_cache_images(
    cache: FrozenTrajectoryCache,
    images: dict[str, dict],
    *,
    prediction_horizon: int,
) -> list[dict]:
    """Join the exact horizon-eligible JSONL subset to cached states and images."""
    if prediction_horizon < 1:
        raise ValueError("prediction horizon must be positive")
    joined = []
    cache_identities = [str(entry["trajectory_id"]) for entry in cache.records]
    if len(set(cache_identities)) != len(cache_identities):
        raise ValueError("sealed cache contains duplicate trajectory identities")
    eligible_identities = {
        identity
        for identity, record in images.items()
        if int(record["action_count"]) >= prediction_horizon
    }
    if set(cache_identities) != eligible_identities:
        missing = sorted(eligible_identities - set(cache_identities))
        extra = sorted(set(cache_identities) - eligible_identities)
        raise ValueError(
            "eligible eval JSONL/cache trajectory identities differ: "
            f"missing_from_cache={missing[:3]}, unexpected_in_cache={extra[:3]}"
        )
    for cache_index, entry in enumerate(cache.records):
        identity = str(entry["trajectory_id"])
        state_count = int(entry["state_count"])
        action_count = int(entry["action_count"])
        source = images[identity]
        paths = source["image_paths"]
        if len(paths) != state_count:
            raise ValueError(
                f"image/state count mismatch for {identity}: "
                f"images={len(paths)}, states={state_count}"
            )
        if int(source["action_count"]) != action_count:
            raise ValueError(
                f"JSONL/cache action count mismatch for {identity}: "
                f"JSONL actions={source['action_count']}, cache actions={action_count}"
            )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing observation images for {identity}: {missing[:3]}"
            )
        joined.append(
            {
                "cache_index": cache_index,
                "trajectory": identity,
                "state_count": state_count,
                "image_paths": paths,
            }
        )
    return joined


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return [round(index * (count - 1) / (limit - 1)) for index in range(limit)]


def select_rows(
    joined: list[dict], *, horizon: int, limit: int = 8
) -> list[dict]:
    """Select deterministic trajectory-diverse rows without looking at predictions."""
    if horizon < 1 or limit < 1:
        raise ValueError("horizon and selection limit must be positive")
    eligible = sorted(
        (row for row in joined if row["state_count"] > horizon),
        key=lambda row: row["trajectory"],
    )
    if not eligible:
        raise ValueError(f"no trajectory has a complete horizon-{horizon} transition")
    selected = []
    for index in _evenly_spaced_indices(len(eligible), min(limit, len(eligible))):
        row = eligible[index]
        window_count = row["state_count"] - horizon
        digest = hashlib.sha256(f"{row['trajectory']}:{horizon}".encode()).digest()
        window_start = int.from_bytes(digest[:8], "big") % window_count
        selected.append(
            {
                **row,
                "window_start": window_start,
                "horizon_step": horizon,
                "future_index": window_start + horizon,
            }
        )
    return selected


def _load_predictor(checkpoint: Path, *, expected_mode: str, device: torch.device):
    run_path = checkpoint.parent / "run.json"
    committed = checkpoint / "COMMITTED"
    predictor_path = checkpoint / "predictor.pt"
    if not run_path.is_file() or not committed.is_file() or not predictor_path.is_file():
        raise FileNotFoundError(f"incomplete frozen-WM checkpoint: {checkpoint}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    training_state = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    complete_path = checkpoint.parent / "COMPLETE"
    if run.get("schema") != "frozen_wm_diagnostic_v1":
        raise ValueError(f"unsupported frozen-WM run identity: {run_path}")
    if run.get("config", {}).get("mode") != expected_mode:
        raise ValueError(
            f"checkpoint mode is not {expected_mode}: {run.get('config', {}).get('mode')}"
        )
    expected_step = int(run.get("config", {}).get("steps", -1))
    terminal_name = complete_path.read_text().strip() if complete_path.is_file() else ""
    if "continuation" in run:
        history = json.loads((checkpoint.parent / "convergence.json").read_text())
        pointers = json.loads((checkpoint.parent / "pointers.json").read_text())
        terminal_step = history["evaluations"][-1]["step"]
        expected_step = int(training_state.get("step", -1))
        valid_terminal = (
            history["status"] == "converged" and history["insufficient"] >= 2
            and len(history["evaluations"]) >= 3
            and terminal_name == f"step_{terminal_step:06d}"
            and checkpoint.resolve() in {
                Path(pointers["last"]).resolve(), Path(pointers["best"]).resolve()
            }
            and expected_step in {terminal_step, history["best_step"]}
            and Path(pointers["last"]).resolve()
            == (checkpoint.parent / terminal_name).resolve()
        )
    else:
        valid_terminal = expected_step == 46 and terminal_name == checkpoint.name
    if (
        not valid_terminal
        or checkpoint.name != f"step_{expected_step:06d}"
        or training_state.get("schema") != "frozen_wm_diagnostic_v1"
        or int(training_state.get("step", -1)) != expected_step
        or training_state.get("run_identity") != run
    ):
        raise ValueError(f"checkpoint is not a verified completed frozen-WM run: {checkpoint}")
    config = GridPredictorConfig(**run["predictor_config"])
    predictor = predictor_type_from_run(run)(config).to(device=device, dtype=torch.float32)
    predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))
    predictor.eval()
    return predictor, config, run


def _checkpoint_identity(checkpoint: Path) -> dict:
    files = ("predictor.pt", "training_state.pt", "COMMITTED")
    missing = [name for name in files if not (checkpoint / name).is_file()]
    run_path = checkpoint.parent / "run.json"
    if missing or not run_path.is_file():
        raise FileNotFoundError(
            f"incomplete frozen-WM checkpoint identity: {checkpoint}; missing={missing}"
        )
    return {
        "path": str(checkpoint.resolve()),
        "predictor_sha256": file_sha256(checkpoint / "predictor.pt"),
        "training_state_sha256": file_sha256(checkpoint / "training_state.pt"),
        "committed_sha256": file_sha256(checkpoint / "COMMITTED"),
        "run_json_sha256": file_sha256(run_path),
    }


@torch.no_grad()
def predict_selected(
    cache: FrozenTrajectoryCache,
    selected: list[dict],
    *,
    stage2_predictor: Predictor,
    dino_predictor: Predictor,
    device: torch.device,
) -> list[dict]:
    output = []
    for row in selected:
        record = cache.load(int(row["cache_index"]))
        start, horizon = int(row["window_start"]), int(row["horizon_step"])
        actions = record["actions"][start : start + horizon][None].to(device)
        empty = torch.empty((1, 0), dtype=torch.long, device=device)
        state_prediction = stage2_predictor.rollout_from_history(
            record["states"][start][None, None].to(device), empty, actions
        )[0, -1].float().cpu()
        dino_prediction = dino_predictor.rollout_from_history(
            record["dino"][start][None, None].to(device), empty, actions
        )[0, -1].float().cpu()
        target = record["dino"][start + horizon].float()
        output.append(
            {
                **row,
                "actions": record["actions"][start : start + horizon].tolist(),
                "target": target,
                "current_dino_copy": record["dino"][start].float(),
                "stage2_state_wm": state_prediction,
                "dino_wm": dino_prediction,
            }
        )
    return output


def _paste_raw(canvas: Image.Image, path: Path, *, x: int, y: int, size: int) -> None:
    with Image.open(path) as source:
        image = source.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        left = x + (size - image.width) // 2
        top = y + (size - image.height) // 2
        canvas.paste(image, (left, top))


def render_pages(
    rows: list[dict],
    *,
    mean: np.ndarray,
    basis: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    error_high: float,
    output: Path,
) -> list[str]:
    columns = (
        "current obs",
        "future obs",
        "true future DINO",
        "current DINO copy",
        "Stage2-state WM",
        "DINO-only WM",
        "state-WM error",
        "DINO-WM error",
    )
    files = []
    for horizon in sorted({int(row["horizon_step"]) for row in rows}):
        page_rows = [row for row in rows if int(row["horizon_step"]) == horizon]
        cell, label_height, row_height = 128, 42, 160
        canvas = Image.new(
            "RGB", (len(columns) * cell, label_height + len(page_rows) * row_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for column, label in enumerate(columns):
            draw.text((column * cell + 4, 8), label, fill="black")
        for row_index, row in enumerate(page_rows):
            y = label_height + row_index * row_height
            paths = row["image_paths"]
            _paste_raw(canvas, paths[row["window_start"]], x=0, y=y, size=cell)
            _paste_raw(canvas, paths[row["future_index"]], x=cell, y=y, size=cell)
            for column, name in enumerate(
                ("target", "current_dino_copy", "stage2_state_wm", "dino_wm"), start=2
            ):
                canvas.paste(
                    feature_image(row[name], mean, basis, low, high, cell),
                    (column * cell, y),
                )
            canvas.paste(
                error_image(row["stage2_state_wm"], row["target"], error_high, cell),
                (6 * cell, y),
            )
            canvas.paste(
                error_image(row["dino_wm"], row["target"], error_high, cell),
                (7 * cell, y),
            )
            label = (
                f"{row['trajectory']} start={row['window_start']} "
                f"future={row['future_index']} actions={row['actions']}"
            )
            draw.text((4, y + cell + 2), label[:150], fill="black")
        path = output / f"horizon_{horizon}.png"
        canvas.save(path)
        files.append(path.name)
    return files


def render(
    *,
    eval_cache: Path,
    eval_jsonl: Path,
    stage2_checkpoint: Path,
    dino_checkpoint: Path,
    output: Path,
    rows_per_horizon: int = 8,
    device: torch.device,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    cache = FrozenTrajectoryCache(eval_cache)
    if (
        cache.manifest.get("prediction_horizon") != 4
        or cache.manifest.get("grid_tokens") != 64
        or cache.manifest.get("state_dim") != 1024
        or cache.manifest.get("identity", {}).get("split") != "eval"
        or cache.manifest.get("identity", {}).get("split_sha256")
        != file_sha256(eval_jsonl)
    ):
        raise ValueError(
            "requested JSONL/cache do not match the sealed eval T4/K64/D1024 contract"
        )
    images, jsonl_record_count = load_image_index(eval_jsonl)
    ineligible_jsonl_identities = sorted(
        identity
        for identity, record in images.items()
        if int(record["action_count"]) < int(cache.manifest["prediction_horizon"])
    )
    joined = join_cache_images(
        cache,
        images,
        prediction_horizon=int(cache.manifest["prediction_horizon"]),
    )
    stage2_predictor, stage2_config, stage2_run = _load_predictor(
        stage2_checkpoint, expected_mode="stage2_state", device=device
    )
    dino_predictor, dino_config, dino_run = _load_predictor(
        dino_checkpoint, expected_mode="dino", device=device
    )
    if predictor_type_from_run(stage2_run) != predictor_type_from_run(dino_run):
        raise ValueError("frozen-WM checkpoints use different predictor kinds")
    if stage2_config != dino_config:
        raise ValueError("frozen-WM checkpoints use different predictor architectures")
    expected_predictor_config = GridPredictorConfig(
        grid_tokens=64,
        emb_dim=1024,
        action_dim=8,
        history_size=1,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
    )
    if stage2_config != expected_predictor_config:
        raise ValueError("checkpoint does not use the production frozen-WM predictor config")
    if (
        stage2_run.get("eval_manifest_sha256") != cache.manifest_sha256
        or dino_run.get("eval_manifest_sha256") != cache.manifest_sha256
    ):
        raise ValueError("checkpoint eval-cache identity differs from requested sealed cache")

    selected = []
    for horizon in range(1, int(cache.manifest["prediction_horizon"]) + 1):
        selected.extend(select_rows(joined, horizon=horizon, limit=rows_per_horizon))
    predicted = predict_selected(
        cache,
        selected,
        stage2_predictor=stage2_predictor,
        dino_predictor=dino_predictor,
        device=device,
    )
    # This follows the existing diagnostic renderer: fit one common basis and
    # scale on only the real targets in the rows that will be displayed.  No
    # model output participates in PCA fitting or percentile selection.
    mean, basis, low, high, explained = pca_basis(predicted)
    slot_errors = np.concatenate(
        [
            (row[name] - row["target"]).square().mean(dim=-1).numpy()
            for row in predicted
            for name in ("stage2_state_wm", "dino_wm")
        ]
    )
    error_high = float(np.percentile(slot_errors, 99))
    output.mkdir(parents=True)
    pca_path = output / "pca_basis.npz"
    np.savez(
        pca_path,
        mean=mean,
        basis=basis,
        low=low,
        high=high,
        explained_variance_ratio=explained,
    )
    files = render_pages(
        predicted,
        mean=mean,
        basis=basis,
        low=low,
        high=high,
        error_high=error_high,
        output=output,
    )
    rendered_files = [
        {"path": name, "sha256": file_sha256(output / name)} for name in files
    ]
    manifest = {
        "schema": SCHEMA,
        "scope": "diagnostic visualization; visual appearance is not model-quality evidence",
        "eval_cache": str(eval_cache.resolve()),
        "eval_cache_manifest_sha256": cache.manifest_sha256,
        "eval_jsonl": str(eval_jsonl.resolve()),
        "eval_jsonl_sha256": file_sha256(eval_jsonl),
        "eval_jsonl_record_count": jsonl_record_count,
        "joined_cache_trajectory_count": len(joined),
        "ineligible_jsonl_trajectory_count": len(ineligible_jsonl_identities),
        "ineligible_jsonl_trajectory_ids": ineligible_jsonl_identities,
        "cache_eligibility_rule": "len(action_indices) >= prediction_horizon",
        "stage2_state_checkpoint": _checkpoint_identity(stage2_checkpoint),
        "dino_checkpoint": _checkpoint_identity(dino_checkpoint),
        "predictor_config": stage2_run["predictor_config"],
        "pca_fit": "all displayed real future DINO targets only",
        "pca_target_count": len(predicted),
        "pca_artifact": pca_path.name,
        "pca_artifact_sha256": file_sha256(pca_path),
        "pca_explained_variance_ratio": explained.tolist(),
        "feature_scale": {
            "percentiles": [1, 99],
            "low": low.tolist(),
            "high": high.tolist(),
            "shared_across": [
                "true future DINO",
                "current DINO copy",
                "Stage2-state WM",
                "DINO-only WM",
            ],
        },
        "error_scale": {
            "percentile": 99,
            "high": error_high,
            "population": "selected rows, both predictors, full-channel per-slot MSE",
        },
        "selection": {
            "rows_per_horizon": rows_per_horizon,
            "rule": (
                "evenly spaced sorted trajectory IDs; window is SHA256(trajectory:horizon) "
                "modulo valid window count; predictions and errors are not used for selection"
            ),
            "rows": [
                {
                    "trajectory_id": row["trajectory"],
                    "window_start": row["window_start"],
                    "horizon_step": row["horizon_step"],
                    "future_observation_index": row["future_index"],
                    "actions": row["actions"],
                    "current_image": str(row["image_paths"][row["window_start"]]),
                    "current_image_sha256": file_sha256(
                        row["image_paths"][row["window_start"]]
                    ),
                    "future_image": str(row["image_paths"][row["future_index"]]),
                    "future_image_sha256": file_sha256(
                        row["image_paths"][row["future_index"]]
                    ),
                    "stage2_state_wm_full_feature_mse": float(
                        (row["stage2_state_wm"] - row["target"]).square().mean()
                    ),
                    "dino_wm_full_feature_mse": float(
                        (row["dino_wm"] - row["target"]).square().mean()
                    ),
                }
                for row in predicted
            ],
        },
        "rendering": {
            "grid": f"{math.isqrt(int(cache.manifest['grid_tokens']))}x"
            f"{math.isqrt(int(cache.manifest['grid_tokens']))}",
            "feature_interpolation": "nearest",
            "raw_observation_thumbnail_interpolation": "lanczos",
            "smoothing": False,
            "files": rendered_files,
        },
        "column_semantics": {
            "Stage2-state WM": (
                "production WM architecture trained on frozen Stage2 states; output is shown "
                "directly in the DINO-target PCA coordinate system"
            ),
            "DINO-only WM": (
                "same production WM architecture trained on real DINO grids only"
            ),
            "error maps": "per-slot mean squared error across all feature channels",
        },
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "COMPLETE").write_text(file_sha256(manifest_path) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    manifest = render(
        eval_cache=args.eval_cache,
        eval_jsonl=args.eval_jsonl,
        stage2_checkpoint=args.stage2_checkpoint,
        dino_checkpoint=args.dino_checkpoint,
        output=args.output,
        rows_per_horizon=args.rows_per_horizon,
        device=torch.device(args.device),
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
