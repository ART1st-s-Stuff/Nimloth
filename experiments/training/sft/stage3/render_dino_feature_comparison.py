"""Render matched Stage2-direct and Stage3 future-DINO feature grids."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def load_export(directory: Path) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    rows: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for path in sorted(directory.glob("rank_*_batch_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("schema") != "stage3_dino_feature_batch_v1":
            raise ValueError(f"unsupported feature export: {path}")
        for index, key in enumerate(payload["keys"]):
            identity = (str(key[0]), int(key[1]))
            if identity in rows:
                raise ValueError(f"duplicate feature identity: {identity}")
            rows[identity] = {
                name: payload[name][index].float()
                for name in ("actions", "predicted", "direct", "online_direct", "dino", "current_dino")
            }
    if not rows:
        raise ValueError(f"no feature exports found in {directory}")
    return rows


def matched_rows(stage2: dict, control: dict, treatment: dict) -> list[dict]:
    if stage2.keys() != control.keys() or control.keys() != treatment.keys():
        raise ValueError("Stage2/control/treatment feature identities differ")
    rows = []
    for identity in sorted(stage2):
        s2, left, right = stage2[identity], control[identity], treatment[identity]
        for source in (left, right):
            if not torch.equal(s2["actions"], source["actions"]):
                raise ValueError(f"action mismatch for {identity}")
            if not torch.equal(s2["dino"], source["dino"]):
                raise ValueError(f"DINO target mismatch for {identity}")
            if not torch.equal(s2["current_dino"], source["current_dino"]):
                raise ValueError(f"current DINO mismatch for {identity}")
        horizon = int(s2["dino"].shape[0])
        for step in range(horizon):
            rows.append({
                "trajectory": identity[0],
                "window_start": identity[1],
                "horizon_step": step + 1,
                "action": int(s2["actions"][step]),
                "target": s2["dino"][step],
                "stage2_direct": s2["direct"][step],
                "copy": s2["current_dino"],
                "control": left["predicted"][step],
                "treatment": right["predicted"][step],
                "control_online_direct": left["online_direct"][step],
                "treatment_online_direct": right["online_direct"][step],
            })
    return rows


def cosine(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(prediction.flatten(1), target.flatten(1)).mean())


def summarize(rows: list[dict]) -> dict:
    result = {"count": len(rows), "models": {}, "by_horizon": {}}
    target = torch.stack([row["target"] for row in rows])
    for name in ("stage2_direct", "copy", "control", "treatment", "control_online_direct", "treatment_online_direct"):
        prediction = torch.stack([row[name] for row in rows])
        centered_prediction = prediction - prediction.mean(dim=0, keepdim=True)
        centered_target = target - target.mean(dim=0, keepdim=True)
        target_var = float(centered_target.square().mean())
        result["models"][name] = {
            "mse": float((prediction - target).square().mean()),
            "cosine": cosine(prediction, target),
            "prediction_mean": float(prediction.mean()),
            "prediction_std": float(prediction.std()),
            "target_mean": float(target.mean()),
            "target_std": float(target.std()),
            "observation_variance_ratio": (
                float(centered_prediction.square().mean()) / target_var if target_var > 0 else None
            ),
            "centered_cosine": cosine(centered_prediction, centered_target),
        }
    for horizon in sorted({row["horizon_step"] for row in rows}):
        subset = [row for row in rows if row["horizon_step"] == horizon]
        result["by_horizon"][str(horizon)] = summarize_without_horizons(subset)
    return result


def summarize_without_horizons(rows: list[dict]) -> dict:
    target = torch.stack([row["target"] for row in rows])
    return {
        name: {
            "mse": float((torch.stack([row[name] for row in rows]) - target).square().mean()),
            "cosine": cosine(torch.stack([row[name] for row in rows]), target),
        }
        for name in ("stage2_direct", "copy", "control", "treatment", "control_online_direct", "treatment_online_direct")
    }


def load_images(jsonl: Path) -> dict[str, list[str]]:
    records = {}
    with jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            records[str(record["id"])] = [str(path) for path in record["image_paths"]]
    return records


def pca_basis(
    rows: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = torch.stack([row["target"] for row in rows]).numpy()
    flat = target.reshape(-1, target.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    centered = flat - mean
    eigenvalues, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    basis = vectors[:, order[:3]]
    projected = (flat - mean) @ basis
    low, high = np.percentile(projected, [1, 99], axis=0)
    explained = eigenvalues[:3] / max(eigenvalues.sum(), 1e-12)
    return mean, basis, low, high, explained


def feature_image(grid: torch.Tensor, mean, basis, low, high, size=128) -> Image.Image:
    projected = (grid.numpy() - mean) @ basis
    rgb = np.clip((projected - low) / np.maximum(high - low, 1e-12), 0, 1)
    side = grid_side(grid)
    image = Image.fromarray(np.uint8(rgb.reshape(side, side, 3) * 255), "RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def error_image(prediction: torch.Tensor, target: torch.Tensor, high: float, size=128) -> Image.Image:
    value = (prediction - target).square().mean(dim=-1).numpy()
    normalized = np.clip(value / max(high, 1e-12), 0, 1)
    rgb = np.stack((normalized, np.sqrt(normalized) * 0.35, 1 - normalized), axis=-1)
    side = grid_side(target)
    return Image.fromarray(np.uint8(rgb.reshape(side, side, 3) * 255), "RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )


def grid_side(grid: torch.Tensor) -> int:
    if grid.ndim != 2 or math.isqrt(grid.shape[0]) ** 2 != grid.shape[0]:
        raise ValueError("feature grid must have shape [square slots, channels]")
    return math.isqrt(grid.shape[0])


def select_page_rows(rows: list[dict], horizon: int, limit: int = 8) -> list[dict]:
    """Round-robin trajectories before showing additional windows of one trajectory."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["horizon_step"] == horizon:
            grouped.setdefault(row["trajectory"], []).append(row)
    selected = []
    for offset in range(max((len(group) for group in grouped.values()), default=0)):
        for trajectory in sorted(grouped):
            group = grouped[trajectory]
            if offset < len(group):
                selected.append(group[offset])
                if len(selected) >= limit:
                    return selected
    return selected


def render(rows: list[dict], images: dict[str, list[str]], output: Path) -> list[str]:
    mean, basis, low, high, explained = pca_basis(rows)
    errors = [
        float((row[name] - row["target"]).square().mean(dim=-1).max())
        for row in rows
        for name in ("stage2_direct", "copy", "control", "treatment")
    ]
    error_high = float(np.percentile(errors, 99))
    columns = ("image", "target", "stage2_direct", "copy", "control", "treatment", "control_error", "treatment_error")
    files = []
    for horizon in sorted({row["horizon_step"] for row in rows}):
        page_rows = select_page_rows(rows, horizon)
        cell, label_h, row_h = 128, 38, 158
        canvas = Image.new("RGB", (len(columns) * cell, label_h + len(page_rows) * row_h), "white")
        draw = ImageDraw.Draw(canvas)
        for col, label in enumerate(columns):
            draw.text((col * cell + 4, 8), label, fill="black")
        for row_index, row in enumerate(page_rows):
            y = label_h + row_index * row_h
            image_index = row["window_start"] + row["horizon_step"]
            raw = Image.open(images[row["trajectory"]][image_index]).convert("RGB")
            raw.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(raw, (0, y))
            for col, name in enumerate(("target", "stage2_direct", "copy", "control", "treatment"), start=1):
                canvas.paste(feature_image(row[name], mean, basis, low, high, cell), (col * cell, y))
            canvas.paste(error_image(row["control"], row["target"], error_high, cell), (6 * cell, y))
            canvas.paste(error_image(row["treatment"], row["target"], error_high, cell), (7 * cell, y))
            draw.text((4, y + cell + 2), f"{row['trajectory']} start={row['window_start']} a={row['action']}", fill="black")
        path = output / f"horizon_{horizon}.png"
        canvas.save(path)
        files.append(path.name)
    (output / "visualization_contract.json").write_text(json.dumps({
        "pca_fit": "all selected real DINO targets only",
        "pca_explained_variance_ratio": explained.tolist(),
        "feature_scale": {"percentiles": [1, 99], "low": low.tolist(), "high": high.tolist()},
        "error_scale": {"percentile": 99, "high": error_high},
        "error_scale_population": "per-row maximum per-position full-channel MSE",
        "row_selection": "round-robin trajectories, then additional windows; up to 8 per horizon",
        "stage2_direct_semantics": "direct encoding sees the target observation; WM predicts it",
        "interpolation": "nearest",
        "files": files,
    }, indent=2), encoding="utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = matched_rows(load_export(args.stage2), load_export(args.control), load_export(args.treatment))
    metrics = summarize(rows)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render(rows, load_images(args.eval_jsonl), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
