"""Paired structural metrics for completed five-column reconstruction strips."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


_ROW_PATTERN = re.compile(r"run_(\d+)_step_(\d+)\.png$")


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def load_strip_columns(
    path: Path,
    *,
    column_count: int = 5,
    image_size: int = 128,
    label_height: int = 18,
) -> torch.Tensor:
    with Image.open(path) as source:
        image = source.convert("RGB")
        expected = (column_count * image_size, label_height + image_size)
        if image.size != expected:
            raise ValueError(f"strip shape mismatch for {path}: {image.size} != {expected}")
        return torch.stack(
            [
                _image_tensor(
                    image.crop(
                        (
                            index * image_size,
                            label_height,
                            (index + 1) * image_size,
                            label_height + image_size,
                        )
                    )
                )
                for index in range(column_count)
            ]
        )


def structural_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> torch.Tensor:
    """Return standard Gaussian-window SSIM for each image in ``[N,C,H,W]``."""

    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("SSIM inputs must have matching [N,C,H,W] shapes")
    coordinates = torch.arange(kernel_size, dtype=left.dtype, device=left.device)
    coordinates = coordinates - (kernel_size - 1) / 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    kernel_2d = torch.outer(gaussian, gaussian)
    channels = left.shape[1]
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    padding = kernel_size // 2
    left_pad = F.pad(left, (padding,) * 4, mode="reflect")
    right_pad = F.pad(right, (padding,) * 4, mode="reflect")
    mu_left = F.conv2d(left_pad, kernel, groups=channels)
    mu_right = F.conv2d(right_pad, kernel, groups=channels)
    mu_left_sq = mu_left.square()
    mu_right_sq = mu_right.square()
    mu_product = mu_left * mu_right
    sigma_left = F.conv2d(left_pad.square(), kernel, groups=channels) - mu_left_sq
    sigma_right = F.conv2d(right_pad.square(), kernel, groups=channels) - mu_right_sq
    sigma_cross = F.conv2d(left_pad * right_pad, kernel, groups=channels) - mu_product
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_product + c1) * (2 * sigma_cross + c2)) / (
        (mu_left_sq + mu_right_sq + c1) * (sigma_left + sigma_right + c2)
    )
    return score.flatten(1).mean(1)


def _per_image_metrics(images: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    error = images - gt
    mse = error.square().flatten(1).mean(1)
    return {
        "l1": error.abs().flatten(1).mean(1),
        "mse": mse,
        "psnr": -10.0 * torch.log10(mse.clamp_min(1e-12)),
        "ssim": structural_similarity(images, gt),
    }


def _mean_metrics(values: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, float]:
    return {name: float(metric[mask].mean()) for name, metric in values.items()}


def compare_strip_directories(old_dir: Path, new_dir: Path) -> dict[str, Any]:
    old_paths = sorted(old_dir.glob("run_*_step_*.png"))
    new_paths = sorted(new_dir.glob("run_*_step_*.png"))
    if not old_paths or [path.name for path in old_paths] != [path.name for path in new_paths]:
        raise ValueError("old/new reconstruction strip inventories differ or are empty")

    old_columns: list[torch.Tensor] = []
    new_columns: list[torch.Tensor] = []
    horizons: list[int] = []
    for old_path, new_path in zip(old_paths, new_paths, strict=True):
        match = _ROW_PATTERN.fullmatch(old_path.name)
        if match is None:
            raise ValueError(f"unexpected reconstruction strip name: {old_path.name}")
        old = load_strip_columns(old_path)
        new = load_strip_columns(new_path)
        if not torch.equal(old[0], new[0]):
            raise ValueError(f"GT column differs between protocols: {old_path.name}")
        old_columns.append(old)
        new_columns.append(new)
        horizons.append(int(match.group(2)))

    old = torch.stack(old_columns)
    new = torch.stack(new_columns)
    gt = old[:, 0]
    horizon_tensor = torch.tensor(horizons, dtype=torch.long)
    result: dict[str, Any] = {
        "status": "completed",
        "protocol": "paired_saved_png_gaussian_ssim_v1",
        "old_dir": str(old_dir.resolve()),
        "new_dir": str(new_dir.resolve()),
        "rows": len(old_paths),
        "gt_columns_exactly_equal": True,
        "image_range": "saved uint8 PNG converted to [0,1]",
        "columns": {},
    }
    for name, column in (("actual", 3), ("predicted", 4)):
        old_metrics = _per_image_metrics(old[:, column], gt)
        new_metrics = _per_image_metrics(new[:, column], gt)
        all_rows = torch.ones(len(old_paths), dtype=torch.bool)
        column_result: dict[str, Any] = {
            "old": _mean_metrics(old_metrics, all_rows),
            "new": _mean_metrics(new_metrics, all_rows),
            "new_minus_old": {
                metric: float((new_metrics[metric] - old_metrics[metric]).mean())
                for metric in old_metrics
            },
            "new_better_frame_fraction": {
                "l1": float((new_metrics["l1"] < old_metrics["l1"]).float().mean()),
                "psnr": float((new_metrics["psnr"] > old_metrics["psnr"]).float().mean()),
                "ssim": float((new_metrics["ssim"] > old_metrics["ssim"]).float().mean()),
            },
            "horizons": {},
        }
        for horizon in sorted(set(horizons)):
            mask = horizon_tensor == horizon
            column_result["horizons"][str(horizon)] = {
                "count": int(mask.sum()),
                "old": _mean_metrics(old_metrics, mask),
                "new": _mean_metrics(new_metrics, mask),
            }
        result["columns"][name] = column_result
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", type=Path, required=True)
    parser.add_argument("--new-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = compare_strip_directories(args.old_dir, args.new_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
