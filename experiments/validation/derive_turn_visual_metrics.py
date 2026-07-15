"""Derive auxiliary per-horizon pixel metrics from saved turn contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

DEFAULT_COLUMNS = ("GT", "Qwen positive", "Frozen State GT", "Vector 1x8192 WM", "Token 8x1024 WM")


def _l1(left: Image.Image, right: Image.Image) -> float:
    means = ImageStat.Stat(ImageChops.difference(left, right)).mean
    return sum(means) / (255 * len(means))


def _row_images(sheet: Image.Image, horizon: int, columns: tuple[str, ...]) -> dict[str, Image.Image]:
    width, strip_height = sheet.width // len(columns), sheet.height // 5
    top = (horizon - 1) * strip_height + 18
    return {name: sheet.crop((index * width, top, (index + 1) * width, horizon * strip_height)) for index, name in enumerate(columns)}


def _row_metrics(images: dict[str, Image.Image], columns: tuple[str, ...]) -> dict[str, float]:
    metrics = {f"{name}_to_GT_l1": _l1(image, images[columns[0]]) for name, image in images.items() if name != columns[0]}
    metrics[f"{columns[3]}_to_{columns[2]}_l1"] = _l1(images[columns[3]], images[columns[2]])
    metrics[f"{columns[4]}_to_{columns[2]}_l1"] = _l1(images[columns[4]], images[columns[2]])
    return metrics


def derive(root: Path) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    columns = tuple(json.loads(metadata_path.read_text())["columns"]) if metadata_path.is_file() else DEFAULT_COLUMNS
    by_horizon = {str(step): [] for step in range(1, 6)}
    for run in range(6):
        with Image.open(root / f"run_{run:02d}.png") as sheet:
            for horizon in range(1, 6):
                images = _row_images(sheet, horizon, columns)
                by_horizon[str(horizon)].append(_row_metrics(images, columns))
    horizons = {key: _average(rows) for key, rows in by_horizon.items()}
    all_rows = [row for rows in by_horizon.values() for row in rows]
    payload = {"status": "completed", "source": "8-bit PNG crops; auxiliary only", "rows": 30, "horizons": horizons, "overall": _average(all_rows)}
    path = root / "visual_horizon_metrics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload), flush=True)
    return payload


def _average(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turn-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    derive(build_parser().parse_args(argv).turn_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
