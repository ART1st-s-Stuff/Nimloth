"""Derive auxiliary per-horizon pixel metrics from saved turn contact sheets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

COLUMNS = ("GT", "Qwen positive", "Frozen State GT", "Vector 1x8192 WM", "Token 8x1024 WM")


def _l1(left: Image.Image, right: Image.Image) -> float:
    means = ImageStat.Stat(ImageChops.difference(left, right)).mean
    return sum(means) / (255 * len(means))


def _row_images(sheet: Image.Image, horizon: int) -> dict[str, Image.Image]:
    width, strip_height = sheet.width // len(COLUMNS), sheet.height // 5
    top = (horizon - 1) * strip_height + 18
    return {name: sheet.crop((index * width, top, (index + 1) * width, horizon * strip_height)) for index, name in enumerate(COLUMNS)}


def _row_metrics(images: dict[str, Image.Image]) -> dict[str, float]:
    metrics = {f"{name}_to_GT_l1": _l1(image, images["GT"]) for name, image in images.items() if name != "GT"}
    metrics["Vector_to_Frozen_State_GT_l1"] = _l1(images["Vector 1x8192 WM"], images["Frozen State GT"])
    metrics["Token_to_Frozen_State_GT_l1"] = _l1(images["Token 8x1024 WM"], images["Frozen State GT"])
    return metrics


def derive(root: Path) -> dict[str, Any]:
    by_horizon = {str(step): [] for step in range(1, 6)}
    for run in range(6):
        with Image.open(root / f"run_{run:02d}.png") as sheet:
            for horizon in range(1, 6):
                by_horizon[str(horizon)].append(_row_metrics(_row_images(sheet, horizon)))
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
