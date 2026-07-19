#!/usr/bin/env python3
"""Fail-fast validation for the fixed 120-task resolution probe dump."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image


_EXPECTED = {
    "navigation_base_train_resolution_probe": "base_train",
    "navigation_common_train_resolution_probe": "common_sense_train",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_dump(path: Path, *, expected_png_size: int) -> dict[str, Any]:
    rows = _load_rows(path)
    if len(rows) != 120:
        raise ValueError(f"expected 120 records, got {len(rows)}")

    keys: set[tuple[str, int]] = set()
    paths: list[Path] = []
    mismatches: list[str] = []
    for index, row in enumerate(rows):
        source = row.get("data_source")
        seed = int(row.get("env_seed"))
        eval_set = row.get("eval_set")
        key = (source, seed)
        if key in keys:
            mismatches.append(f"row {index}: duplicate key {key}")
        keys.add(key)
        expected_eval_set = _EXPECTED.get(source)
        expected_uid = f"{source}:{seed}:{expected_eval_set}"
        config_id = str(row.get("config_id", ""))
        metrics = row.get("metrics")
        if (
            expected_eval_set is None
            or seed not in range(1, 61)
            or eval_set != expected_eval_set
            or row.get("uid") != expected_uid
            or f"eval_set={expected_eval_set}," not in config_id
            or not isinstance(metrics, dict)
            or "success" not in metrics
        ):
            mismatches.append(
                f"row {index}: key={key} eval_set={eval_set!r} "
                f"uid={row.get('uid')!r} config_id={config_id!r}"
            )
        image_paths = row.get("image_paths")
        if not isinstance(image_paths, list) or not image_paths:
            mismatches.append(f"row {index}: missing image_paths")
        else:
            paths.extend(Path(image_path) for image_path in image_paths)

    expected_keys = {
        (source, seed) for source in _EXPECTED for seed in range(1, 61)
    }
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        mismatches.append(f"paired keys differ: missing={missing[:5]} extra={extra[:5]}")
    if mismatches:
        raise ValueError("stable metadata mismatch: " + "; ".join(mismatches[:10]))

    size_counts: Counter[tuple[int, int]] = Counter()
    for image_path in paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"missing rollout image: {image_path}")
        with Image.open(image_path) as image:
            size_counts[image.size] += 1
            if image.mode != "RGB":
                raise ValueError(f"rollout image is not RGB: {image_path} mode={image.mode}")
    expected_size = (expected_png_size, expected_png_size)
    if set(size_counts) != {expected_size}:
        raise ValueError(
            f"rollout image size mismatch: expected={expected_size}, "
            f"observed={dict(size_counts)}"
        )
    return {
        "records": len(rows),
        "unique_keys": len(keys),
        "metadata_mismatches": 0,
        "image_references": len(paths),
        "image_size": f"{expected_png_size}x{expected_png_size}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--expected-png-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_dump(args.jsonl, expected_png_size=args.expected_png_size)
    text = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
