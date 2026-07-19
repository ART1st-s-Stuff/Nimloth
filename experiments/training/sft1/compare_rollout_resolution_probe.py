#!/usr/bin/env python3
"""Compare paired rollout success for the 504px and 252px model-input paths."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from math import comb
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("data_source")
            seed = row.get("env_seed")
            if not isinstance(source, str) or seed is None:
                raise ValueError(f"{path}:{line_number} lacks data_source/env_seed")
            key = (source, int(seed))
            if key in rows:
                raise ValueError(f"{path}:{line_number} duplicates {key}")
            rows[key] = row
    return rows


def _is_success(row: dict[str, Any]) -> bool:
    return float(row.get("traj_success", 0.0) or 0.0) >= 1.0


def _png_size_counts(jsonl_path: Path) -> dict[str, int]:
    image_root = jsonl_path.parent / f"image_{jsonl_path.stem}"
    counts: Counter[str] = Counter()
    for path in sorted(image_root.glob("images_*/*.png")):
        with Image.open(path) as image:
            counts[f"{image.width}x{image.height}"] += 1
    if not counts:
        raise FileNotFoundError(f"no PNG images under {image_root}")
    return dict(sorted(counts.items()))


def _mcnemar_exact(old_only: int, new_only: int) -> float:
    discordant = old_only + new_only
    if discordant == 0:
        return 1.0
    lower = min(old_only, new_only)
    one_sided = sum(comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * one_sided)


def compare_runs(old_jsonl: Path, new_jsonl: Path) -> dict[str, Any]:
    old_jsonl = old_jsonl.resolve()
    new_jsonl = new_jsonl.resolve()
    old_rows = _load_rows(old_jsonl)
    new_rows = _load_rows(new_jsonl)
    if set(old_rows) != set(new_rows):
        missing_new = sorted(set(old_rows) - set(new_rows))
        missing_old = sorted(set(new_rows) - set(old_rows))
        raise ValueError(
            f"paired keys differ: missing_new={missing_new[:5]} missing_old={missing_old[:5]}"
        )

    paired = {
        "both_success": 0,
        "old_only_success": 0,
        "new_only_success": 0,
        "both_failure": 0,
    }
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"records": 0, "old_success": 0, "new_success": 0}
    )
    for key in sorted(old_rows):
        old_success = _is_success(old_rows[key])
        new_success = _is_success(new_rows[key])
        if old_success and new_success:
            paired["both_success"] += 1
        elif old_success:
            paired["old_only_success"] += 1
        elif new_success:
            paired["new_only_success"] += 1
        else:
            paired["both_failure"] += 1
        source = key[0]
        by_source[source]["records"] += 1
        by_source[source]["old_success"] += int(old_success)
        by_source[source]["new_success"] += int(new_success)

    total = len(old_rows)
    old_successes = sum(_is_success(row) for row in old_rows.values())
    new_successes = sum(_is_success(row) for row in new_rows.values())
    source_summaries: dict[str, Any] = {}
    for source, counts in sorted(by_source.items()):
        records = counts["records"]
        source_summaries[source] = {
            **counts,
            "old_success_rate": counts["old_success"] / records,
            "new_success_rate": counts["new_success"] / records,
            "success_rate_delta": (
                counts["new_success"] - counts["old_success"]
            )
            / records,
        }

    return {
        "records": total,
        "old": {
            "jsonl": str(old_jsonl),
            "success": old_successes,
            "success_rate": old_successes / total if total else 0.0,
            "png_size_counts": _png_size_counts(old_jsonl),
        },
        "new": {
            "jsonl": str(new_jsonl),
            "success": new_successes,
            "success_rate": new_successes / total if total else 0.0,
            "png_size_counts": _png_size_counts(new_jsonl),
        },
        "success_rate_delta": (
            (new_successes - old_successes) / total if total else 0.0
        ),
        "paired": paired,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(
            paired["old_only_success"], paired["new_only_success"]
        ),
        "by_data_source": source_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-jsonl", type=Path, required=True)
    parser.add_argument("--new-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runs(args.old_jsonl, args.new_jsonl)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
