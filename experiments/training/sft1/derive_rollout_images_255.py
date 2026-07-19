#!/usr/bin/env python3
"""Create a non-destructive 255x255 copy of converted rollout records."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

OUTPUT_SIZE = (255, 255)


def _output_image_path(output_root: Path, source: Path) -> Path:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()
    return output_root / "images" / digest[:2] / f"{digest}.png"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _resize_one(task: tuple[str, str]) -> tuple[str, bool]:
    source = Path(task[0])
    output = Path(task[1])
    with Image.open(source) as image:
        source_size = f"{image.width}x{image.height}"
        if output.is_file():
            try:
                with Image.open(output) as existing:
                    if existing.mode == "RGB" and existing.size == OUTPUT_SIZE:
                        return source_size, True
            except OSError:
                pass

        converted = image.convert("RGB")
        if converted.size != OUTPUT_SIZE:
            converted = converted.resize(OUTPUT_SIZE, resample=Image.Resampling.BICUBIC)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        converted.save(temporary, format="PNG")
        os.replace(temporary, output)
    return source_size, False


def _ordered_results(
    tasks: list[tuple[str, str]], workers: int
) -> Iterable[tuple[str, bool]]:
    if workers == 1:
        return map(_resize_one, tasks)
    executor = ProcessPoolExecutor(max_workers=workers)

    def iterate() -> Iterable[tuple[str, bool]]:
        try:
            yield from executor.map(_resize_one, tasks, chunksize=32)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    return iterate()


def derive_dataset(source_root: Path, output_root: Path, *, workers: int) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if source_root == output_root:
        raise ValueError("output root must differ from source root")
    if workers < 1:
        raise ValueError("workers must be positive")

    jsonl_paths = sorted(source_root.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"no top-level JSONL files under {source_root}")

    records_by_name: dict[str, list[dict[str, Any]]] = {}
    source_images: dict[str, Path] = {}
    image_references = 0
    record_count = 0
    for jsonl_path in jsonl_paths:
        records: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_paths = record.get("image_paths")
                if not isinstance(image_paths, list) or not all(
                    isinstance(path, str) for path in image_paths
                ):
                    raise ValueError(
                        f"{jsonl_path}:{line_number} has invalid image_paths"
                    )
                for raw_path in image_paths:
                    source = Path(raw_path).expanduser().resolve()
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"{jsonl_path}:{line_number} missing image {source}"
                        )
                    source_images[str(source)] = source
                image_references += len(image_paths)
                record_count += 1
                records.append(record)
        records_by_name[jsonl_path.name] = records

    output_paths = {
        source_key: _output_image_path(output_root, source)
        for source_key, source in source_images.items()
    }
    tasks = [
        (source_key, str(output_paths[source_key]))
        for source_key in sorted(source_images)
    ]
    source_size_counts: Counter[str] = Counter()
    reused_images = 0
    for source_size, reused in _ordered_results(tasks, workers):
        source_size_counts[source_size] += 1
        reused_images += int(reused)

    for jsonl_name, records in records_by_name.items():
        output_lines: list[str] = []
        for record in records:
            rewritten = dict(record)
            rewritten["image_paths"] = [
                str(output_paths[str(Path(path).expanduser().resolve())])
                for path in record["image_paths"]
            ]
            output_lines.append(json.dumps(rewritten, ensure_ascii=False))
        _atomic_write_text(
            output_root / jsonl_name,
            "\n".join(output_lines) + ("\n" if output_lines else ""),
        )

    manifest: dict[str, Any] = {
        "source_records_root": str(source_root),
        "output_records_root": str(output_root),
        "jsonl_files": len(jsonl_paths),
        "records": record_count,
        "image_references": image_references,
        "unique_images": len(source_images),
        "reused_images": reused_images,
        "source_size_counts": dict(sorted(source_size_counts.items())),
        "output_size": list(OUTPUT_SIZE),
        "mode": "RGB",
        "resampling": "PIL.Image.Resampling.BICUBIC",
        "source_images_preserved": True,
    }
    _atomic_write_text(
        output_root / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-records-root", type=Path, required=True)
    parser.add_argument("--output-records-root", type=Path, required=True)
    parser.add_argument(
        "--workers", type=int, default=min(32, os.cpu_count() or 1)
    )
    args = parser.parse_args()
    manifest = derive_dataset(
        args.source_records_root,
        args.output_records_root,
        workers=args.workers,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
