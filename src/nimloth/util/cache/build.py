"""构建 compact 和 legacy 两种 Qwen transition 预处理缓存。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.util.distributed import is_main
from nimloth.util.cache.encoding import (
    encode_qwen_item_from_image_grids,
    encode_transition_item,
)
from nimloth.util.cache.schema import (
    CE_MASK_VERSION,
    COMPACT_CACHE_FORMAT,
    DEFAULT_MIN_PIXELS,
    LEGACY_CACHE_FORMAT,
    TRANSITION_EXPANSION_VERSION,
    cache_fingerprint,
    safe_cache_name,
    transition_sample_id,
)
from nimloth.rollout.transitions import (
    TransitionJsonlDataset,
    TransitionSample,
    transition_training_item,
)

_CACHE_PROCESSOR: AutoProcessor | None = None
_CACHE_MAX_LENGTH = 0
_CACHE_LATENT_TOKEN_COUNT = 1
_CACHE_MASK_LATENT_QUERY_LABELS = True
_COMPACT_PATH_TO_IMAGE_INDEX: dict[str, int] = {}
_COMPACT_IMAGE_GRIDS: list[list[int]] = []


def _init_cache_worker(
    model_path: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> None:
    global _CACHE_PROCESSOR, _CACHE_MAX_LENGTH, _CACHE_LATENT_TOKEN_COUNT, _CACHE_MASK_LATENT_QUERY_LABELS
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    add_special_tokens(processor.tokenizer, latent_token_count=latent_token_count)
    _CACHE_PROCESSOR = processor
    _CACHE_MAX_LENGTH = max_length
    _CACHE_LATENT_TOKEN_COUNT = int(latent_token_count)
    _CACHE_MASK_LATENT_QUERY_LABELS = bool(mask_latent_query_labels)


def _init_compact_cache_worker(
    model_path: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
    latent_token_count: int,
    mask_latent_query_labels: bool,
    path_to_image_index: dict[str, int],
    image_grids: list[list[int]],
) -> None:
    global _COMPACT_PATH_TO_IMAGE_INDEX, _COMPACT_IMAGE_GRIDS
    _init_cache_worker(
        model_path,
        min_pixels,
        max_pixels,
        max_length,
        latent_token_count,
        mask_latent_query_labels,
    )
    _COMPACT_PATH_TO_IMAGE_INDEX = path_to_image_index
    _COMPACT_IMAGE_GRIDS = image_grids


def _cache_one_transition(task: tuple[dict[str, Any], str]) -> tuple[str, bool, str]:
    item, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_transition_item(
            item,
            _CACHE_PROCESSOR,
            _CACHE_MAX_LENGTH,
            latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
            mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
        )
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return item["id"], True, ""
    except Exception as exc:  # noqa: BLE001
        return item["id"], False, str(exc)


def _cache_image_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported compact cache image dtype: {name}") from exc


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _cache_one_image_shard(task: tuple[list[str], str, str]) -> tuple[str, bool, str]:
    image_paths, out_path, dtype_name = task
    try:
        assert _CACHE_PROCESSOR is not None
        dtype = _cache_image_dtype(dtype_name)
        pixels: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        offsets = [0]
        for image_path in image_paths:
            with Image.open(image_path) as image:
                encoded = _CACHE_PROCESSOR.image_processor(
                    images=[image.convert("RGB")],
                    return_tensors="pt",
                )
            pixel_values = encoded["pixel_values"].to(dtype=dtype).contiguous()
            grid = encoded["image_grid_thw"].to(dtype=torch.long).reshape(1, 3).contiguous()
            expected_rows = int(grid[0].prod().item())
            if pixel_values.shape[0] != expected_rows:
                raise ValueError(
                    f"pixel/grid mismatch for {image_path}: {pixel_values.shape[0]} != {expected_rows}"
                )
            pixels.append(pixel_values)
            grids.append(grid)
            offsets.append(offsets[-1] + pixel_values.shape[0])
        payload = {
            "pixel_values": torch.cat(pixels, dim=0) if pixels else torch.empty((0, 0), dtype=dtype),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "image_grid_thw": torch.cat(grids, dim=0) if grids else torch.empty((0, 3), dtype=torch.long),
        }
        _atomic_torch_save(payload, Path(out_path))
        return out_path, True, ""
    except Exception as exc:  # noqa: BLE001
        return out_path, False, str(exc)


def _cache_one_compact_transition_shard(
    task: tuple[
        list[TransitionSample],
        str,
        set[tuple[str, int]],
    ],
) -> tuple[str, bool, str]:
    samples, out_path, dedicated_next_keys = task
    try:
        assert _CACHE_PROCESSOR is not None
        entries: list[dict[str, Any]] = []
        for sample in samples:
            item = transition_training_item(sample)
            image_indices = [
                _COMPACT_PATH_TO_IMAGE_INDEX[str(Path(path).resolve())]
                for path in sample.prefix_image_paths
            ]
            grids = torch.tensor(
                [_COMPACT_IMAGE_GRIDS[index] for index in image_indices],
                dtype=torch.long,
            ).reshape(-1, 3)
            current_enc = encode_qwen_item_from_image_grids(
                item["messages"],
                grids,
                _CACHE_PROCESSOR,
                _CACHE_MAX_LENGTH,
                include_labels=True,
                latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
                mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
            )
            current_enc["image_indices"] = torch.tensor(image_indices, dtype=torch.int32)
            entry = {
                "id": item["id"],
                "record_id": item["record_id"],
                "step_index": item["step_index"],
                "action_index": item["action_index"],
                "action_value_target": item["action_value_target"],
                "success": item["success"],
                "current_enc": current_enc,
            }
            sample_key = (sample.record_id, sample.step_index)
            if sample_key in dedicated_next_keys:
                next_paths = sample.next_prefix_image_paths
                if sample.next_prefix_messages is None or next_paths is None:
                    raise ValueError(
                        "dedicated next state is missing for "
                        f"{sample.record_id}:{sample.step_index}"
                    )
                next_image_indices = [
                    _COMPACT_PATH_TO_IMAGE_INDEX[str(Path(path).resolve())]
                    for path in next_paths
                ]
                next_grids = torch.tensor(
                    [_COMPACT_IMAGE_GRIDS[index] for index in next_image_indices],
                    dtype=torch.long,
                ).reshape(-1, 3)
                next_enc = encode_qwen_item_from_image_grids(
                    item["next_messages"],
                    next_grids,
                    _CACHE_PROCESSOR,
                    _CACHE_MAX_LENGTH,
                    include_labels=False,
                    latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
                    mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
                )
                next_enc["image_indices"] = torch.tensor(
                    next_image_indices,
                    dtype=torch.int32,
                )
                entry["next_enc"] = next_enc
            entries.append(entry)
        _atomic_torch_save({"entries": entries}, Path(out_path))
        return out_path, True, ""
    except Exception as exc:  # noqa: BLE001
        return out_path, False, str(exc)


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _run_bounded_cache_tasks(
    pool: ProcessPoolExecutor,
    function,
    tasks: list[Any],
    *,
    max_pending: int,
) -> list[tuple[str, bool, str]]:
    iterator = iter(tasks)
    pending = set()
    results: list[tuple[str, bool, str]] = []

    def fill() -> None:
        while len(pending) < max_pending:
            try:
                task = next(iterator)
            except StopIteration:
                break
            pending.add(pool.submit(function, task))

    fill()
    while pending:
        done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
        pending = set(still_pending)
        results.extend(future.result() for future in done)
        fill()
    return results


def _compact_image_source_fingerprint(image_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for image_path in image_paths:
        stat = Path(image_path).stat()
        digest.update(image_path.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def _compact_cache_files_ready(
    cache_dir: Path,
    *,
    image_shards: int,
    transition_shards: int,
) -> bool:
    if not (cache_dir / "image_index.json").is_file():
        return False
    return all(
        (cache_dir / "images" / f"shard_{index:05d}.pt").is_file()
        for index in range(image_shards)
    ) and all(
        (cache_dir / "transitions" / f"shard_{index:05d}.pt").is_file()
        for index in range(transition_shards)
    )


def build_compact_transition_preprocess_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    model_path: Path,
    processor: AutoProcessor,
    max_length: int,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_records: int = -1,
    success_only: bool = False,
    preprocess_workers: int = 4,
    force: bool = False,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    image_dtype: str = "bfloat16",
    image_shard_size: int = 128,
    transition_shard_size: int = 256,
) -> None:
    """Build a deduplicated cache: each image once, token prefixes in mmap shards."""

    _cache_image_dtype(image_dtype)
    image_shard_size = max(1, int(image_shard_size))
    transition_shard_size = max(1, int(transition_shard_size))
    samples = TransitionJsonlDataset(
        jsonl_path,
        max_records=max_records,
        success_only=success_only,
        value_gamma=value_gamma,
    ).samples
    sample_keys = {
        (sample.record_id, sample.step_index)
        for sample in samples
    }
    dedicated_next_keys = {
        (sample.record_id, sample.step_index)
        for sample in samples
        if sample.next_prefix_messages is not None
        and sample.next_prefix_image_paths is not None
        and (sample.record_id, sample.step_index + 1) not in sample_keys
    }

    unique_image_paths: list[str] = []
    path_to_image_index: dict[str, int] = {}
    cumulative_image_refs = 0
    for sample in samples:
        cumulative_image_refs += len(sample.prefix_image_paths)
        referenced_paths = list(sample.prefix_image_paths)
        if (sample.record_id, sample.step_index) in dedicated_next_keys:
            referenced_paths.extend(sample.next_prefix_image_paths or ())
            cumulative_image_refs += len(sample.next_prefix_image_paths or ())
        for raw_path in referenced_paths:
            image_path = str(Path(raw_path).resolve())
            if image_path not in path_to_image_index:
                path_to_image_index[image_path] = len(unique_image_paths)
                unique_image_paths.append(image_path)
    image_source_fingerprint = _compact_image_source_fingerprint(unique_image_paths)
    base_fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        value_gamma=value_gamma,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype=image_dtype,
        processor_source=str(model_path.resolve()),
    )
    fingerprint = hashlib.sha256(
        (
            f"{base_fingerprint}|{image_source_fingerprint}|"
            f"image_shard={image_shard_size}|transition_shard={transition_shard_size}"
        ).encode()
    ).hexdigest()[:16]
    image_chunks = _chunked(unique_image_paths, image_shard_size)
    transition_count = len(samples)
    image_shard_count = len(image_chunks)
    transition_shard_count = math.ceil(transition_count / transition_shard_size) if transition_count else 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    build_state_path = cache_dir / "build_state.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"compact cache fingerprint mismatch at {cache_dir}; use a new directory or --force-rebuild-cache"
            )
        if (
            manifest.get("format") == COMPACT_CACHE_FORMAT
            and int(manifest.get("count", -1)) == transition_count
            and int(manifest.get("image_shard_size", -1)) == image_shard_size
            and int(manifest.get("transition_shard_size", -1)) == transition_shard_size
            and _compact_cache_files_ready(
                cache_dir,
                image_shards=image_shard_count,
                transition_shards=transition_shard_count,
            )
        ):
            build_state_path.unlink(missing_ok=True)
            if is_main():
                print(
                    json.dumps(
                        {
                            "preprocess_cache": "hit_compact",
                            "dir": str(cache_dir),
                            "transitions": transition_count,
                            "unique_images": len(unique_image_paths),
                        }
                    )
                )
            return
    if force:
        shutil.rmtree(cache_dir / "images", ignore_errors=True)
        shutil.rmtree(cache_dir / "transitions", ignore_errors=True)
        for path in (cache_dir / "image_index.json", manifest_path, build_state_path):
            path.unlink(missing_ok=True)

    expected_build_state = {
        "format": COMPACT_CACHE_FORMAT,
        "fingerprint": fingerprint,
        "count": transition_count,
        "unique_images": len(unique_image_paths),
        "image_shard_size": image_shard_size,
        "transition_shard_size": transition_shard_size,
    }
    if build_state_path.is_file():
        build_state = json.loads(build_state_path.read_text(encoding="utf-8"))
        if build_state != expected_build_state:
            raise RuntimeError(
                f"partial compact cache fingerprint mismatch at {cache_dir}; "
                "use a new directory or --force-rebuild-cache"
            )
    else:
        stale_artifacts = (
            any((cache_dir / "images").glob("*.pt"))
            or any((cache_dir / "transitions").glob("*.pt"))
            or (cache_dir / "image_index.json").exists()
        )
        if stale_artifacts:
            raise RuntimeError(
                f"compact cache has untracked partial files at {cache_dir}; "
                "use --force-rebuild-cache once"
            )
        state_tmp = build_state_path.with_suffix(".json.tmp")
        state_tmp.write_text(json.dumps(expected_build_state, indent=2), encoding="utf-8")
        os.replace(state_tmp, build_state_path)

    workers = max(1, int(preprocess_workers))
    image_tasks = [
        (chunk, str(cache_dir / "images" / f"shard_{index:05d}.pt"), image_dtype)
        for index, chunk in enumerate(image_chunks)
        if not (cache_dir / "images" / f"shard_{index:05d}.pt").is_file()
    ]
    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build_compact_images",
                    "dir": str(cache_dir),
                    "transitions": transition_count,
                    "cumulative_image_refs": cumulative_image_refs,
                    "unique_images": len(unique_image_paths),
                    "image_reuse_factor": cumulative_image_refs / max(len(unique_image_paths), 1),
                    "image_dtype": image_dtype,
                    "image_shards": image_shard_count,
                    "to_build": len(image_tasks),
                    "workers": workers,
                }
            )
        )
    if image_tasks:
        failures: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
            ),
        ) as pool:
            results = _run_bounded_cache_tasks(
                pool,
                _cache_one_image_shard,
                image_tasks,
                max_pending=workers * 2,
            )
            for shard, ok, error in results:
                if not ok:
                    failures.append(f"{shard}: {error}")
        if failures:
            raise RuntimeError(f"compact image cache failed for {len(failures)} shards; first={failures[0]}")

    image_locations: list[dict[str, Any]] = []
    image_grids: list[list[int]] = []
    for shard_index, paths in enumerate(image_chunks):
        shard_path = cache_dir / "images" / f"shard_{shard_index:05d}.pt"
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        grids = shard["image_grid_thw"]
        if grids.shape[0] != len(paths):
            raise ValueError(f"image shard/index mismatch: {shard_path}")
        for local_index, (path, grid) in enumerate(zip(paths, grids, strict=True)):
            image_locations.append(
                {
                    "path": path,
                    "shard": shard_index,
                    "index": local_index,
                    "grid_thw": [int(value) for value in grid.tolist()],
                }
            )
            image_grids.append([int(value) for value in grid.tolist()])
    image_index_path = cache_dir / "image_index.json"
    image_index_tmp = image_index_path.with_suffix(".json.tmp")
    image_index_tmp.write_text(
        json.dumps({"format": COMPACT_CACHE_FORMAT, "images": image_locations}),
        encoding="utf-8",
    )
    os.replace(image_index_tmp, image_index_path)

    transition_chunks = _chunked(samples, transition_shard_size)
    transition_tasks = [
        (
            chunk,
            str(cache_dir / "transitions" / f"shard_{index:05d}.pt"),
            dedicated_next_keys,
        )
        for index, chunk in enumerate(transition_chunks)
        if not (cache_dir / "transitions" / f"shard_{index:05d}.pt").is_file()
    ]
    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build_compact_tokens",
                    "dir": str(cache_dir),
                    "transition_shards": transition_shard_count,
                    "to_build": len(transition_tasks),
                    "workers": workers,
                }
            )
        )
    if transition_tasks:
        failures = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_compact_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
                path_to_image_index,
                image_grids,
            ),
        ) as pool:
            results = _run_bounded_cache_tasks(
                pool,
                _cache_one_compact_transition_shard,
                transition_tasks,
                max_pending=workers * 2,
            )
            for shard, ok, error in results:
                if not ok:
                    failures.append(f"{shard}: {error}")
        if failures:
            raise RuntimeError(f"compact token cache failed for {len(failures)} shards; first={failures[0]}")

    image_bytes = sum(path.stat().st_size for path in (cache_dir / "images").glob("*.pt"))
    transition_bytes = sum(path.stat().st_size for path in (cache_dir / "transitions").glob("*.pt"))
    manifest = {
        "fingerprint": fingerprint,
        "base_fingerprint": base_fingerprint,
        "format": COMPACT_CACHE_FORMAT,
        "count": transition_count,
        "unique_images": len(unique_image_paths),
        "cumulative_image_refs": cumulative_image_refs,
        "image_reuse_factor": cumulative_image_refs / max(len(unique_image_paths), 1),
        "image_source_fingerprint": image_source_fingerprint,
        "max_length": max_length,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "value_gamma": value_gamma,
        "latent_token_count": latent_token_count,
        "latent_query_mode": "inject" if mask_latent_query_labels else "generate",
        "mask_latent_query_labels": mask_latent_query_labels,
        "image_dtype": image_dtype,
        "image_shard_size": image_shard_size,
        "transition_shard_size": transition_shard_size,
        "image_shards": image_shard_count,
        "transition_shards": transition_shard_count,
        "image_bytes": image_bytes,
        "transition_bytes": transition_bytes,
        "total_bytes": image_bytes + transition_bytes + image_index_path.stat().st_size,
        "ce_mask_version": CE_MASK_VERSION,
        "transition_expansion_version": TRANSITION_EXPANSION_VERSION,
        "dir": str(cache_dir),
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    build_state_path.unlink(missing_ok=True)
    if is_main():
        print(json.dumps({"preprocess_cache": "done_compact", **manifest}))


def build_transition_preprocess_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    model_path: Path,
    processor: AutoProcessor,
    max_length: int,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_records: int = -1,
    success_only: bool = False,
    preprocess_workers: int = 4,
    force: bool = False,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> None:
    samples = TransitionJsonlDataset(
        jsonl_path,
        max_records=max_records,
        success_only=success_only,
        value_gamma=value_gamma,
    ).samples
    fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        value_gamma=value_gamma,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
        processor_source=str(model_path.resolve()),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("count") == len(samples)
            and manifest.get("max_length") == max_length
        ):
            missing = sum(
                1
                for sample in samples
                if not (cache_dir / f"{safe_cache_name(transition_sample_id(sample))}.pt").is_file()
            )
            if missing == 0:
                if is_main():
                    print(json.dumps({"preprocess_cache": "hit", "dir": str(cache_dir), "count": len(samples)}))
                return

    tasks: list[tuple[dict[str, Any], str]] = []
    for sample in samples:
        item = collate_transition_training_items([sample])[0]
        out_path = cache_dir / f"{safe_cache_name(item['id'])}.pt"
        if not force and out_path.is_file():
            continue
        tasks.append((item, str(out_path)))

    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build",
                    "dir": str(cache_dir),
                    "fingerprint": fingerprint,
                    "total": len(samples),
                    "to_build": len(tasks),
                    "workers": preprocess_workers,
                }
            )
        )

    if tasks:
        workers = max(1, preprocess_workers)
        failures: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
            ),
        ) as pool:
            futures = [pool.submit(_cache_one_transition, task) for task in tasks]
            for fut in as_completed(futures):
                sample_id, ok, err = fut.result()
                if not ok:
                    failures.append(f"{sample_id}: {err}")
        if failures:
            raise RuntimeError(f"preprocess cache failed for {len(failures)} samples; first={failures[0]}")

    if is_main():
        total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
        print(json.dumps({"preprocess_cache": "done", "dir": str(cache_dir), "count": len(samples), "total_bytes": total_bytes}))

    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "count": len(samples),
                "max_length": max_length,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "value_gamma": value_gamma,
                "latent_token_count": latent_token_count,
                "latent_query_mode": "inject" if mask_latent_query_labels else "generate",
                "mask_latent_query_labels": mask_latent_query_labels,
                "ce_mask_version": CE_MASK_VERSION,
                "transition_expansion_version": TRANSITION_EXPANSION_VERSION,
                "dir": str(cache_dir),
                "total_bytes": sum(path.stat().st_size for path in cache_dir.glob("*.pt")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
