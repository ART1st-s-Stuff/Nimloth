"""Validate and hardlink immutable image shards, never cached text encodings."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from .schema import COMPACT_CACHE_FORMAT

IMAGE_PROCESSOR_IDENTITY_SCHEMA = "qwen_image_processor_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_processor_identity(
    processor: Any,
    *,
    max_pixels: int,
    min_pixels: int,
) -> dict[str, Any]:
    """Return a tokenizer-independent identity for the effective image processor."""

    image_processor = processor.image_processor
    actual_bounds = (
        int(image_processor.min_pixels),
        int(image_processor.max_pixels),
    )
    expected_bounds = (int(min_pixels), int(max_pixels))
    if actual_bounds != expected_bounds:
        raise ValueError(
            "effective image processor pixel bounds mismatch: "
            f"actual={actual_bounds}, expected={expected_bounds}"
        )
    config = image_processor.to_dict()
    payload = {
        "schema": IMAGE_PROCESSOR_IDENTITY_SCHEMA,
        "class": (
            f"{type(image_processor).__module__}."
            f"{type(image_processor).__qualname__}"
        ),
        "config": config,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("image processor config is not canonically serializable") from exc
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_image_reuse(
    source: Path, *, paths: list[str], source_fingerprint: str,
    visual_identity: dict[str, Any], legacy_base_fingerprint: str | None,
    image_dtype: str, max_pixels: int,
    min_pixels: int, image_shard_size: int, pixel_width: int, merge_size: int,
) -> dict[str, Any]:
    """Bind the old completed cache to exact source/order/preprocessing settings."""
    source = source.resolve()
    if (source / "build_state.json").exists():
        raise ValueError("image reuse source is an unfinished cache")
    manifest_path = source / "manifest.json"
    index_path = source / "image_index.json"
    if manifest_path.is_symlink() or index_path.is_symlink():
        raise ValueError("image reuse source metadata must not be symlinks")
    manifest = json.loads(manifest_path.read_text())
    count = (len(paths) + image_shard_size - 1) // image_shard_size
    expected = dict(format=COMPACT_CACHE_FORMAT,
                    image_source_fingerprint=source_fingerprint, image_dtype=image_dtype,
                    max_pixels=max_pixels, min_pixels=min_pixels,
                    image_shard_size=image_shard_size, image_shards=count, unique_images=len(paths))
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"image reuse source identity mismatch: {mismatches}")
    source_visual_identity = manifest.get("image_processor_identity")
    if source_visual_identity is None:
        if legacy_base_fingerprint is None:
            raise ValueError(
                "legacy image reuse source lacks image_processor_identity and "
                "cannot be bound to its original processor"
            )
        if manifest.get("base_fingerprint") != legacy_base_fingerprint:
            raise ValueError("legacy image reuse source base fingerprint mismatch")
    elif source_visual_identity != visual_identity:
        raise ValueError("image reuse source visual processor identity mismatch")

    transition_count = int(manifest.get("transition_shards", -1))
    expected_transition_files = {
        f"shard_{index:05d}.pt" for index in range(transition_count)
    }
    actual_transition_files = {
        path.name for path in (source / "transitions").glob("*.pt")
    }
    if transition_count < 0 or actual_transition_files != expected_transition_files:
        raise ValueError("image reuse source transition shards are incomplete")

    index = json.loads(index_path.read_text())
    locations = index["images"]
    if (
        index.get("format") != COMPACT_CACHE_FORMAT
        or len(locations) != len(paths)
        or [row["path"] for row in locations] != paths
    ):
        raise ValueError("image reuse source ordered image paths mismatch")
    expected_image_files = {f"shard_{index:05d}.pt" for index in range(count)}
    actual_image_files = {path.name for path in (source / "images").glob("*.pt")}
    if actual_image_files != expected_image_files:
        raise ValueError("image reuse source image shards are incomplete")
    files = []
    dtype = getattr(torch, image_dtype)
    for shard_index in range(count):
        name = f"shard_{shard_index:05d}.pt"
        path = source / "images" / name
        if path.is_symlink():
            raise ValueError(f"image reuse source shard must not be a symlink: {name}")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        pixels, grids, offsets = (payload[key] for key in ("pixel_values", "image_grid_thw", "offsets"))
        expected_count = min(image_shard_size, len(paths) - shard_index * image_shard_size)
        if (pixels.dtype != dtype or pixels.ndim != 2 or grids.dtype != torch.long
                or grids.shape != (expected_count, 3) or offsets.dtype != torch.long
                or offsets.shape != (expected_count + 1,) or torch.any(grids <= 0)
                or pixels.shape[1] != pixel_width or torch.any(grids[:, 1:] % merge_size)):
            raise ValueError(f"image reuse shard dtype/shape mismatch: {name}")
        if (int(offsets[0]) != 0 or int(offsets[-1]) != pixels.shape[0]
                or not torch.equal(offsets[1:] - offsets[:-1], grids.prod(-1))):
            raise ValueError(f"image reuse shard grid/pixel offsets mismatch: {name}")
        for local in range(expected_count):
            row = locations[shard_index * image_shard_size + local]
            if row["shard"] != shard_index or row["index"] != local or row["grid_thw"] != grids[local].tolist():
                raise ValueError(f"image reuse grid/index mismatch: {name}")
        stat = path.stat()
        files.append({"name": name, "bytes": stat.st_size, "sha256": file_sha256(path),
                      "device": stat.st_dev, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns})
        del payload, pixels, grids, offsets
    return {"source": str(source), "manifest_sha256": file_sha256(manifest_path),
            "image_index_sha256": file_sha256(index_path),
            "image_processor_identity": visual_identity, "files": files}


def link_verified_images(reuse: dict[str, Any], destination: Path) -> None:
    """Link after the destination's ordinary build-state guard has succeeded."""
    source = Path(reuse["source"])
    if file_sha256(source / "manifest.json") != reuse["manifest_sha256"]:
        raise ValueError("image reuse source manifest changed during verification")
    if file_sha256(source / "image_index.json") != reuse["image_index_sha256"]:
        raise ValueError("image reuse source index changed during verification")
    target = destination / "images"
    target.mkdir(parents=True, exist_ok=True)
    for entry in reuse["files"]:
        origin = source / "images" / entry["name"]
        output = target / entry["name"]
        stat = origin.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (
                entry["device"], entry["inode"], entry["bytes"], entry["mtime_ns"]):
            raise ValueError("image reuse source shard changed after verification")
        if file_sha256(origin) != entry["sha256"]:
            raise ValueError("image reuse source shard hash changed after verification")
        if output.exists():
            if not output.samefile(origin):
                raise ValueError("partial image reuse file is not the verified source inode")
        else:
            os.link(origin, output)
