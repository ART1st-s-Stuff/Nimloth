"""Compressed Qwen visual-feature cache for RCDM reconstruction.

This cache stores global image features extracted by the frozen Qwen2.5-VL
visual encoder.  It is used as an oracle-image-feature baseline for RCDM: if
RCDM can reconstruct from these visual features but not from SFT2 latent state
vectors, the bottleneck is likely the SFT2 state representation rather than the
RCDM image decoder itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch

QWEN_VISION_CACHE_VERSION = "rcdm_qwen_vision_cache_v1"
Compression = Literal["gzip", "none"]
FeatureDType = Literal["float16", "bfloat16", "float32"]


def _path_stat_payload(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def qwen_vision_cache_fingerprint(
    *,
    jsonl_path: Path,
    model_path: Path,
    image_role: str,
    max_pixels: int,
    min_pixels: int,
    success_only: bool,
    max_records: int,
    feature_dtype: FeatureDType,
) -> str:
    payload = "|".join(
        [
            QWEN_VISION_CACHE_VERSION,
            _path_stat_payload(jsonl_path),
            str(model_path.resolve()),
            image_role,
            str(max_pixels),
            str(min_pixels),
            str(success_only),
            str(max_records),
            feature_dtype,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _torch_dtype(name: FeatureDType) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported feature dtype: {name}")


def _shard_name(index: int, compression: Compression) -> str:
    suffix = ".pt.gz" if compression == "gzip" else ".pt"
    return f"shard_{index:06d}{suffix}"


def _save_payload(payload: dict[str, Any], path: Path, compression: Compression) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compression == "gzip":
        with gzip.open(path, "wb", compresslevel=3) as f:
            torch.save(payload, f)
    elif compression == "none":
        torch.save(payload, path)
    else:
        raise ValueError(f"unsupported compression: {compression}")


def _load_payload(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


@lru_cache(maxsize=8192)
def _load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


@torch.no_grad()
def encode_qwen_visual_features(
    *,
    qwen_model,
    processor,
    image_paths: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Return mean-pooled Qwen visual encoder features for one or more images."""

    images = [_load_rgb_image(str(path)).copy() for path in image_paths]
    enc = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(device)
    image_grid_thw = enc["image_grid_thw"].to(device)
    image_features = qwen_model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    if isinstance(image_features, torch.Tensor):
        split_sizes = (image_grid_thw.prod(-1) // qwen_model.model.visual.spatial_merge_size**2).tolist()
        image_features = torch.split(image_features, split_sizes)
    pooled = [features.float().mean(dim=0) for features in image_features]
    return torch.stack(pooled, dim=0)


@dataclass(frozen=True)
class RCDMQwenVisionCacheManifest:
    cache_dir: Path
    count: int
    cond_dim: int
    feature_dtype: FeatureDType
    compression: Compression
    shard_size: int
    shards: list[dict[str, Any]]
    fingerprint: str

    @classmethod
    def load(cls, cache_dir: Path) -> "RCDMQwenVisionCacheManifest":
        data = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            cache_dir=cache_dir,
            count=int(data["count"]),
            cond_dim=int(data["cond_dim"]),
            feature_dtype=data["feature_dtype"],
            compression=data["compression"],
            shard_size=int(data["shard_size"]),
            shards=list(data["shards"]),
            fingerprint=str(data["fingerprint"]),
        )

    def write(self, extra: dict[str, Any]) -> None:
        payload = {
            **extra,
            "version": QWEN_VISION_CACHE_VERSION,
            "count": self.count,
            "cond_dim": self.cond_dim,
            "feature_dtype": self.feature_dtype,
            "compression": self.compression,
            "shard_size": self.shard_size,
            "shards": self.shards,
            "fingerprint": self.fingerprint,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def qwen_vision_cache_ready(cache_dir: Path) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = RCDMQwenVisionCacheManifest.load(cache_dir)
    except Exception:
        return False
    return all((cache_dir / str(shard["file"])).is_file() for shard in manifest.shards)


@torch.no_grad()
def build_rcdm_qwen_vision_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    split_name: str,
    model_path: Path,
    processor,
    qwen_model,
    device: torch.device,
    max_pixels: int,
    min_pixels: int,
    image_role: str = "current",
    max_records: int = -1,
    success_only: bool = False,
    batch_size: int = 4,
    shard_size: int = 4096,
    compression: Compression = "gzip",
    feature_dtype: FeatureDType = "float16",
    force: bool = False,
) -> RCDMQwenVisionCacheManifest:
    """Precompute compressed Qwen visual features for one split."""

    if image_role not in {"current", "next"}:
        raise ValueError("image_role must be 'current' or 'next'")
    fingerprint = qwen_vision_cache_fingerprint(
        jsonl_path=jsonl_path,
        model_path=model_path,
        image_role=image_role,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        success_only=success_only,
        max_records=max_records,
        feature_dtype=feature_dtype,
    )
    if not force and qwen_vision_cache_ready(cache_dir):
        manifest = RCDMQwenVisionCacheManifest.load(cache_dir)
        if manifest.fingerprint == fingerprint:
            print(json.dumps({"rcdm_qwen_vision_cache": "hit", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
            return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("shard_*.pt*"):
        old.unlink()

    ds = TransitionQwenDataset(jsonl_path, max_records=max_records, success_only=success_only)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_transition_batch)
    target_dtype = _torch_dtype(feature_dtype)
    shard_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    count = 0
    shard_index = 0
    cond_dim = -1

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        cond = torch.stack([row.pop("cond_emb") for row in shard_rows]).to(dtype=target_dtype)
        filename = _shard_name(shard_index, compression)
        _save_payload({"cond_emb": cond, "rows": shard_rows}, cache_dir / filename, compression)
        shards.append({"file": filename, "count": len(shard_rows)})
        shard_index += 1
        shard_rows = []

    qwen_model.eval()
    for items in loader:
        image_key = "current_image_path" if image_role == "current" else "next_image_path"
        image_paths = [str(item[image_key]) for item in items]
        features = encode_qwen_visual_features(qwen_model=qwen_model, processor=processor, image_paths=image_paths, device=device).detach().cpu()
        if cond_dim < 0:
            cond_dim = int(features.shape[-1])
        for item, feature, target_path in zip(items, features, image_paths, strict=True):
            shard_rows.append(
                {
                    "id": str(item.get("id", count)),
                    "record_id": str(item.get("record_id", "")),
                    "step_index": int(item.get("step_index", -1)),
                    "action_index": int(item["action_index"]),
                    "success": bool(item.get("success", False)),
                    "current_image_path": str(item["current_image_path"]),
                    "next_image_path": str(item["next_image_path"]),
                    "target_image_path": str(target_path),
                    "image_role": image_role,
                    "cond_emb": feature,
                }
            )
            count += 1
            if len(shard_rows) >= shard_size:
                flush()
    flush()

    manifest = RCDMQwenVisionCacheManifest(
        cache_dir=cache_dir,
        count=count,
        cond_dim=cond_dim,
        feature_dtype=feature_dtype,
        compression=compression,
        shard_size=shard_size,
        shards=shards,
        fingerprint=fingerprint,
    )
    total_bytes = sum((cache_dir / str(shard["file"])).stat().st_size for shard in shards)
    manifest.write(
        {
            "split": split_name,
            "jsonl_path": str(jsonl_path),
            "model_path": str(model_path),
            "image_role": image_role,
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
            "success_only": success_only,
            "max_records": max_records,
            "total_bytes": total_bytes,
        }
    )
    print(json.dumps({"rcdm_qwen_vision_cache": "done", "split": split_name, "dir": str(cache_dir), "count": count, "total_bytes": total_bytes}))
    return manifest


class RCDMQwenVisionCacheDataset(Dataset):
    """Dataset over compressed Qwen-vision feature cache shards."""

    def __init__(self, cache_dir: Path) -> None:
        self.manifest = RCDMQwenVisionCacheManifest.load(cache_dir)
        self.cache_dir = cache_dir
        self.index: list[tuple[int, int]] = []
        for shard_idx, shard in enumerate(self.manifest.shards):
            self.index.extend((shard_idx, row_idx) for row_idx in range(int(shard["count"])))

    def __len__(self) -> int:
        return len(self.index)

    @lru_cache(maxsize=4)
    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        shard = self.manifest.shards[shard_idx]
        return _load_payload(self.cache_dir / str(shard["file"]))

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_idx, row_idx = self.index[index]
        payload = self._load_shard(shard_idx)
        row = dict(payload["rows"][row_idx])
        row["cond_emb"] = payload["cond_emb"][row_idx]
        return row


def collate_rcdm_qwen_vision_cache_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cond_emb": torch.stack([item["cond_emb"] for item in batch]),
        "action_index": torch.tensor([int(item["action_index"]) for item in batch], dtype=torch.long),
        "id": [str(item["id"]) for item in batch],
        "record_id": [str(item.get("record_id", "")) for item in batch],
        "step_index": [int(item.get("step_index", -1)) for item in batch],
        "success": [bool(item.get("success", False)) for item in batch],
        "current_image_path": [str(item["current_image_path"]) for item in batch],
        "next_image_path": [str(item["next_image_path"]) for item in batch],
        "target_image_path": [str(item["target_image_path"]) for item in batch],
        "image_role": [str(item.get("image_role", "current")) for item in batch],
    }
