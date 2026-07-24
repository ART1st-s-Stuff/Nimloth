"""只读 DINO spatial-grid target cache。

本模块只负责证明并读取 frozen teacher 预计算目标。它不参与 Qwen forward，
也不拥有 SFT2 loss 或 world-model decoder。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


DINO_GRID_CACHE_FORMAT = "dino_grid_sharded_v1"


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class DINOIdentity:
    """能唯一证明 cached target teacher 的身份。"""

    source: str
    revision: str
    processor_fingerprint: str
    hidden_size: int


DINOV2_LARGE_IDENTITY = DINOIdentity(
    source="facebook/dinov2-large",
    revision="47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    processor_fingerprint="7d65a7de8788e87d",
    hidden_size=1024,
)


def _image_index(cache_split_dir: Path) -> tuple[list[str], str]:
    index_path = cache_split_dir / "image_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"DINO grid cache requires compact image index: {index_path}"
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if payload.get("format") != "dedup_sharded_v1" or not isinstance(images, list):
        raise ValueError(f"invalid compact image index: {index_path}")
    paths = [str(Path(entry["path"]).resolve()) for entry in images]
    return paths, _json_fingerprint(paths)


def _parent_cache_identity(cache_split_dir: Path) -> dict[str, str]:
    manifest_path = cache_split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"DINO grid cache requires compact manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dedup_sharded_v1":
        raise ValueError(
            f"DINO grid target requires compact Qwen cache: {cache_split_dir}"
        )
    return {
        "parent_fingerprint": str(manifest.get("fingerprint", "")),
        "image_source_fingerprint": str(
            manifest.get("image_source_fingerprint", "")
        ),
    }


def _load_grid_shard(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - older torch fallback
        payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(f"invalid DINO grid feature shard: {path}")
    return features


class CachedDINOGridTargets:
    """从经过 lineage 校验的 mmap sidecar 读取 next-image DINO grid。"""

    def __init__(
        self,
        *,
        identity: DINOIdentity,
        grid_size: int,
        path_to_feature: dict[str, tuple[torch.Tensor, int]],
        cache_fingerprint: str,
    ) -> None:
        self.identity = identity
        self.grid_size = int(grid_size)
        self.path_to_feature = path_to_feature
        self.cache_fingerprint = str(cache_fingerprint)

    @property
    def grid_tokens(self) -> int:
        return self.grid_size**2

    @classmethod
    def from_cache_root(
        cls,
        cache_root: str | Path,
        *,
        identity: DINOIdentity,
        grid_size: int = 4,
    ) -> "CachedDINOGridTargets":
        cache_root = Path(cache_root)
        path_to_feature: dict[str, tuple[torch.Tensor, int]] = {}
        fingerprints: list[str] = []
        grid_tokens = int(grid_size) ** 2

        for split in ("train", "val"):
            split_dir = cache_root / split
            sidecar = split_dir / f"dino_grid{grid_size}"
            manifest_path = sidecar / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"required DINO grid cache missing: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            claimed = manifest.get("fingerprint")
            fingerprint_payload = {
                key: value
                for key, value in manifest.items()
                if key != "fingerprint"
            }
            if claimed != _json_fingerprint(fingerprint_payload):
                raise ValueError(
                    f"DINO grid cache manifest fingerprint mismatch: {sidecar}"
                )
            if (
                manifest.get("format") != DINO_GRID_CACHE_FORMAT
                or manifest.get("identity") != asdict(identity)
                or int(manifest.get("grid_size", -1)) != int(grid_size)
                or int(manifest.get("grid_tokens", -1)) != grid_tokens
            ):
                raise ValueError(
                    f"DINO grid cache teacher or grid identity mismatch: {sidecar}"
                )

            paths, image_index_fingerprint = _image_index(split_dir)
            parent = _parent_cache_identity(split_dir)
            if (
                manifest.get("image_index_fingerprint")
                != image_index_fingerprint
                or manifest.get("parent_fingerprint")
                != parent["parent_fingerprint"]
                or manifest.get("image_source_fingerprint")
                != parent["image_source_fingerprint"]
            ):
                raise ValueError(
                    f"DINO grid cache parent/image lineage mismatch: {sidecar}"
                )

            shard_size = int(manifest.get("shard_size", 0))
            shard_count = int(manifest.get("shards", -1))
            expected_shards = math.ceil(len(paths) / shard_size) if shard_size else -1
            if (
                int(manifest.get("count", -1)) != len(paths)
                or manifest.get("feature_dtype") != "float32"
                or shard_size < 1
                or shard_count != expected_shards
            ):
                raise ValueError(
                    f"invalid DINO grid cache shard metadata: {sidecar}"
                )

            shards: list[torch.Tensor] = []
            for index in range(shard_count):
                tensor = _load_grid_shard(sidecar / f"shard_{index:05d}.pt")
                expected_rows = min(shard_size, len(paths) - index * shard_size)
                expected_shape = (
                    expected_rows,
                    grid_tokens,
                    identity.hidden_size,
                )
                if tensor.shape != expected_shape or tensor.dtype != torch.float32:
                    raise ValueError(
                        "DINO grid cache feature shape/dtype mismatch: "
                        f"{sidecar}; got={tuple(tensor.shape)}/{tensor.dtype}, "
                        f"expected={expected_shape}/float32"
                    )
                shards.append(tensor)

            for index, path in enumerate(paths):
                path_to_feature.setdefault(
                    path,
                    (shards[index // shard_size], index % shard_size),
                )
            fingerprints.append(str(claimed))

        return cls(
            identity=identity,
            grid_size=grid_size,
            path_to_feature=path_to_feature,
            cache_fingerprint=_json_fingerprint(fingerprints),
        )

    @torch.no_grad()
    def load(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not paths:
            raise ValueError("DINO grid supervision requires at least one image")
        rows: list[torch.Tensor] = []
        for raw_path in paths:
            path = str(Path(raw_path).resolve())
            location = self.path_to_feature.get(path)
            if location is None:
                raise KeyError(f"DINO grid cache missing image: {path}")
            shard, index = location
            rows.append(shard[index])
        return (
            torch.stack(rows)
            .to(device=device, dtype=torch.float32, non_blocking=True)
            .detach()
        )


__all__ = [
    "CachedDINOGridTargets",
    "DINOIdentity",
    "DINO_GRID_CACHE_FORMAT",
    "DINOV2_LARGE_IDENTITY",
]
