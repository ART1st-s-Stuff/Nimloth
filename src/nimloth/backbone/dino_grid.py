"""DINO spatial-grid target 的在线 teacher 与只读 cache。

本模块只负责产生或读取 frozen teacher target。它不参与 Qwen forward，
也不拥有 SFT2/RL loss 或 world-model 参数。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import torch
from PIL import Image


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


class DINOGridTargets(Protocol):
    """按图像路径读取 frozen DINO spatial-grid target。"""

    identity: DINOIdentity
    grid_size: int

    def load(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        ...


def _processor_fingerprint(processor: Any) -> str:
    to_dict = getattr(processor, "to_dict", None)
    payload = (
        to_dict()
        if callable(to_dict)
        else {"class": type(processor).__qualname__}
    )
    return _json_fingerprint(payload)


class FrozenDINOGridTargets:
    """在线计算并缓存 frozen DINOv2 的 row-major pooled patch grid。"""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        image_processor: Any,
        identity: DINOIdentity,
        grid_size: int = 4,
        batch_size: int = 32,
    ) -> None:
        self.model = model.requires_grad_(False).eval()
        self.image_processor = image_processor
        self.identity = identity
        self.grid_size = int(grid_size)
        self.batch_size = int(batch_size)
        self._cached_targets: dict[str, torch.Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        identity: DINOIdentity,
        *,
        device: torch.device,
        dtype: torch.dtype,
        grid_size: int = 4,
        batch_size: int = 32,
    ) -> "FrozenDINOGridTargets":
        """按固定 revision 加载 RL 使用的 frozen DINO teacher。"""

        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(
            identity.source,
            revision=identity.revision,
            trust_remote_code=True,
        )
        if _processor_fingerprint(processor) != identity.processor_fingerprint:
            raise ValueError("loaded DINO processor does not match its identity")
        model = AutoModel.from_pretrained(
            identity.source,
            revision=identity.revision,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device=device, dtype=dtype)
        return cls(
            model=model,
            image_processor=processor,
            identity=identity,
            grid_size=grid_size,
            batch_size=batch_size,
        )

    @property
    def grid_tokens(self) -> int:
        return self.grid_size**2

    def _encode(self, paths: Sequence[str]) -> torch.Tensor:
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return self._encode_images(images)

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        if not images or any(not isinstance(image, Image.Image) for image in images):
            raise ValueError("DINO grid image batch must contain PIL images")
        rgb_images = [image.convert("RGB") for image in images]
        processed = self.image_processor(images=rgb_images, return_tensors="pt")
        model_parameter = next(self.model.parameters())
        pixel_values = processed["pixel_values"].to(
            device=model_parameter.device,
            dtype=model_parameter.dtype,
        )
        hidden = self.model(pixel_values=pixel_values).last_hidden_state
        patch_size = int(self.model.config.patch_size)
        patch_height = int(pixel_values.shape[-2]) // patch_size
        patch_width = int(pixel_values.shape[-1]) // patch_size
        patch_count = patch_height * patch_width
        spatial_tokens = hidden[:, -patch_count:, :].reshape(
            len(paths),
            patch_height,
            patch_width,
            self.identity.hidden_size,
        )
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            spatial_tokens.permute(0, 3, 1, 2).float(),
            (self.grid_size, self.grid_size),
        )
        return pooled.permute(0, 2, 3, 1).reshape(
            len(images),
            self.grid_tokens,
            self.identity.hidden_size,
        )

    @torch.no_grad()
    def load_images(
        self,
        images: Sequence[Image.Image],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode in-memory rollout observations without temporary files."""

        return self._encode_images(images).detach().to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

    @torch.no_grad()
    def load(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        resolved = [str(Path(path).resolve()) for path in paths]
        missing = tuple(
            dict.fromkeys(
                path for path in resolved if path not in self._cached_targets
            )
        )
        for start in range(0, len(missing), self.batch_size):
            current_paths = missing[start : start + self.batch_size]
            current_targets = self._encode(current_paths).detach().cpu()
            self._cached_targets.update(
                zip(current_paths, current_targets, strict=True)
            )
        return torch.stack(
            [self._cached_targets[path] for path in resolved]
        ).to(device=device, dtype=torch.float32, non_blocking=True)


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
    "DINOGridTargets",
    "DINOIdentity",
    "DINO_GRID_CACHE_FORMAT",
    "DINOV2_LARGE_IDENTITY",
    "FrozenDINOGridTargets",
]
