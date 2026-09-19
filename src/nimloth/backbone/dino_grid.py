"""DINO spatial-grid target 的在线 teacher 与只读 cache。

本模块只负责产生或读取 frozen teacher target。它不参与 Qwen forward，
也不拥有 SFT2/RL loss 或 world-model 参数。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from PIL import Image

DINO_GRID_CACHE_FORMAT = "dino_grid_sharded_v1"
STANDALONE_DINO_GRID_CACHE_FORMAT = "dino_grid_images_v1"
STANDALONE_DINO_STATE_CACHE_FORMAT = "dino_spatial_cls_images_v2"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    ) -> torch.Tensor: ...


def _processor_fingerprint(processor: Any) -> str:
    to_dict = getattr(processor, "to_dict", None)
    payload = (
        to_dict() if callable(to_dict) else {"class": type(processor).__qualname__}
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
        include_cls: bool = False,
    ) -> None:
        self.model = model.requires_grad_(False).eval()
        self.image_processor = image_processor
        self.identity = identity
        self.grid_size = int(grid_size)
        self.batch_size = int(batch_size)
        self.include_cls = bool(include_cls)
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
        include_cls: bool = False,
    ) -> FrozenDINOGridTargets:
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
            include_cls=include_cls,
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
            len(images),
            patch_height,
            patch_width,
            self.identity.hidden_size,
        )
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            spatial_tokens.permute(0, 3, 1, 2).float(),
            (self.grid_size, self.grid_size),
        )
        spatial = pooled.permute(0, 2, 3, 1).reshape(
            len(images),
            self.grid_tokens,
            self.identity.hidden_size,
        )
        if not self.include_cls:
            return spatial
        cls = hidden[:, 0, :].float().unsqueeze(1)
        if cls.shape != (len(images), 1, self.identity.hidden_size):
            raise ValueError("DINO CLS target has unexpected shape")
        return torch.cat((spatial, cls), dim=1)

    @property
    def state_tokens(self) -> int:
        return self.grid_tokens + int(self.include_cls)

    @torch.no_grad()
    def load_images(
        self,
        images: Sequence[Image.Image],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode in-memory rollout observations without temporary files."""

        return (
            self._encode_images(images)
            .detach()
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )
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
            dict.fromkeys(path for path in resolved if path not in self._cached_targets)
        )
        for start in range(0, len(missing), self.batch_size):
            current_paths = missing[start : start + self.batch_size]
            current_targets = self._encode(current_paths).detach().cpu()
            self._cached_targets.update(
                zip(current_paths, current_targets, strict=True)
            )
        return torch.stack([self._cached_targets[path] for path in resolved]).to(
            device=device, dtype=torch.float32, non_blocking=True
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
        "image_source_fingerprint": str(manifest.get("image_source_fingerprint", "")),
    }


def _load_grid_shard(
    path: Path,
    *,
    require_spatial_cls: bool = False,
) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - older torch fallback
        payload = torch.load(path, map_location="cpu", weights_only=True)
    has_spatial_cls = {"spatial_features", "cls_features"} <= set(payload)
    if require_spatial_cls and (not has_spatial_cls or "features" in payload):
        raise ValueError(
            f"DINO spatial/CLS shard must store explicit spatial_features and "
            f"cls_features tensors: {path}"
        )
    features = payload.get("features")
    if features is None and has_spatial_cls:
        spatial = payload["spatial_features"]
        cls = payload["cls_features"]
        if (
            not isinstance(spatial, torch.Tensor)
            or not isinstance(cls, torch.Tensor)
            or cls.ndim != 2
            or spatial.ndim != 3
            or spatial.shape[0] != cls.shape[0]
            or spatial.shape[-1] != cls.shape[-1]
        ):
            raise ValueError(f"invalid DINO spatial/CLS feature shard: {path}")
        features = torch.cat((spatial, cls.unsqueeze(1)), dim=1)
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
        shard_references: dict[int, tuple] | None = None,
        include_cls: bool = False,
    ) -> None:
        self.identity = identity
        self.grid_size = int(grid_size)
        self.path_to_feature = path_to_feature
        self.cache_fingerprint = str(cache_fingerprint)
        self._shard_references = shard_references
        self.include_cls = bool(include_cls)

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    @classmethod
    def _shard_reference(cls, path: Path, tensor: torch.Tensor) -> tuple:
        path = path.resolve()
        return (str(path), cls._file_identity(path), tuple(tensor.shape), tensor.dtype)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._shard_references is not None:
            # Do not send mmap tensors through Torch's multiprocessing reducer:
            # it copies whole shards into /dev/shm. Workers reopen the exact files
            # already validated by the parent, without repeating the full audit.
            state["path_to_feature"] = {
                path: (id(shard), row)
                for path, (shard, row) in self.path_to_feature.items()
            }
        return state

    def __setstate__(self, state):
        references = state.get("_shard_references")
        if references is not None:
            shards = {}
            for key, (filename, signature, shape, dtype) in references.items():
                path = Path(filename)
                if self._file_identity(path) != signature:
                    raise ValueError(f"validated DINO shard changed before worker load: {path}")
                tensor = _load_grid_shard(
                    path,
                    require_spatial_cls=bool(state.get("include_cls", False)),
                )
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                        or self._file_identity(path) != signature):
                    raise ValueError(f"validated DINO shard changed during worker load: {path}")
                shards[key] = tensor
            state["path_to_feature"] = {
                path: (shards[key], row)
                for path, (key, row) in state["path_to_feature"].items()
            }
            state["_shard_references"] = {
                id(shards[key]): reference for key, reference in references.items()
            }
        self.__dict__.update(state)

    @property
    def grid_tokens(self) -> int:
        return self.grid_size**2

    @property
    def state_tokens(self) -> int:
        return self.grid_tokens + int(self.include_cls)

    @classmethod
    def from_cache_root(
        cls,
        cache_root: str | Path,
        *,
        identity: DINOIdentity,
        grid_size: int = 4,
        _allow_incomplete: bool = False,
    ) -> CachedDINOGridTargets:
        cache_root = Path(cache_root)
        if (cache_root / "manifest.json").is_file():
            result = cls._from_standalone(cache_root, identity, grid_size)
            if not _allow_incomplete and (
                not (cache_root / "COMPLETED").is_file()
                or (cache_root / "COMPLETED").read_text().strip()
                != result.cache_fingerprint
            ):
                raise ValueError("standalone DINO cache is not complete")
            return result
        path_to_feature: dict[str, tuple[torch.Tensor, int]] = {}
        shard_references = {}
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
                key: value for key, value in manifest.items() if key != "fingerprint"
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
                manifest.get("image_index_fingerprint") != image_index_fingerprint
                or manifest.get("parent_fingerprint") != parent["parent_fingerprint"]
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
                raise ValueError(f"invalid DINO grid cache shard metadata: {sidecar}")

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
                shard_references[id(tensor)] = cls._shard_reference(
                    sidecar / f"shard_{index:05d}.pt", tensor
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
            shard_references=shard_references,
        )

    @classmethod
    def _from_standalone(cls, root: Path, identity: DINOIdentity, grid_size: int):
        manifest = json.loads((root / "manifest.json").read_text())
        claimed = manifest.get("fingerprint")
        if claimed != _json_fingerprint(
            {k: v for k, v in manifest.items() if k != "fingerprint"}
        ):
            raise ValueError("standalone DINO manifest fingerprint mismatch")
        cache_format = manifest.get("format")
        include_cls = cache_format == STANDALONE_DINO_STATE_CACHE_FORMAT
        if (
            cache_format not in {
                STANDALONE_DINO_GRID_CACHE_FORMAT,
                STANDALONE_DINO_STATE_CACHE_FORMAT,
            }
            or manifest.get("identity") != asdict(identity)
            or manifest.get("grid_size") != grid_size
            or manifest.get("feature_dtype") != "float32"
        ):
            raise ValueError("standalone DINO teacher/grid identity mismatch")
        if include_cls:
            if (
                manifest.get("spatial_tokens") != grid_size**2
                or manifest.get("global_tokens") != 1
                or manifest.get("state_tokens") != grid_size**2 + 1
                or manifest.get("global_role") != "dino_cls"
                or manifest.get("ordering") != "row_major_spatial_then_global"
                or manifest.get("processor_fingerprint")
                != identity.processor_fingerprint
            ):
                raise ValueError("standalone DINO spatial/CLS layout mismatch")
            build_commit = manifest.get("build_commit")
            if (
                not isinstance(build_commit, str)
                or len(build_commit) != 40
                or any(character not in "0123456789abcdef" for character in build_commit)
            ):
                raise ValueError("standalone DINO spatial/CLS build commit is invalid")
            expected_parent = _json_fingerprint(
                {"images": manifest.get("images"), "splits": manifest.get("splits")}
            )
            if manifest.get("parent_data_fingerprint") != expected_parent:
                raise ValueError("standalone DINO parent data fingerprint mismatch")
        splits = manifest.get("splits", {})
        if set(splits) != {"train", "val"}:
            raise ValueError("standalone DINO requires train and val lineage")
        for split in splits.values():
            if file_sha256(split["jsonl"]) != split["sha256"]:
                raise ValueError("standalone DINO source JSONL changed")
        images = manifest["images"]
        paths = [entry["path"] for entry in images]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError("standalone DINO image index empty or duplicated")
        for entry in images:
            if (
                str(Path(entry["path"]).resolve()) != entry["path"]
                or file_sha256(entry["path"]) != entry["sha256"]
            ):
                raise ValueError("standalone DINO source image changed")
        for split in splits.values():
            indices = split["image_indices"]
            if not indices or any(
                type(i) is not int or i < 0 or i >= len(images) for i in indices
            ):
                raise ValueError("standalone DINO invalid split image references")
        if {i for split in splits.values() for i in split["image_indices"]} != set(
            range(len(images))
        ):
            raise ValueError("standalone DINO unreferenced image")
        mapping = {}
        shard_references = {}
        offset = 0
        for shard in manifest["shards"]:
            path = root / shard["file"]
            if path.parent != root or file_sha256(path) != shard["sha256"]:
                raise ValueError("standalone DINO shard hash/path mismatch")
            features = _load_grid_shard(path, require_spatial_cls=include_cls)
            if (
                features.shape != (
                    shard["count"],
                    grid_size**2 + int(include_cls),
                    identity.hidden_size,
                )
                or features.dtype != torch.float32
                or not torch.isfinite(features).all()
            ):
                raise ValueError(
                    "standalone DINO shard shape/dtype/finiteness mismatch"
                )
            shard_references[id(features)] = cls._shard_reference(path, features)
            for row in range(len(features)):
                if offset >= len(paths):
                    raise ValueError("standalone DINO too many shard rows")
                mapping[paths[offset]] = (features, row)
                offset += 1
        if offset != len(paths):
            raise ValueError("standalone DINO missing shard rows")
        return cls(
            identity=identity,
            grid_size=grid_size,
            path_to_feature=mapping,
            cache_fingerprint=claimed,
            shard_references=shard_references,
            include_cls=include_cls,
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
    "DINOV2_LARGE_IDENTITY",
    "DINO_GRID_CACHE_FORMAT",
    "STANDALONE_DINO_STATE_CACHE_FORMAT",
    "CachedDINOGridTargets",
    "DINOGridTargets",
    "DINOIdentity",
    "FrozenDINOGridTargets",
]
