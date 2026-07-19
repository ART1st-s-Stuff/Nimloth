"""Frozen DINO-family supervision and exact precomputed CLS targets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn


DEFAULT_DINO_MODEL = "facebook/dinov2-large"
DINO_CACHE_FORMAT = "dino_cls_sharded_v1"


def _json_fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _processor_fingerprint(processor: Any) -> str:
    to_dict = getattr(processor, "to_dict", None)
    payload = to_dict() if callable(to_dict) else {"class": type(processor).__qualname__}
    return _json_fingerprint(payload)


def _local_source_revision(source: str) -> str:
    path = Path(source)
    if not path.is_dir():
        return ""
    entries = []
    for candidate in sorted(path.iterdir()):
        if candidate.name.endswith((".json", ".safetensors", ".bin")):
            stat = candidate.stat()
            entries.append((candidate.name, stat.st_size, stat.st_mtime_ns))
    return _json_fingerprint(entries) if entries else ""


@dataclass(frozen=True)
class DINOIdentity:
    """Immutable identity needed to prove cached targets use the requested teacher."""

    source: str
    revision: str
    processor_fingerprint: str
    hidden_size: int


def resolve_dino_identity(source: str | Path) -> DINOIdentity:
    """Resolve model revision and processor config without loading model weights."""

    from transformers import AutoConfig, AutoImageProcessor

    source_str = str(source)
    config = AutoConfig.from_pretrained(source_str, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(source_str, trust_remote_code=True)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError("DINO model config must define hidden_size")
    revision = str(getattr(config, "_commit_hash", None) or _local_source_revision(source_str))
    return DINOIdentity(
        source=source_str,
        revision=revision,
        processor_fingerprint=_processor_fingerprint(processor),
        hidden_size=int(hidden_size),
    )


class FrozenDINOEncoder(nn.Module):
    """Extract detached global (CLS) features from RGB observations.

    The encoder owns no trainable path: the explicitly selected DINO-family
    model remains in eval mode and its weights are frozen. Nimloth directly
    aligns the projected query state to the final CLS token, so no trainable
    alignment head can absorb the supervision.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        image_processor: Any,
        source: str,
        identity: DINOIdentity | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.source = str(source)
        hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DINO model config must define hidden_size")
        self.hidden_size = int(hidden_size)
        revision = str(
            getattr(getattr(model, "config", None), "_commit_hash", None)
            or _local_source_revision(self.source)
        )
        self.identity = identity or DINOIdentity(
            source=self.source,
            revision=revision,
            processor_fingerprint=_processor_fingerprint(image_processor),
            hidden_size=self.hidden_size,
        )
        if self.identity.hidden_size != self.hidden_size or self.identity.source != self.source:
            raise ValueError("DINO identity does not match loaded model")
        self.model.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path = DEFAULT_DINO_MODEL,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> FrozenDINOEncoder:
        """Load the explicitly requested DINO checkpoint and processor.

        Loading failures are intentionally propagated; the encoder never
        silently substitutes another model or DINO generation.
        """

        from transformers import AutoImageProcessor, AutoModel

        source_str = str(source)
        image_processor = AutoImageProcessor.from_pretrained(source_str, trust_remote_code=True)
        model = AutoModel.from_pretrained(source_str, trust_remote_code=True, torch_dtype=dtype)
        model.to(device=device, dtype=dtype)
        return cls(model=model, image_processor=image_processor, source=source_str)

    def train(self, mode: bool = True) -> FrozenDINOEncoder:
        """Keep the frozen teacher in eval mode even when its parent trains."""

        super().train(False)
        return self

    def _encode_image_paths_hidden(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if not paths:
            raise ValueError("DINO alignment requires at least one image path")
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))

        processed = self.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, non_blocking=True)
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = pixel_values.dtype
        if pixel_values.is_floating_point():
            pixel_values = pixel_values.to(dtype=model_dtype)

        outputs = self.model(pixel_values=pixel_values)
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None or hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            shape = None if hidden is None else tuple(hidden.shape)
            raise ValueError(
                "DINO model must return last_hidden_state with shape "
                f"(B, tokens, {self.hidden_size}), got {shape}"
            )
        return hidden, (int(pixel_values.shape[-2]), int(pixel_values.shape[-1]))

    @torch.no_grad()
    def encode_image_paths(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return final-layer CLS features with shape ``(B, hidden_size)``."""

        hidden, _image_hw = self._encode_image_paths_hidden(paths, device=device)
        return hidden[:, 0, :].detach().float()

    @torch.no_grad()
    def encode_image_paths_grid(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
        grid_size: int = 4,
    ) -> torch.Tensor:
        """Pool final spatial patch tokens into a row-major ``grid_size²`` grid."""

        if grid_size < 1:
            raise ValueError("DINO grid_size must be positive")
        hidden, (image_height, image_width) = self._encode_image_paths_hidden(paths, device=device)
        patch_size = getattr(getattr(self.model, "config", None), "patch_size", None)
        if patch_size is None:
            raise ValueError("DINO patch-grid alignment requires model.config.patch_size")
        patch_size = int(patch_size)
        patch_height = image_height // patch_size
        patch_width = image_width // patch_size
        patch_count = patch_height * patch_width
        if patch_count < 1 or hidden.shape[1] < patch_count + 1:
            raise ValueError(
                "DINO patch token count is incompatible with processed image/grid: "
                f"hidden_tokens={hidden.shape[1]}, image={image_height}x{image_width}, patch={patch_size}"
            )
        # Taking the final H*W tokens is robust to optional register tokens
        # inserted between CLS and spatial patches by some DINO-family models.
        patches = hidden[:, -patch_count:, :]
        patches = patches.reshape(hidden.shape[0], patch_height, patch_width, self.hidden_size)
        patches = patches.permute(0, 3, 1, 2).float()
        pooled = torch.nn.functional.adaptive_avg_pool2d(patches, (grid_size, grid_size))
        return pooled.permute(0, 2, 3, 1).reshape(hidden.shape[0], grid_size * grid_size, self.hidden_size).detach()


def _source_image_fingerprint(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for image_path in paths:
        stat = Path(image_path).stat()
        digest.update(str(image_path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def _image_index(cache_split_dir: Path) -> tuple[list[str], str]:
    index_path = cache_split_dir / "image_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"DINO cache requires compact image index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if payload.get("format") != "dedup_sharded_v1" or not isinstance(images, list):
        raise ValueError(f"invalid compact image index: {index_path}")
    paths = [str(Path(entry["path"]).resolve()) for entry in images]
    return paths, _json_fingerprint(paths)


def _parent_cache_identity(cache_split_dir: Path) -> dict[str, str]:
    manifest_path = cache_split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DINO cache requires compact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "dedup_sharded_v1":
        raise ValueError(f"DINO target cache requires compact Qwen cache: {cache_split_dir}")
    return {
        "parent_fingerprint": str(manifest.get("fingerprint", "")),
        "image_source_fingerprint": str(manifest.get("image_source_fingerprint", "")),
    }


def _load_feature_shard(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - old torch fallback
        payload = torch.load(path, map_location="cpu", weights_only=True)
    features = payload.get("features")
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"invalid DINO feature shard: {path}")
    return features


def _build_dino_split_cache(
    *,
    cache_split_dir: Path,
    encoder: Any,
    device: torch.device,
    batch_size: int,
    shard_size: int,
    force: bool,
) -> dict[str, Any]:
    paths, image_index_fingerprint = _image_index(cache_split_dir)
    parent = _parent_cache_identity(cache_split_dir)
    current_image_source_fingerprint = _source_image_fingerprint(paths)
    if current_image_source_fingerprint != parent["image_source_fingerprint"]:
        raise ValueError(
            f"compact cache image source changed before DINO cache build: {cache_split_dir}"
        )
    identity = encoder.identity
    expected = {
        "format": DINO_CACHE_FORMAT,
        "identity": asdict(identity),
        **parent,
        "image_index_fingerprint": image_index_fingerprint,
        "count": len(paths),
        "feature_dtype": "float32",
        "shard_size": shard_size,
        "shards": math.ceil(len(paths) / shard_size) if paths else 0,
    }
    expected["fingerprint"] = _json_fingerprint(expected)
    output_dir = cache_split_dir / "dino_cls"
    manifest_path = output_dir / "manifest.json"
    build_state_path = output_dir / "build_state.json"

    if force:
        shutil.rmtree(output_dir, ignore_errors=True)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ready = manifest == expected and all(
            (output_dir / f"shard_{index:05d}.pt").is_file()
            for index in range(expected["shards"])
        )
        if ready:
            return manifest
        raise RuntimeError(f"DINO cache fingerprint/files mismatch: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if build_state_path.is_file():
        state = json.loads(build_state_path.read_text(encoding="utf-8"))
        if state != expected:
            raise RuntimeError(f"partial DINO cache identity mismatch: {output_dir}")
    else:
        if any(output_dir.glob("shard_*.pt")):
            raise RuntimeError(f"untracked partial DINO cache: {output_dir}")
        tmp = build_state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        os.replace(tmp, build_state_path)

    for shard_index in range(expected["shards"]):
        shard_path = output_dir / f"shard_{shard_index:05d}.pt"
        start = shard_index * shard_size
        chunk = paths[start : start + shard_size]
        if shard_path.is_file():
            existing = _load_feature_shard(shard_path)
            if existing.shape != (len(chunk), identity.hidden_size) or existing.dtype != torch.float32:
                raise ValueError(f"invalid resumable DINO feature shard: {shard_path}")
            continue
        rows = []
        for offset in range(0, len(chunk), batch_size):
            batch_paths = chunk[offset : offset + batch_size]
            features = encoder.encode_image_paths(batch_paths, device=device)
            if features.shape != (len(batch_paths), identity.hidden_size):
                raise ValueError(
                    f"DINO cache feature shape mismatch: {tuple(features.shape)} != "
                    f"({len(batch_paths)}, {identity.hidden_size})"
                )
            if not torch.isfinite(features.float()).all():
                raise ValueError("DINO cache refuses non-finite feature")
            rows.append(features.detach().to(device="cpu", dtype=torch.float32))
        shard_features = torch.cat(rows, dim=0) if rows else torch.empty((0, identity.hidden_size), dtype=torch.float32)
        tmp = shard_path.with_suffix(".pt.tmp")
        torch.save({"features": shard_features}, tmp)
        os.replace(tmp, shard_path)

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    build_state_path.unlink(missing_ok=True)
    return expected


def build_dino_feature_cache(
    *,
    cache_root: Path,
    encoder: Any,
    device: torch.device,
    batch_size: int = 32,
    shard_size: int = 1024,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build resumable float32 CLS sidecars for compact train/val image caches."""

    if batch_size < 1 or shard_size < 1:
        raise ValueError("DINO cache batch_size and shard_size must be positive")
    manifests = {}
    for split in ("train", "val"):
        manifests[split] = _build_dino_split_cache(
            cache_split_dir=Path(cache_root) / split,
            encoder=encoder,
            device=device,
            batch_size=int(batch_size),
            shard_size=int(shard_size),
            force=force,
        )
    return manifests


class CachedDINOEncoder(nn.Module):
    """Serve exact frozen DINO CLS targets from validated mmap sidecars."""

    def __init__(
        self,
        *,
        identity: DINOIdentity,
        path_to_feature: dict[str, tuple[torch.Tensor, int]],
        cache_fingerprint: str,
    ) -> None:
        super().__init__()
        self.identity = identity
        self.source = identity.source
        self.hidden_size = identity.hidden_size
        self.path_to_feature = path_to_feature
        self.cache_fingerprint = cache_fingerprint
        super().train(False)

    @classmethod
    def from_cache_root(
        cls,
        cache_root: str | Path,
        *,
        identity: DINOIdentity,
    ) -> CachedDINOEncoder:
        path_to_feature: dict[str, tuple[torch.Tensor, int]] = {}
        fingerprints = []
        for split in ("train", "val"):
            split_dir = Path(cache_root) / split
            sidecar = split_dir / "dino_cls"
            manifest_path = sidecar / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"required DINO feature cache missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            claimed_fingerprint = manifest.get("fingerprint")
            fingerprint_payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
            if claimed_fingerprint != _json_fingerprint(fingerprint_payload):
                raise ValueError(f"DINO cache manifest fingerprint mismatch: {sidecar}")
            if manifest.get("format") != DINO_CACHE_FORMAT or manifest.get("identity") != asdict(identity):
                raise ValueError(f"DINO cache identity mismatch: {sidecar}")
            paths, image_index_fingerprint = _image_index(split_dir)
            parent = _parent_cache_identity(split_dir)
            if (
                manifest.get("image_index_fingerprint") != image_index_fingerprint
                or manifest.get("parent_fingerprint") != parent["parent_fingerprint"]
                or manifest.get("image_source_fingerprint") != parent["image_source_fingerprint"]
            ):
                raise ValueError(f"DINO cache parent/image fingerprint mismatch: {sidecar}")
            shard_size = int(manifest.get("shard_size", 0))
            shard_count = int(manifest.get("shards", -1))
            if (
                int(manifest.get("count", -1)) != len(paths)
                or manifest.get("feature_dtype") != "float32"
                or shard_size < 1
                or shard_count != (math.ceil(len(paths) / shard_size) if paths else 0)
            ):
                raise ValueError(f"invalid DINO cache shard metadata: {sidecar}")
            shards = []
            for index in range(shard_count):
                tensor = _load_feature_shard(sidecar / f"shard_{index:05d}.pt")
                expected_rows = min(shard_size, len(paths) - index * shard_size)
                if tensor.shape != (expected_rows, identity.hidden_size) or tensor.dtype != torch.float32:
                    raise ValueError(f"DINO cache feature shape/dtype mismatch: {sidecar}")
                shards.append(tensor)
            for index, path in enumerate(paths):
                location = (shards[index // shard_size], index % shard_size)
                path_to_feature.setdefault(path, location)
            fingerprints.append(str(manifest.get("fingerprint", "")))
        return cls(
            identity=identity,
            path_to_feature=path_to_feature,
            cache_fingerprint=_json_fingerprint(fingerprints),
        )

    def train(self, mode: bool = True) -> CachedDINOEncoder:
        super().train(False)
        return self

    @torch.no_grad()
    def encode_image_paths(
        self,
        paths: Sequence[str | Path],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not paths:
            raise ValueError("DINO alignment requires at least one image path")
        rows = []
        for raw_path in paths:
            path = str(Path(raw_path).resolve())
            location = self.path_to_feature.get(path)
            if location is None:
                raise KeyError(f"DINO cache missing image: {path}")
            tensor, index = location
            rows.append(tensor[index])
        return torch.stack(rows).to(device=device, non_blocking=True).float().detach()
