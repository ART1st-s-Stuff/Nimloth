"""On-disk cache for direct Qwen vision-encoder tokens."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from nimloth.representation_ablation.vision_tokens import extract_qwen_vision_tokens
from nimloth.training.sft2.dataset import TransitionQwenDataset

CACHE_VERSION = "qwen_vision_token_cache_v1"
CacheDType = Literal["float16", "bfloat16", "float32"]


def _path_stat_payload(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def _torch_dtype(name: CacheDType) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported cache dtype: {name}")


def _checkpoint_payload(path: Path) -> str:
    if path.is_dir():
        parts = []
        for name in (
            "config.json",
            "adapter_config.json",
            "adapter_model.safetensors",
            "vision_full_state.pt",
        ):
            child = path / name
            if child.exists():
                parts.append(_path_stat_payload(child))
        return "|".join(parts) or str(path.resolve())
    return _path_stat_payload(path)


def _image_paths_from_jsonl(jsonl_path: Path, *, max_records: int, success_only: bool) -> list[str]:
    ds = TransitionQwenDataset(jsonl_path, max_records=max_records, success_only=success_only)
    paths: list[str] = []
    seen: set[str] = set()
    for sample in ds.samples:
        for key in ("current_image_path", "next_image_path"):
            value = getattr(sample, key, None)
            if value is None:
                continue
            path = str(value)
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def cache_fingerprint(
    *,
    jsonl_path: Path,
    qwen_checkpoint: Path,
    max_records: int,
    success_only: bool,
    max_pixels: int,
    expected_num_tokens: int,
    token_dim: int,
    dtype: CacheDType,
) -> str:
    payload = "|".join(
        [
            CACHE_VERSION,
            _path_stat_payload(jsonl_path),
            _checkpoint_payload(qwen_checkpoint),
            str(max_records),
            str(success_only),
            str(max_pixels),
            str(expected_num_tokens),
            str(token_dim),
            dtype,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class VisionTokenCacheManifest:
    cache_dir: Path
    count: int
    num_tokens: int
    token_dim: int
    dtype: CacheDType
    shard_size: int
    shards: list[dict[str, Any]]
    entries: dict[str, tuple[str, int]]
    fingerprint: str

    @classmethod
    def load(cls, cache_dir: Path) -> "VisionTokenCacheManifest":
        data = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            cache_dir=cache_dir,
            count=int(data["count"]),
            num_tokens=int(data["num_tokens"]),
            token_dim=int(data["token_dim"]),
            dtype=data["dtype"],
            shard_size=int(data["shard_size"]),
            shards=list(data["shards"]),
            entries={str(k): (str(v[0]), int(v[1])) for k, v in dict(data["entries"]).items()},
            fingerprint=str(data["fingerprint"]),
        )

    def write(self, extra: dict[str, Any]) -> None:
        payload = {
            **extra,
            "version": CACHE_VERSION,
            "count": self.count,
            "num_tokens": self.num_tokens,
            "token_dim": self.token_dim,
            "dtype": self.dtype,
            "shard_size": self.shard_size,
            "shards": self.shards,
            "entries": self.entries,
            "fingerprint": self.fingerprint,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def vision_token_cache_ready(cache_dir: Path) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = VisionTokenCacheManifest.load(cache_dir)
    except Exception:
        return False
    return all((cache_dir / str(shard["file"])).is_file() for shard in manifest.shards)


@torch.no_grad()
def build_vision_token_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    split_name: str,
    qwen_checkpoint: Path,
    model,
    processor,
    device: torch.device,
    max_pixels: int,
    expected_num_tokens: int,
    token_dim: int,
    max_records: int = -1,
    success_only: bool = False,
    batch_size: int = 32,
    shard_size: int = 1024,
    dtype: CacheDType = "float16",
    force: bool = False,
) -> VisionTokenCacheManifest:
    fingerprint = cache_fingerprint(
        jsonl_path=jsonl_path,
        qwen_checkpoint=qwen_checkpoint,
        max_records=max_records,
        success_only=success_only,
        max_pixels=max_pixels,
        expected_num_tokens=expected_num_tokens,
        token_dim=token_dim,
        dtype=dtype,
    )
    if not force and vision_token_cache_ready(cache_dir):
        manifest = VisionTokenCacheManifest.load(cache_dir)
        if manifest.fingerprint == fingerprint:
            print(json.dumps({"vision_token_cache": "hit", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
            return manifest

    paths = _image_paths_from_jsonl(jsonl_path, max_records=max_records, success_only=success_only)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("shard_*.pt"):
        old.unlink()
    target_dtype = _torch_dtype(dtype)
    shards: list[dict[str, Any]] = []
    entries: dict[str, tuple[str, int]] = {}
    shard_paths: list[str] = []
    shard_tokens: list[torch.Tensor] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_paths, shard_tokens, shard_index
        if not shard_paths:
            return
        filename = f"shard_{shard_index:06d}.pt"
        tokens = torch.stack(shard_tokens, dim=0).to(dtype=target_dtype)
        torch.save({"image_paths": shard_paths, "tokens": tokens}, cache_dir / filename)
        for offset, path in enumerate(shard_paths):
            entries[path] = (filename, offset)
        shards.append({"file": filename, "count": len(shard_paths)})
        shard_paths = []
        shard_tokens = []
        shard_index += 1

    model.eval()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        tokens = extract_qwen_vision_tokens(
            model,
            processor,
            batch_paths,
            device=device,
            max_pixels=max_pixels,
            expected_num_tokens=expected_num_tokens,
        ).detach().cpu()
        if tokens.shape[1] != expected_num_tokens or tokens.shape[2] != token_dim:
            raise ValueError(
                f"unexpected token shape {tuple(tokens.shape)}; expected (B, {expected_num_tokens}, {token_dim})"
            )
        for path, token in zip(batch_paths, tokens, strict=True):
            shard_paths.append(path)
            shard_tokens.append(token)
            if len(shard_paths) >= shard_size:
                flush()
    flush()

    manifest = VisionTokenCacheManifest(
        cache_dir=cache_dir,
        count=len(paths),
        num_tokens=expected_num_tokens,
        token_dim=token_dim,
        dtype=dtype,
        shard_size=shard_size,
        shards=shards,
        entries=entries,
        fingerprint=fingerprint,
    )
    manifest.write(
        {
            "split": split_name,
            "jsonl_path": str(jsonl_path),
            "qwen_checkpoint": str(qwen_checkpoint),
            "max_records": max_records,
            "success_only": success_only,
            "max_pixels": max_pixels,
        }
    )
    print(json.dumps({"vision_token_cache": "built", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
    return manifest


class VisionTokenCache:
    """Lazy shard loader for cached Qwen vision tokens."""

    def __init__(self, cache_dir: Path, *, device: torch.device | None = None) -> None:
        self.manifest = VisionTokenCacheManifest.load(cache_dir)
        self.cache_dir = Path(cache_dir)
        self.device = device
        self._loaded: dict[str, torch.Tensor] = {}

    def _load_shard(self, filename: str) -> torch.Tensor:
        if filename not in self._loaded:
            payload = torch.load(self.cache_dir / filename, map_location="cpu", weights_only=False)
            self._loaded[filename] = payload["tokens"]
        return self._loaded[filename]

    def get(self, image_path: str | Path) -> torch.Tensor:
        key = str(image_path)
        if key not in self.manifest.entries:
            raise KeyError(f"image path not found in vision-token cache: {key}")
        filename, offset = self.manifest.entries[key]
        token = self._load_shard(filename)[offset].float()
        if self.device is not None:
            token = token.to(self.device)
        return token

    def get_many(self, image_paths: Sequence[str | Path]) -> torch.Tensor:
        return torch.stack([self.get(path) for path in image_paths], dim=0)
