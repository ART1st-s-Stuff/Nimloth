"""Compressed state-embedding cache for RCDM SFT2 reconstruction.

This cache stores the expensive part of post-hoc RCDM training:
``StateProjector(Qwen <|latent_state|>)``.  It intentionally keeps image
contents out of the cache and stores only image paths, so the cache stays small
and RCDM can still choose the target image resolution at training time.
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
from torch.utils.data import DataLoader, Dataset

from nimloth.training.common.qwen_batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.training.sft2.qwen_latent import extract_qwen_latents
from nimloth.wm.state_proj import StateProjector

STATE_CACHE_VERSION = "rcdm_state_cache_v1"
Compression = Literal["gzip", "none"]
StateDType = Literal["float16", "bfloat16", "float32"]


def _path_stat_payload(path: Path) -> str:
    stat = path.stat()
    return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"


def _checkpoint_payload(path: Path) -> str:
    if path.is_dir():
        parts = []
        for name in ("predictor.pt", "config.json"):
            child = path / name
            if child.exists():
                parts.append(_path_stat_payload(child))
        return "|".join(parts) or str(path.resolve())
    return _path_stat_payload(path)


def state_cache_fingerprint(
    *,
    jsonl_path: Path,
    model_path: Path,
    state_proj_checkpoint: Path,
    wm_checkpoint: Path,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    vocab_size: int,
    success_only: bool,
    max_records: int,
    state_dtype: StateDType,
) -> str:
    payload = "|".join(
        [
            STATE_CACHE_VERSION,
            _path_stat_payload(jsonl_path),
            str(model_path.resolve()),
            _checkpoint_payload(state_proj_checkpoint),
            _checkpoint_payload(wm_checkpoint),
            str(max_length),
            str(max_pixels),
            str(min_pixels),
            str(vocab_size),
            str(success_only),
            str(max_records),
            state_dtype,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _torch_dtype(name: StateDType) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported state dtype: {name}")


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


@dataclass(frozen=True)
class RCDMStateCacheManifest:
    cache_dir: Path
    count: int
    cond_dim: int
    state_dtype: StateDType
    compression: Compression
    shard_size: int
    shards: list[dict[str, Any]]
    fingerprint: str

    @classmethod
    def load(cls, cache_dir: Path) -> "RCDMStateCacheManifest":
        path = cache_dir / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            cache_dir=cache_dir,
            count=int(data["count"]),
            cond_dim=int(data["cond_dim"]),
            state_dtype=data["state_dtype"],
            compression=data["compression"],
            shard_size=int(data["shard_size"]),
            shards=list(data["shards"]),
            fingerprint=str(data["fingerprint"]),
        )

    def write(self, extra: dict[str, Any]) -> None:
        payload = {
            **extra,
            "version": STATE_CACHE_VERSION,
            "count": self.count,
            "cond_dim": self.cond_dim,
            "state_dtype": self.state_dtype,
            "compression": self.compression,
            "shard_size": self.shard_size,
            "shards": self.shards,
            "fingerprint": self.fingerprint,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def state_cache_ready(cache_dir: Path) -> bool:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = RCDMStateCacheManifest.load(cache_dir)
    except Exception:
        return False
    return all((cache_dir / str(shard["file"])).is_file() for shard in manifest.shards)


@torch.no_grad()
def build_rcdm_state_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    split_name: str,
    model_path: Path,
    state_proj_checkpoint: Path,
    wm_checkpoint: Path,
    processor,
    qwen_model,
    token_id_map: dict[str, int],
    state_proj: StateProjector,
    device: torch.device,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    max_records: int = -1,
    success_only: bool = False,
    batch_size: int = 1,
    shard_size: int = 4096,
    compression: Compression = "gzip",
    state_dtype: StateDType = "float16",
    force: bool = False,
) -> RCDMStateCacheManifest:
    """Precompute and compressed-save SFT2 state embeddings for one split."""

    fingerprint = state_cache_fingerprint(
        jsonl_path=jsonl_path,
        model_path=model_path,
        state_proj_checkpoint=state_proj_checkpoint,
        wm_checkpoint=wm_checkpoint,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        success_only=success_only,
        max_records=max_records,
        state_dtype=state_dtype,
    )
    if not force and state_cache_ready(cache_dir):
        manifest = RCDMStateCacheManifest.load(cache_dir)
        if manifest.fingerprint == fingerprint:
            print(json.dumps({"rcdm_state_cache": "hit", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
            return manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob("shard_*.pt*"):
        old.unlink()

    ds = TransitionQwenDataset(jsonl_path, max_records=max_records, success_only=success_only)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transition_batch,
    )
    target_dtype = _torch_dtype(state_dtype)
    shard_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    count = 0
    shard_index = 0
    cond_dim = -1

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        states = torch.stack([row.pop("state_emb") for row in shard_rows]).to(dtype=target_dtype)
        payload = {
            "state_emb": states,
            "rows": shard_rows,
        }
        filename = _shard_name(shard_index, compression)
        _save_payload(payload, cache_dir / filename, compression)
        shards.append({"file": filename, "count": len(shard_rows)})
        shard_index += 1
        shard_rows = []

    qwen_model.eval()
    state_proj.eval()
    for items in loader:
        enc = build_qwen_batch(items, processor, max_length=max_length)
        hidden, _ = extract_qwen_latents(qwen_model, enc, token_id_map, device)
        states = state_proj(hidden).detach().float().cpu()
        if cond_dim < 0:
            cond_dim = int(states.shape[-1])
        for item, state in zip(items, states, strict=True):
            shard_rows.append(
                {
                    "id": str(item.get("id", count)),
                    "record_id": str(item.get("record_id", "")),
                    "step_index": int(item.get("step_index", -1)),
                    "action_index": int(item["action_index"]),
                    "success": bool(item.get("success", False)),
                    "current_image_path": str(item["current_image_path"]),
                    "next_image_path": str(item["next_image_path"]),
                    "state_emb": state,
                }
            )
            count += 1
            if len(shard_rows) >= shard_size:
                flush()
    flush()

    manifest = RCDMStateCacheManifest(
        cache_dir=cache_dir,
        count=count,
        cond_dim=cond_dim,
        state_dtype=state_dtype,
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
            "state_proj_checkpoint": str(state_proj_checkpoint),
            "wm_checkpoint": str(wm_checkpoint),
            "max_length": max_length,
            "max_pixels": max_pixels,
            "min_pixels": min_pixels,
            "success_only": success_only,
            "max_records": max_records,
            "total_bytes": total_bytes,
        }
    )
    print(json.dumps({"rcdm_state_cache": "done", "split": split_name, "dir": str(cache_dir), "count": count, "total_bytes": total_bytes}))
    return manifest


class RCDMStateCacheDataset(Dataset):
    """Dataset over compressed RCDM state-cache shards."""

    def __init__(self, cache_dir: Path) -> None:
        self.manifest = RCDMStateCacheManifest.load(cache_dir)
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
        row["state_emb"] = payload["state_emb"][row_idx]
        return row


def collate_rcdm_state_cache_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_emb": torch.stack([item["state_emb"] for item in batch]),
        "action_index": torch.tensor([int(item["action_index"]) for item in batch], dtype=torch.long),
        "id": [str(item["id"]) for item in batch],
        "record_id": [str(item.get("record_id", "")) for item in batch],
        "step_index": [int(item.get("step_index", -1)) for item in batch],
        "success": [bool(item.get("success", False)) for item in batch],
        "current_image_path": [str(item["current_image_path"]) for item in batch],
        "next_image_path": [str(item["next_image_path"]) for item in batch],
    }
