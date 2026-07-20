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
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Subset

from nimloth.backbone.qwen25vl.batch import build_qwen_batch
from nimloth.training.sft2.dataset import TransitionQwenDataset, collate_transition_batch
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.wm.state_proj import StateProjector

STATE_CACHE_VERSION = "rcdm_state_cache_v2"
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
    latent_token_count: int,
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
            str(latent_token_count),
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


def _shard_name(index: int, compression: Compression, *, rank: int | None = None) -> str:
    suffix = ".pt.gz" if compression == "gzip" else ".pt"
    if rank is None:
        return f"shard_{index:06d}{suffix}"
    return f"shard_r{rank:03d}_{index:06d}{suffix}"


def contiguous_rank_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Split an ordered dataset into balanced contiguous rank ranges."""

    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return total * rank // world_size, total * (rank + 1) // world_size


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
    latent_token_count: int,
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
        latent_token_count=latent_token_count,
        vocab_size=len(processor.tokenizer),
        success_only=success_only,
        max_records=max_records,
        state_dtype=state_dtype,
    )
    distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    cache_hit = False
    if rank == 0 and not force and state_cache_ready(cache_dir):
        manifest = RCDMStateCacheManifest.load(cache_dir)
        cache_hit = manifest.fingerprint == fingerprint
    if distributed:
        hit_payload = [cache_hit]
        dist.broadcast_object_list(hit_payload, src=0)
        cache_hit = bool(hit_payload[0])
    if cache_hit:
        manifest = RCDMStateCacheManifest.load(cache_dir)
        if rank == 0:
            print(json.dumps({"rcdm_state_cache": "hit", "split": split_name, "dir": str(cache_dir), "count": manifest.count}))
        return manifest

    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for old in cache_dir.glob("shard_*.pt*"):
            old.unlink()
        (cache_dir / "manifest.json").unlink(missing_ok=True)
    if distributed:
        dist.barrier()

    ds = TransitionQwenDataset(jsonl_path, max_records=max_records, success_only=success_only)
    sample_start, sample_end = contiguous_rank_bounds(len(ds), rank, world_size)
    rank_ds = Subset(ds, range(sample_start, sample_end))
    loader = DataLoader(
        rank_ds,
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
        filename = _shard_name(
            shard_index,
            compression,
            rank=rank if distributed else None,
        )
        _save_payload(payload, cache_dir / filename, compression)
        shards.append({"file": filename, "count": len(shard_rows)})
        shard_index += 1
        shard_rows = []

    qwen_model.eval()
    state_proj.eval()
    for items in loader:
        enc = build_qwen_batch(
            items,
            processor,
            max_length=max_length,
            latent_token_count=latent_token_count,
        )
        hidden, _ = extract_qwen_latents(
            qwen_model,
            enc,
            token_id_map,
            device,
            latent_token_count=latent_token_count,
        )
        states = state_proj(hidden).detach().float().cpu()
        if cond_dim < 0:
            cond_dim = int(states.shape[-1])
        for item, state in zip(items, states, strict=True):
            shard_rows.append(
                {
                    "id": str(item.get("id", sample_start + count)),
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

    local_total_bytes = sum((cache_dir / str(shard["file"])).stat().st_size for shard in shards)
    local_result = {
        "rank": rank,
        "sample_start": sample_start,
        "sample_end": sample_end,
        "count": count,
        "cond_dim": cond_dim,
        "shards": shards,
        "total_bytes": local_total_bytes,
    }
    if distributed:
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_result)
        rank_results = [result for result in gathered if result is not None]
    else:
        rank_results = [local_result]

    rank_results.sort(key=lambda result: int(result["rank"]))
    cond_dims = {int(result["cond_dim"]) for result in rank_results if int(result["count"]) > 0}
    if len(cond_dims) != 1:
        raise ValueError(f"distributed RCDM cache cond_dim mismatch: {sorted(cond_dims)}")
    if rank == 0:
        merged_shards = [
            shard
            for result in rank_results
            for shard in result["shards"]
        ]
        merged_count = sum(int(result["count"]) for result in rank_results)
        total_bytes = sum(int(result["total_bytes"]) for result in rank_results)
        manifest = RCDMStateCacheManifest(
            cache_dir=cache_dir,
            count=merged_count,
            cond_dim=next(iter(cond_dims)),
            state_dtype=state_dtype,
            compression=compression,
            shard_size=shard_size,
            shards=merged_shards,
            fingerprint=fingerprint,
        )
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
                "latent_token_count": latent_token_count,
                "success_only": success_only,
                "max_records": max_records,
                "cache_build_world_size": world_size,
                "rank_ranges": [
                    {
                        "rank": int(result["rank"]),
                        "start": int(result["sample_start"]),
                        "end": int(result["sample_end"]),
                        "count": int(result["count"]),
                    }
                    for result in rank_results
                ],
                "total_bytes": total_bytes,
            }
        )
        print(json.dumps({
            "rcdm_state_cache": "done",
            "split": split_name,
            "dir": str(cache_dir),
            "count": merged_count,
            "total_bytes": total_bytes,
            "world_size": world_size,
        }))
    if distributed:
        dist.barrier()
    return RCDMStateCacheManifest.load(cache_dir)


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
