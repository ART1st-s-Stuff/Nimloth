"""Qwen transition 预处理缓存的 Dataset 与 mmap reader。"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from nimloth.util.cache.schema import (
    COMPACT_CACHE_FORMAT,
    safe_cache_name,
    transition_sample_id,
)
from nimloth.agent import bind_image_placeholders
from nimloth.rollout.transitions import TransitionContextIndex, TransitionSample


class _MmapShardStore:
    """Per-DataLoader-worker LRU of mmap-backed torch shards."""

    def __init__(self, root: Path, *, max_open_shards: int = 2) -> None:
        self.root = root
        self.max_open_shards = max(1, int(max_open_shards))
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def get(self, shard_index: int) -> dict[str, Any]:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        path = self.root / f"shard_{shard_index:05d}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing compact cache shard: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover - older torch fallback
            payload = torch.load(path, map_location="cpu", weights_only=True)
        self._cache[shard_index] = payload
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return payload


class CompactCachedTransitionCollator:
    """Materialize deduplicated pixels in DataLoader workers, then pre-batch them."""

    def __init__(self, cache_dir: Path, *, max_open_shards: int = 2) -> None:
        self.cache_dir = cache_dir
        index_path = cache_dir / "image_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing compact image index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        self.image_locations = list(index["images"])
        self.max_open_shards = max(1, int(max_open_shards))
        self._image_store: _MmapShardStore | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_image_store"] = None
        return state

    def _store(self) -> _MmapShardStore:
        if self._image_store is None:
            self._image_store = _MmapShardStore(
                self.cache_dir / "images",
                max_open_shards=self.max_open_shards,
            )
        return self._image_store

    def _materialize_encoding(self, compact: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {key: value for key, value in compact.items() if key != "image_indices"}
        image_indices = compact.get("image_indices")
        if image_indices is None or image_indices.numel() == 0:
            return out
        pixel_rows: list[torch.Tensor] = []
        for image_index in image_indices.tolist():
            location = self.image_locations[int(image_index)]
            shard = self._store().get(int(location["shard"]))
            local_index = int(location["index"])
            offsets = shard["offsets"]
            start = int(offsets[local_index].item())
            end = int(offsets[local_index + 1].item())
            pixel_rows.append(shard["pixel_values"][start:end])
        out["pixel_values"] = torch.cat(pixel_rows, dim=0).contiguous()
        return out

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        materialized: dict[int, dict[str, torch.Tensor]] = {}
        current_rows: list[dict[str, torch.Tensor]] = []
        items: list[dict[str, Any]] = []
        for entry in batch:
            cache_index = int(entry["cache_index"])
            row = self._materialize_encoding(entry["current_enc"])
            materialized[cache_index] = row
            current_rows.append(row)
            items.append(
                {
                    "id": entry["id"],
                    "record_id": entry.get("record_id", ""),
                    "step_index": entry.get("step_index", 0),
                    "action_index": entry["action_index"],
                    "action_value_target": entry["action_value_target"],
                    "success": entry["success"],
                    "messages": entry.get("messages"),
                    "next_messages": entry.get("next_messages"),
                    "context_length": entry.get("context_length"),
                    "is_current_step": entry.get("is_current_step"),
                }
            )

        next_rows: list[dict[str, torch.Tensor] | None] = []
        for entry in batch:
            next_index = entry.get("next_cache_index")
            if next_index is None:
                compact_next = entry.get("next_enc")
                next_rows.append(
                    self._materialize_encoding(compact_next)
                    if compact_next is not None
                    else None
                )
                continue
            next_index = int(next_index)
            row = materialized.get(next_index)
            if row is None:
                compact_next = entry.get("next_enc")
                if compact_next is None:
                    raise ValueError("compact next encoding missing for WM-eligible transition")
                row = self._materialize_encoding(compact_next)
                materialized[next_index] = row
            next_rows.append(row)

        return {
            "items": items,
            "current_enc_rows": current_rows,
            "next_enc_rows": next_rows,
        }


class CachedTransitionDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        samples: list[TransitionSample],
        *,
        max_open_shards: int = 2,
    ):
        self.cache_dir = cache_dir
        self.samples = samples
        manifest_path = cache_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
        self.is_compact = self.manifest.get("format") == COMPACT_CACHE_FORMAT
        self.transition_shard_size = int(self.manifest.get("transition_shard_size", 0) or 0)
        self._transition_store = (
            _MmapShardStore(cache_dir / "transitions", max_open_shards=max_open_shards)
            if self.is_compact
            else None
        )
        self._sample_index = {
            (sample.record_id, sample.step_index): index
            for index, sample in enumerate(samples)
        }
        if self.is_compact and self.transition_shard_size <= 0:
            raise ValueError(f"invalid compact transition_shard_size in {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def cache_path_for_sample(self, sample: TransitionSample) -> Path:
        return self.cache_dir / f"{safe_cache_name(transition_sample_id(sample))}.pt"

    def _compact_entry(self, index: int) -> dict[str, Any]:
        if self._transition_store is None:
            raise RuntimeError("compact transition store is not initialized")
        shard_index = index // self.transition_shard_size
        local_index = index % self.transition_shard_size
        shard = self._transition_store.get(shard_index)
        entries = shard["entries"]
        if local_index >= len(entries):
            raise IndexError(f"compact cache index {index} missing from shard {shard_index}")
        return dict(entries[local_index])

    def __getitem__(self, index: int | TransitionContextIndex) -> dict[str, Any]:
        context_index = index if isinstance(index, TransitionContextIndex) else None
        if context_index is not None:
            index = context_index.sample_index
        sample = self.samples[index]
        if self.is_compact:
            entry = self._compact_entry(index)
            if entry.get("id") != transition_sample_id(sample):
                raise ValueError(
                    f"compact cache/sample mismatch at {index}: {entry.get('id')!r} != "
                    f"{transition_sample_id(sample)!r}"
                )
            entry["cache_index"] = index
            next_index = None
            if sample.next_prefix_messages is not None:
                next_index = self._sample_index.get((sample.record_id, sample.step_index + 1))
            entry["next_cache_index"] = next_index
            if next_index is not None:
                entry["next_enc"] = self._compact_entry(next_index)["current_enc"]
            else:
                entry["next_enc"] = entry.get("next_enc")
        else:
            cache_path = self.cache_path_for_sample(sample)
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing preprocess cache: {cache_path}")
            entry = torch.load(cache_path, map_location="cpu", weights_only=True)
        entry["messages"] = bind_image_placeholders(
            sample.prefix_messages,
            sample.prefix_image_paths,
        )
        if sample.next_prefix_messages is not None and sample.next_prefix_image_paths is not None:
            entry["next_messages"] = bind_image_placeholders(
                sample.next_prefix_messages,
                sample.next_prefix_image_paths,
            )
        if context_index is not None:
            entry["context_length"] = context_index.context_length
            entry["is_current_step"] = context_index.is_current_step
        return entry
