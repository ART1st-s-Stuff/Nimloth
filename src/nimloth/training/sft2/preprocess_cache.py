"""Disk preprocess cache for SFT2 transition prefixes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.common.dist import is_main
from nimloth.training.common.qwen_batch import (
    _labels_for_text_rows,
    _message_cache_key,
    _render_messages,
    assistant_char_spans,
    encode_qwen_item,
)
from nimloth.wm.collate import messages_with_image_paths, transition_collate_for_qwen
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample, expand_record_transitions, load_jsonl_records

CE_MASK_VERSION = "last_assistant_span_v1"
TRANSITION_EXPANSION_VERSION = "wm_expand_v1"
DEFAULT_MIN_PIXELS = 3136
COMPACT_CACHE_FORMAT = "dedup_sharded_v1"
LEGACY_CACHE_FORMAT = "legacy_transition_v1"


def safe_cache_name(sample_id: str) -> str:
    return sample_id.replace("/", "__").replace(" ", "_")


def transition_sample_id(sample: TransitionSample) -> str:
    return f"{sample.record_id}:{sample.step_index}"


def cache_fingerprint(
    jsonl_path: Path,
    *,
    max_length: int,
    max_pixels: int,
    min_pixels: int,
    vocab_size: int,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    cache_format: str = LEGACY_CACHE_FORMAT,
    image_dtype: str = "float32",
    processor_source: str = "",
) -> str:
    stat = jsonl_path.stat()
    payload = "|".join(
        [
            str(jsonl_path.resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            str(max_length),
            str(max_pixels),
            str(min_pixels),
            str(vocab_size),
            str(value_gamma),
            str(latent_token_count),
            "inject" if mask_latent_query_labels else "generate",
            str(mask_latent_query_labels),
            cache_format,
            image_dtype,
            processor_source,
            CE_MASK_VERSION,
            TRANSITION_EXPANSION_VERSION,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_transition_item(
    item: dict[str, Any],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    current_enc = encode_qwen_item(
        item["messages"],
        processor,
        max_length,
        include_labels=True,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    next_enc = None
    if item.get("next_messages"):
        next_enc = encode_qwen_item(
            item["next_messages"],
            processor,
            max_length,
            include_labels=False,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
    return {
        "id": item["id"],
        "record_id": item.get("record_id", ""),
        "step_index": item.get("step_index", 0),
        "action_index": item["action_index"],
        "action_value_target": item["action_value_target"],
        "success": item["success"],
        "current_enc": current_enc,
        "next_enc": next_enc,
    }


def _expand_qwen_image_tokens(
    text: str,
    image_grid_thw: torch.Tensor,
    processor: AutoProcessor,
) -> str:
    """Expand Qwen image placeholders exactly as Qwen2_5_VLProcessor.__call__."""

    image_token = str(processor.image_token)
    merge_length = int(processor.image_processor.merge_size) ** 2
    expanded = text
    for grid in image_grid_thw:
        if image_token not in expanded:
            raise ValueError("fewer image placeholders than cached image grids")
        token_count = int(grid.prod().item()) // merge_length
        expanded = expanded.replace(image_token, "<|placeholder|>" * token_count, 1)
    if image_token in expanded:
        raise ValueError("more image placeholders than cached image grids")
    return expanded.replace("<|placeholder|>", image_token)


def encode_qwen_item_from_image_grids(
    messages: list[dict[str, Any]],
    image_grid_thw: torch.Tensor,
    processor: AutoProcessor,
    max_length: int,
    *,
    include_labels: bool = True,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode text/labels without reprocessing pixels.

    The image grids come from a per-image cache. Image-token expansion mirrors
    the Hugging Face processor, so input ids are identical to online encoding.
    """

    text = _render_messages(
        messages,
        processor,
        add_generation_prompt=False,
        latent_token_count=latent_token_count,
    )
    grids = image_grid_thw.to(dtype=torch.long, device="cpu").reshape(-1, 3).contiguous()
    expanded_text = _expand_qwen_image_tokens(text, grids, processor)
    enc = processor.tokenizer(
        [expanded_text],
        padding=False,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    out: dict[str, torch.Tensor] = {
        key: value.squeeze(0).contiguous()
        for key, value in enc.items()
        if isinstance(value, torch.Tensor)
    }
    if grids.numel():
        out["image_grid_thw"] = grids
    if include_labels:
        labels = _labels_for_text_rows(
            processor,
            enc["input_ids"],
            [text],
            [assistant_char_spans(messages, processor, latent_token_count=latent_token_count)],
            max_length,
            latent_token_count=latent_token_count,
            mask_latent_query_labels=mask_latent_query_labels,
        )
        out["labels"] = labels.squeeze(0).contiguous()
    return out


def collate_cached_encodings(batch: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    if len(batch) == 1:
        return {k: (v.unsqueeze(0) if v.ndim == 1 else v) for k, v in batch[0].items()}
    out: dict[str, torch.Tensor] = {}
    if "input_ids" in batch[0]:
        out["input_ids"] = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=pad_token_id,
        )
    if "attention_mask" in batch[0]:
        out["attention_mask"] = pad_sequence(
            [item["attention_mask"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
    if "labels" in batch[0]:
        out["labels"] = pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
    for key in ("pixel_values", "image_grid_thw"):
        if key in batch[0]:
            tensors = []
            for item in batch:
                tensor = item[key]
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                tensors.append(tensor)
            out[key] = torch.cat(tensors, dim=0)
    return out


def _collate_next_encoding_bundle(
    items: list[dict[str, Any]],
    next_rows: list[dict[str, torch.Tensor] | None],
    *,
    pad_token_id: int,
) -> dict[str, Any] | None:
    unique_rows: list[dict[str, torch.Tensor]] = []
    unique_keys: list[str] = []
    seen: set[str] = set()
    for item, row in zip(items, next_rows, strict=True):
        messages = item.get("next_messages")
        if not messages or row is None:
            continue
        key = _message_cache_key(messages)
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
        unique_rows.append(row)
    if not unique_rows:
        return None
    return {
        "keys": unique_keys,
        "enc": collate_cached_encodings(unique_rows, pad_token_id),
    }


def collate_cached_transition_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    current_rows: list[dict[str, torch.Tensor]] = []
    next_rows: list[dict[str, torch.Tensor] | None] = []
    for entry in batch:
        items.append(
            {
                "id": entry["id"],
                "record_id": entry.get("record_id", ""),
                "step_index": entry.get("step_index", 0),
                "action_index": entry["action_index"],
                "action_value_target": entry["action_value_target"],
                "success": entry["success"],
                "current_image_path": entry.get("current_image_path"),
                "next_image_path": entry.get("next_image_path"),
                "messages": entry.get("messages"),
                "next_messages": entry.get("next_messages"),
            }
        )
        current_rows.append(entry["current_enc"])
        next_rows.append(entry.get("next_enc"))
    return {
        "items": items,
        "current_enc": collate_cached_encodings(current_rows, pad_token_id),
        "current_enc_rows": current_rows,
        "next_enc_rows": next_rows,
        "next_enc_bundle": _collate_next_encoding_bundle(
            items,
            next_rows,
            pad_token_id=pad_token_id,
        ),
    }


def unpack_transition_batch(
    batch,
    processor,
    max_length: int,
    *,
    pad_token_id: int | None = None,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    Any,
]:
    from nimloth.training.common.qwen_batch import build_qwen_batch

    if isinstance(batch, dict) and "current_enc" in batch:
        items = batch["items"]
        enc = batch["current_enc"]
        next_rows = batch.get("next_enc_bundle", batch.get("next_enc_rows"))
        return items, enc, next_rows
    items = batch
    enc = build_qwen_batch(
        items,
        processor,
        max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    return items, enc, None


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

    def __init__(self, cache_dir: Path, *, pad_token_id: int, max_open_shards: int = 2) -> None:
        self.cache_dir = cache_dir
        self.pad_token_id = int(pad_token_id)
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
                    "current_image_path": entry.get("current_image_path"),
                    "next_image_path": entry.get("next_image_path"),
                    "messages": entry.get("messages"),
                    "next_messages": entry.get("next_messages"),
                }
            )

        next_rows: list[dict[str, torch.Tensor] | None] = []
        for entry in batch:
            next_index = entry.get("next_cache_index")
            if next_index is None:
                next_rows.append(None)
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
            "current_enc": collate_cached_encodings(current_rows, self.pad_token_id),
            "current_enc_rows": current_rows,
            "next_enc_rows": next_rows,
            "next_enc_bundle": _collate_next_encoding_bundle(
                items,
                next_rows,
                pad_token_id=self.pad_token_id,
            ),
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

    def __getitem__(self, index: int) -> dict[str, Any]:
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
            entry["next_enc"] = self._compact_entry(next_index)["current_enc"] if next_index is not None else None
        else:
            cache_path = self.cache_path_for_sample(sample)
            if not cache_path.is_file():
                raise FileNotFoundError(f"missing preprocess cache: {cache_path}")
            entry = torch.load(cache_path, map_location="cpu", weights_only=True)
        entry["current_image_path"] = sample.current_image_path
        entry["next_image_path"] = sample.next_image_path
        entry["messages"] = messages_with_image_paths(
            sample.prefix_messages,
            sample.prefix_image_paths,
        )
        if sample.next_prefix_messages is not None and sample.next_prefix_image_paths is not None:
            entry["next_messages"] = messages_with_image_paths(
                sample.next_prefix_messages,
                sample.next_prefix_image_paths,
            )
        return entry


_CACHE_PROCESSOR: AutoProcessor | None = None
_CACHE_MAX_LENGTH = 0
_CACHE_LATENT_TOKEN_COUNT = 1
_CACHE_MASK_LATENT_QUERY_LABELS = True
_COMPACT_PATH_TO_IMAGE_INDEX: dict[str, int] = {}
_COMPACT_IMAGE_GRIDS: list[list[int]] = []


def _init_cache_worker(
    model_path: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> None:
    global _CACHE_PROCESSOR, _CACHE_MAX_LENGTH, _CACHE_LATENT_TOKEN_COUNT, _CACHE_MASK_LATENT_QUERY_LABELS
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    add_special_tokens(processor.tokenizer, latent_token_count=latent_token_count)
    _CACHE_PROCESSOR = processor
    _CACHE_MAX_LENGTH = max_length
    _CACHE_LATENT_TOKEN_COUNT = int(latent_token_count)
    _CACHE_MASK_LATENT_QUERY_LABELS = bool(mask_latent_query_labels)


def _init_compact_cache_worker(
    model_path: str,
    min_pixels: int,
    max_pixels: int,
    max_length: int,
    latent_token_count: int,
    mask_latent_query_labels: bool,
    path_to_image_index: dict[str, int],
    image_grids: list[list[int]],
) -> None:
    global _COMPACT_PATH_TO_IMAGE_INDEX, _COMPACT_IMAGE_GRIDS
    _init_cache_worker(
        model_path,
        min_pixels,
        max_pixels,
        max_length,
        latent_token_count,
        mask_latent_query_labels,
    )
    _COMPACT_PATH_TO_IMAGE_INDEX = path_to_image_index
    _COMPACT_IMAGE_GRIDS = image_grids


def _cache_one_transition(task: tuple[dict[str, Any], str]) -> tuple[str, bool, str]:
    item, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_transition_item(
            item,
            _CACHE_PROCESSOR,
            _CACHE_MAX_LENGTH,
            latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
            mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
        )
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return item["id"], True, ""
    except Exception as exc:  # noqa: BLE001
        return item["id"], False, str(exc)


def _cache_image_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported compact cache image dtype: {name}") from exc


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _cache_one_image_shard(task: tuple[list[str], str, str]) -> tuple[str, bool, str]:
    image_paths, out_path, dtype_name = task
    try:
        assert _CACHE_PROCESSOR is not None
        dtype = _cache_image_dtype(dtype_name)
        pixels: list[torch.Tensor] = []
        grids: list[torch.Tensor] = []
        offsets = [0]
        for image_path in image_paths:
            with Image.open(image_path) as image:
                encoded = _CACHE_PROCESSOR.image_processor(
                    images=[image.convert("RGB")],
                    return_tensors="pt",
                )
            pixel_values = encoded["pixel_values"].to(dtype=dtype).contiguous()
            grid = encoded["image_grid_thw"].to(dtype=torch.long).reshape(1, 3).contiguous()
            expected_rows = int(grid[0].prod().item())
            if pixel_values.shape[0] != expected_rows:
                raise ValueError(
                    f"pixel/grid mismatch for {image_path}: {pixel_values.shape[0]} != {expected_rows}"
                )
            pixels.append(pixel_values)
            grids.append(grid)
            offsets.append(offsets[-1] + pixel_values.shape[0])
        payload = {
            "pixel_values": torch.cat(pixels, dim=0) if pixels else torch.empty((0, 0), dtype=dtype),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "image_grid_thw": torch.cat(grids, dim=0) if grids else torch.empty((0, 3), dtype=torch.long),
        }
        _atomic_torch_save(payload, Path(out_path))
        return out_path, True, ""
    except Exception as exc:  # noqa: BLE001
        return out_path, False, str(exc)


def _cache_one_compact_transition_shard(
    task: tuple[list[TransitionSample], str],
) -> tuple[str, bool, str]:
    samples, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        entries: list[dict[str, Any]] = []
        for sample in samples:
            item = transition_collate_for_qwen([sample])[0]
            image_indices = [
                _COMPACT_PATH_TO_IMAGE_INDEX[str(Path(path).resolve())]
                for path in sample.prefix_image_paths
            ]
            grids = torch.tensor(
                [_COMPACT_IMAGE_GRIDS[index] for index in image_indices],
                dtype=torch.long,
            ).reshape(-1, 3)
            current_enc = encode_qwen_item_from_image_grids(
                item["messages"],
                grids,
                _CACHE_PROCESSOR,
                _CACHE_MAX_LENGTH,
                include_labels=True,
                latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
                mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
            )
            current_enc["image_indices"] = torch.tensor(image_indices, dtype=torch.int32)
            entries.append(
                {
                    "id": item["id"],
                    "record_id": item["record_id"],
                    "step_index": item["step_index"],
                    "action_index": item["action_index"],
                    "action_value_target": item["action_value_target"],
                    "success": item["success"],
                    "current_enc": current_enc,
                }
            )
        _atomic_torch_save({"entries": entries}, Path(out_path))
        return out_path, True, ""
    except Exception as exc:  # noqa: BLE001
        return out_path, False, str(exc)


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _run_bounded_cache_tasks(
    pool: ProcessPoolExecutor,
    function,
    tasks: list[Any],
    *,
    max_pending: int,
) -> list[tuple[str, bool, str]]:
    iterator = iter(tasks)
    pending = set()
    results: list[tuple[str, bool, str]] = []

    def fill() -> None:
        while len(pending) < max_pending:
            try:
                task = next(iterator)
            except StopIteration:
                break
            pending.add(pool.submit(function, task))

    fill()
    while pending:
        done, still_pending = wait(pending, return_when=FIRST_COMPLETED)
        pending = set(still_pending)
        results.extend(future.result() for future in done)
        fill()
    return results


def _compact_image_source_fingerprint(image_paths: list[str]) -> str:
    digest = hashlib.sha256()
    for image_path in image_paths:
        stat = Path(image_path).stat()
        digest.update(image_path.encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def _compact_cache_files_ready(
    cache_dir: Path,
    *,
    image_shards: int,
    transition_shards: int,
) -> bool:
    if not (cache_dir / "image_index.json").is_file():
        return False
    return all(
        (cache_dir / "images" / f"shard_{index:05d}.pt").is_file()
        for index in range(image_shards)
    ) and all(
        (cache_dir / "transitions" / f"shard_{index:05d}.pt").is_file()
        for index in range(transition_shards)
    )


def build_compact_transition_preprocess_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    model_path: Path,
    processor: AutoProcessor,
    max_length: int,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_records: int = -1,
    success_only: bool = False,
    preprocess_workers: int = 4,
    force: bool = False,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
    image_dtype: str = "bfloat16",
    image_shard_size: int = 128,
    transition_shard_size: int = 256,
) -> None:
    """Build a deduplicated cache: each image once, token prefixes in mmap shards."""

    _cache_image_dtype(image_dtype)
    image_shard_size = max(1, int(image_shard_size))
    transition_shard_size = max(1, int(transition_shard_size))
    samples = TransitionJsonlDataset(
        jsonl_path,
        max_records=max_records,
        success_only=success_only,
        value_gamma=value_gamma,
    ).samples

    unique_image_paths: list[str] = []
    path_to_image_index: dict[str, int] = {}
    cumulative_image_refs = 0
    for sample in samples:
        cumulative_image_refs += len(sample.prefix_image_paths)
        # Include the post-action observation even for terminal transitions.
        # It is not present in the final Qwen prefix, but decoded-DINO SFT2
        # supervises every transition against this exact next RGB.
        source_paths = [*sample.prefix_image_paths, sample.next_image_path]
        for raw_path in source_paths:
            image_path = str(Path(raw_path).resolve())
            if image_path not in path_to_image_index:
                path_to_image_index[image_path] = len(unique_image_paths)
                unique_image_paths.append(image_path)
    image_source_fingerprint = _compact_image_source_fingerprint(unique_image_paths)
    base_fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        value_gamma=value_gamma,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype=image_dtype,
        processor_source=str(model_path.resolve()),
    )
    fingerprint = hashlib.sha256(
        (
            f"{base_fingerprint}|{image_source_fingerprint}|"
            f"image_shard={image_shard_size}|transition_shard={transition_shard_size}"
        ).encode()
    ).hexdigest()[:16]
    image_chunks = _chunked(unique_image_paths, image_shard_size)
    transition_count = len(samples)
    image_shard_count = len(image_chunks)
    transition_shard_count = math.ceil(transition_count / transition_shard_size) if transition_count else 0

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    build_state_path = cache_dir / "build_state.json"
    if manifest_path.is_file() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"compact cache fingerprint mismatch at {cache_dir}; use a new directory or --force-rebuild-cache"
            )
        if (
            manifest.get("format") == COMPACT_CACHE_FORMAT
            and int(manifest.get("count", -1)) == transition_count
            and int(manifest.get("image_shard_size", -1)) == image_shard_size
            and int(manifest.get("transition_shard_size", -1)) == transition_shard_size
            and _compact_cache_files_ready(
                cache_dir,
                image_shards=image_shard_count,
                transition_shards=transition_shard_count,
            )
        ):
            build_state_path.unlink(missing_ok=True)
            if is_main():
                print(
                    json.dumps(
                        {
                            "preprocess_cache": "hit_compact",
                            "dir": str(cache_dir),
                            "transitions": transition_count,
                            "unique_images": len(unique_image_paths),
                        }
                    )
                )
            return
    if force:
        shutil.rmtree(cache_dir / "images", ignore_errors=True)
        shutil.rmtree(cache_dir / "transitions", ignore_errors=True)
        for path in (cache_dir / "image_index.json", manifest_path, build_state_path):
            path.unlink(missing_ok=True)

    expected_build_state = {
        "format": COMPACT_CACHE_FORMAT,
        "fingerprint": fingerprint,
        "count": transition_count,
        "unique_images": len(unique_image_paths),
        "image_shard_size": image_shard_size,
        "transition_shard_size": transition_shard_size,
    }
    if build_state_path.is_file():
        build_state = json.loads(build_state_path.read_text(encoding="utf-8"))
        if build_state != expected_build_state:
            raise RuntimeError(
                f"partial compact cache fingerprint mismatch at {cache_dir}; "
                "use a new directory or --force-rebuild-cache"
            )
    else:
        stale_artifacts = (
            any((cache_dir / "images").glob("*.pt"))
            or any((cache_dir / "transitions").glob("*.pt"))
            or (cache_dir / "image_index.json").exists()
        )
        if stale_artifacts:
            raise RuntimeError(
                f"compact cache has untracked partial files at {cache_dir}; "
                "use --force-rebuild-cache once"
            )
        state_tmp = build_state_path.with_suffix(".json.tmp")
        state_tmp.write_text(json.dumps(expected_build_state, indent=2), encoding="utf-8")
        os.replace(state_tmp, build_state_path)

    workers = max(1, int(preprocess_workers))
    image_tasks = [
        (chunk, str(cache_dir / "images" / f"shard_{index:05d}.pt"), image_dtype)
        for index, chunk in enumerate(image_chunks)
        if not (cache_dir / "images" / f"shard_{index:05d}.pt").is_file()
    ]
    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build_compact_images",
                    "dir": str(cache_dir),
                    "transitions": transition_count,
                    "cumulative_image_refs": cumulative_image_refs,
                    "unique_images": len(unique_image_paths),
                    "image_reuse_factor": cumulative_image_refs / max(len(unique_image_paths), 1),
                    "image_dtype": image_dtype,
                    "image_shards": image_shard_count,
                    "to_build": len(image_tasks),
                    "workers": workers,
                }
            )
        )
    if image_tasks:
        failures: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
            ),
        ) as pool:
            results = _run_bounded_cache_tasks(
                pool,
                _cache_one_image_shard,
                image_tasks,
                max_pending=workers * 2,
            )
            for shard, ok, error in results:
                if not ok:
                    failures.append(f"{shard}: {error}")
        if failures:
            raise RuntimeError(f"compact image cache failed for {len(failures)} shards; first={failures[0]}")

    image_locations: list[dict[str, Any]] = []
    image_grids: list[list[int]] = []
    for shard_index, paths in enumerate(image_chunks):
        shard_path = cache_dir / "images" / f"shard_{shard_index:05d}.pt"
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:  # pragma: no cover
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        grids = shard["image_grid_thw"]
        if grids.shape[0] != len(paths):
            raise ValueError(f"image shard/index mismatch: {shard_path}")
        for local_index, (path, grid) in enumerate(zip(paths, grids, strict=True)):
            image_locations.append(
                {
                    "path": path,
                    "shard": shard_index,
                    "index": local_index,
                    "grid_thw": [int(value) for value in grid.tolist()],
                }
            )
            image_grids.append([int(value) for value in grid.tolist()])
    image_index_path = cache_dir / "image_index.json"
    image_index_tmp = image_index_path.with_suffix(".json.tmp")
    image_index_tmp.write_text(
        json.dumps({"format": COMPACT_CACHE_FORMAT, "images": image_locations}),
        encoding="utf-8",
    )
    os.replace(image_index_tmp, image_index_path)

    transition_chunks = _chunked(samples, transition_shard_size)
    transition_tasks = [
        (chunk, str(cache_dir / "transitions" / f"shard_{index:05d}.pt"))
        for index, chunk in enumerate(transition_chunks)
        if not (cache_dir / "transitions" / f"shard_{index:05d}.pt").is_file()
    ]
    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build_compact_tokens",
                    "dir": str(cache_dir),
                    "transition_shards": transition_shard_count,
                    "to_build": len(transition_tasks),
                    "workers": workers,
                }
            )
        )
    if transition_tasks:
        failures = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_compact_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
                path_to_image_index,
                image_grids,
            ),
        ) as pool:
            results = _run_bounded_cache_tasks(
                pool,
                _cache_one_compact_transition_shard,
                transition_tasks,
                max_pending=workers * 2,
            )
            for shard, ok, error in results:
                if not ok:
                    failures.append(f"{shard}: {error}")
        if failures:
            raise RuntimeError(f"compact token cache failed for {len(failures)} shards; first={failures[0]}")

    image_bytes = sum(path.stat().st_size for path in (cache_dir / "images").glob("*.pt"))
    transition_bytes = sum(path.stat().st_size for path in (cache_dir / "transitions").glob("*.pt"))
    manifest = {
        "fingerprint": fingerprint,
        "base_fingerprint": base_fingerprint,
        "format": COMPACT_CACHE_FORMAT,
        "count": transition_count,
        "unique_images": len(unique_image_paths),
        "cumulative_image_refs": cumulative_image_refs,
        "image_reuse_factor": cumulative_image_refs / max(len(unique_image_paths), 1),
        "image_source_fingerprint": image_source_fingerprint,
        "max_length": max_length,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
        "value_gamma": value_gamma,
        "latent_token_count": latent_token_count,
        "latent_query_mode": "inject" if mask_latent_query_labels else "generate",
        "mask_latent_query_labels": mask_latent_query_labels,
        "image_dtype": image_dtype,
        "image_shard_size": image_shard_size,
        "transition_shard_size": transition_shard_size,
        "image_shards": image_shard_count,
        "transition_shards": transition_shard_count,
        "image_bytes": image_bytes,
        "transition_bytes": transition_bytes,
        "total_bytes": image_bytes + transition_bytes + image_index_path.stat().st_size,
        "ce_mask_version": CE_MASK_VERSION,
        "transition_expansion_version": TRANSITION_EXPANSION_VERSION,
        "dir": str(cache_dir),
    }
    manifest_tmp = manifest_path.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    build_state_path.unlink(missing_ok=True)
    if is_main():
        print(json.dumps({"preprocess_cache": "done_compact", **manifest}))


def build_transition_preprocess_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    model_path: Path,
    processor: AutoProcessor,
    max_length: int,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_records: int = -1,
    success_only: bool = False,
    preprocess_workers: int = 4,
    force: bool = False,
    value_gamma: float = 1.0,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> None:
    samples = TransitionJsonlDataset(
        jsonl_path,
        max_records=max_records,
        success_only=success_only,
        value_gamma=value_gamma,
    ).samples
    fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        value_gamma=value_gamma,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
        processor_source=str(model_path.resolve()),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") == fingerprint
            and manifest.get("count") == len(samples)
            and manifest.get("max_length") == max_length
        ):
            missing = sum(
                1
                for sample in samples
                if not (cache_dir / f"{safe_cache_name(transition_sample_id(sample))}.pt").is_file()
            )
            if missing == 0:
                if is_main():
                    print(json.dumps({"preprocess_cache": "hit", "dir": str(cache_dir), "count": len(samples)}))
                return

    tasks: list[tuple[dict[str, Any], str]] = []
    for sample in samples:
        item = transition_collate_for_qwen([sample])[0]
        out_path = cache_dir / f"{safe_cache_name(item['id'])}.pt"
        if not force and out_path.is_file():
            continue
        tasks.append((item, str(out_path)))

    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build",
                    "dir": str(cache_dir),
                    "fingerprint": fingerprint,
                    "total": len(samples),
                    "to_build": len(tasks),
                    "workers": preprocess_workers,
                }
            )
        )

    if tasks:
        workers = max(1, preprocess_workers)
        failures: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
            ),
        ) as pool:
            futures = [pool.submit(_cache_one_transition, task) for task in tasks]
            for fut in as_completed(futures):
                sample_id, ok, err = fut.result()
                if not ok:
                    failures.append(f"{sample_id}: {err}")
        if failures:
            raise RuntimeError(f"preprocess cache failed for {len(failures)} samples; first={failures[0]}")

    if is_main():
        total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
        print(json.dumps({"preprocess_cache": "done", "dir": str(cache_dir), "count": len(samples), "total_bytes": total_bytes}))

    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "count": len(samples),
                "max_length": max_length,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "value_gamma": value_gamma,
                "latent_token_count": latent_token_count,
                "latent_query_mode": "inject" if mask_latent_query_labels else "generate",
                "mask_latent_query_labels": mask_latent_query_labels,
                "ce_mask_version": CE_MASK_VERSION,
                "transition_expansion_version": TRANSITION_EXPANSION_VERSION,
                "dir": str(cache_dir),
                "total_bytes": sum(path.stat().st_size for path in cache_dir.glob("*.pt")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def encode_trajectory_record(
    record: dict[str, Any],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    from nimloth.training.sft2.trajectory_once import encode_full_trajectory

    steps = expand_record_transitions(record)
    if not steps:
        raise ValueError(f"record {record.get('id')!r} produced no transitions")
    full_enc, _text = encode_full_trajectory(
        steps,
        processor,
        max_length,
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
    )
    return {
        "record_id": str(record.get("id", "")),
        "num_steps": len(steps),
        "full_enc": full_enc,
        "step_items": transition_collate_for_qwen(steps),
    }


class CachedTrajectoryDataset(Dataset):
    def __init__(self, cache_dir: Path, samples: list[TransitionSample]) -> None:
        from nimloth.training.sft2.trajectory_batching import build_record_trajectory_batches

        self.cache_dir = cache_dir
        self.samples = samples
        self._record_index_lists = build_record_trajectory_batches(samples)

    def __len__(self) -> int:
        return len(self._record_index_lists)

    def cache_path_for_record(self, record_id: str) -> Path:
        return self.cache_dir / f"{safe_cache_name(record_id)}.pt"

    def __getitem__(self, index: int) -> dict[str, Any]:
        steps = [self.samples[i] for i in self._record_index_lists[index]]
        record_id = steps[0].record_id
        cache_path = self.cache_path_for_record(record_id)
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing trajectory preprocess cache: {cache_path}")
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        return {
            "transition_samples": steps,
            "items": cached["step_items"],
            "full_enc": cached["full_enc"],
        }


def _cache_one_trajectory(task: tuple[dict[str, Any], str]) -> tuple[str, bool, str]:
    record, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_trajectory_record(
            record,
            _CACHE_PROCESSOR,
            _CACHE_MAX_LENGTH,
            latent_token_count=_CACHE_LATENT_TOKEN_COUNT,
            mask_latent_query_labels=_CACHE_MASK_LATENT_QUERY_LABELS,
        )
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return str(record.get("id", "")), True, ""
    except Exception as exc:  # noqa: BLE001
        return str(record.get("id", "")), False, str(exc)


def build_trajectory_preprocess_cache(
    *,
    jsonl_path: Path,
    cache_dir: Path,
    model_path: Path,
    processor: AutoProcessor,
    max_length: int,
    max_pixels: int,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_records: int = -1,
    success_only: bool = False,
    preprocess_workers: int = 4,
    force: bool = False,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> None:
    records = load_jsonl_records(jsonl_path, max_records=max_records)
    if success_only:
        records = [record for record in records if record.get("success", False)]
    fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=max_length,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
        vocab_size=len(processor.tokenizer),
        latent_token_count=latent_token_count,
        mask_latent_query_labels=mask_latent_query_labels,
        processor_source=str(model_path.resolve()),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not force and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint and manifest.get("count") == len(records):
            missing = sum(
                1
                for record in records
                if not (cache_dir / f"{safe_cache_name(str(record.get('id', '')))}.pt").is_file()
            )
            if missing == 0:
                if is_main():
                    print(json.dumps({"preprocess_cache": "hit", "dir": str(cache_dir), "records": len(records)}))
                return

    tasks: list[tuple[dict[str, Any], str]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        out_path = cache_dir / f"{safe_cache_name(record_id)}.pt"
        if not force and out_path.is_file():
            continue
        tasks.append((record, str(out_path)))

    if is_main():
        print(
            json.dumps(
                {
                    "preprocess_cache": "build_trajectory",
                    "dir": str(cache_dir),
                    "fingerprint": fingerprint,
                    "records": len(records),
                    "to_build": len(tasks),
                    "workers": preprocess_workers,
                }
            )
        )

    if tasks:
        workers = max(1, preprocess_workers)
        failures: list[str] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_cache_worker,
            initargs=(
                str(model_path),
                min_pixels,
                max_pixels,
                max_length,
                latent_token_count,
                mask_latent_query_labels,
            ),
        ) as pool:
            futures = [pool.submit(_cache_one_trajectory, task) for task in tasks]
            for fut in as_completed(futures):
                record_id, ok, err = fut.result()
                if not ok:
                    failures.append(f"{record_id}: {err}")
        if failures:
            raise RuntimeError(f"trajectory preprocess cache failed for {len(failures)} records; first={failures[0]}")

    total_bytes = sum(path.stat().st_size for path in cache_dir.glob("*.pt"))
    manifest_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "granularity": "trajectory",
                "count": len(records),
                "max_length": max_length,
                "max_pixels": max_pixels,
                "min_pixels": min_pixels,
                "latent_token_count": latent_token_count,
                "latent_query_mode": "inject" if mask_latent_query_labels else "generate",
                "mask_latent_query_labels": mask_latent_query_labels,
                "ce_mask_version": CE_MASK_VERSION,
                "transition_expansion_version": TRANSITION_EXPANSION_VERSION,
                "dir": str(cache_dir),
                "total_bytes": total_bytes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if is_main():
        print(json.dumps({"preprocess_cache": "done", "dir": str(cache_dir), "records": len(records), "total_bytes": total_bytes}))
