"""Disk preprocess cache for SFT2 transition prefixes."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor

from nimloth.latent import add_special_tokens
from nimloth.training.common.dist import is_main
from nimloth.training.common.qwen_batch import encode_qwen_item
from nimloth.wm.collate import messages_with_image_paths, transition_collate_for_qwen
from nimloth.wm.dataset import TransitionJsonlDataset, TransitionSample, expand_record_transitions, load_jsonl_records

CE_MASK_VERSION = "last_assistant_span_v1"
TRANSITION_EXPANSION_VERSION = "wm_expand_v1"
DEFAULT_MIN_PIXELS = 3136


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
            CE_MASK_VERSION,
            TRANSITION_EXPANSION_VERSION,
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode_transition_item(
    item: dict[str, Any],
    processor: AutoProcessor,
    max_length: int,
) -> dict[str, Any]:
    current_enc = encode_qwen_item(item["messages"], processor, max_length, include_labels=True)
    next_enc = None
    if item.get("next_messages"):
        next_enc = encode_qwen_item(item["next_messages"], processor, max_length, include_labels=False)
    return {
        "id": item["id"],
        "action_index": item["action_index"],
        "action_value_target": item["action_value_target"],
        "success": item["success"],
        "current_enc": current_enc,
        "next_enc": next_enc,
    }


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
                "action_index": entry["action_index"],
                "action_value_target": entry["action_value_target"],
                "success": entry["success"],
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
    }


def unpack_transition_batch(
    batch,
    processor,
    max_length: int,
    *,
    pad_token_id: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, torch.Tensor] | None] | None,
]:
    from nimloth.training.common.qwen_batch import build_qwen_batch

    if isinstance(batch, dict) and "current_enc" in batch:
        items = batch["items"]
        enc = batch["current_enc"]
        next_rows = batch.get("next_enc_rows")
        return items, enc, next_rows
    items = batch
    enc = build_qwen_batch(items, processor, max_length)
    return items, enc, None


class CachedTransitionDataset(Dataset):
    def __init__(self, cache_dir: Path, samples: list[TransitionSample]):
        self.cache_dir = cache_dir
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def cache_path_for_sample(self, sample: TransitionSample) -> Path:
        return self.cache_dir / f"{safe_cache_name(transition_sample_id(sample))}.pt"

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cache_path = self.cache_path_for_sample(sample)
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing preprocess cache: {cache_path}")
        entry = torch.load(cache_path, map_location="cpu", weights_only=True)
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


def _init_cache_worker(model_path: str, min_pixels: int, max_pixels: int, max_length: int) -> None:
    global _CACHE_PROCESSOR, _CACHE_MAX_LENGTH
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.image_processor.min_pixels = min_pixels
    processor.image_processor.max_pixels = max_pixels
    add_special_tokens(processor.tokenizer)
    _CACHE_PROCESSOR = processor
    _CACHE_MAX_LENGTH = max_length


def _cache_one_transition(task: tuple[dict[str, Any], str]) -> tuple[str, bool, str]:
    item, out_path = task
    try:
        assert _CACHE_PROCESSOR is not None
        encoded = encode_transition_item(item, _CACHE_PROCESSOR, _CACHE_MAX_LENGTH)
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(encoded, path)
        return item["id"], True, ""
    except Exception as exc:  # noqa: BLE001
        return item["id"], False, str(exc)


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
            initargs=(str(model_path), min_pixels, max_pixels, max_length),
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
) -> dict[str, Any]:
    from nimloth.training.sft2.trajectory_once import encode_full_trajectory

    steps = expand_record_transitions(record)
    if not steps:
        raise ValueError(f"record {record.get('id')!r} produced no transitions")
    full_enc, _text = encode_full_trajectory(steps, processor, max_length)
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
        encoded = encode_trajectory_record(record, _CACHE_PROCESSOR, _CACHE_MAX_LENGTH)
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
            initargs=(str(model_path), min_pixels, max_pixels, max_length),
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
