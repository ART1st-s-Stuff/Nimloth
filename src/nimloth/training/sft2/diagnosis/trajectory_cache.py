"""Research-only preprocess cache for non-equivalent trajectory packing."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from nimloth.backbone.qwen25vl.transition import transition_collate_for_qwen
from nimloth.util.distributed import is_main
from nimloth.util.cache import build as cache_build
from nimloth.util.cache.schema import (
    CE_MASK_VERSION,
    DEFAULT_MIN_PIXELS,
    TRANSITION_EXPANSION_VERSION,
    cache_fingerprint,
    safe_cache_name,
)
from nimloth.rollout.transitions import (
    TransitionSample,
    expand_record_transitions,
    load_jsonl_records,
)

def encode_trajectory_record(
    record: dict[str, Any],
    processor: AutoProcessor,
    max_length: int,
    *,
    latent_token_count: int = 1,
    mask_latent_query_labels: bool = True,
) -> dict[str, Any]:
    from nimloth.training.sft2.diagnosis.trajectory_once import encode_full_trajectory

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
        from nimloth.training.sft2.diagnosis.trajectory_batching import build_record_trajectory_batches

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
        assert cache_build._CACHE_PROCESSOR is not None
        encoded = encode_trajectory_record(
            record,
            cache_build._CACHE_PROCESSOR,
            cache_build._CACHE_MAX_LENGTH,
            latent_token_count=cache_build._CACHE_LATENT_TOKEN_COUNT,
            mask_latent_query_labels=cache_build._CACHE_MASK_LATENT_QUERY_LABELS,
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
            initializer=cache_build._init_cache_worker,
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
