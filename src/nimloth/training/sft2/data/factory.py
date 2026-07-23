"""Build SFT2 datasets, preprocess-cache adapters, and data loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch.utils.data import DataLoader

from nimloth.training.sft2.batch import SFT2BatchBuilder
from nimloth.util.distributed import is_main
from nimloth.util.cache import (
    COMPACT_CACHE_FORMAT,
    COMPACT_CACHE_FORMAT_V1,
    SUPPORTED_COMPACT_CACHE_FORMATS,
    LEGACY_CACHE_FORMAT,
    LEGACY_TRANSITION_EXPANSION_VERSION,
    TRANSITION_EXPANSION_VERSION,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    build_compact_transition_preprocess_cache,
    build_transition_preprocess_cache,
    cache_fingerprint,
)
from nimloth.training.sft2.data.samplers import (
    OnlineHistoryBatchSampler,
)
from nimloth.rollout.transitions import TransitionJsonlDataset, TransitionSample


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_samples: list[TransitionSample]
    val_samples: list[TransitionSample]
    train_batch_sampler: OnlineHistoryBatchSampler
    val_batch_sampler: OnlineHistoryBatchSampler


def _dataloader_workers(config: Any) -> int:
    if config.dataloader_workers >= 0:
        return int(config.dataloader_workers)
    return 4 if config.preprocess_cache_dir is not None else 0


def _load_transition_samples(config: Any) -> tuple[list[TransitionSample], list[TransitionSample]]:
    train_samples = TransitionJsonlDataset(
        config.train_jsonl,
        max_records=config.max_train_records,
        success_only=config.success_only,
        value_gamma=config.value_gamma,
    ).samples
    val_samples = TransitionJsonlDataset(
        config.val_jsonl,
        max_records=config.max_val_records,
        value_gamma=config.value_gamma,
    ).samples
    return train_samples, val_samples


def _verify_cache_manifest(
    *,
    cache_dir: Path,
    jsonl_path: Path,
    expected_count: int,
    config: Any,
    processor,
) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        mode = "required prebuilt" if config.require_prebuilt_cache else "built"
        raise FileNotFoundError(f"{mode} preprocess cache missing manifest: {cache_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compact = config.preprocess_cache_format == "compact"
    actual_format = str(manifest.get("format", ""))
    if compact and actual_format not in SUPPORTED_COMPACT_CACHE_FORMATS:
        raise ValueError(
            f"unsupported compact preprocess cache format: {actual_format!r}"
        )
    expected_fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=config.max_length,
        max_pixels=config.max_pixels,
        min_pixels=3136,
        vocab_size=len(processor.tokenizer),
        value_gamma=config.value_gamma,
        latent_token_count=config.latent_token_count,
        mask_latent_query_labels=config.mask_latent_query_labels,
        cache_format=actual_format if compact else LEGACY_CACHE_FORMAT,
        image_dtype=config.preprocess_cache_image_dtype if compact else "float32",
        processor_source=str(Path(config.model).resolve()),
        transition_expansion_version=(
            LEGACY_TRANSITION_EXPANSION_VERSION
            if actual_format == COMPACT_CACHE_FORMAT_V1
            else TRANSITION_EXPANSION_VERSION
        ),
    )
    actual_fingerprint = manifest.get("base_fingerprint") if compact else manifest.get("fingerprint")
    if actual_fingerprint != expected_fingerprint or int(manifest.get("count", -1)) != expected_count:
        raise ValueError(
            f"preprocess cache fingerprint/count mismatch: {cache_dir}; "
            "rebuild the CPU cache for this model, dataset, and config"
        )


def _build_or_open_cached_datasets(
    config: Any,
    batch_builder: SFT2BatchBuilder,
    train_samples: list[TransitionSample],
    val_samples: list[TransitionSample],
):
    processor = batch_builder.processor
    cache_root = Path(config.preprocess_cache_dir)
    train_cache_dir = cache_root / "train"
    val_cache_dir = cache_root / "val"
    compact = config.preprocess_cache_format == "compact"
    builder = build_compact_transition_preprocess_cache if compact else build_transition_preprocess_cache

    build_kwargs = {
        "model_path": config.model,
        "processor": processor,
        "max_length": config.max_length,
        "max_pixels": config.max_pixels,
        "min_pixels": 3136,
        "preprocess_workers": config.preprocess_workers,
        "force": config.force_rebuild_cache,
        "value_gamma": config.value_gamma,
        "latent_token_count": config.latent_token_count,
        "mask_latent_query_labels": config.mask_latent_query_labels,
    }
    compact_kwargs = (
        {
            "image_dtype": config.preprocess_cache_image_dtype,
            "image_shard_size": config.preprocess_cache_image_shard_size,
            "transition_shard_size": config.preprocess_cache_transition_shard_size,
        }
        if compact
        else {}
    )
    if is_main() and not config.require_prebuilt_cache:
        builder(
            jsonl_path=config.train_jsonl,
            cache_dir=train_cache_dir,
            max_records=config.max_train_records,
            success_only=config.success_only,
            **build_kwargs,
            **compact_kwargs,
        )
        builder(
            jsonl_path=config.val_jsonl,
            cache_dir=val_cache_dir,
            max_records=config.max_val_records,
            success_only=False,
            **build_kwargs,
            **compact_kwargs,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    _verify_cache_manifest(
        cache_dir=train_cache_dir,
        jsonl_path=config.train_jsonl,
        expected_count=len(train_samples),
        config=config,
        processor=processor,
    )
    _verify_cache_manifest(
        cache_dir=val_cache_dir,
        jsonl_path=config.val_jsonl,
        expected_count=len(val_samples),
        config=config,
        processor=processor,
    )

    train_dataset = CachedTransitionDataset(
        train_cache_dir,
        train_samples,
        max_open_shards=config.preprocess_cache_shard_lru,
        processor=processor,
        max_length=config.max_length,
        latent_token_count=config.latent_token_count,
        mask_latent_query_labels=config.mask_latent_query_labels,
    )
    val_dataset = CachedTransitionDataset(
        val_cache_dir,
        val_samples,
        max_open_shards=config.preprocess_cache_shard_lru,
        processor=processor,
        max_length=config.max_length,
        latent_token_count=config.latent_token_count,
        mask_latent_query_labels=config.mask_latent_query_labels,
    )
    if train_dataset.is_compact != val_dataset.is_compact:
        raise ValueError("train/val preprocess cache formats differ")

    if train_dataset.is_compact:
        collate_train = CompactCachedTransitionCollator(
            train_cache_dir,
            max_open_shards=config.preprocess_cache_shard_lru,
        )
        collate_val = CompactCachedTransitionCollator(
            val_cache_dir,
            max_open_shards=config.preprocess_cache_shard_lru,
        )
    else:
        collate_train = batch_builder.collate_cached_transition_batch
        collate_val = collate_train
    return train_dataset, val_dataset, collate_train, collate_val


def build_data_bundle(
    config: Any,
    batch_builder: SFT2BatchBuilder,
    *,
    rank: int,
    world_size: int,
) -> DataBundle:
    """Construct the complete SFT2 data plane from one validated config."""

    train_samples, val_samples = _load_transition_samples(config)
    if config.preprocess_cache_dir is None:
        train_dataset = TransitionJsonlDataset.from_samples(train_samples)
        val_dataset = TransitionJsonlDataset.from_samples(val_samples)
        train_collate = batch_builder.collate_transition_samples
        val_collate = batch_builder.collate_transition_samples
    else:
        train_dataset, val_dataset, train_collate, val_collate = _build_or_open_cached_datasets(
            config,
            batch_builder,
            train_samples,
            val_samples,
        )

    workers = _dataloader_workers(config)
    loader_kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=max(1, int(config.dataloader_prefetch_factor)),
        )

    if config.batch_mode != "trajectory_online_cache":
        raise ValueError(
            "SFT2 requires batch_mode='trajectory_online_cache' so detached "
            "history states are written before use"
        )
    train_batch_sampler = OnlineHistoryBatchSampler(
        train_samples,
        history_size=config.history_size,
        batch_size=config.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        pad_to_equal_batches=True,
    )
    val_batch_sampler = OnlineHistoryBatchSampler(
        val_samples,
        history_size=config.history_size,
        batch_size=config.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
        pad_to_equal_batches=False,
    )
    if train_batch_sampler.window_count == 0:
        raise ValueError(
            "SFT2 training data has no transition with a real next state: "
            f"history_size={config.history_size}"
        )
    if val_batch_sampler.window_count == 0:
        raise ValueError(
            "SFT2 validation data has no transition with a real next state: "
            f"history_size={config.history_size}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "online_history_cache_sampler": "trajectory_lane_v1",
                    "train_padding_batches_rank0": train_batch_sampler.padding_batch_count,
                }
            )
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=train_collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        collate_fn=val_collate,
        **loader_kwargs,
    )
    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_samples=train_samples,
        val_samples=val_samples,
        train_batch_sampler=train_batch_sampler,
        val_batch_sampler=val_batch_sampler,
    )
