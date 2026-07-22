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
    LEGACY_CACHE_FORMAT,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    build_compact_transition_preprocess_cache,
    build_transition_preprocess_cache,
    cache_fingerprint,
)
from nimloth.training.sft2.data.samplers import (
    TrajectoryWindowBatchSampler,
)
from nimloth.rollout.transitions import TransitionJsonlDataset, TransitionSample


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_samples: list[TransitionSample]
    val_samples: list[TransitionSample]
    train_batch_sampler: TrajectoryWindowBatchSampler
    val_batch_sampler: TrajectoryWindowBatchSampler


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
    expected_fingerprint = cache_fingerprint(
        jsonl_path,
        max_length=config.max_length,
        max_pixels=config.max_pixels,
        min_pixels=3136,
        vocab_size=len(processor.tokenizer),
        value_gamma=config.value_gamma,
        latent_token_count=config.latent_token_count,
        mask_latent_query_labels=config.mask_latent_query_labels,
        cache_format=COMPACT_CACHE_FORMAT if compact else LEGACY_CACHE_FORMAT,
        image_dtype=config.preprocess_cache_image_dtype if compact else "float32",
        processor_source=str(Path(config.model).resolve()),
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
    )
    val_dataset = CachedTransitionDataset(
        val_cache_dir,
        val_samples,
        max_open_shards=config.preprocess_cache_shard_lru,
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

    if config.batch_mode not in {
        "random",
        "trajectory",
        "trajectory_image_budget",
    }:
        raise ValueError(f"unsupported SFT2 batch mode: {config.batch_mode!r}")
    image_budget = (
        int(config.max_images_per_batch)
        if config.batch_mode == "trajectory_image_budget"
        else None
    )
    row_budget = (
        int(config.max_steps_per_trajectory)
        if config.batch_mode == "trajectory_image_budget"
        else None
    )
    train_batch_sampler = TrajectoryWindowBatchSampler(
        train_samples,
        history_size=config.history_size,
        batch_size=config.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        shuffle_windows=config.batch_mode == "random",
        seed=config.seed,
        max_images_per_batch=image_budget,
        max_transition_rows_per_batch=row_budget,
        pad_to_equal_batches=True,
    )
    val_batch_sampler = TrajectoryWindowBatchSampler(
        val_samples,
        history_size=config.history_size,
        batch_size=config.batch_size,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
        max_images_per_batch=image_budget,
        max_transition_rows_per_batch=row_budget,
        pad_to_equal_batches=False,
    )
    if train_batch_sampler.window_count == 0:
        raise ValueError(
            "SFT2 training data has no complete LeWM windows: "
            f"history_size={config.history_size}"
        )
    if val_batch_sampler.window_count == 0:
        raise ValueError(
            "SFT2 validation data has no complete LeWM windows: "
            f"history_size={config.history_size}"
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
