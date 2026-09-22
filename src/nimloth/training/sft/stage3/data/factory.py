"""Build SFT2 datasets, preprocess-cache adapters, and data loaders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch.utils.data import DataLoader

from nimloth.training.sft.stage3.batch import SFT2BatchBuilder
from nimloth.util.distributed import is_main
from nimloth.util.cache import (
    COMPACT_CACHE_FORMAT,
    TRANSITION_EXPANSION_VERSION,
    CachedTransitionDataset,
    CompactCachedTransitionCollator,
    build_compact_transition_preprocess_cache,
    cache_fingerprint,
)
from nimloth.training.sft.stage3.data.samplers import TrajectoryBatchSampler
from nimloth.training.sft.stage3.data.trajectory import TrajectoryDataset, TrajectoryCollator
from nimloth.rollout.transitions import TransitionJsonlDataset, TransitionSample


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    train_samples: list[TransitionSample]
    val_samples: list[TransitionSample]
    train_batch_sampler: TrajectoryBatchSampler
    val_batch_sampler: TrajectoryBatchSampler


def _dataloader_workers(config: Any) -> int:
    if config.dataloader_workers >= 0:
        return int(config.dataloader_workers)
    return 4 if config.preprocess_cache_dir is not None else 0


def _cache_processor_source(config: Any) -> Path:
    source = getattr(config, "preprocess_cache_processor_source", None)
    if source is None:
        return Path(config.model)
    if not config.require_prebuilt_cache:
        raise ValueError(
            "preprocess_cache_processor_source is only valid for a required "
            "prebuilt cache"
        )
    return Path(source)


def _load_transition_samples(
    config: Any,
) -> tuple[list[TransitionSample], list[TransitionSample]]:
    if config.success_only:
        raise ValueError("Stage 3 requires all trajectories; only LM uses success masking")
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
    allow_prefix_subset: bool,
    config: Any,
    processor,
) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        mode = "required prebuilt" if config.require_prebuilt_cache else "built"
        raise FileNotFoundError(
            f"{mode} preprocess cache missing manifest: {cache_dir}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_format = str(manifest.get("format", ""))
    if actual_format != COMPACT_CACHE_FORMAT:
        raise ValueError(
            "SFT2 requires the current compact preprocess cache format "
            f"{COMPACT_CACHE_FORMAT!r}; got {actual_format!r}. Rebuild the cache "
            "from the migrated trajectory JSONL"
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
        cache_format=COMPACT_CACHE_FORMAT,
        image_dtype=config.preprocess_cache_image_dtype,
        processor_source=str(_cache_processor_source(config).resolve()),
        transition_expansion_version=TRANSITION_EXPANSION_VERSION,
    )
    actual_fingerprint = manifest.get("base_fingerprint")
    actual_count = int(manifest.get("count", -1))
    count_matches = actual_count == expected_count or (
        allow_prefix_subset and actual_count > expected_count
    )
    if actual_fingerprint != expected_fingerprint or not count_matches:
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
    cache_processor_source = _cache_processor_source(config)
    cache_root = Path(config.preprocess_cache_dir)
    train_cache_dir = cache_root / "train"
    val_cache_dir = cache_root / "val"
    build_kwargs = {
        "model_path": cache_processor_source,
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
    compact_kwargs = {
        "image_dtype": config.preprocess_cache_image_dtype,
        "image_shard_size": config.preprocess_cache_image_shard_size,
        "transition_shard_size": config.preprocess_cache_transition_shard_size,
    }
    reuse_root = getattr(config, "preprocess_cache_reuse_image_root", None)
    reuse_processor_source = getattr(
        config,
        "preprocess_cache_reuse_processor_source",
        None,
    )
    if is_main() and not config.require_prebuilt_cache:
        build_compact_transition_preprocess_cache(
            jsonl_path=config.train_jsonl,
            cache_dir=train_cache_dir,
            max_records=config.max_train_records,
            success_only=config.success_only,
            reuse_image_cache=(Path(reuse_root) / "train" if reuse_root is not None else None),
            reuse_image_processor_source=reuse_processor_source,
            **build_kwargs,
            **compact_kwargs,
        )
        build_compact_transition_preprocess_cache(
            jsonl_path=config.val_jsonl,
            cache_dir=val_cache_dir,
            max_records=config.max_val_records,
            success_only=False,
            reuse_image_cache=(Path(reuse_root) / "val" if reuse_root is not None else None),
            reuse_image_processor_source=reuse_processor_source,
            **build_kwargs,
            **compact_kwargs,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    _verify_cache_manifest(
        cache_dir=train_cache_dir,
        jsonl_path=config.train_jsonl,
        expected_count=len(train_samples),
        # max_records is applied before expansion, so an unfiltered truncated
        # dataset is exactly a transition prefix of the full cache.
        allow_prefix_subset=(
            config.max_train_records > 0 and not config.success_only
        ),
        config=config,
        processor=processor,
    )
    _verify_cache_manifest(
        cache_dir=val_cache_dir,
        jsonl_path=config.val_jsonl,
        expected_count=len(val_samples),
        allow_prefix_subset=config.max_val_records > 0,
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
    collate_train = CompactCachedTransitionCollator(
        train_cache_dir,
        max_open_shards=config.preprocess_cache_shard_lru,
    )
    collate_val = CompactCachedTransitionCollator(
        val_cache_dir,
        max_open_shards=config.preprocess_cache_shard_lru,
    )
    return train_dataset, val_dataset, collate_train, collate_val


def build_data_bundle(
    config: Any,
    batch_builder: SFT2BatchBuilder,
    *,
    rank: int,
    world_size: int,
) -> DataBundle:
    """Construct the complete SFT2 data plane from one validated config."""

    if int(config.history_size) != 1:
        raise ValueError("Stage3 requires history_size=1")
    train_samples, val_samples = _load_transition_samples(config)
    horizon = int(config.prediction_horizon)
    train_cache = val_cache = train_materializer = val_materializer = None
    if config.preprocess_cache_dir is not None:
        train_cache, val_cache, train_materializer, val_materializer = _build_or_open_cached_datasets(
            config, batch_builder, train_samples, val_samples)
    train_dataset = TrajectoryDataset(train_samples, prediction_horizon=horizon, cached=train_cache)
    val_dataset = TrajectoryDataset(val_samples, prediction_horizon=horizon, cached=val_cache)
    train_collate = TrajectoryCollator(batch_builder.input_builder, prediction_horizon=horizon,
                                      cache_collator=train_materializer)
    val_collate = TrajectoryCollator(batch_builder.input_builder, prediction_horizon=horizon,
                                    cache_collator=val_materializer)
    workers = _dataloader_workers(config)
    loader_kwargs: dict[str, Any] = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=max(1, int(config.dataloader_prefetch_factor)))
    train_batch_sampler = TrajectoryBatchSampler(train_dataset, batch_size=config.batch_size,
        num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed,
        pad_to_equal_batches=True)
    val_batch_sampler = TrajectoryBatchSampler(val_dataset, batch_size=config.batch_size,
        num_replicas=world_size, rank=rank, shuffle=False, seed=config.seed,
        pad_to_equal_batches=True)
    if is_main():
        print(json.dumps({"stage3_sampler": "complete_trajectory_v1",
            "prediction_horizon": horizon,
            "train_trajectories": len(train_dataset), "train_windows": train_dataset.window_count,
            "val_trajectories": len(val_dataset), "val_windows": val_dataset.window_count,
            "train_omitted_short_trajectories": train_dataset.omitted_short_trajectories,
            "val_omitted_short_trajectories": val_dataset.omitted_short_trajectories,
            "train_padding_batches_rank0": train_batch_sampler.padding_batch_count}))
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler,
                             collate_fn=train_collate, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler,
                           collate_fn=val_collate, **loader_kwargs)
    return DataBundle(train_loader, val_loader, train_samples, val_samples,
                      train_batch_sampler, val_batch_sampler)
