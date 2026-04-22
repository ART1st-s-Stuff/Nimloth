"""WM latent 预编码与缓存复用的统一入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.data.dataset import WMDataset
from src.wm.encoders import WMImageEncoder


def resolve_manifest_path(manifest_path: str) -> Path:
    candidate = Path(manifest_path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if len(parts) >= 3 and parts[-2] == "latest":
        group_dir = Path(*parts[:-2])
        meta_path = group_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            latest = metadata.get("latest")
            if isinstance(latest, str):
                latest_path = group_dir / latest / parts[-1]
                if latest_path.exists():
                    return latest_path
    return candidate


def build_latent_cache_path(manifest_path: Path, wm_name: str) -> Path:
    stem = manifest_path.stem
    return manifest_path.parent / f"{stem}.latents.{wm_name}.pt"


def build_wm_dataset_with_cache(
    *,
    manifest_path: Path,
    wm_name: str,
    latent_dim: int,
    action_dim: int,
    history_len: int,
    rollout_steps: int,
    image_encoder: WMImageEncoder | None,
    temporal_stride: int | tuple[int, int] = 1,
    encoder_num_workers: int,
    encoder_batch_size: int,
    expected_num_patches: int = 0,
    expected_token_dim: int = 0,
    on_latent_progress: Callable[[int, int], None] | None = None,
) -> tuple[WMDataset, Path]:
    latent_cache_path = build_latent_cache_path(manifest_path, wm_name)
    dataset = WMDataset(
        manifest_path=str(manifest_path),
        latent_dim=latent_dim,
        action_dim=action_dim,
        history_len=history_len,
        rollout_steps=rollout_steps,
        temporal_stride=temporal_stride,
        image_encoder=image_encoder,
        latent_cache_path=str(latent_cache_path),
        encoder_num_workers=encoder_num_workers,
        encoder_batch_size=encoder_batch_size,
        expected_num_patches=expected_num_patches,
        expected_token_dim=expected_token_dim,
        on_latent_progress=on_latent_progress,
    )
    return dataset, latent_cache_path

