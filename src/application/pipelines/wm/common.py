"""WM 训练/评估公共构建逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from src.train.latent_cache import build_wm_dataset_with_cache
from src.train.manifest_resolver import resolve_manifest_for_split
from src.wm.encoder import build_wm_image_encoder


def resolve_wm_manifest(
    *,
    manifests_cfg: dict,
    split: str,
    outputs_root: str,
    dataset_name: str,
) -> Path:
    return resolve_manifest_for_split(
        manifests_cfg=manifests_cfg,
        split=split,
        outputs_root=outputs_root,
        dataset_name=dataset_name,
    )


def build_wm_split_loader(
    *,
    split: str,
    manifest_path: Path,
    wm_cfg,
    dataset_cfg,
    train_cfg,
    temporal_stride: int | tuple[int, int],
    encoder_for_split: Any = None,
) -> DataLoader | None:
    if not manifest_path.exists():
        return None
    encoder = encoder_for_split if encoder_for_split is not None else build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset, _ = build_wm_dataset_with_cache(
        run_dir=manifest_path,
        wm_name=str(wm_cfg.name),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        temporal_stride=temporal_stride,
        image_encoder=encoder,
        encoder_num_workers=int(train_cfg.encoder_num_workers),
        encoder_batch_size=int(train_cfg.encoder_batch_size),
        expected_num_patches=int(getattr(wm_cfg.encoder, "num_patches", 0)),
        expected_token_dim=(
            int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 1))
            if int(getattr(wm_cfg.encoder, "num_patches", 0)) > 0
            else 0
        ),
    )
    dataset.disable_encoder_after_warmup()
    num_workers = int(train_cfg.num_workers)
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=(split == "train"),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
