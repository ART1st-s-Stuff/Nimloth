"""语义对齐公共构建逻辑。"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.latent_cache import infer_latent_cache_path_from_manifest
from src.train.manifest_resolver import resolve_manifest_for_split
from src.vlm.qwen_adapter import QwenVLMAdapter


def collate_semantic_batch(batch: list[dict]) -> dict:
    return {
        "z_t": torch.stack([item["z_t"] for item in batch], dim=0),
        "z_t_pos": torch.stack([item["z_t_pos"] for item in batch], dim=0),
        "z_t_neg": torch.stack([item["z_t_neg"] for item in batch], dim=0),
        "image_path": [str(item["image_path"]) for item in batch],
        "pos_image_path": [str(item["pos_image_path"]) for item in batch],
        "task_text": [str(item["task_text"]) for item in batch],
        "env_context": [str(item["env_context"]) for item in batch],
    }


def resolve_split_manifest(
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


def build_semantic_dataset(
    *,
    manifest_path: Path,
    wm_cfg: DictConfig,
    dataset_cfg: DictConfig,
    image_encoder,
    positive_k: int,
    negative_gap: int,
    enable_cot_target: bool = False,
) -> SemanticAlignDataset:
    resolved_cache = infer_latent_cache_path_from_manifest(str(manifest_path), str(wm_cfg.name))
    return SemanticAlignDataset(
        manifest_path=str(manifest_path),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=image_encoder,
        positive_k=positive_k,
        negative_gap=negative_gap,
        enable_cot_target=enable_cot_target,
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )


def build_semantic_loader(
    *,
    dataset: SemanticAlignDataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_semantic_batch,
    )


def build_qwen_vlm_adapter(
    *,
    vlm_cfg: DictConfig,
    train_cfg: DictConfig,
    latent_dim: int,
) -> QwenVLMAdapter:
    model_name = str(OmegaConf.select(vlm_cfg, "model.hf_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct"))
    max_new_tokens = int(OmegaConf.select(vlm_cfg, "model.max_new_tokens", default=128))
    return QwenVLMAdapter(
        model_name=model_name,
        latent_dim=latent_dim,
        enabled=bool(vlm_cfg.get("enabled", False) and train_cfg.use_vlm_for_st),
        fallback_enabled=bool(vlm_cfg.get("fallback_enabled", True)),
        max_new_tokens=max_new_tokens,
    )
