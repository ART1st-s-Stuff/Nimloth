"""dev 语义对齐脚本公共能力。"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.application.pipelines.semantic.common import build_semantic_dataset, resolve_split_manifest


def load_cfg(*, wm_override: str = "cfm_dinov2m") -> DictConfig:
    config_dir = Path("/home/jincai_guo/atst/flower/configs").resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=[f"wm={wm_override}"])


def build_dataset_from_cfg(
    *,
    cfg: DictConfig,
    split: str,
    use_encoder,
):
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    train_cfg = cfg.pipeline.train.semantic_align
    manifests_cfg = dict(dataset_cfg.get("manifests", {}))
    manifest = resolve_split_manifest(
        manifests_cfg=manifests_cfg,
        split=split,
        outputs_root=str(train_cfg.operation.outputs_root),
        dataset_name=str(dataset_cfg.name),
    )
    return build_semantic_dataset(
        manifest_path=manifest,
        wm_cfg=wm_cfg,
        dataset_cfg=dataset_cfg,
        image_encoder=use_encoder,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
    )
