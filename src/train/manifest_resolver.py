"""训练/评估入口的 manifest 路径解析工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.train.latent_cache import resolve_manifest_path, resolve_split_manifest_path


def resolve_manifest_for_split(
    *,
    manifests_cfg: Any,
    split: str,
    outputs_root: str,
    dataset_name: str,
) -> Path:
    """优先使用显式 manifests 配置，否则按 outputs_root 推导。"""
    manifests = dict(manifests_cfg or {})
    if split in manifests and str(manifests[split]):
        return resolve_manifest_path(str(manifests[split]))
    return resolve_split_manifest_path(
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        split=split,
    )
