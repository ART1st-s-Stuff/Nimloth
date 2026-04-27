"""UMAP 缓存公共工具。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import umap


def ensure_cache_dirs(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)


def umap_cache_key(feature_name: str, points: list[list[float]]) -> str:
    payload = json.dumps({"feature_name": feature_name, "points": points}, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def load_umap_cache(cache_dir: Path, feature_name: str, cache_key: str) -> list[list[float]] | None:
    ensure_cache_dirs(cache_dir)
    cache_file = cache_dir / f"{feature_name}_{cache_key}.json"
    if not cache_file.exists():
        return None
    return json.loads(cache_file.read_text(encoding="utf-8"))


def save_umap_cache(cache_dir: Path, feature_name: str, cache_key: str, embedded: list[list[float]]) -> None:
    ensure_cache_dirs(cache_dir)
    cache_file = cache_dir / f"{feature_name}_{cache_key}.json"
    cache_file.write_text(json.dumps(embedded), encoding="utf-8")


def compute_umap_3d(
    *,
    cache_dir: Path,
    points: list[list[float]],
    feature_name: str,
) -> tuple[list[list[float]], str | None]:
    if len(points) < 3:
        return points, "样本数少于 3，跳过 UMAP 降维。"
    cache_key = umap_cache_key(feature_name=feature_name, points=points)
    cached = load_umap_cache(cache_dir, feature_name=feature_name, cache_key=cache_key)
    if cached is not None:
        return cached, None
    reducer = umap.UMAP(n_components=3, random_state=42)
    embedded = reducer.fit_transform(np.asarray(points, dtype=np.float32))
    embedded_list = embedded.tolist()
    save_umap_cache(cache_dir, feature_name=feature_name, cache_key=cache_key, embedded=embedded_list)
    return embedded_list, None
