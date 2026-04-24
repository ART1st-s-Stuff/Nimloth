"""WM latent 预编码与缓存复用的统一入口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from src.data.dataset import WMDataset
from src.wm.encoders import WMImageEncoder


def resolve_manifest_path(manifest_path: str) -> Path:
    """解析 manifest 路径，支持 latest 软链接模式。"""
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


def resolve_split_manifest_path(
    *,
    outputs_root: str,
    dataset_name: str,
    split: str,
    manifest_filename: str = "manifest.jsonl",
) -> Path:
    """解析各 split 的 manifest 路径。

    采集输出的目录结构为：
        {outputs_root}/{dataset_name}/{split}/{run_dir}/manifest_worker_*.jsonl

    返回 run 目录路径，Dataset 会自动读取其中的 manifest_worker 文件。
    """
    base = Path(outputs_root) / dataset_name / split

    # 查找 latest run
    meta_path = base / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        latest = metadata.get("latest")
        if isinstance(latest, str):
            latest_dir = base / latest
            if latest_dir.is_dir():
                return latest_dir

    # 退化为按目录时间戳取最新
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name not in ("images", "val", "test")]
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)

    return base


def resolve_split_run_dir(
    *,
    outputs_root: str,
    dataset_name: str,
    split: str,
) -> Path:
    """解析各 split 的最新 run 目录路径。

    采集输出的目录结构为：
        {outputs_root}/{dataset_name}/{split}/{run_dir}/
            manifest_worker_*.jsonl
            images/
            ...
    """
    return resolve_split_manifest_path(
        outputs_root=outputs_root,
        dataset_name=dataset_name,
        split=split,
        manifest_filename="",  # 不检查具体文件，只找目录
    )


def build_latent_cache_path(manifest_path: Path, wm_name: str) -> Path:
    stem = manifest_path.stem
    return manifest_path.parent / f"{stem}.latents.{wm_name}.pt"


def build_latent_cache_dir(run_dir: Path, wm_name: str) -> Path:
    """构建分块 latent cache 目录路径。

    结构：
        {run_dir}/{stem}.latents.{wm_name}/
            episode_{scene}_{id}.pt
            episode_{scene}_{id}.ready
            ...
    """
    stem = run_dir.name  # 使用 run 目录名作为 stem
    return run_dir / f"{stem}.latents.{wm_name}"


def build_episode_cache_path(cache_dir: Path, episode_key: str) -> Path:
    """单个 episode 的 cache 文件路径。

    episode_key 格式为 "{scene}_{episode_id}"，例如 "FloorPlan1_0"
    """
    # 将 scene 转为合法文件名：转小写，空格/特殊字符替换为下划线
    safe_key = episode_key.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"episode_{safe_key}.pt"


def build_episode_ready_path(cache_dir: Path, episode_key: str) -> Path:
    """episode 就绪 marker 文件路径。"""
    safe_key = episode_key.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return cache_dir / f"episode_{safe_key}.ready"


def list_completed_episodes(cache_dir: Path) -> set[str]:
    """列出所有已完成（带 .ready marker）的 episode keys。

    Returns:
        set of episode keys (e.g., {"FloorPlan1_0", "FloorPlan2_1", ...})
    """
    if not cache_dir.exists():
        return set()
    completed = set()
    for f in cache_dir.iterdir():
        if f.suffix == ".ready" and f.stem.startswith("episode_"):
            # 从文件名提取 episode key：episode_{key}.ready -> {key}
            episode_key = f.stem.replace("episode_", "", 1)  # 只替换第一个
            if episode_key:
                completed.add(episode_key)
    return completed


def build_wm_dataset_with_cache(
    *,
    run_dir: Path,
    wm_name: str,
    latent_dim: int,
    action_dim: int,
    history_len: int,
    image_encoder: WMImageEncoder | None,
    temporal_stride: int | tuple[int, int] = 1,
    encoder_num_workers: int,
    encoder_batch_size: int,
    expected_num_patches: int = 0,
    expected_token_dim: int = 0,
    on_latent_progress: Callable[[int, int], None] | None = None,
    lazy_mode: bool = False,
    encoder_queue: Any = None,
    chunk_mode: bool = True,
) -> tuple[WMDataset, Path]:
    """构建使用 latent cache 的 WM 数据集。

    Args:
        run_dir: run 目录路径，包含 manifest_worker_*.jsonl 文件
        wm_name: world model 名称，用于构建 cache 目录名
        ...
    """
    if chunk_mode:
        latent_cache_dir = build_latent_cache_dir(run_dir, wm_name)
        # 分块模式：latent_cache_path 指向分块目录本身
        # Dataset 会检测这个路径是目录还是文件
        dataset = WMDataset(
            manifest_path=str(run_dir),
            latent_dim=latent_dim,
            action_dim=action_dim,
            history_len=history_len,
            temporal_stride=temporal_stride,
            image_encoder=image_encoder,
            latent_cache_path=str(latent_cache_dir),
            encoder_num_workers=encoder_num_workers,
            encoder_batch_size=encoder_batch_size,
            expected_num_patches=expected_num_patches,
            expected_token_dim=expected_token_dim,
            on_latent_progress=on_latent_progress,
            lazy_mode=lazy_mode,
            encoder_queue=encoder_queue,
        )
        return dataset, latent_cache_dir
    else:
        latent_cache_path = run_dir / f"{run_dir.name}.latents.{wm_name}.pt"
        dataset = WMDataset(
            manifest_path=str(run_dir),
            latent_dim=latent_dim,
            action_dim=action_dim,
            history_len=history_len,
            temporal_stride=temporal_stride,
            image_encoder=image_encoder,
            latent_cache_path=str(latent_cache_path),
            encoder_num_workers=encoder_num_workers,
            encoder_batch_size=encoder_batch_size,
            expected_num_patches=expected_num_patches,
            expected_token_dim=expected_token_dim,
            on_latent_progress=on_latent_progress,
            lazy_mode=lazy_mode,
            encoder_queue=encoder_queue,
        )
        return dataset, latent_cache_path

