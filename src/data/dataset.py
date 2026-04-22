"""训练所需数据集定义。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset
from src.wm.encoders import WMImageEncoder

logger = logging.getLogger(__name__)


def build_env_context(metadata: dict[str, Any]) -> str:
    """基于 metadata 生成任务无关环境信息描述。"""
    scene = metadata.get("scene")
    distance = metadata.get("target_distance")
    collided = metadata.get("collided")
    grasped = metadata.get("grasped")
    parts: list[str] = []
    if scene is not None:
        parts.append(f"scene={scene}")
    if isinstance(distance, (int, float)):
        parts.append(f"target_distance={float(distance):.3f}m")
    if isinstance(collided, bool):
        parts.append(f"collided={'yes' if collided else 'no'}")
    if isinstance(grasped, bool):
        parts.append(f"grasped={'yes' if grasped else 'no'}")
    if not parts:
        return "env=unknown"
    return " | ".join(parts)


class WMDataset(Dataset):
    """从 manifest 构造序列训练样本。"""

    def __init__(
        self,
        manifest_path: str,
        latent_dim: int,
        action_dim: int,
        history_len: int,
        image_encoder: WMImageEncoder | None = None,
        latent_cache_path: str | None = None,
        encoder_num_workers: int = 1,
        encoder_batch_size: int = 32,
        on_latent_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = max(1, history_len)
        self.image_encoder = image_encoder
        self.latent_cache_path = Path(latent_cache_path) if latent_cache_path else None
        self.encoder_num_workers = max(1, int(encoder_num_workers))
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.on_latent_progress = on_latent_progress
        # 训练样本存在大量重叠帧，按 image_path 缓存可显著减少重复编码。
        self._latent_cache: dict[str, torch.Tensor] = {}
        self.samples = []
        self._index_pairs: list[tuple[int, int]] = []
        path = Path(manifest_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.samples.append(json.loads(line))
        episode_to_indices: dict[int, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            episode_id = int(sample.get("episode_id", -1))
            episode_to_indices.setdefault(episode_id, []).append(idx)
        for episode_indices in episode_to_indices.values():
            for end_idx in range(self.history_len - 1, len(episode_indices) - 1):
                history_last_global_idx = episode_indices[end_idx]
                target_global_idx = episode_indices[end_idx + 1]
                self._index_pairs.append((history_last_global_idx, target_global_idx))
        self._warmup_latent_cache()

    def __len__(self) -> int:
        return len(self._index_pairs)

    def _encode_latent(self, sample: dict[str, Any]) -> torch.Tensor:
        image_path = str(sample["image_path"])
        cached = self._latent_cache.get(image_path)
        if cached is not None:
            return cached
        if self.image_encoder is None:
            raise RuntimeError(f"latent 缓存缺失且 encoder 不可用: {image_path}")
        latent = self.image_encoder.encode_image_path(image_path).z
        self._latent_cache[image_path] = latent
        return latent

    def disable_encoder_after_warmup(self) -> None:
        """预编码完成后关闭 encoder，便于 DataLoader 多进程并行读取缓存。"""
        self.image_encoder = None

    def _warmup_latent_cache(self) -> None:
        if self.image_encoder is None:
            return
        cache_path = self.latent_cache_path
        if cache_path is not None and cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            latents = payload.get("latents", {}) if isinstance(payload, dict) else {}
            if isinstance(latents, dict):
                for key, value in latents.items():
                    if isinstance(key, str) and isinstance(value, torch.Tensor):
                        self._latent_cache[key] = value.detach().cpu()
        unique_paths = {str(sample["image_path"]) for sample in self.samples if "image_path" in sample}
        missing_paths = [path for path in unique_paths if path not in self._latent_cache]
        if not missing_paths:
            logger.info("latent 缓存已命中: %d 张图像，无需预编码。", len(self._latent_cache))
            if self.on_latent_progress is not None:
                self.on_latent_progress(0, 0)
            return
        logger.info(
            "开始预编码 latent: 总图像=%d, 待编码=%d, workers=%d, batch_size=%d",
            len(unique_paths),
            len(missing_paths),
            self.encoder_num_workers,
            self.encoder_batch_size,
        )
        flush_every = 500
        for start in range(0, len(missing_paths), self.encoder_batch_size):
            batch_paths = missing_paths[start : start + self.encoder_batch_size]
            batch_outputs = self.image_encoder.encode_image_paths(batch_paths)
            for image_path, output in zip(batch_paths, batch_outputs, strict=True):
                self._latent_cache[image_path] = output.z.detach().cpu()
            done = start + len(batch_paths)
            if self.on_latent_progress is not None:
                self.on_latent_progress(done, len(missing_paths))
            if done % flush_every == 0:
                logger.info("latent 预编码进度: %d/%d", done, len(missing_paths))
                self._save_latent_cache(cache_path)
        self._save_latent_cache(cache_path)
        logger.info("latent 预编码完成: 新增=%d, 缓存总量=%d", len(missing_paths), len(self._latent_cache))

    def _save_latent_cache(self, cache_path: Path | None) -> None:
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent_dim": self.latent_dim,
                "latents": self._latent_cache,
            },
            cache_path,
        )

    def _build_action_vec(self, sample: dict[str, Any]) -> torch.Tensor:
        move = float(sample.get("move_ahead_distance", 0.0))
        yaw = float(sample.get("delta_yaw", 0.0))
        pitch = float(sample.get("delta_pitch", 0.0))
        action = torch.tensor([move, yaw, pitch], dtype=torch.float32)
        if self.action_dim <= 3:
            return action[: self.action_dim]
        padded = torch.zeros(self.action_dim, dtype=torch.float32)
        padded[:3] = action
        return padded

    def __getitem__(self, idx: int) -> dict[str, Any]:
        history_last_idx, target_idx = self._index_pairs[idx]
        curr = self.samples[history_last_idx]
        nxt = self.samples[target_idx]
        history_start_idx = history_last_idx - (self.history_len - 1)
        history_samples = self.samples[history_start_idx : history_last_idx + 1]
        z_history = torch.stack([self._encode_latent(sample) for sample in history_samples], dim=0)
        action_history = torch.stack([self._build_action_vec(sample) for sample in history_samples], dim=0)
        z_next = self._encode_latent(nxt)
        gt_action = self._build_action_vec(curr)
        env_context = build_env_context(curr.get("metadata", {}))
        return {
            "z_history": z_history,
            "action_history": action_history,
            "z_next": z_next,
            "gt_action": gt_action,
            "env_context": env_context,
        }

