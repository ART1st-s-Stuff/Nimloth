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
        rollout_steps: int = 1,
        temporal_stride: int | tuple[int, int] = 1,
        image_encoder: WMImageEncoder | None = None,
        latent_cache_path: str | None = None,
        encoder_num_workers: int = 1,
        encoder_batch_size: int = 32,
        expected_num_patches: int = 0,
        expected_token_dim: int = 0,
        on_latent_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = max(1, history_len)
        self.rollout_steps = max(1, int(rollout_steps))
        self.temporal_stride_min, self.temporal_stride_max = self._normalize_temporal_stride(temporal_stride)
        self.image_encoder = image_encoder
        self.latent_cache_path = Path(latent_cache_path) if latent_cache_path else None
        self.encoder_num_workers = max(1, int(encoder_num_workers))
        self.encoder_batch_size = max(1, int(encoder_batch_size))
        self.expected_num_patches = max(0, int(expected_num_patches))
        self.expected_token_dim = max(0, int(expected_token_dim))
        self.on_latent_progress = on_latent_progress
        # 训练样本存在大量重叠帧，按 image_path 缓存可显著减少重复编码。
        self._latent_cache: dict[str, torch.Tensor] = {}
        self.samples = []
        self._training_indices: list[dict[str, list[int]]] = []
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
            min_history_last = self.history_len - 1
            max_history_last = len(episode_indices) - 2
            for history_last_local_idx in range(min_history_last, max_history_last + 1):
                max_valid_stride = (len(episode_indices) - 1 - history_last_local_idx) // self.rollout_steps
                if max_valid_stride < self.temporal_stride_min:
                    continue
                chosen_stride = self._sample_stride(max_valid_stride=max_valid_stride)
                history_start_local_idx = history_last_local_idx - (self.history_len - 1)
                history_global_indices = episode_indices[history_start_local_idx : history_last_local_idx + 1]
                future_global_indices = [
                    episode_indices[history_last_local_idx + step * chosen_stride]
                    for step in range(1, self.rollout_steps + 1)
                ]
                action_source_indices = [
                    episode_indices[history_last_local_idx + step * chosen_stride - 1]
                    for step in range(1, self.rollout_steps + 1)
                ]
                self._training_indices.append(
                    {
                        "history_indices": history_global_indices,
                        "future_indices": future_global_indices,
                        "action_source_indices": action_source_indices,
                        "temporal_stride": [chosen_stride],
                    }
                )
        self._warmup_latent_cache()

    @staticmethod
    def _normalize_temporal_stride(temporal_stride: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(temporal_stride, tuple):
            if len(temporal_stride) != 2:
                raise ValueError("temporal_stride 区间配置必须是二元组 (min, max)。")
            stride_min = int(temporal_stride[0])
            stride_max = int(temporal_stride[1])
        else:
            stride_min = int(temporal_stride)
            stride_max = int(temporal_stride)
        stride_min = max(1, stride_min)
        stride_max = max(stride_min, stride_max)
        return stride_min, stride_max

    def _sample_stride(self, *, max_valid_stride: int) -> int:
        hi = min(self.temporal_stride_max, int(max_valid_stride))
        lo = min(self.temporal_stride_min, hi)
        if lo >= hi:
            return lo
        return int(torch.randint(low=lo, high=hi + 1, size=(1,)).item())

    def __len__(self) -> int:
        return len(self._training_indices)

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
            cached_latent_dim = payload.get("latent_dim") if isinstance(payload, dict) else None
            if cached_latent_dim is not None and int(cached_latent_dim) != int(self.latent_dim):
                logger.warning(
                    "检测到 latent 缓存维度不匹配，忽略旧缓存: cache_dim=%s, expected_dim=%s, cache=%s",
                    str(cached_latent_dim),
                    str(self.latent_dim),
                    str(cache_path),
                )
            elif isinstance(latents, dict):
                for key, value in latents.items():
                    if not (isinstance(key, str) and isinstance(value, torch.Tensor)):
                        continue
                    matches_flat_dim = int(value.numel()) == int(self.latent_dim)
                    matches_patch_layout = (
                        self.expected_num_patches > 0
                        and self.expected_token_dim > 0
                        and value.dim() == 2
                        and int(value.size(0)) == int(self.expected_num_patches)
                        and int(value.size(1)) == int(self.expected_token_dim)
                    )
                    if matches_flat_dim or matches_patch_layout:
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
        index_data = self._training_indices[idx]
        history_samples = [self.samples[i] for i in index_data["history_indices"]]
        future_samples = [self.samples[i] for i in index_data["future_indices"]]
        action_source_samples = [self.samples[i] for i in index_data["action_source_indices"]]
        z_history = torch.stack([self._encode_latent(sample) for sample in history_samples], dim=0)
        action_history = torch.stack([self._build_action_vec(sample) for sample in history_samples], dim=0)
        z_future = torch.stack([self._encode_latent(sample) for sample in future_samples], dim=0)
        gt_action_future = torch.stack([self._build_action_vec(sample) for sample in action_source_samples], dim=0)
        z_next = z_future[0]
        gt_action = gt_action_future[0]
        env_context = build_env_context(history_samples[-1].get("metadata", {}))
        return {
            "z_history": z_history,
            "action_history": action_history,
            "z_next": z_next,
            "gt_action": gt_action,
            "z_future": z_future,
            "gt_action_future": gt_action_future,
            "env_context": env_context,
            "temporal_stride": torch.tensor(index_data["temporal_stride"], dtype=torch.int64),
        }

