"""训练所需数据集定义。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from src.wm.encoders import WMImageEncoder


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
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = max(1, history_len)
        self.image_encoder = image_encoder
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

    def __len__(self) -> int:
        return len(self._index_pairs)

    def _encode_latent(self, sample: dict[str, Any]) -> torch.Tensor:
        if self.image_encoder is None:
            return torch.randn(self.latent_dim) * 0.1 + float(sample["step_id"]) * 0.01
        return self.image_encoder.encode_image_path(str(sample["image_path"])).z

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

