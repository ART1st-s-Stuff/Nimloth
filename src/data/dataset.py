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
    """从 manifest 构造 (z_t, a_t, z_{t+1}) 训练样本。"""

    def __init__(
        self,
        manifest_path: str,
        latent_dim: int,
        action_dim: int,
        image_encoder: WMImageEncoder | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.image_encoder = image_encoder
        self.samples = []
        path = Path(manifest_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return max(0, len(self.samples) - 1)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        curr = self.samples[idx]
        nxt = self.samples[idx + 1]
        if self.image_encoder is None:
            # 无 encoder 时保留可复现的伪 latent，便于最小链路调试。
            z_t = torch.randn(self.latent_dim) * 0.1 + curr["step_id"] * 0.01
            z_next = torch.randn(self.latent_dim) * 0.1 + nxt["step_id"] * 0.01
        else:
            z_t = self.image_encoder.encode_image_path(str(curr["image_path"])).z
            z_next = self.image_encoder.encode_image_path(str(nxt["image_path"])).z
        action = torch.zeros(self.action_dim)
        action[int(curr["action_id"]) % self.action_dim] = 1.0
        env_context = build_env_context(curr.get("metadata", {}))
        return {"z_t": z_t, "action": action, "z_next": z_next, "env_context": env_context}

