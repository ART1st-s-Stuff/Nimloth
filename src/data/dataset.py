"""训练所需数据集定义。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class WMDataset(Dataset):
    """从 manifest 构造 (z_t, a_t, z_{t+1}) 训练样本。"""

    def __init__(self, manifest_path: str, latent_dim: int, action_dim: int) -> None:
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.samples = []
        path = Path(manifest_path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return max(0, len(self.samples) - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        curr = self.samples[idx]
        nxt = self.samples[idx + 1]
        # 首版使用可复现的伪 latent，后续替换为真实 encoder 输出。
        z_t = torch.randn(self.latent_dim) * 0.1 + curr["step_id"] * 0.01
        z_next = torch.randn(self.latent_dim) * 0.1 + nxt["step_id"] * 0.01
        action = torch.zeros(self.action_dim)
        action[int(curr["action_id"]) % self.action_dim] = 1.0
        return {"z_t": z_t, "action": action, "z_next": z_next}

