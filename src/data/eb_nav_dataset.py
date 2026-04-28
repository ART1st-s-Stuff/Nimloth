"""EB-Nav 数据集适配器。

将 EmbodiedBench/EB-Nav_trajectory_dataset 转换为项目训练所需格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.wm.encoder import WMImageEncoder


# 动作映射
ACTION_MAP = {
    0: [0.25, 0, 0],      # Move forward
    1: [-0.25, 0, 0],     # Move backward
    2: [0, 0, -0.25],     # Move right (Qwen 用负值表示右)
    3: [0, 0, 0.25],      # Move left
    4: [0, -90, 0],        # Rotate right 90°
    5: [0, 90, 0],          # Rotate left 90°
    6: [0, 0, 0],           # Tilt up (俯仰，可能不用)
    7: [0, 0, 0],           # Tilt down (俯仰，可能不用)
}


class EBNavDataset(Dataset):
    """EB-Nav 数据集，用于 WM 训练和 Value Head 训练。

    数据格式：
    {
        "model_name": "claude-3-5-sonnet-...",
        "instruction": "navigate to the Bread...",
        "trajectory": [
            {
                "visual_description": "...",
                "reasoning_and_reflection": "...",  # CoT
                "language_plan": "...",
                "executable_plan": [
                    {
                        "step_id": 1,
                        "img_path": "...",
                        "action": [0, "Move forward by 0.25"],
                        "action_success": true,
                    }
                ],
                "input_image_path": "..."
            }
        ]
    }

    转换为项目格式：
    - image_path: 图像路径
    - action: 连续动作 [move, yaw, pitch]
    - instruction: 导航指令
    - cot: Chain-of-thought reasoning
    """

    def __init__(
        self,
        json_path: str,
        images_base_dir: str | None = None,
        latent_dim: int = 4096,
        action_dim: int = 3,
        history_len: int = 4,
        image_encoder: WMImageEncoder | None = None,
        split: str = "train",
    ) -> None:
        self.json_path = Path(json_path)
        self.images_base_dir = Path(images_base_dir) if images_base_dir else self.json_path.parent / "images"
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.image_encoder = image_encoder
        self.split = split

        # 加载数据
        with open(self.json_path) as f:
            self.data = json.load(f)

        # 构建样本列表：(episode_idx, step_idx)
        self.samples: list[tuple[int, int]] = []
        for ep_idx, episode in enumerate(self.data):
            for step_idx in range(len(episode["trajectory"])):
                self.samples.append((ep_idx, step_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_action(self, action: list) -> list[float]:
        """将动作转换为连续动作向量 [move, yaw, pitch]"""
        action_id = action[0]
        action_vec = ACTION_MAP.get(action_id, [0, 0, 0])
        return action_vec[: self.action_dim]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, step_idx = self.samples[idx]
        episode = self.data[ep_idx]
        step = episode["trajectory"][step_idx]

        # 获取图像路径
        plan = step["executable_plan"][0] if step["executable_plan"] else {}
        img_path = plan.get("img_path", step.get("input_image_path", ""))
        if img_path and not img_path.startswith("/"):
            img_path = str(self.images_base_dir / img_path)

        # 获取动作
        action_list = plan.get("action", [0, ""])
        action = self._parse_action(action_list)

        return {
            "image_path": img_path,
            "action": action,
            "instruction": episode.get("instruction", ""),
            "cot": step.get("reasoning_and_reflection", ""),
            "visual_description": step.get("visual_description", ""),
            "language_plan": step.get("language_plan", ""),
            "episode_id": episode.get("episode_id", str(ep_idx)),
            "model_name": episode.get("model_name", ""),
            "step_id": step_idx,
            "success": episode.get("success", 0),
        }


class EBNavSequenceDataset(Dataset):
    """EB-Nav 序列数据集，返回历史帧和未来预测目标。

    用于训练 WM 的 dynamics prediction。
    """

    def __init__(
        self,
        json_path: str,
        images_base_dir: str | None = None,
        latent_dim: int = 4096,
        action_dim: int = 3,
        history_len: int = 4,
        temporal_stride: int = 1,
        image_encoder: WMImageEncoder | None = None,
        split: str = "train",
    ) -> None:
        self.json_path = Path(json_path)
        self.images_base_dir = Path(images_base_dir) if images_base_dir else self.json_path.parent / "images"
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.temporal_stride = temporal_stride
        self.image_encoder = image_encoder
        self.split = split

        # 加载数据
        with open(self.json_path) as f:
            self.data = json.load(f)

        # 构建序列样本
        self.sequences: list[dict] = []
        for ep_idx, episode in enumerate(self.data):
            trajectory = episode["trajectory"]
            num_steps = len(trajectory)

            for start in range(num_steps - history_len - temporal_stride + 1):
                history_steps = trajectory[start : start + history_len]
                future_steps = trajectory[start + history_len : start + history_len + temporal_stride]

                # 构建历史和未来的图像路径、动作
                history_images = []
                history_actions = []
                for step in history_steps:
                    plan = step["executable_plan"][0] if step["executable_plan"] else {}
                    img_path = plan.get("img_path", step.get("input_image_path", ""))
                    if img_path and not img_path.startswith("/"):
                        img_path = str(self.images_base_dir / img_path)
                    history_images.append(img_path)
                    action_list = plan.get("action", [0, ""])
                    action_vec = [0, 0, 0]
                    action_id = action_list[0] if isinstance(action_list, list) else 0
                    action_vec = ACTION_MAP.get(action_id, [0, 0, 0])
                    history_actions.append(action_vec[:action_dim])

                future_images = []
                future_actions = []
                for step in future_steps:
                    plan = step["executable_plan"][0] if step["executable_plan"] else {}
                    img_path = plan.get("img_path", step.get("input_image_path", ""))
                    if img_path and not img_path.startswith("/"):
                        img_path = str(self.images_base_dir / img_path)
                    future_images.append(img_path)
                    action_list = plan.get("action", [0, ""])
                    action_vec = [0, 0, 0]
                    action_id = action_list[0] if isinstance(action_list, list) else 0
                    action_vec = ACTION_MAP.get(action_id, [0, 0, 0])
                    future_actions.append(action_vec[:action_dim])

                self.sequences.append({
                    "episode_idx": ep_idx,
                    "instruction": episode.get("instruction", ""),
                    "history_images": history_images,
                    "history_actions": history_actions,
                    "future_images": future_images,
                    "future_actions": future_actions,
                    "model_name": episode.get("model_name", ""),
                    "success": episode.get("success", 0),
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.sequences[idx]
