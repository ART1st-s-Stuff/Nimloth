"""EB-Nav 数据集适配器。

将 EmbodiedBench/EB-Nav_trajectory_dataset 转换为项目训练所需格式。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torch.utils.data import Dataset

if TYPE_CHECKING:
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

ACTION_NAMES = {
    0: "Move forward by 0.25",
    1: "Move backward by 0.25",
    2: "Move rightward by 0.25",
    3: "Move leftward by 0.25",
    4: "Rotate to the right by 90 degrees",
    5: "Rotate to the left by 90 degrees",
    6: "Tilt the camera upward by 30 degrees",
    7: "Tilt the camera downward by 30 degrees",
}

LATENT_STATE_MARKER = "<LATENT_STATE>"


def resolve_eb_nav_image_path(img_path: str, images_base_dir: str | Path) -> str:
    """Resolve an EB-Nav image path while preserving absolute paths."""
    if not img_path:
        return ""
    path = Path(img_path)
    if path.is_absolute():
        return str(path)
    base = Path(images_base_dir)
    candidate = base / path
    if candidate.exists():
        return str(candidate)
    if base.name == "images" and path.parts and path.parts[0] == "images":
        return str(base.parent / path)
    return str(candidate)


def get_eb_nav_action_id(plan: dict[str, Any]) -> int:
    action = plan.get("action", [0, ""])
    if isinstance(action, list) and action:
        try:
            return int(action[0])
        except (TypeError, ValueError):
            return 0
    return 0


def build_action_prior(action_id: int, smoothing: float = 0.05) -> list[float]:
    """Build an 8-way label-smoothed expert action prior."""
    num_actions = len(ACTION_NAMES)
    action_id = int(action_id)
    smoothing = min(max(float(smoothing), 0.0), 1.0)
    if action_id not in ACTION_NAMES:
        return [1.0 / num_actions for _ in range(num_actions)]
    off_value = smoothing / float(num_actions - 1) if num_actions > 1 else 0.0
    values = [off_value for _ in range(num_actions)]
    values[action_id] = 1.0 - smoothing
    return values


def build_planner_response(
    *,
    cot: str,
    action_id: int,
    smoothing: float = 0.05,
) -> dict[str, Any]:
    """Build the fixed Qwen planner response schema used by the SFT stage."""
    probabilities = build_action_prior(action_id=action_id, smoothing=smoothing)
    sorted_actions = sorted(
        (
            {
                "action_id": int(idx),
                "name": ACTION_NAMES[int(idx)],
                "score": float(score),
            }
            for idx, score in enumerate(probabilities)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "cot": cot or "",
        "planner_trigger": True,
        "latent_state": LATENT_STATE_MARKER,
        "action_prior": {
            "probabilities": probabilities,
            "top_actions": sorted_actions[:3],
        },
    }


def compute_eb_nav_reward(
    *,
    action_success: bool,
    env_feedback: str = "",
    is_terminal: bool = False,
    episode_success: bool | float | int = False,
    step_cost: float = -0.01,
    success_action_reward: float = 0.05,
    failed_action_reward: float = -0.20,
    terminal_success_reward: float = 1.0,
    terminal_failure_reward: float = -0.5,
) -> float:
    """Default EB-Nav step reward from available success/failure fields."""
    feedback = str(env_feedback or "").lower()
    success = bool(action_success)
    reward = float(step_cost)
    if success and "invalid" not in feedback and "blocking" not in feedback:
        reward += float(success_action_reward)
    else:
        reward += float(failed_action_reward)
    if is_terminal:
        reward += float(terminal_success_reward if bool(episode_success) else terminal_failure_reward)
    return float(reward)


def load_reward_cache(reward_cache_path: str | Path | None) -> dict[tuple[int, int, int], float]:
    """Load a JSON/JSONL reward cache keyed by (episode_idx, trajectory_step_idx, plan_idx)."""
    if not reward_cache_path:
        return {}
    path = Path(reward_cache_path)
    if not path.exists():
        return {}
    records: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with open(path) as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        loaded = json.load(open(path))
        records = loaded.get("records", loaded) if isinstance(loaded, dict) else loaded
    cache: dict[tuple[int, int, int], float] = {}
    for item in records:
        key = (
            int(item.get("episode_idx", 0)),
            int(item.get("trajectory_step_idx", item.get("step_idx", 0))),
            int(item.get("plan_idx", 0)),
        )
        cache[key] = float(item.get("reward", 0.0))
    return cache


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
        reward_cache_path: str | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.images_base_dir = Path(images_base_dir) if images_base_dir else self.json_path.parent / "images"
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.image_encoder = image_encoder
        self.split = split
        self.reward_cache = load_reward_cache(reward_cache_path)

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

    def _get_reward(
        self,
        *,
        ep_idx: int,
        step_idx: int,
        plan_idx: int,
        plan: dict[str, Any],
        episode: dict[str, Any],
    ) -> float:
        cached = self.reward_cache.get((ep_idx, step_idx, plan_idx))
        if cached is not None:
            return cached
        return compute_eb_nav_reward(
            action_success=bool(plan.get("action_success", False)),
            env_feedback=str(plan.get("env_feedback", "")),
            is_terminal=step_idx == len(episode.get("trajectory", [])) - 1,
            episode_success=episode.get("success", 0),
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep_idx, step_idx = self.samples[idx]
        episode = self.data[ep_idx]
        step = episode["trajectory"][step_idx]

        # 获取图像路径
        plan = step["executable_plan"][0] if step["executable_plan"] else {}
        img_path = plan.get("img_path", step.get("input_image_path", ""))
        img_path = resolve_eb_nav_image_path(img_path, self.images_base_dir)

        # 获取动作
        action_list = plan.get("action", [0, ""])
        action = self._parse_action(action_list)
        action_id = get_eb_nav_action_id(plan)
        reward = self._get_reward(
            ep_idx=ep_idx,
            step_idx=step_idx,
            plan_idx=0,
            plan=plan,
            episode=episode,
        )

        return {
            "image_path": img_path,
            "action": action,
            "action_id": action_id,
            "action_text": ACTION_NAMES.get(action_id, action_list[1] if len(action_list) > 1 else ""),
            "action_success": bool(plan.get("action_success", False)),
            "env_feedback": str(plan.get("env_feedback", "")),
            "reward": reward,
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
        reward_cache_path: str | None = None,
    ) -> None:
        self.json_path = Path(json_path)
        self.images_base_dir = Path(images_base_dir) if images_base_dir else self.json_path.parent / "images"
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.temporal_stride = temporal_stride
        self.image_encoder = image_encoder
        self.split = split
        self.reward_cache = load_reward_cache(reward_cache_path)

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
                    img_path = resolve_eb_nav_image_path(img_path, self.images_base_dir)
                    history_images.append(img_path)
                    action_list = plan.get("action", [0, ""])
                    action_vec = [0, 0, 0]
                    action_id = action_list[0] if isinstance(action_list, list) else 0
                    action_vec = ACTION_MAP.get(action_id, [0, 0, 0])
                    history_actions.append(action_vec[:action_dim])

                future_images = []
                future_actions = []
                future_rewards = []
                future_action_ids = []
                for future_offset, step in enumerate(future_steps):
                    absolute_step_idx = start + history_len + future_offset
                    plan = step["executable_plan"][0] if step["executable_plan"] else {}
                    img_path = plan.get("img_path", step.get("input_image_path", ""))
                    img_path = resolve_eb_nav_image_path(img_path, self.images_base_dir)
                    future_images.append(img_path)
                    action_list = plan.get("action", [0, ""])
                    action_vec = [0, 0, 0]
                    action_id = action_list[0] if isinstance(action_list, list) else 0
                    action_vec = ACTION_MAP.get(action_id, [0, 0, 0])
                    future_actions.append(action_vec[:action_dim])
                    future_action_ids.append(int(action_id))
                    cached = self.reward_cache.get((ep_idx, absolute_step_idx, 0))
                    if cached is None:
                        cached = compute_eb_nav_reward(
                            action_success=bool(plan.get("action_success", False)),
                            env_feedback=str(plan.get("env_feedback", "")),
                            is_terminal=absolute_step_idx == num_steps - 1,
                            episode_success=episode.get("success", 0),
                        )
                    future_rewards.append(float(cached))

                self.sequences.append({
                    "episode_idx": ep_idx,
                    "episode_id": episode.get("episode_id", str(ep_idx)),
                    "instruction": episode.get("instruction", ""),
                    "history_images": history_images,
                    "history_actions": history_actions,
                    "future_images": future_images,
                    "future_actions": future_actions,
                    "future_action_ids": future_action_ids,
                    "future_rewards": future_rewards,
                    "model_name": episode.get("model_name", ""),
                    "success": episode.get("success", 0),
                })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.sequences[idx]
