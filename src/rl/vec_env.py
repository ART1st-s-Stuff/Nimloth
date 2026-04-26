"""LatentVecEnv - 基于 latent cache 的向量化环境。

使用已有的 latent cache 数据构建 RL 环境，
每个环境对应一个 episode，支持并行采样。
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class EnvStepResult:
    """单步执行结果。"""

    obs_z: Tensor
    obs_s: Tensor | None
    reward: Tensor
    done: Tensor
    info: dict[str, Any]


class LatentVecEnv:
    """基于 latent cache 的向量化环境。

    使用预编码的 latent 数据模拟环境交互：
    - 每个 env 对应一个 episode
    - step() 执行动作，返回下一 latent 和奖励
    - 奖励基于动作与 GT 的相似度计算（简化版本）
    """

    def __init__(
        self,
        manifest_path: str,
        latent_cache_dir: str,
        num_envs: int,
        history_len: int,
        num_patches: int,
        token_dim: int,
        action_dim: int,
        semantic_dim: int = 0,
        device: str = "cuda",
        max_episode_length: int = 50,
        include_semantic: bool = False,
        wm_ckpt_path: str | None = None,
        use_wm_prediction: bool = False,
        reward_type: str = "action_match",
        vlm_adapter_ckpt: str | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            manifest_path: 数据 manifest 路径（支持目录/.jsonl/.pt）
            latent_cache_dir: latent cache 文件或目录路径
            num_envs: 并行环境数
            history_len: 使用的历史长度
            num_patches: latent patch 数
            token_dim: 每个 patch 的维度
            action_dim: 动作维度
            semantic_dim: VLM 语义特征维度
            device: 计算设备
            max_episode_length: 最大 episode 长度
            include_semantic: 是否包含语义特征
            wm_ckpt_path: WM checkpoint 路径（用于预测下一 latent）
            use_wm_prediction: 是否使用 WM 预测下一 latent
            reward_type: 奖励类型
            vlm_adapter_ckpt: VLM adapter checkpoint
            seed: 随机种子
        """
        self.manifest_path = manifest_path
        self.latent_cache_path = Path(latent_cache_dir)
        self.num_envs = num_envs
        self.history_len = history_len
        self.num_patches = num_patches
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.semantic_dim = semantic_dim
        self.device = device
        self.max_episode_length = max_episode_length
        self.include_semantic = include_semantic
        self.use_wm_prediction = use_wm_prediction
        self.reward_type = reward_type
        self._needs_z_history_reinit = False

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # 加载 latent cache
        self.latent_cache: dict[str, Tensor] = {}
        self.episodes: dict[str, list[str]] = {}  # episode_key -> sorted image_paths
        self._load_latent_cache()
        logger.info("加载了 %d 个 latent cache 条目, %d 个 episodes",
                    len(self.latent_cache), len(self.episodes))

        if len(self.episodes) == 0:
            raise ValueError(f"没有找到有效的 episodes")

        # 当前 episode 列表（用于随机采样）
        self.episode_keys = list(self.episodes.keys())

        # 环境状态
        self._reset_envs()

    def _parse_episode_key(self, path_str: str) -> tuple[str, int, int]:
        """从图像路径解析 episode key, episode_id, step。

        支持格式: floorplan6_ep0001_step0156.png
        Returns: (episode_key, episode_id, step)
        """
        # 匹配 floorplan6_ep0001_step0156.png
        # episode_key 不包含 step 数字，这样同一 episode 的所有 step 会归到一组
        match = re.search(r"(floorplan\d+_ep(\d+))_step(\d+)\.png", path_str)
        if match:
            episode_key = match.group(1)  # floorplan6_ep0001 (不包含 step)
            episode_id = int(match.group(2))
            step = int(match.group(3))
            return episode_key, episode_id, step
        return "", -1, -1

    def _load_latent_cache(self) -> None:
        """加载 latent cache 和 episode 结构。"""
        cache_path = self.latent_cache_path
        if not cache_path.exists():
            logger.warning("latent cache 不存在: %s", cache_path)
            return

        if cache_path.is_file() and cache_path.suffix == ".pt":
            # 单文件格式
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                latents = payload.get("latents", {})
                if isinstance(latents, dict):
                    self._build_episodes_from_latents(latents)
                    # 自动检测 token_dim
                    if latents:
                        sample = list(latents.values())[0]
                        if sample.dim() == 2:
                            detected_num_patches, detected_token_dim = sample.shape
                            if self.num_patches != detected_num_patches or self.token_dim != detected_token_dim:
                                logger.info("检测到 latent 维度: patches=%d, token_dim=%d (之前配置: patches=%d, token_dim=%d)",
                                    detected_num_patches, detected_token_dim, self.num_patches, self.token_dim)
                                self.num_patches = detected_num_patches
                                self.token_dim = detected_token_dim
                                # 更新 z_history 形状（延迟初始化，需要在 _reset_envs 后使用）
                                self._needs_z_history_reinit = True
            except Exception as exc:
                logger.warning("加载 %s 失败: %s", cache_path, exc)
        elif cache_path.is_dir():
            # 分块格式
            for pt_file in sorted(cache_path.glob("episode_*.pt")):
                try:
                    payload = torch.load(pt_file, map_location="cpu", weights_only=False)
                    latents = payload.get("latents", {})
                    if isinstance(latents, dict):
                        self.latent_cache.update(latents)
                        self._build_episodes_from_latents(latents)
                except Exception as exc:
                    logger.warning("加载 %s 失败: %s", pt_file, exc)
            # 也查找单文件 latent cache
            for pt_file in sorted(cache_path.glob("*.latents.pt")):
                try:
                    payload = torch.load(pt_file, map_location="cpu", weights_only=False)
                    latents = payload.get("latents", {})
                    if isinstance(latents, dict):
                        self._build_episodes_from_latents(latents)
                except Exception:
                    pass
        else:
            logger.warning("未知的 latent cache 格式: %s", cache_path)

    def _build_episodes_from_latents(self, latents: dict[str, Tensor]) -> None:
        """从 latent keys 构建 episode 结构。

        图像路径格式: .../floorplan6_ep0001_step0156.png
        """
        episode_data: dict[str, list[tuple[int, str]]] = {}
        for path_str in latents.keys():
            episode_key, episode_id, step = self._parse_episode_key(path_str)
            if episode_key and step >= 0:
                if episode_key not in episode_data:
                    episode_data[episode_key] = []
                episode_data[episode_key].append((step, path_str))

        # 排序并存储
        for episode_key, steps_paths in episode_data.items():
            sorted_pairs = sorted(steps_paths, key=lambda x: x[0])
            sorted_paths = [p for _, p in sorted_pairs]
            self.episodes[episode_key] = sorted_paths
            for path_str in sorted_paths:
                if path_str not in self.latent_cache:
                    self.latent_cache[path_str] = latents[path_str]

    def _reset_envs(self) -> None:
        """重置所有环境状态。"""
        # 为每个 env 分配一个 episode
        num_available = len(self.episode_keys)
        if num_available < self.num_envs:
            logger.warning(
                "episode 数量 (%d) 少于 env 数量 (%d)，将重复采样",
                num_available, self.num_envs
            )
            episode_indices = (
                random.sample(range(num_available), num_available) *
                ((self.num_envs // num_available) + 1)
            )[:self.num_envs]
        else:
            episode_indices = random.sample(range(num_available), self.num_envs)

        # 当前 step 索引（episode 内的位置）
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long)
        # 当前 z_history（初始为全零）- 使用当前 num_patches, token_dim
        self.z_history = torch.zeros(
            self.num_envs, self.history_len, self.num_patches, self.token_dim,
            device=self.device, dtype=torch.float32,
        )
        # 当前语义特征
        self.semantic = torch.zeros(
            self.num_envs, self.semantic_dim,
            device=self.device, dtype=torch.float32,
        ) if self.semantic_dim > 0 else None
        # episode 分配
        self.env_episode_keys = [self.episode_keys[i] for i in episode_indices]
        self.env_step_indices = torch.zeros(self.num_envs, dtype=torch.long)  # episode 内的 index
        # 如果之前检测到不同的 token_dim，重新初始化
        if self._needs_z_history_reinit:
            self._needs_z_history_reinit = False

    def _get_latent(self, path_str: str) -> Tensor:
        """获取指定路径的 latent。"""
        if path_str in self.latent_cache:
            z = self.latent_cache[path_str]
        else:
            # Fallback: 返回随机 latent
            z = torch.randn(self.num_patches, self.token_dim, device="cpu")
        return z

    def reset(self) -> tuple[Tensor, Tensor | None]:
        """重置所有环境。

        Returns:
            z_history: [num_envs, H, P, D]
            semantic: [num_envs, D_s] 或 None
        """
        self._reset_envs()

        # 加载初始 latent
        for i in range(self.num_envs):
            ep_key = self.env_episode_keys[i]
            step_idx = self.env_step_indices[i].item()
            paths = self.episodes[ep_key]
            if step_idx < len(paths):
                z = self._get_latent(paths[step_idx])
                self.z_history[i, -1] = z.to(self.z_history.device)

        return self.z_history.clone(), self.semantic.clone() if self.semantic is not None else None

    def step(self, actions: Tensor) -> EnvStepResult:
        """执行一步交互。

        Args:
            actions: [num_envs, A] 动作

        Returns:
            EnvStepResult: 包含下一观察、奖励、done、info
        """
        actions = actions.to(self.device)
        B = actions.size(0)

        # 计算奖励（简化：基于动作大小作为探索奖励）
        rewards = torch.zeros(B, device=self.device)
        for i in range(B):
            action_norm = actions[i].norm().item()
            # 鼓励动作接近零的动作距离（简化）
            reward = -action_norm * 0.01
            rewards[i] = reward

        # 更新 step 索引
        self.current_step += 1
        self.env_step_indices += 1

        # 检查 done
        dones = torch.zeros(B, dtype=torch.bool, device=self.device)
        for i in range(B):
            ep_key = self.env_episode_keys[i]
            paths = self.episodes[ep_key]
            if self.env_step_indices[i] >= len(paths) - 1:
                dones[i] = True
            elif self.current_step[i] >= self.max_episode_length:
                dones[i] = True

        # 更新 z_history
        new_z = torch.zeros(B, 1, self.num_patches, self.token_dim, device=self.device, dtype=torch.float32)
        for i in range(B):
            ep_key = self.env_episode_keys[i]
            step_idx = self.env_step_indices[i].item()
            paths = self.episodes[ep_key]
            if step_idx < len(paths) and not dones[i]:
                z = self._get_latent(paths[step_idx])
                new_z[i, 0] = z.to(new_z.device)
            else:
                # Episode 结束或超出范围：使用随机 latent
                new_z[i, 0] = torch.randn(self.num_patches, self.token_dim, device=self.device) * 0.1

        # 移位
        self.z_history = torch.cat([self.z_history[:, 1:], new_z], dim=1)

        # 更新语义特征（简化：使用全零）
        if self.semantic is not None:
            self.semantic = torch.zeros_like(self.semantic)

        return EnvStepResult(
            obs_z=self.z_history.clone(),
            obs_s=self.semantic.clone() if self.semantic is not None else None,
            reward=rewards,
            done=dones,
            info={"step": self.current_step.clone()},
        )

    @property
    def batch_size(self) -> int:
        return self.num_envs

    def close(self) -> None:
        """关闭环境（清理资源）。"""
        pass

    def render(self, mode: str = "rgb_array") -> None:
        """渲染（暂不支持）。"""
        raise NotImplementedError("LatentVecEnv 不支持渲染")


class DummyVecEnv:
    """单进程 dummy 环境，用于测试。"""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_patches: int,
        token_dim: int,
        history_len: int,
        num_envs: int = 1,
        semantic_dim: int = 0,
        device: str = "cuda",
        max_episode_length: int = 50,
    ) -> None:
        self.num_envs = num_envs
        self.num_patches = num_patches
        self.token_dim = token_dim
        self.history_len = history_len
        self.semantic_dim = semantic_dim
        self.device = device
        self.max_episode_length = max_episode_length

        self.z_history = torch.zeros(
            num_envs, history_len, num_patches, token_dim,
            device=device, dtype=torch.float32,
        )
        self.semantic = torch.zeros(
            num_envs, semantic_dim, device=device, dtype=torch.float32,
        ) if semantic_dim > 0 else None
        self.current_step = torch.zeros(num_envs, dtype=torch.long)

    def reset(self) -> tuple[Tensor, Tensor | None]:
        self.current_step.zero_()
        self.z_history = torch.randn_like(self.z_history) * 0.1
        if self.semantic is not None:
            self.semantic.zero_()
        return self.z_history, self.semantic

    def step(self, actions: Tensor) -> EnvStepResult:
        self.current_step += 1
        dones = self.current_step >= self.max_episode_length

        # 简单奖励：鼓励动作接近零（简化测试）
        rewards = -actions.norm(dim=-1) * 0.01

        # 随机更新 latent
        new_z = torch.randn(self.num_envs, 1, self.num_patches, self.token_dim, device=self.device) * 0.1
        self.z_history = torch.cat([self.z_history[:, 1:], new_z], dim=1)

        if self.semantic is not None:
            self.semantic = torch.randn_like(self.semantic) * 0.1

        return EnvStepResult(
            obs_z=self.z_history.clone(),
            obs_s=self.semantic.clone() if self.semantic is not None else None,
            reward=rewards,
            done=dones,
            info={"step": self.current_step.clone()},
        )

    @property
    def batch_size(self) -> int:
        return self.num_envs

    def close(self) -> None:
        pass
