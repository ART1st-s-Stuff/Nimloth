"""RolloutStorage - 经验回放存储。

用于存储 PPO 训练过程中的 rollout 数据：
observations, actions, rewards, dones, values, log_probs 等。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class RolloutStorage:
    """存储一个 rollout 的所有数据。

    使用循环缓冲区，支持固定步数和并行环境。
    """

    num_steps: int
    num_envs: int
    latent_dim: int
    action_dim: int
    semantic_dim: int = 0
    num_patches: int = 0
    token_dim: int = 0
    history_len: int = 0
    device: str = "cuda"

    # 循环缓冲区
    z_history: Tensor | None = None
    semantic: Tensor | None = None
    actions: Tensor | None = None
    rewards: Tensor | None = None
    dones: Tensor | None = None
    values: Tensor | None = None
    log_probs: Tensor | None = None
    advantages: Tensor | None = None
    returns: Tensor | None = None

    # 当前写入位置
    step: int = 0
    _initialized: bool = field(default=False, init=False)

    def _shape_z(self) -> list[int]:
        """z_history 的形状。"""
        if self.num_patches > 0 and self.token_dim > 0:
            return [self.num_steps, self.num_envs, self.history_len, self.num_patches, self.token_dim]
        # 扁平化格式
        return [self.num_steps, self.num_envs, self.latent_dim]

    def _shape_semantic(self) -> list[int]:
        if self.semantic_dim > 0:
            return [self.num_steps, self.num_envs, self.semantic_dim]
        return [0]

    def initialize(self) -> None:
        """初始化所有缓冲区。"""
        if self._initialized:
            return

        self.z_history = torch.zeros(
            self.num_steps + 1, self.num_envs,
            self.history_len, self.num_patches, self.token_dim,
            device=self.device, dtype=torch.float32,
        )
        if self.semantic_dim > 0:
            self.semantic = torch.zeros(
                self.num_steps + 1, self.num_envs, self.semantic_dim,
                device=self.device, dtype=torch.float32,
            )
        else:
            self.semantic = None

        self.actions = torch.zeros(
            self.num_steps, self.num_envs, self.action_dim,
            device=self.device, dtype=torch.float32,
        )
        self.rewards = torch.zeros(
            self.num_steps, self.num_envs,
            device=self.device, dtype=torch.float32,
        )
        self.dones = torch.zeros(
            self.num_steps, self.num_envs,
            device=self.device, dtype=torch.bool,
        )
        self.values = torch.zeros(
            self.num_steps + 1, self.num_envs,
            device=self.device, dtype=torch.float32,
        )
        self.log_probs = torch.zeros(
            self.num_steps, self.num_envs,
            device=self.device, dtype=torch.float32,
        )
        self.advantages = torch.zeros(
            self.num_steps, self.num_envs,
            device=self.device, dtype=torch.float32,
        )
        self.returns = torch.zeros(
            self.num_steps + 1, self.num_envs,
            device=self.device, dtype=torch.float32,
        )

        self._initialized = True
        self.step = 0

    def insert(
        self,
        z_history: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
        value: Tensor,
        log_prob: Tensor,
        semantic: Tensor | None = None,
    ) -> None:
        """插入一个时间步的数据。

        Args:
            z_history: [num_envs, H, P, D] 或 [num_envs, latent_dim]
            action: [num_envs, A]
            reward: [num_envs]
            done: [num_envs]
            value: [num_envs]
            log_prob: [num_envs]
            semantic: [num_envs, D_s]
        """
        if not self._initialized:
            self.initialize()

        s = self.step
        self.z_history[s] = z_history.to(self.z_history.device)
        if semantic is not None and self.semantic is not None:
            self.semantic[s] = semantic.to(self.semantic.device)
        self.actions[s] = action.to(self.actions.device)
        self.rewards[s] = reward.to(self.rewards.device)
        self.dones[s] = done.to(self.dones.device)
        self.values[s] = value.to(self.values.device)
        self.log_probs[s] = log_prob.to(self.log_probs.device)

        self.step = (self.step + 1) % self.num_steps

    def after_step(self, last_values: Tensor) -> None:
        """完成一个 rollout 后，计算 advantages 和 returns。

        Args:
            last_values: [num_envs] 最后状态的价值估计
        """
        self.values[self.step] = last_values.to(self.values.device)
        self.compute_returns(gamma=0.99, gae_lambda=0.95)

    def compute_returns(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = False,
    ) -> None:
        """计算 GAE 优势函数和回报。

        使用 Generalized Advantage Estimation (GAE):
        A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}
        其中 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

        Returns 使用 TD(0) 估计: R_t = r_t + gamma * R_{t+1}
        """
        gae = 0.0
        for t in reversed(range(self.num_steps)):
            # next_value = V(s_{t+1})，最后一步用 last_values
            if t == self.num_steps - 1:
                next_value = self.values[self.step]  # 最后状态的价值
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * (~self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (~self.dones[t]) * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

        # 优势归一化（可选）
        if normalize_advantages and self.num_envs > 1:
            flat_adv = self.advantages.flatten()
            mean = flat_adv.mean()
            std = flat_adv.std()
            # 使用 clone 避免 in-place 操作
            self.advantages = (self.advantages - mean) / (std + 1e-8)

    def get_statistics(self) -> dict[str, float]:
        """返回当前 rollout 的统计信息。"""
        if not self._initialized:
            return {}
        return {
            "reward_mean": self.rewards.mean().item(),
            "reward_std": self.rewards.std().item(),
            "reward_min": self.rewards.min().item(),
            "reward_max": self.rewards.max().item(),
            "advantage_mean": self.advantages.mean().item(),
            "advantage_std": self.advantages.std().item(),
            "return_mean": self.returns.mean().item(),
            "value_loss_init": (self.returns[:-1] - self.values[:-1]).pow(2).mean().item(),
        }

    def feed_forward_generator(self, mini_batch_size: int):
        """生成器：返回 feed-forward 方式的 mini-batches。

        将 rollout 数据展平后随机采样。

        Args:
            mini_batch_size: 每个 mini-batch 的样本数

        Yields:
            dict: 包含每个 batch 的数据
        """
        if not self._initialized:
            raise RuntimeError("RolloutStorage 未初始化")

        num_samples = self.num_steps * self.num_envs
        indices = torch.randperm(num_samples, device=self.device)

        for start in range(0, num_samples, mini_batch_size):
            end = min(start + mini_batch_size, num_samples)
            batch_indices = indices[start:end]

            # 解析 indices
            t_indices = batch_indices // self.num_envs
            env_indices = batch_indices % self.num_envs

            # 收集数据
            batch = {
                "z_history": self.z_history[t_indices, env_indices],
                "actions": self.actions[t_indices, env_indices],
                "old_log_probs": self.log_probs[t_indices, env_indices],
                "advantages": self.advantages[t_indices, env_indices],
                "returns": self.returns[t_indices, env_indices],
            }
            if self.semantic is not None:
                batch["semantic"] = self.semantic[t_indices, env_indices]

            yield batch

    def recurrent_generator(self, mini_batch_size: int, sequence_length: int):
        """生成器：返回 recurrent (LSTM) 方式的 mini-batches。

        Args:
            mini_batch_size: 每个 mini-batch 的 batch 数（不是样本数）
            sequence_length: 每个序列的长度

        Yields:
            dict: 包含每个 sequence batch 的数据
        """
        # TODO: 支持 RNN hidden state
        raise NotImplementedError("Recurrent generator 尚未实现")

    def to(self, device: str) -> "RolloutStorage":
        """将存储移动到指定设备。"""
        self.device = device
        if self.z_history is not None:
            self.z_history = self.z_history.to(device)
        if self.semantic is not None:
            self.semantic = self.semantic.to(device)
        if self.actions is not None:
            self.actions = self.actions.to(device)
        if self.rewards is not None:
            self.rewards = self.rewards.to(device)
        if self.dones is not None:
            self.dones = self.dones.to(device)
        if self.values is not None:
            self.values = self.values.to(device)
        if self.log_probs is not None:
            self.log_probs = self.log_probs.to(device)
        if self.advantages is not None:
            self.advantages = self.advantages.to(device)
        if self.returns is not None:
            self.returns = self.returns.to(device)
        return self
