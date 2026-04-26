"""PPO Learner - PPO 训练器。

核心功能：
1. collect_experience: 与环境交互收集 rollout
2. update: 执行 PPO 更新
3. compute_advantages: 计算 GAE 优势函数
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.rl.storage import RolloutStorage

if TYPE_CHECKING:
    from src.rl.policy_model import PolicyModel
    from src.rl.value_network import ValueNetwork
    from src.rl.vec_env import LatentVecEnv

logger = logging.getLogger(__name__)


@dataclass
class PPOTrainingStats:
    """PPO 训练统计。"""

    policy_loss: float
    value_loss: float
    entropy_loss: float
    total_loss: float
    kl_divergence: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float


class PPOLearner:
    """Proximal Policy Optimization 训练器。"""

    def __init__(
        self,
        policy: PolicyModel,
        value_net: ValueNetwork,
        lr: float = 3e-4,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        mini_batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_advantages: bool = True,
        clip_low: float = -10.0,
        clip_high: float = 10.0,
        target_kl: float | None = None,
        device: str = "cuda",
    ) -> None:
        """
        Args:
            policy: 策略网络
            value_net: Value 网络
            lr: 学习率
            epsilon: PPO 裁剪参数
            value_coef: Value loss 系数
            entropy_coef: 熵正则系数
            max_grad_norm: 梯度裁剪阈值
            num_epochs: 每次 rollout 的更新轮数
            mini_batch_size: Mini batch 大小
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数
            clip_advantages: 是否裁剪 advantages
            clip_low: advantages 裁剪下界
            clip_high: advantages 裁剪上界
            target_kl: KL 散度目标（用于早停）
            device: 计算设备
        """
        self.policy = policy
        self.value_net = value_net
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_advantages = clip_advantages
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.target_kl = target_kl
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=lr * 0.1
        )

        # 统计
        self._step_count = 0

    def collect_experience(
        self,
        env: LatentVecEnv,
        storage: RolloutStorage,
        num_steps: int | None = None,
    ) -> dict:
        """收集 rollout 经验。

        Args:
            env: 向量化环境
            storage: 经验存储
            num_steps: 收集步数（默认为 storage.num_steps）

        Returns:
            收集统计信息
        """
        if num_steps is None:
            num_steps = storage.num_steps

        # 重置环境
        obs_z, obs_s = env.reset()
        storage.initialize()
        storage.z_history[0] = obs_z.to(storage.z_history.device)
        if obs_s is not None and storage.semantic is not None:
            storage.semantic[0] = obs_s.to(storage.semantic.device)

        # 初始 value
        with torch.no_grad():
            init_value = self.value_net(obs_z.to(self.device), obs_s.to(self.device) if obs_s is not None else None)
            storage.values[0] = init_value

        total_rewards = []
        episode_lengths = []
        episode_rewards = torch.zeros(env.num_envs, device=self.device)
        episode_count = torch.zeros(env.num_envs, device=self.device)

        for step in range(num_steps):
            # 选择动作
            with torch.no_grad():
                action, log_prob, _ = self.policy.act(
                    obs_z.to(self.device),
                    obs_s.to(self.device) if obs_s is not None else None,
                    deterministic=False,
                )
                value = self.value_net(
                    obs_z.to(self.device),
                    obs_s.to(self.device) if obs_s is not None else None,
                )

            # 执行动作
            result = env.step(action.cpu())
            reward = result.reward.to(self.device)
            done = result.done.to(self.device)
            next_obs_z = result.obs_z.to(self.device)
            next_obs_s = result.obs_s.to(self.device) if result.obs_s is not None else None

            # 更新 episode 统计
            episode_rewards += reward
            for i in range(env.num_envs):
                if done[i]:
                    episode_lengths.append(step + 1)
                    total_rewards.append(episode_rewards[i].item())
                    episode_rewards[i] = 0.0
                    episode_count[i] += 1

            # 存储
            storage.insert(
                z_history=obs_z,
                action=action.cpu(),
                reward=reward.cpu(),
                done=done.cpu(),
                value=value.cpu().detach(),
                log_prob=log_prob.cpu().detach(),
                semantic=obs_s.cpu() if obs_s is not None else None,
            )

            obs_z = next_obs_z
            obs_s = next_obs_s

            if step == num_steps - 1:
                # 最后一步的 value（用于 bootstrap）
                with torch.no_grad():
                    last_value = self.value_net(
                        obs_z.to(self.device),
                        obs_s.to(self.device) if obs_s is not None else None,
                    )
                storage.after_step(last_value.cpu().detach())

        # 计算 advantages
        storage.compute_returns(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_advantages=True,
        )

        return {
            "reward_mean": sum(total_rewards) / max(len(total_rewards), 1),
            "reward_max": max(total_rewards) if total_rewards else 0.0,
            "episode_length_mean": sum(episode_lengths) / max(len(episode_lengths), 1),
            "num_episodes": len(total_rewards),
        }

    def update(self, storage: RolloutStorage) -> PPOTrainingStats:
        """执行 PPO 更新。

        Args:
            storage: 经验存储（包含 advantages 和 returns）

        Returns:
            训练统计
        """
        stats = PPOTrainingStats(
            policy_loss=0.0,
            value_loss=0.0,
            entropy_loss=0.0,
            total_loss=0.0,
            kl_divergence=0.0,
            clip_fraction=0.0,
            explained_variance=0.0,
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )

        num_updates = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0

        for epoch in range(self.num_epochs):
            for batch in storage.feed_forward_generator(self.mini_batch_size):
                batch_z = batch["z_history"].to(self.device)
                batch_action = batch["actions"].to(self.device)
                batch_old_log_prob = batch["old_log_probs"].to(self.device)
                batch_advantages = batch["advantages"].to(self.device)
                batch_returns = batch["returns"].to(self.device)
                batch_semantic = batch.get("semantic")
                if batch_semantic is not None:
                    batch_semantic = batch_semantic.to(self.device)

                # 优势裁剪
                if self.clip_advantages:
                    batch_advantages = batch_advantages.clamp(self.clip_low, self.clip_high)

                # === Policy 更新 ===
                new_log_prob, entropy = self.policy.evaluate_actions(
                    batch_z, batch_semantic, batch_action
                )
                ratio = torch.exp(new_log_prob - batch_old_log_prob)

                # PPO 裁剪
                surr1 = ratio * batch_advantages
                surr2 = ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 熵正则
                entropy_loss = -entropy.mean()

                # === Value 更新 ===
                value_pred = self.value_net(batch_z, batch_semantic)
                value_loss = torch.nn.functional.mse_loss(
                    value_pred, batch_returns, reduction="mean"
                )

                # === 总损失 ===
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # 统计
                with torch.no_grad():
                    approx_kl = (batch_old_log_prob - new_log_prob).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.epsilon).float().mean()

                    # explained variance
                    var_returns = batch_returns.var()
                    if var_returns > 1e-8:
                        ev = 1 - (value_pred - batch_returns).var() / var_returns
                    else:
                        ev = 0.0

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                num_updates += 1

                # KL 早停
                if self.target_kl is not None and approx_kl.item() > 1.5 * self.target_kl:
                    logger.info(
                        "Early stopping: KL=%.4f > target=%.4f (epoch %d)",
                        approx_kl.item(), self.target_kl, epoch
                    )
                    break

        # 平均统计
        if num_updates > 0:
            stats.policy_loss = total_policy_loss / num_updates
            stats.value_loss = total_value_loss / num_updates
            stats.entropy_loss = total_entropy_loss / num_updates
            stats.total_loss = stats.policy_loss + self.value_coef * stats.value_loss + self.entropy_coef * stats.entropy_loss
            stats.kl_divergence = total_kl / num_updates
            stats.clip_fraction = total_clip_frac / num_updates
            stats.learning_rate = self.optimizer.param_groups[0]["lr"]

        self._step_count += 1
        self.lr_scheduler.step()

        return stats

    def save_checkpoint(self, path: str, step: int, extra: dict | None = None) -> None:
        """保存 checkpoint。"""
        ckpt = {
            "step": step,
            "policy_state": self.policy.state_dict(),
            "value_state": self.value_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "lr_scheduler_state": self.lr_scheduler.state_dict(),
            "epsilon": self.epsilon,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
        }
        if extra is not None:
            ckpt.update(extra)
        torch.save(ckpt, path)
        logger.info("Checkpoint saved to %s (step=%d)", path, step)

    def load_checkpoint(self, path: str) -> int:
        """加载 checkpoint。"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.value_net.load_state_dict(ckpt["value_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state"])
        if "epsilon" in ckpt:
            self.epsilon = ckpt["epsilon"]
        logger.info("Checkpoint loaded from %s (step=%d)", path, ckpt["step"])
        return ckpt["step"]
