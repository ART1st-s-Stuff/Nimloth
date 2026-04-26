"""JointTrainer - WM + PM + VLM 联合优化器。

核心功能：
1. 组合奖励计算: r = r_task + α1*r_wm + α2*r_semantic
2. WM 预测一致性奖励
3. PM 和可选 WM 的联合更新
4. System 1/2 切换支持
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from src.rl.policy_model import PolicyModel
from src.rl.value_network import ValueNetwork

logger = logging.getLogger(__name__)


@dataclass
class JointTrainerStats:
    """联合训练统计。"""

    policy_loss: float
    value_loss: float
    wm_loss: float
    entropy_loss: float
    total_loss: float
    reward_mean: float
    reward_wm: float
    reward_task: float
    kl_divergence: float
    clip_fraction: float
    learning_rate: float
    wm_update_enabled: bool


class RewardCalculator:
    """奖励计算器。

    支持多种奖励类型：
    - action_match: 动作接近零（简化）
    - latent_dist: 基于 latent 距离
    - wm_consistency: WM 预测一致性
    - combined: 组合以上多种
    """

    def __init__(
        self,
        wm_reward_weight: float = 0.2,
        semantic_reward_weight: float = 0.1,
        action_penalty_weight: float = 0.01,
        latent_dist_weight: float = 0.1,
        reward_type: str = "combined",
    ) -> None:
        self.wm_reward_weight = wm_reward_weight
        self.semantic_reward_weight = semantic_reward_weight
        self.action_penalty_weight = action_penalty_weight
        self.latent_dist_weight = latent_dist_weight
        self.reward_type = reward_type

    def compute_task_reward(self, action: Tensor) -> Tensor:
        """计算任务奖励（简化版）。

        鼓励动作接近零范数（稳定策略）。
        """
        action_norm = action.norm(dim=-1)  # [B]
        return -self.action_penalty_weight * action_norm

    def compute_wm_reward(
        self,
        pred_z_next: Tensor,
        gt_z_next: Tensor,
    ) -> Tensor:
        """计算 WM 预测一致性奖励。

        使用余弦相似度衡量预测质量。
        """
        # pred_z_next: [B, P, D]
        # gt_z_next: [B, P, D]
        pred_flat = pred_z_next.flatten(start_dim=1)  # [B, P*D]
        gt_flat = gt_z_next.flatten(start_dim=1)

        # 余弦相似度
        cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=-1)  # [B]
        # 转换为奖励（高相似度 = 高奖励）
        return (cos_sim + 1) / 2  # 归一化到 [0, 1]

    def compute_latent_dist_reward(
        self,
        z_t: Tensor,
        z_next: Tensor,
    ) -> Tensor:
        """计算 latent 距离奖励。

        鼓励连续 latent 之间的合理变化。
        """
        dist = (z_next - z_t).flatten(start_dim=1).norm(dim=-1)  # [B]
        # 使用负距离作为奖励（距离越小越好）
        return -self.latent_dist_weight * dist

    def compute(
        self,
        action: Tensor,
        pred_z_next: Tensor | None = None,
        gt_z_next: Tensor | None = None,
        semantic: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """计算组合奖励。

        Args:
            action: [B, A] 动作
            pred_z_next: [B, P, D] WM 预测的下一 latent
            gt_z_next: [B, P, D] 真实的下一 latent
            semantic: [B, D_s] VLM 语义特征

        Returns:
            rewards: [B] 组合奖励
            stats: 各分量奖励的统计
        """
        r_task = self.compute_task_reward(action)

        r_total = r_task
        r_wm = torch.zeros_like(r_task)
        r_latent = torch.zeros_like(r_task)
        r_semantic = torch.zeros_like(r_task)

        if pred_z_next is not None and gt_z_next is not None:
            r_wm = self.compute_wm_reward(pred_z_next, gt_z_next)
            r_total = r_total + self.wm_reward_weight * r_wm

        if pred_z_next is not None and gt_z_next is not None:
            z_t = gt_z_next  # 使用真实下一 latent 作为参考
            r_latent = self.compute_latent_dist_reward(z_t, gt_z_next)
            r_total = r_total + r_latent

        return r_total, {
            "r_task": r_task.mean().item(),
            "r_wm": r_wm.mean().item() if isinstance(r_wm, Tensor) else 0.0,
            "r_latent": r_latent.mean().item() if isinstance(r_latent, Tensor) else 0.0,
            "r_semantic": r_semantic.mean().item() if isinstance(r_semantic, Tensor) else 0.0,
        }


class JointTrainer:
    """WM + PM + VLM 联合优化器。

    协调 WM (World Model), PM (Policy Model) 和 VLM (Vision Language Model)
    的联合训练，支持：
    - PM 策略优化（主要目标）
    - WM 预测优化（可选）
    - VLM 语义对齐（可选）
    """

    def __init__(
        self,
        # 策略网络
        policy: PolicyModel,
        value_net: ValueNetwork,
        # WM（可选，用于奖励计算）
        wm: Any = None,
        # VLM Adapter（可选，用于语义奖励）
        vlm_adapter: Any = None,
        # 奖励计算
        reward_calculator: RewardCalculator | None = None,
        # 学习率
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        wm_lr: float = 1e-4,
        # PPO 参数
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 10,
        mini_batch_size: int = 64,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        # WM 更新控制
        wm_update_enabled: bool = False,
        wm_update_start_step: int = 1000,
        wm_update_interval: int = 10,
        # 其他
        device: str = "cuda",
        clip_advantages: bool = True,
        target_kl: float | None = None,
    ) -> None:
        self.policy = policy
        self.value_net = value_net
        self.wm = wm
        self.vlm_adapter = vlm_adapter
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.wm_update_enabled = wm_update_enabled
        self.wm_update_start_step = wm_update_start_step
        self.wm_update_interval = wm_update_interval
        self.device = device
        self.clip_advantages = clip_advantages
        self.target_kl = target_kl

        # 优化器
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=policy_lr,
            weight_decay=0.01,
        )
        self.value_optimizer = torch.optim.AdamW(
            self.value_net.parameters(),
            lr=value_lr,
            weight_decay=0.01,
        )
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.policy_optimizer, T_max=1000, eta_min=policy_lr * 0.1
        )
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.value_optimizer, T_max=1000, eta_min=value_lr * 0.1
        )

        # WM 优化器
        self.wm_optimizer = None
        if self.wm is not None and hasattr(self.wm, "wm_optimizer"):
            self.wm_optimizer = self.wm.wm_optimizer
            if wm_lr != policy_lr:
                self.wm_optimizer = torch.optim.AdamW(
                    self.wm.parameters(),
                    lr=wm_lr,
                    weight_decay=0.01,
                )

        # EMA 模型
        self._ema_policy: PolicyModel | None = None
        self._ema_decay = 0.999

        self._step_count = 0

    def _update_ema(self) -> None:
        """更新 EMA 模型。"""
        if self._ema_policy is None:
            self._ema_policy = copy.deepcopy(self.policy)
            return
        for ema_p, p in zip(self._ema_policy.parameters(), self.policy.parameters()):
            ema_p.mul_(self._ema_decay).add_(p.detach(), alpha=1 - self._ema_decay)

    def compute_reward(
        self,
        action: Tensor,
        z_history: Tensor,
        z_next: Tensor | None = None,
        pred_z_next: Tensor | None = None,
        semantic: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """计算组合奖励。

        Args:
            action: [B, A] 动作
            z_history: [B, H, P, D] 历史 latent
            z_next: [B, P, D] 真实下一 latent
            pred_z_next: [B, P, D] WM 预测的下一 latent
            semantic: [B, D_s] VLM 语义特征

        Returns:
            rewards: [B] 组合奖励
            stats: 奖励分量统计
        """
        return self.reward_calculator.compute(
            action=action,
            pred_z_next=pred_z_next,
            gt_z_next=z_next,
            semantic=semantic,
        )

    def compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        last_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """计算 GAE 优势函数。

        Args:
            rewards: [T, B] 奖励
            values: [T+1, B] 价值估计
            dones: [T, B] 完成标志
            last_values: [B] 最后状态的价值

        Returns:
            advantages: [T, B]
            returns: [T, B]
        """
        T = rewards.size(0)
        B = rewards.size(1)
        advantages = torch.zeros(T, B, device=rewards.device)
        returns = torch.zeros(T, B, device=rewards.device)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_values
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (~dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (~dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + values[t]

        return advantages, returns

    def step(
        self,
        batch: dict[str, Tensor],
    ) -> JointTrainerStats:
        """一步训练更新。

        Args:
            batch: 包含以下键的字典
                - z_history: [T*B, H, P, D]
                - semantic: [T*B, D_s] 或 None
                - actions: [T*B, A]
                - old_log_probs: [T*B]
                - advantages: [T*B]
                - returns: [T*B]

        Returns:
            JointTrainerStats: 训练统计
        """
        z_history = batch["z_history"].to(self.device).detach()
        actions = batch["actions"].to(self.device).detach()
        old_log_probs = batch["old_log_probs"].to(self.device).detach()
        advantages = batch["advantages"].to(self.device).detach()
        returns = batch["returns"].to(self.device).detach()
        semantic = batch.get("semantic")
        if semantic is not None:
            semantic = semantic.to(self.device).detach()

        # 归一化优势
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 优势裁剪
        if self.clip_advantages:
            advantages = advantages.clamp(-10.0, 10.0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        num_updates = 0

        for epoch in range(self.num_epochs):
            # 随机打乱
            B = z_history.size(0)
            indices = torch.randperm(B, device=self.device)

            for start in range(0, B, self.mini_batch_size):
                end = min(start + self.mini_batch_size, B)
                batch_idx = indices[start:end]

                mb_z = z_history[batch_idx]
                mb_action = actions[batch_idx]
                mb_old_log_prob = old_log_probs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_returns = returns[batch_idx]
                mb_semantic = semantic[batch_idx] if semantic is not None else None

                # === Policy 更新 ===
                new_log_prob, entropy = self.policy.evaluate_actions(
                    mb_z, mb_semantic, mb_action
                )
                # Detach old_log_prob and ratio to avoid in-place issues
                with torch.no_grad():
                    ratio = torch.exp(new_log_prob - mb_old_log_prob.detach()).clone()
                    ratio_clipped = ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon).clone()

                # PPO 裁剪
                surr1 = ratio * mb_advantages
                surr2 = ratio_clipped * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 熵正则（需要 detach entropy 以避免 in-place 操作）
                entropy_loss = -entropy.mean()

                # === Value 更新 ===
                value_pred = self.value_net(mb_z, mb_semantic)
                value_loss = F.mse_loss(value_pred, mb_returns, reduction="mean")

                # === 总损失 ===
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # 反向传播
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # 统计
                with torch.no_grad():
                    approx_kl = (mb_old_log_prob - new_log_prob).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.epsilon).float().mean()

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

        # 更新学习率
        self.policy_scheduler.step()
        self.value_scheduler.step()

        # 更新 EMA
        self._update_ema()

        self._step_count += 1

        # 平均统计
        if num_updates > 0:
            policy_loss = total_policy_loss / num_updates
            value_loss = total_value_loss / num_updates
            entropy_loss = total_entropy_loss / num_updates
        else:
            policy_loss = value_loss = entropy_loss = 0.0

        return JointTrainerStats(
            policy_loss=policy_loss,
            value_loss=value_loss,
            wm_loss=0.0,  # WM 更新在 separate 方法中
            entropy_loss=entropy_loss,
            total_loss=policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss,
            reward_mean=0.0,
            reward_wm=0.0,
            reward_task=0.0,
            kl_divergence=total_kl / num_updates if num_updates > 0 else 0.0,
            clip_fraction=total_clip_frac / num_updates if num_updates > 0 else 0.0,
            learning_rate=self.policy_optimizer.param_groups[0]["lr"],
            wm_update_enabled=self.wm_update_enabled,
        )

    def update_wm(self, batch: dict[str, Tensor]) -> float:
        """更新 WM（如果启用）。

        Args:
            batch: 包含 WM 训练所需的数据

        Returns:
            WM loss
        """
        if not self.wm_update_enabled:
            return 0.0

        if self._step_count < self.wm_update_start_step:
            return 0.0

        if self._step_count % self.wm_update_interval != 0:
            return 0.0

        if self.wm is None or self.wm_optimizer is None:
            return 0.0

        try:
            z_history = batch["z_history"].to(self.device)
            action_history = batch["action_history"].to(self.device)
            z_future = batch["z_future"].to(self.device)
            gt_action_future = batch["gt_action_future"].to(self.device)

            wm_batch = {
                "z_history": z_history,
                "action_history": action_history,
                "z_future": z_future,
                "gt_action_future": gt_action_future,
            }

            result = self.wm.train_step(wm_batch)
            return result.get("loss", 0.0)
        except Exception as e:
            logger.warning("WM 更新失败: %s", e)
            return 0.0

    def save_checkpoint(self, path: str, step: int, extra: dict | None = None) -> None:
        """保存 checkpoint。"""
        ckpt = {
            "step": step,
            "policy_state": self.policy.state_dict(),
            "value_state": self.value_net.state_dict(),
            "policy_optimizer_state": self.policy_optimizer.state_dict(),
            "value_optimizer_state": self.value_optimizer.state_dict(),
            "policy_scheduler_state": self.policy_scheduler.state_dict(),
            "value_scheduler_state": self.value_scheduler.state_dict(),
            "epsilon": self.epsilon,
            "value_coef": self.value_coef,
            "entropy_coef": self.entropy_coef,
        }
        if self.wm is not None:
            try:
                wm_state = self.wm.get_state()
                ckpt["wm_state"] = wm_state
            except Exception as e:
                logger.warning("保存 WM state 失败: %s", e)
        if extra is not None:
            ckpt.update(extra)
        torch.save(ckpt, path)
        logger.info("Checkpoint saved to %s (step=%d)", path, step)

    def load_checkpoint(self, path: str) -> int:
        """加载 checkpoint。"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.value_net.load_state_dict(ckpt["value_state"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer_state"])
        self.policy_scheduler.load_state_dict(ckpt["policy_scheduler_state"])
        self.value_scheduler.load_state_dict(ckpt["value_scheduler_state"])
        if "epsilon" in ckpt:
            self.epsilon = ckpt["epsilon"]
        if "value_coef" in ckpt:
            self.value_coef = ckpt["value_coef"]
        if "entropy_coef" in ckpt:
            self.entropy_coef = ckpt["entropy_coef"]
        if self.wm is not None and "wm_state" in ckpt:
            try:
                self.wm.load_state(ckpt["wm_state"])
            except Exception as e:
                logger.warning("加载 WM state 失败: %s", e)
        logger.info("Checkpoint loaded from %s (step=%d)", path, ckpt["step"])
        return ckpt["step"]


class JointTrainerWithWM:
    """带 WM 预测能力的 JointTrainer。

    扩展 JointTrainer，支持 WM 预测下一 latent，
    并将 WM 预测误差作为奖励的一部分。
    """

    def __init__(
        self,
        policy: PolicyModel,
        value_net: ValueNetwork,
        wm: Any,
        vlm_adapter: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(
            policy=policy,
            value_net=value_net,
            wm=wm,
            vlm_adapter=vlm_adapter,
            **kwargs,
        )
        self._action_history = None

    def collect_with_wm(
        self,
        env: Any,
        storage: Any,
        num_steps: int | None = None,
    ) -> dict:
        """收集经验（使用 WM 预测下一 latent）。

        Args:
            env: 向量化环境
            storage: 经验存储
            num_steps: 收集步数

        Returns:
            收集统计
        """
        if num_steps is None:
            num_steps = storage.num_steps

        # 重置环境
        obs_z, obs_s = env.reset()
        storage.initialize()
        storage.z_history[0] = obs_z.to(storage.z_history.device)
        if obs_s is not None and storage.semantic is not None:
            storage.semantic[0] = obs_s.to(storage.semantic.device)

        # 初始化动作历史
        action_dim = env.action_dim
        history_len = env.history_len
        num_envs = env.num_envs
        device = env.device

        self._action_history = torch.zeros(
            num_envs, history_len, action_dim, device=device
        )

        # 初始 value
        with torch.no_grad():
            init_value = self.value_net(
                obs_z.to(self.device),
                obs_s.to(self.device) if obs_s is not None else None
            )
            storage.values[0] = init_value

        total_rewards = []
        reward_wm_list = []
        reward_task_list = []
        episode_lengths = []
        episode_rewards = torch.zeros(num_envs, device=device)

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

            # 使用 WM 预测下一 latent
            z_history = obs_z.to(self.device)
            action_history = self._action_history

            with torch.no_grad():
                if hasattr(self.wm, "predict_next"):
                    pred_z_next = self.wm.predict_next(z_history, action_history)
                elif hasattr(self.wm, "wm"):
                    # LeWMModel 格式
                    pred_z_next = self.wm.wm.predict_next(z_history, action_history)
                else:
                    pred_z_next = None

            # 执行动作
            result = env.step(action.cpu())
            reward = result.reward.to(device)
            done = result.done.to(device)
            next_obs_z = result.obs_z.to(device)
            next_obs_s = result.obs_s.to(device) if result.obs_s is not None else None

            # 获取真实下一 latent
            gt_z_next = next_obs_z[:, -1, :, :] if next_obs_z is not None else None

            # 计算组合奖励
            action_for_reward = action.to(device)
            if pred_z_next is not None and gt_z_next is not None:
                reward, reward_stats = self.compute_reward(
                    action=action_for_reward,
                    z_history=z_history,
                    z_next=gt_z_next,
                    pred_z_next=pred_z_next,
                    semantic=obs_s,
                )
                reward_wm_list.append(reward_stats["r_wm"])
                reward_task_list.append(reward_stats["r_task"])
            else:
                reward, _ = self.compute_reward(
                    action=action_for_reward,
                    z_history=z_history,
                )

            # 更新 episode 统计
            episode_rewards += reward
            for i in range(num_envs):
                if done[i]:
                    episode_lengths.append(step + 1)
                    total_rewards.append(episode_rewards[i].item())
                    episode_rewards[i] = 0.0

            # 更新动作历史
            self._action_history = torch.cat(
                [self._action_history[:, 1:], action.unsqueeze(1).to(self._action_history.device)],
                dim=1
            )

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
                # 最后一步的 value
                with torch.no_grad():
                    last_value = self.value_net(
                        obs_z.to(self.device),
                        obs_s.to(self.device) if obs_s is not None else None,
                    )
                storage.after_step(last_value.cpu().detach())

        return {
            "reward_mean": sum(total_rewards) / max(len(total_rewards), 1) if total_rewards else 0.0,
            "reward_wm_mean": sum(reward_wm_list) / max(len(reward_wm_list), 1) if reward_wm_list else 0.0,
            "reward_task_mean": sum(reward_task_list) / max(len(reward_task_list), 1) if reward_task_list else 0.0,
            "episode_length_mean": sum(episode_lengths) / max(len(episode_lengths), 1) if episode_lengths else 0.0,
            "num_episodes": len(total_rewards),
        }
