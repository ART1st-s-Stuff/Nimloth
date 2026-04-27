"""Joint Training Entry - WM + PM + VLM 联合训练入口。

使用 JointTrainer 实现：
- PM 策略优化（主要）
- WM 预测一致性奖励
- 可选的 WM 在线更新

使用方法:
    uv run python src/rl/train_joint.py
    uv run python src/rl/train_joint.py rl.num_iterations=100 rl.wm_update_enabled=true
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.joint_trainer import JointTrainer, RewardCalculator, JointTrainerStats
from src.rl.policy_model import PolicyModel
from src.rl.storage import RolloutStorage
from src.rl.value_network import ValueNetwork
from src.rl.vec_env import LatentVecEnv, DummyVecEnv
from src.rl.train_utils import (
    EnvConfig,
    RLModelConfig,
    setup_logging,
    get_device,
    build_models_with_env,
    build_environment,
)

logger = logging.getLogger(__name__)


@dataclass
class JointRLConfig:
    """联合训练配置。"""

    # 算法选择
    algorithm: str = "ppo"

    # 环境参数
    num_envs: int = 8
    num_steps: int = 128
    max_episode_length: int = 50

    # 数据源
    manifest_path: str = "data/ai2thor/train"
    latent_cache_dir: str = "data/ai2thor/latents"

    # PPO 参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    mini_batch_size: int = 64

    # 奖励权重
    wm_reward_weight: float = 0.2
    semantic_reward_weight: float = 0.1
    action_penalty_weight: float = 0.01
    latent_dist_weight: float = 0.1
    reward_type: str = "combined"

    # 模型参数
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    action_std_init: float = 0.5

    # VLM 语义
    use_vlm: bool = False
    semantic_dim: int = 0

    # 训练控制
    num_iterations: int = 1000
    eval_every: int = 10
    save_every: int = 50
    wm_freeze_steps: int = 500

    # WM 联合训练
    wm_update_enabled: bool = False
    wm_update_interval: int = 10
    wm_lr: float = 1e-4

    # 输出
    outputs_root: str = "models/rl_joint"
    run_name: str = "rl_joint_default"

    # 设备
    device: str = "cuda"


@dataclass
class JointEnvConfig:
    """环境配置。"""

    latent_dim: int = 512
    action_dim: int = 3
    num_patches: int = 16
    token_dim: int = 384
    history_len: int = 4
    semantic_dim: int = 0
    max_episode_length: int = 50


@dataclass
class JointWMConfig:
    """WM 配置。"""

    type: str = "cfm"
    latent_dim: int = 512
    num_patches: int = 16
    token_dim: int = 384
    history_len: int = 4
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    action_dim: int = 3
    ckpt_path: str | None = None


def build_reward_calculator(cfg: JointRLConfig) -> RewardCalculator:
    """构建奖励计算器。"""
    return RewardCalculator(
        wm_reward_weight=cfg.wm_reward_weight,
        semantic_reward_weight=cfg.semantic_reward_weight,
        action_penalty_weight=cfg.action_penalty_weight,
        latent_dist_weight=cfg.latent_dist_weight,
        reward_type=cfg.reward_type,
    )


def build_models(
    cfg: JointRLConfig,
    env_cfg: JointEnvConfig,
    device: str,
) -> tuple[PolicyModel, ValueNetwork]:
    """构建策略网络和 Value 网络。"""
    expected_latent_dim = env_cfg.num_patches * env_cfg.token_dim

    policy = PolicyModel(
        latent_dim=expected_latent_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        history_len=env_cfg.history_len,
        num_patches=env_cfg.num_patches,
        token_dim=env_cfg.token_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        semantic_dim=env_cfg.semantic_dim if cfg.use_vlm else 0,
        action_std_init=cfg.action_std_init,
        use_vlm=cfg.use_vlm,
    )

    value_net = ValueNetwork(
        latent_dim=expected_latent_dim,
        hidden_dim=cfg.hidden_dim,
        history_len=env_cfg.history_len,
        num_patches=env_cfg.num_patches,
        token_dim=env_cfg.token_dim,
        num_layers=max(2, cfg.num_layers // 2),
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        semantic_dim=env_cfg.semantic_dim if cfg.use_vlm else 0,
        use_vlm=cfg.use_vlm,
    )

    return policy.to(device), value_net.to(device)


def build_models_from_env(
    cfg: JointRLConfig,
    env: LatentVecEnv,
    device: str,
) -> tuple[PolicyModel, ValueNetwork]:
    """使用环境维度构建模型。"""
    actual_patches = env.num_patches
    actual_token_dim = env.token_dim
    expected_latent_dim = actual_patches * actual_token_dim

    policy = PolicyModel(
        latent_dim=expected_latent_dim,
        action_dim=env.action_dim,
        hidden_dim=cfg.hidden_dim,
        history_len=env.history_len,
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        semantic_dim=env.semantic_dim if cfg.use_vlm else 0,
        action_std_init=cfg.action_std_init,
        use_vlm=cfg.use_vlm,
    )

    value_net = ValueNetwork(
        latent_dim=expected_latent_dim,
        hidden_dim=cfg.hidden_dim,
        history_len=env.history_len,
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=max(2, cfg.num_layers // 2),
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        semantic_dim=env.semantic_dim if cfg.use_vlm else 0,
        use_vlm=cfg.use_vlm,
    )

    return policy.to(device), value_net.to(device)


def build_environment(
    cfg: JointRLConfig,
    env_cfg: JointEnvConfig,
    device: str,
) -> LatentVecEnv | DummyVecEnv:
    """构建环境。"""
    manifest_path = Path(cfg.manifest_path)
    latent_cache_dir = Path(cfg.latent_cache_dir)

    if not manifest_path.exists() or not latent_cache_dir.exists():
        logger.warning(
            "数据不存在，使用 DummyVecEnv 进行测试。manifest=%s, cache=%s",
            manifest_path, latent_cache_dir
        )
        return DummyVecEnv(
            latent_dim=env_cfg.latent_dim,
            action_dim=env_cfg.action_dim,
            num_patches=env_cfg.num_patches,
            token_dim=env_cfg.token_dim,
            history_len=env_cfg.history_len,
            num_envs=cfg.num_envs,
            semantic_dim=env_cfg.semantic_dim,
            device=device,
            max_episode_length=cfg.max_episode_length,
        )

    return LatentVecEnv(
        manifest_path=str(manifest_path),
        latent_cache_dir=str(latent_cache_dir),
        num_envs=cfg.num_envs,
        history_len=env_cfg.history_len,
        num_patches=env_cfg.num_patches,
        token_dim=env_cfg.token_dim,
        action_dim=env_cfg.action_dim,
        semantic_dim=env_cfg.semantic_dim if cfg.use_vlm else 0,
        device=device,
        max_episode_length=cfg.max_episode_length,
        reward_type=cfg.reward_type,
    )


def collect_experience(
    trainer: JointTrainer,
    env: LatentVecEnv | DummyVecEnv,
    storage: RolloutStorage,
    num_steps: int | None = None,
) -> dict:
    """收集经验。"""
    if num_steps is None:
        num_steps = storage.num_steps

    device = trainer.device

    # 重置环境
    obs_z, obs_s = env.reset()
    storage.initialize()
    storage.z_history[0] = obs_z.to(storage.z_history.device)
    if obs_s is not None and storage.semantic is not None:
        storage.semantic[0] = obs_s.to(storage.semantic.device)

    # 初始 value
    with torch.no_grad():
        init_value = trainer.value_net(
            obs_z.to(device),
            obs_s.to(device) if obs_s is not None else None
        )
        storage.values[0] = init_value

    total_rewards = []
    episode_lengths = []
    episode_rewards = torch.zeros(env.num_envs, device=device)

    for step in range(num_steps):
        # 选择动作
        with torch.no_grad():
            action, log_prob, _ = trainer.policy.act(
                obs_z.to(device),
                obs_s.to(device) if obs_s is not None else None,
                deterministic=False,
            )
            value = trainer.value_net(
                obs_z.to(device),
                obs_s.to(device) if obs_s is not None else None,
            )

        # 执行动作
        result = env.step(action.cpu())
        reward = result.reward.to(device)
        done = result.done.to(device)
        next_obs_z = result.obs_z.to(device)
        next_obs_s = result.obs_s.to(device) if result.obs_s is not None else None

        # 更新 episode 统计
        episode_rewards += reward
        for i in range(env.num_envs):
            if done[i]:
                episode_lengths.append(step + 1)
                total_rewards.append(episode_rewards[i].item())
                episode_rewards[i] = 0.0

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
                last_value = trainer.value_net(
                    obs_z.to(device),
                    obs_s.to(device) if obs_s is not None else None,
                )
            storage.after_step(last_value.cpu().detach())

    return {
        "reward_mean": sum(total_rewards) / max(len(total_rewards), 1) if total_rewards else 0.0,
        "reward_max": max(total_rewards) if total_rewards else 0.0,
        "episode_length_mean": sum(episode_lengths) / max(len(episode_lengths), 1) if episode_lengths else 0.0,
        "num_episodes": len(total_rewards),
    }


def print_train_summary(
    iteration: int,
    collect_stats: dict,
    train_stats: JointTrainerStats,
    storage_stats: dict,
    elapsed: float,
    wm_loss: float = 0.0,
) -> None:
    """打印训练摘要。"""
    logger.info(
        "Iter %d | Reward=%.3f | PolicyLoss=%.4f | ValueLoss=%.4f | "
        "WMLoss=%.4f | EntropyLoss=%.4f | KL=%.4f | ClipFrac=%.3f | "
        "LR=%.6f | Time=%.1fs",
        iteration,
        collect_stats.get("reward_mean", 0.0),
        train_stats.policy_loss,
        train_stats.value_loss,
        wm_loss,
        train_stats.entropy_loss,
        train_stats.kl_divergence,
        train_stats.clip_fraction,
        train_stats.learning_rate,
        elapsed,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="rl_joint")
def main(cfg: DictConfig) -> None:
    """主训练循环。"""
    # 解析配置
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    rl_cfg = JointRLConfig(**cfg_dict.get("rl", {}))
    env_cfg = JointEnvConfig(**cfg_dict.get("env", {}))
    wm_cfg = JointWMConfig(**cfg_dict.get("wm", {}))

    # 覆盖设备
    device = "cuda" if torch.cuda.is_available() and rl_cfg.device == "cuda" else "cpu"
    logger.info("使用设备: %s", device)
    logger.info("RL 配置: %s", OmegaConf.to_yaml(rl_cfg))
    logger.info("环境配置: %s", OmegaConf.to_yaml(env_cfg))

    # 设置日志和输出目录
    run_dir = setup_logging(rl_cfg.run_name, rl_cfg.outputs_root)
    logger.info("输出目录: %s", run_dir)

    # 保存配置
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    # 构建环境（先构建，以便获取实际的 num_patches/token_dim）
    env = build_environment(rl_cfg, env_cfg, device)
    logger.info("环境检测到 latent 维度: patches=%d, token_dim=%d", env.num_patches, env.token_dim)

    # 构建模型
    policy, value_net = build_models_from_env(rl_cfg, env, device)
    num_params = sum(p.numel() for p in policy.parameters())
    num_value_params = sum(p.numel() for p in value_net.parameters())
    logger.info("Policy 参数: %d, Value 参数: %d", num_params, num_value_params)

    # 构建奖励计算器
    reward_calculator = build_reward_calculator(rl_cfg)

    # 构建存储
    storage = RolloutStorage(
        num_steps=rl_cfg.num_steps,
        num_envs=rl_cfg.num_envs,
        latent_dim=env_cfg.latent_dim,
        action_dim=env_cfg.action_dim,
        semantic_dim=env_cfg.semantic_dim if rl_cfg.use_vlm else 0,
        num_patches=env.num_patches,
        token_dim=env.token_dim,
        history_len=env_cfg.history_len,
        device=device,
    )

    # 构建 JointTrainer
    trainer = JointTrainer(
        policy=policy,
        value_net=value_net,
        wm=None,  # 暂不使用 WM
        vlm_adapter=None,
        reward_calculator=reward_calculator,
        policy_lr=rl_cfg.lr,
        value_lr=rl_cfg.lr * 3,
        wm_lr=rl_cfg.wm_lr,
        epsilon=rl_cfg.epsilon,
        value_coef=rl_cfg.value_coef,
        entropy_coef=rl_cfg.entropy_coef,
        max_grad_norm=rl_cfg.max_grad_norm,
        num_epochs=rl_cfg.num_epochs,
        mini_batch_size=rl_cfg.mini_batch_size,
        gamma=rl_cfg.gamma,
        gae_lambda=rl_cfg.gae_lambda,
        wm_update_enabled=rl_cfg.wm_update_enabled,
        wm_update_start_step=rl_cfg.wm_freeze_steps,
        wm_update_interval=rl_cfg.wm_update_interval,
        device=device,
    )

    # 训练循环
    start_time = time.time()
    global_step = 0

    logger.info("开始联合训练 (%d iterations)...", rl_cfg.num_iterations)

    for iteration in range(1, rl_cfg.num_iterations + 1):
        iter_start = time.time()

        # 1. 收集经验
        collect_start = time.time()
        collect_stats = collect_experience(trainer, env, storage)
        collect_time = time.time() - collect_start

        # 2. 计算优势
        with torch.no_grad():
            T, B = storage.num_steps, storage.num_envs
            advantages, returns = trainer.compute_advantages(
                storage.rewards[:-1],  # [T-1, B]
                storage.values[:-1],  # [T, B]
                storage.dones[:-1],   # [T-1, B]
                storage.values[-1],  # [B]
            )

            # 展平存储到 batch (所有使用 T-1 以保持一致)
            T_B = (T - 1) * B
            flat_advantages = advantages.flatten()
            flat_returns = returns.flatten()

        # 准备 batch (所有使用 T-1 以保持与 advantages 一致)
        flat_z_history = storage.z_history[:-2].reshape(T_B, env.history_len, env.num_patches, env.token_dim)
        flat_actions = storage.actions[:-1].reshape(T_B, env_cfg.action_dim)
        flat_old_log_probs = storage.log_probs[:-1].reshape(T_B)
        flat_semantic = storage.semantic[:-2].reshape(T_B, env_cfg.semantic_dim) if storage.semantic is not None else None

        batch = {
            "z_history": flat_z_history,
            "actions": flat_actions,
            "old_log_probs": flat_old_log_probs,
            "advantages": flat_advantages,
            "returns": flat_returns,
        }
        if flat_semantic is not None:
            batch["semantic"] = flat_semantic

        # 3. 训练更新
        train_start = time.time()
        train_stats = trainer.step(batch)
        train_time = time.time() - train_start

        global_step += rl_cfg.num_steps * rl_cfg.num_envs

        # 4. 日志
        storage_stats = storage.get_statistics()
        elapsed = time.time() - iter_start

        if iteration % rl_cfg.eval_every == 0 or iteration == 1:
            print_train_summary(
                iteration, collect_stats, train_stats, storage_stats, elapsed
            )

        # 5. 保存 checkpoint
        if iteration % rl_cfg.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_{iteration:06d}.pt"
            trainer.save_checkpoint(
                str(ckpt_path),
                step=iteration,
                extra={
                    "collect_stats": collect_stats,
                    "storage_stats": storage_stats,
                    "total_steps": global_step,
                },
            )

    # 最终保存
    final_path = run_dir / "checkpoint_final.pt"
    trainer.save_checkpoint(
        str(final_path),
        step=rl_cfg.num_iterations,
        extra={"total_steps": global_step},
    )

    total_time = time.time() - start_time
    logger.info(
        "训练完成！总计 %d iterations, %d steps, %.1f 秒",
        rl_cfg.num_iterations, global_step, total_time
    )

    env.close()


if __name__ == "__main__":
    main()
