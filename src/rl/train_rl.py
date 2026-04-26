"""RL 训练入口。

支持 PPO 训练，使用 latent cache 数据作为环境。
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

from src.rl.policy_model import PolicyModel
from src.rl.ppo_learner import PPOLearner
from src.rl.storage import RolloutStorage
from src.rl.value_network import ValueNetwork
from src.rl.vec_env import DummyVecEnv, LatentVecEnv

logger = logging.getLogger(__name__)

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class RLConfig:
    """RL 训练配置。"""

    # 算法选择
    algorithm: str = "ppo"
    # 环境参数
    num_envs: int = 16
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
    # 奖励类型
    reward_type: str = "action_match"
    # 设备
    device: str = "cuda"
    # 训练步数
    num_iterations: int = 1000
    eval_every: int = 10
    save_every: int = 50
    # 输出
    outputs_root: str = "models"
    run_name: str = "rl_ppo"
    # 模型参数
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    action_std_init: float = 0.5
    # VLM 语义
    use_vlm: bool = False
    semantic_dim: int = 0
    # 共享编码器
    shared_encoder: bool = True
    # Warmup
    wm_freeze_steps: int = 1000


@dataclass
class WMConfig:
    """WM 配置（用于获取 latent 维度）。"""

    latent_dim: int = 512
    num_patches: int = 0
    token_dim: int = 0
    history_len: int = 4
    ckpt_path: str | None = None


@dataclass
class EnvConfig:
    """环境配置。"""

    latent_dim: int = 512
    action_dim: int = 3
    num_patches: int = 16
    token_dim: int = 32
    history_len: int = 4
    semantic_dim: int = 0
    max_episode_length: int = 50


def setup_logging(run_name: str, outputs_root: str) -> Path:
    """设置日志。"""
    run_dir = Path(outputs_root) / run_name / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )
    return run_dir


def build_models(cfg: RLConfig, env_cfg: EnvConfig, device: str) -> tuple[PolicyModel, ValueNetwork]:
    """构建策略网络和 Value 网络。"""

    # 计算 latent 维度
    if env_cfg.num_patches > 0 and env_cfg.token_dim > 0:
        expected_latent_dim = env_cfg.num_patches * env_cfg.token_dim
    else:
        expected_latent_dim = env_cfg.latent_dim

    policy = PolicyModel(
        latent_dim=expected_latent_dim,
        action_dim=env_cfg.action_dim,
        hidden_dim=cfg.hidden_dim,
        history_len=env_cfg.history_len,
        num_patches=env_cfg.num_patches or 16,
        token_dim=env_cfg.token_dim or 32,
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
        num_patches=env_cfg.num_patches or 16,
        token_dim=env_cfg.token_dim or 32,
        num_layers=max(2, cfg.num_layers // 2),
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        semantic_dim=env_cfg.semantic_dim if cfg.use_vlm else 0,
        use_vlm=cfg.use_vlm,
    )

    return policy.to(device), value_net.to(device)


def build_environment(cfg: RLConfig, env_cfg: EnvConfig, device: str) -> LatentVecEnv:
    """构建向量化环境。"""

    # 如果 manifest 或 cache 不存在，使用 dummy 环境
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
            num_patches=env_cfg.num_patches or 16,
            token_dim=env_cfg.token_dim or 32,
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
        num_patches=env_cfg.num_patches or 16,
        token_dim=env_cfg.token_dim or 32,
        action_dim=env_cfg.action_dim,
        semantic_dim=env_cfg.semantic_dim if cfg.use_vlm else 0,
        device=device,
        max_episode_length=cfg.max_episode_length,
        reward_type=cfg.reward_type,
    )


def print_train_summary(
    iteration: int,
    collect_stats: dict,
    train_stats: Any,
    storage_stats: dict,
    elapsed: float,
) -> None:
    """打印训练摘要。"""
    logger.info(
        "Iter %d | Reward=%.3f | PolicyLoss=%.4f | ValueLoss=%.4f | "
        "EntropyLoss=%.4f | KL=%.4f | ClipFrac=%.3f | LR=%.6f | "
        "Time=%.1fs",
        iteration,
        collect_stats.get("reward_mean", 0.0),
        train_stats.policy_loss,
        train_stats.value_loss,
        train_stats.entropy_loss,
        train_stats.kl_divergence,
        train_stats.clip_fraction,
        train_stats.learning_rate,
        elapsed,
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="rl_default")
def main(cfg: DictConfig) -> None:
    """主训练循环。"""
    # 解析配置
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    rl_cfg = RLConfig(**cfg_dict.get("rl", {}))
    env_cfg = EnvConfig(**cfg_dict.get("env", {}))
    wm_cfg = WMConfig(**cfg_dict.get("wm", {}))

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

    # 构建模型
    policy, value_net = build_models(rl_cfg, env_cfg, device)
    num_params = sum(p.numel() for p in policy.parameters())
    num_value_params = sum(p.numel() for p in value_net.parameters())
    logger.info(
        "Policy 参数: %d, Value 参数: %d",
        num_params, num_value_params
    )

    # 构建环境和存储
    env = build_environment(rl_cfg, env_cfg, device)

    storage = RolloutStorage(
        num_steps=rl_cfg.num_steps,
        num_envs=rl_cfg.num_envs,
        latent_dim=env_cfg.latent_dim,
        action_dim=env_cfg.action_dim,
        semantic_dim=env_cfg.semantic_dim if rl_cfg.use_vlm else 0,
        num_patches=env_cfg.num_patches or 16,
        token_dim=env_cfg.token_dim or 32,
        history_len=env_cfg.history_len,
        device=device,
    )

    # 构建 PPO Learner
    learner = PPOLearner(
        policy=policy,
        value_net=value_net,
        lr=rl_cfg.lr,
        epsilon=rl_cfg.epsilon,
        value_coef=rl_cfg.value_coef,
        entropy_coef=rl_cfg.entropy_coef,
        max_grad_norm=rl_cfg.max_grad_norm,
        num_epochs=rl_cfg.num_epochs,
        mini_batch_size=rl_cfg.mini_batch_size,
        gamma=rl_cfg.gamma,
        gae_lambda=rl_cfg.gae_lambda,
        device=device,
    )

    # 训练循环
    start_time = time.time()
    global_step = 0

    logger.info("开始训练 (%d iterations)...", rl_cfg.num_iterations)

    for iteration in range(1, rl_cfg.num_iterations + 1):
        iter_start = time.time()

        # 1. 收集经验
        collect_start = time.time()
        collect_stats = learner.collect_experience(env, storage)
        collect_time = time.time() - collect_start

        # 2. PPO 更新
        train_start = time.time()
        train_stats = learner.update(storage)
        train_time = time.time() - train_start

        global_step += rl_cfg.num_steps * rl_cfg.num_envs

        # 3. 日志
        storage_stats = storage.get_statistics()
        elapsed = time.time() - iter_start

        if iteration % rl_cfg.eval_every == 0 or iteration == 1:
            print_train_summary(iteration, collect_stats, train_stats, storage_stats, elapsed)

        # 4. 保存 checkpoint
        if iteration % rl_cfg.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_{iteration:06d}.pt"
            learner.save_checkpoint(
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
    learner.save_checkpoint(
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
