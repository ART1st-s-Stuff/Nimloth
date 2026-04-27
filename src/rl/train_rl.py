"""RL 训练入口。

支持 PPO 训练，使用 latent cache 数据作为环境。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.rl import (
    EnvConfig,
    PPOLearner,
    RolloutStorage,
    build_environment,
    build_models_with_env,
    get_device,
    setup_logging,
    LatentVecEnv,
)

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """RL 训练配置。"""

    algorithm: str = "ppo"
    num_envs: int = 16
    num_steps: int = 128
    max_episode_length: int = 50
    manifest_path: str = "data/ai2thor/train"
    latent_cache_dir: str = "data/ai2thor/latents"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    num_epochs: int = 10
    mini_batch_size: int = 64
    reward_type: str = "action_match"
    device: str = "cuda"
    num_iterations: int = 1000
    eval_every: int = 10
    save_every: int = 50
    outputs_root: str = "models"
    run_name: str = "ppo"
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    action_std_init: float = 0.5
    use_vlm: bool = False
    semantic_dim: int = 0
    shared_encoder: bool = True
    wm_freeze_steps: int = 1000


@dataclass
class WMConfig:
    """WM 配置（用于获取 latent 维度）。"""

    latent_dim: int = 512
    num_patches: int = 0
    token_dim: int = 0
    history_len: int = 4
    ckpt_path: str | None = None


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
    OmegaConf.resolve(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    rl_cfg = RLConfig(**cfg_dict.get("rl", {}))
    env_cfg = EnvConfig(**cfg_dict.get("env", {}))
    WMConfig(**cfg_dict.get("wm", {}))

    device = get_device(rl_cfg.device)
    logger.info("使用设备: %s", device)
    logger.info("RL 配置: %s", OmegaConf.to_yaml(rl_cfg))
    logger.info("环境配置: %s", OmegaConf.to_yaml(env_cfg))

    run_dir = setup_logging(rl_cfg.run_name, rl_cfg.outputs_root)
    logger.info("输出目录: %s", run_dir)

    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    env = build_environment(
        manifest_path=rl_cfg.manifest_path,
        latent_cache_dir=rl_cfg.latent_cache_dir,
        env_config=env_cfg,
        model_config=_make_model_config(rl_cfg),
        num_envs=rl_cfg.num_envs,
        device=device,
        max_episode_length=rl_cfg.max_episode_length,
        reward_type=rl_cfg.reward_type,
    )
    logger.info("环境检测到 latent 维度: patches=%d, token_dim=%d", env.num_patches, env.token_dim)

    policy, value_net = build_models_with_env(
        env=env,
        action_dim=env_cfg.action_dim,
        env_config=env_cfg,
        model_config=_make_model_config(rl_cfg),
        device=device,
    )
    num_params = sum(p.numel() for p in policy.parameters())
    num_value_params = sum(p.numel() for p in value_net.parameters())
    logger.info("Policy 参数: %d, Value 参数: %d", num_params, num_value_params)

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

    start_time = time.time()
    global_step = 0

    logger.info("开始训练 (%d iterations)...", rl_cfg.num_iterations)

    for iteration in range(1, rl_cfg.num_iterations + 1):
        iter_start = time.time()

        collect_stats = learner.collect_experience(env, storage)
        train_stats = learner.update(storage)

        global_step += rl_cfg.num_steps * rl_cfg.num_envs

        storage_stats = storage.get_statistics()
        elapsed = time.time() - iter_start

        if iteration % rl_cfg.eval_every == 0 or iteration == 1:
            print_train_summary(iteration, collect_stats, train_stats, storage_stats, elapsed)

        if iteration % rl_cfg.save_every == 0:
            ckpt_path = run_dir / f"checkpoint_{iteration:06d}.pt"
            learner.save_checkpoint(
                str(ckpt_path),
                step=iteration,
                extra={"collect_stats": collect_stats, "storage_stats": storage_stats, "total_steps": global_step},
            )

    final_path = run_dir / "checkpoint_final.pt"
    learner.save_checkpoint(str(final_path), step=rl_cfg.num_iterations, extra={"total_steps": global_step})

    total_time = time.time() - start_time
    logger.info("训练完成！总计 %d iterations, %d steps, %.1f 秒", rl_cfg.num_iterations, global_step, total_time)

    env.close()


def _make_model_config(rl_cfg: RLConfig):
    """从 RLConfig 创建 RLModelConfig。"""
    from src.rl.train_utils import RLModelConfig
    return RLModelConfig(
        hidden_dim=rl_cfg.hidden_dim,
        num_layers=rl_cfg.num_layers,
        num_heads=rl_cfg.num_heads,
        dropout=rl_cfg.dropout,
        action_std_init=rl_cfg.action_std_init,
        use_vlm=rl_cfg.use_vlm,
        semantic_dim=rl_cfg.semantic_dim,
    )


if __name__ == "__main__":
    main()