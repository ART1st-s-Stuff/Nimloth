"""RL 训练工具函数。

提供训练入口共享的日志、模型构建、环境构建等工具函数。
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from src.rl.policy_model import PolicyModel
    from src.rl.value_network import ValueNetwork
    from src.rl.vec_env import LatentVecEnv, DummyVecEnv


def setup_logging(run_name: str, outputs_root: str, experiment_name: str | None = None) -> Path:
    """设置日志。

    目录结构: {outputs_root}/{experiment_name or run_name}/{timestamp}
    例如: models/rl/default/2026-04-26_22-00-00

    Args:
        run_name: 任务名称（作为子目录）
        outputs_root: 输出根目录
        experiment_name: 可选的实验名称（放在 run_name 下）

    Returns:
        run_dir: 实际的运行目录路径
    """
    if experiment_name:
        run_dir = Path(outputs_root) / experiment_name / run_name / time.strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_dir = Path(outputs_root) / run_name / time.strftime("%Y-%m-%d_%H-%M-%S")
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


def get_device(preferred: str = "cuda") -> str:
    """获取计算设备。"""
    import torch
    return "cuda" if torch.cuda.is_available() and preferred == "cuda" else "cpu"


def compute_episode_stats(
    rewards: list[float],
    episode_lengths: list[int],
) -> dict[str, float]:
    """计算 episode 统计信息。"""
    if not rewards:
        return {
            "reward_mean": 0.0,
            "reward_max": 0.0,
            "episode_length_mean": 0.0,
            "num_episodes": 0,
        }
    return {
        "reward_mean": sum(rewards) / len(rewards),
        "reward_max": max(rewards),
        "episode_length_mean": sum(episode_lengths) / len(episode_lengths),
        "num_episodes": len(rewards),
    }


def detach_tensor(x: "Tensor") -> "Tensor":
    """安全地 detach tensor，避免 in-place 操作问题。"""
    return x.detach().clone()


# =============================================================================
# 模型构建工具
# =============================================================================

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


@dataclass
class RLModelConfig:
    """RL 模型配置。"""
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    action_std_init: float = 0.5
    use_vlm: bool = False
    semantic_dim: int = 0


def build_models(
    latent_dim: int,
    action_dim: int,
    env_config: EnvConfig,
    model_config: RLModelConfig,
    device: str,
) -> tuple["PolicyModel", "ValueNetwork"]:
    """构建策略网络和 Value 网络。

    Args:
        latent_dim: latent 总维度 (num_patches * token_dim)
        action_dim: 动作维度
        env_config: 环境配置
        model_config: 模型配置
        device: 计算设备

    Returns:
        (policy, value_net)
    """
    from src.rl.policy_model import PolicyModel
    from src.rl.value_network import ValueNetwork

    policy = PolicyModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        hidden_dim=model_config.hidden_dim,
        history_len=env_config.history_len,
        num_patches=env_config.num_patches,
        token_dim=env_config.token_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        semantic_dim=model_config.semantic_dim if model_config.use_vlm else 0,
        action_std_init=model_config.action_std_init,
        use_vlm=model_config.use_vlm,
    )

    value_net = ValueNetwork(
        latent_dim=latent_dim,
        hidden_dim=model_config.hidden_dim,
        history_len=env_config.history_len,
        num_patches=env_config.num_patches,
        token_dim=env_config.token_dim,
        num_layers=max(2, model_config.num_layers // 2),
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        semantic_dim=model_config.semantic_dim if model_config.use_vlm else 0,
        use_vlm=model_config.use_vlm,
    )

    return policy.to(device), value_net.to(device)


def build_models_with_env(
    env: "LatentVecEnv",
    action_dim: int,
    env_config: EnvConfig,
    model_config: RLModelConfig,
    device: str,
) -> tuple["PolicyModel", "ValueNetwork"]:
    """构建策略网络和 Value 网络（使用环境的实际维度）。

    Args:
        env: 向量化环境（用于获取实际维度）
        action_dim: 动作维度
        env_config: 环境配置
        model_config: 模型配置
        device: 计算设备

    Returns:
        (policy, value_net)
    """
    from src.rl.policy_model import PolicyModel
    from src.rl.value_network import ValueNetwork

    actual_patches = env.num_patches
    actual_token_dim = env.token_dim
    expected_latent_dim = actual_patches * actual_token_dim

    policy = PolicyModel(
        latent_dim=expected_latent_dim,
        action_dim=action_dim,
        hidden_dim=model_config.hidden_dim,
        history_len=env.history_len,
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        semantic_dim=model_config.semantic_dim if model_config.use_vlm else 0,
        action_std_init=model_config.action_std_init,
        use_vlm=model_config.use_vlm,
    )

    value_net = ValueNetwork(
        latent_dim=expected_latent_dim,
        hidden_dim=model_config.hidden_dim,
        history_len=env.history_len,
        num_patches=actual_patches,
        token_dim=actual_token_dim,
        num_layers=max(2, model_config.num_layers // 2),
        num_heads=model_config.num_heads,
        dropout=model_config.dropout,
        semantic_dim=model_config.semantic_dim if model_config.use_vlm else 0,
        use_vlm=model_config.use_vlm,
    )

    return policy.to(device), value_net.to(device)


def build_environment(
    manifest_path: str,
    latent_cache_dir: str,
    env_config: EnvConfig,
    model_config: RLModelConfig,
    num_envs: int,
    device: str,
    max_episode_length: int = 50,
    reward_type: str = "action_match",
) -> "LatentVecEnv | DummyVecEnv":
    """构建向量化环境。

    Args:
        manifest_path: 数据 manifest 路径
        latent_cache_dir: latent cache 目录
        env_config: 环境配置
        model_config: 模型配置
        num_envs: 并行环境数
        device: 计算设备
        max_episode_length: 最大 episode 长度
        reward_type: 奖励类型

    Returns:
        LatentVecEnv 或 DummyVecEnv（数据不存在时）
    """
    from src.rl.vec_env import DummyVecEnv, LatentVecEnv

    manifest_path = Path(manifest_path)
    latent_cache_dir = Path(latent_cache_dir)

    if not manifest_path.exists() or not latent_cache_dir.exists():
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "数据不存在，使用 DummyVecEnv 进行测试。manifest=%s, cache=%s",
            manifest_path, latent_cache_dir
        )
        return DummyVecEnv(
            latent_dim=env_config.latent_dim,
            action_dim=env_config.action_dim,
            num_patches=env_config.num_patches,
            token_dim=env_config.token_dim,
            history_len=env_config.history_len,
            num_envs=num_envs,
            semantic_dim=env_config.semantic_dim,
            device=device,
            max_episode_length=max_episode_length,
        )

    return LatentVecEnv(
        manifest_path=str(manifest_path),
        latent_cache_dir=str(latent_cache_dir),
        num_envs=num_envs,
        history_len=env_config.history_len,
        num_patches=env_config.num_patches,
        token_dim=env_config.token_dim,
        action_dim=env_config.action_dim,
        semantic_dim=model_config.semantic_dim if model_config.use_vlm else 0,
        device=device,
        max_episode_length=max_episode_length,
        reward_type=reward_type,
    )


def count_parameters(model: "PolicyModel | ValueNetwork") -> int:
    """计算模型参数数量。"""
    return sum(p.numel() for p in model.parameters())