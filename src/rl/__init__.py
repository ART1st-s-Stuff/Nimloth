"""RL (Reinforcement Learning) 模块。

提供 PPO/GRPO 训练所需的核心组件：
- PolicyModel: 策略网络 (PM)
- ValueNetwork: Value 函数网络
- Storage: 经验回放
- VecEnv: 向量化环境
- PPOLearner: PPO 训练器
- JointTrainer: WM + PM + VLM 联合训练器
"""

from src.rl.policy_model import PolicyModel
from src.rl.value_network import ValueNetwork
from src.rl.storage import RolloutStorage
from src.rl.vec_env import LatentVecEnv, DummyVecEnv
from src.rl.ppo_learner import PPOLearner
from src.rl.joint_trainer import JointTrainer, RewardCalculator, JointTrainerStats
from src.rl.train_utils import (
    EnvConfig,
    RLModelConfig,
    setup_logging,
    get_device,
    build_models,
    build_models_with_env,
    build_environment,
    compute_episode_stats,
)

__all__ = [
    # 核心组件
    "PolicyModel",
    "ValueNetwork",
    "RolloutStorage",
    "LatentVecEnv",
    "DummyVecEnv",
    "PPOLearner",
    "JointTrainer",
    "RewardCalculator",
    "JointTrainerStats",
    # 训练工具
    "EnvConfig",
    "RLModelConfig",
    "setup_logging",
    "get_device",
    "build_models",
    "build_models_with_env",
    "build_environment",
    "compute_episode_stats",
]
