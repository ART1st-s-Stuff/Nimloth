"""训练阶段无关的 rollout 记录、来源、持久化和 transition 接口。"""

from nimloth.rollout.batch import TransitionBatch, TransitionBatchBuilder
from nimloth.rollout.encoding import EncodedTrajectory, EncodedTransition, RolloutEncoder
from nimloth.rollout.from_agent import trajectory_from_agent_episode
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.source import JSONLRolloutCollector, RolloutCollector
from nimloth.rollout.storage import load_trajectories, save_trajectories
from nimloth.rollout.validation import validate_rollout_trajectory

__all__ = [
    "JSONLRolloutCollector",
    "EncodedTrajectory",
    "EncodedTransition",
    "RolloutCollector",
    "RolloutTrajectory",
    "RolloutEncoder",
    "TransitionBatch",
    "TransitionBatchBuilder",
    "trajectory_from_agent_episode",
    "load_trajectories",
    "save_trajectories",
    "validate_rollout_trajectory",
]
