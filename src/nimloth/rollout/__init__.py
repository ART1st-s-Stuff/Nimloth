"""训练阶段无关的 rollout 记录、来源、持久化和 transition 接口。"""

from nimloth.rollout.collector import VAGENNavigationRolloutCollector
from nimloth.rollout.encoding import (
    EncodedRolloutTransition,
    encode_rollout_transitions,
    encode_trajectory_states,
)
from nimloth.rollout.schema import RolloutTrajectory, validate_rollout_trajectory
from nimloth.rollout.source import JSONLRolloutCollector, RolloutCollector
from nimloth.rollout.storage import load_trajectories, save_trajectories

__all__ = [
    "JSONLRolloutCollector",
    "EncodedRolloutTransition",
    "RolloutCollector",
    "RolloutTrajectory",
    "VAGENNavigationRolloutCollector",
    "encode_rollout_transitions",
    "encode_trajectory_states",
    "load_trajectories",
    "save_trajectories",
    "validate_rollout_trajectory",
]
