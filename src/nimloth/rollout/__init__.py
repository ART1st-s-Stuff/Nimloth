"""训练阶段无关的 rollout 记录、来源、持久化和 transition 接口。"""

from nimloth.rollout.batch import TransitionBatch, TransitionBatchBuilder
from nimloth.rollout.from_agent import trajectory_from_agent_episode
from nimloth.rollout.fresh import FreshJSONLRolloutCollector, FreshRolloutManifest
from nimloth.rollout.merge import merge_fresh_rollout_shards
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.source import JSONLRolloutCollector, RolloutCollector
from nimloth.rollout.storage import load_trajectories, save_trajectories
from nimloth.rollout.validation import validate_rollout_trajectory
from nimloth.rollout.windows import (
    TrajectoryWindow,
    count_trajectory_windows,
    sample_trajectory_windows,
)

__all__ = [
    "JSONLRolloutCollector",
    "FreshJSONLRolloutCollector",
    "FreshRolloutManifest",
    "RolloutCollector",
    "RolloutTrajectory",
    "TrajectoryWindow",
    "TransitionBatch",
    "TransitionBatchBuilder",
    "trajectory_from_agent_episode",
    "load_trajectories",
    "merge_fresh_rollout_shards",
    "count_trajectory_windows",
    "sample_trajectory_windows",
    "save_trajectories",
    "validate_rollout_trajectory",
]
