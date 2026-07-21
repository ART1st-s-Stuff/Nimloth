"""RL rollout source 的启动约束测试。"""

from __future__ import annotations

import pytest

from nimloth.rollout import JSONLRolloutCollector
from nimloth.training.rl.rollout_runtime import validate_collector_configuration


def test_static_jsonl_cannot_drive_ppo_actor() -> None:
    with pytest.raises(ValueError, match="PPO actor requires fresh trajectories"):
        validate_collector_configuration(
            actor_enabled=True,
            train_collector=JSONLRolloutCollector(),
            eval_collector=None,
            validation_enabled=False,
        )


def test_validation_requires_an_independent_collector() -> None:
    with pytest.raises(ValueError, match="separate eval collector"):
        validate_collector_configuration(
            actor_enabled=False,
            train_collector=JSONLRolloutCollector(),
            eval_collector=None,
            validation_enabled=True,
        )
