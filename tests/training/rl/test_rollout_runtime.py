"""RL rollout source 的启动约束测试。"""

from __future__ import annotations

import pytest

from nimloth.rollout import JSONLRolloutCollector
from nimloth.training.rl.rollout_runtime import (
    validate_collector_configuration,
    validate_online_policy_configuration,
    validate_planning_initialization,
)


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


def test_planning_policy_rejects_qwen_ppo_replay() -> None:
    with pytest.raises(ValueError, match="planner behavior probability replay"):
        validate_online_policy_configuration(
            actor_enabled=True,
            planning_enabled=True,
        )

    validate_online_policy_configuration(
        actor_enabled=False,
        planning_enabled=True,
    )


def test_online_planning_requires_trained_model_modules() -> None:
    with pytest.raises(ValueError, match="requires a resumed RL checkpoint"):
        validate_planning_initialization(
            planning_enabled=True,
            online_policy_needed=True,
            resume_loaded=False,
            wm_checkpoint=None,
            state_proj_checkpoint=None,
            value_head_checkpoint=None,
        )

    validate_planning_initialization(
        planning_enabled=True,
        online_policy_needed=True,
        resume_loaded=True,
        wm_checkpoint=None,
        state_proj_checkpoint=None,
        value_head_checkpoint=None,
    )
