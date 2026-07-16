"""RL rollout schema and dataset-split safety tests."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from experiments.training.rl.rollout_env import validate_split, validate_trajectories
from nimloth.training.rl.rollout import (
    EnvRolloutCollector,
    RolloutTrajectory,
    validate_rl_policy_protocol,
)


def _trajectory() -> RolloutTrajectory:
    return RolloutTrajectory(
        record_id="train-1",
        image_paths=["before.png", "after.png"],
        action_indices=[0],
        action_names=["moveahead"],
        action_log_probs=[[-2.0] * 8],
        nav_instruction="Move near the couch.",
        split="train",
    )


def test_rl_policy_protocol_requires_inject_queries() -> None:
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=1,
        nimloth_latent_query_mode="inject",
    )) == 1
    assert validate_rl_policy_protocol(SimpleNamespace(
        nimloth_latent_token_count=8,
        nimloth_latent_query_mode="inject",
    )) == 8
    with pytest.raises(ValueError, match="requires an inject checkpoint"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=1,
            nimloth_latent_query_mode="generate",
        ))
    with pytest.raises(ValueError, match="at least one latent query"):
        validate_rl_policy_protocol(SimpleNamespace(
            nimloth_latent_token_count=0,
            nimloth_latent_query_mode="inject",
        ))


def test_training_split_requires_training_dataset() -> None:
    validate_split("base_train", "train")
    with pytest.raises(ValueError, match="refusing to label eval dataset"):
        validate_split("base", "train")
    with pytest.raises(ValueError, match="must use --split train"):
        validate_split("base_train", "eval")


def test_env_collector_enforces_training_dataset() -> None:
    EnvRolloutCollector(None, None, "http://env", None,
                        eval_sets=("base_train",), split="train")
    with pytest.raises(ValueError, match=r"requires \*_train datasets"):
        EnvRolloutCollector(None, None, "http://env", None,
                            eval_sets=("base",), split="train")
    EnvRolloutCollector(None, None, "http://env", None,
                        eval_sets=("base",), split="validation")
    with pytest.raises(ValueError, match=r"forbids \*_train datasets"):
        EnvRolloutCollector(None, None, "http://env", None,
                            eval_sets=("base_train",), split="validation")


def test_complete_trajectory_schema_passes() -> None:
    validate_trajectories([_trajectory()])


def test_missing_final_observation_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.image_paths.pop()
    with pytest.raises(RuntimeError, match="images=1 but actions=1"):
        validate_trajectories([trajectory])


def test_missing_action_log_probs_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.action_log_probs.clear()
    with pytest.raises(RuntimeError, match="action_log_probs=0 but actions=1"):
        validate_trajectories([trajectory])
