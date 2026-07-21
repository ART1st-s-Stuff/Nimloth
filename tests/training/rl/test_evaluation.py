"""RL held-out rollout 指标与 checkpoint 选择测试。"""

from __future__ import annotations

from nimloth.rollout import RolloutTrajectory
from nimloth.training.rl.evaluation import (
    evaluate_rollout_collector,
    summarize_rollouts,
)


class _Collector:
    def __init__(self, trajectories: list[RolloutTrajectory]) -> None:
        self._trajectories = trajectories
        self.output_dir = None

    def collect(self, *, num_episodes, max_steps_per_episode, output_dir):
        assert num_episodes == 2
        assert max_steps_per_episode == 4
        self.output_dir = output_dir
        return self._trajectories


def _trajectories() -> list[RolloutTrajectory]:
    return [
        RolloutTrajectory(
            record_id="success",
            action_indices=[0, 1],
            success=True,
            reward=10.0,
        ),
        RolloutTrajectory(
            record_id="failure",
            action_indices=[0],
            success=False,
            reward=-1.0,
        ),
    ]


def test_summarize_rollouts_uses_episode_level_metrics() -> None:
    assert summarize_rollouts(_trajectories()) == {
        "success_rate": 0.5,
        "avg_reward": 4.5,
        "avg_steps": 1.5,
    }


def test_evaluation_uses_the_explicit_checkpoint_metric(tmp_path) -> None:
    collector = _Collector(_trajectories())
    result = evaluate_rollout_collector(
        collector,
        num_episodes=2,
        max_steps_per_episode=4,
        output_dir=tmp_path / "validation",
        checkpoint_metric="avg_reward",
    )

    assert result is not None
    assert result.selected_metric == 4.5
    assert result.metrics["val_success_rate"] == 0.5
    assert collector.output_dir == tmp_path / "validation"
