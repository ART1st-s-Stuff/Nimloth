"""RL 阶段的 rollout 指标汇总与 held-out evaluation。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nimloth.rollout import RolloutCollector, RolloutTrajectory


@dataclass(frozen=True)
class RLEvaluationResult:
    """一次 held-out rollout 的指标与 checkpoint 选择值。"""

    metrics: dict[str, float]
    selected_metric: float
    num_episodes: int


def summarize_rollouts(
    trajectories: list[RolloutTrajectory],
) -> dict[str, float]:
    """汇总不依赖 optimizer 的 episode 级指标。"""

    if not trajectories:
        return {}
    count = len(trajectories)
    return {
        "success_rate": sum(item.success for item in trajectories) / count,
        "avg_reward": sum(item.reward for item in trajectories) / count,
        "avg_steps": sum(item.num_steps for item in trajectories) / count,
    }


def evaluate_rollout_collector(
    collector: RolloutCollector,
    *,
    num_episodes: int,
    max_steps_per_episode: int,
    output_dir: Path,
    checkpoint_metric: str,
) -> RLEvaluationResult | None:
    """从独立 collector 采集 validation，并选择显式 checkpoint 指标。"""

    trajectories = collector.collect(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        output_dir=output_dir,
    )
    summary = summarize_rollouts(trajectories)
    if not summary:
        return None
    if checkpoint_metric not in {"success_rate", "avg_reward"}:
        raise ValueError(f"unsupported RL checkpoint metric {checkpoint_metric!r}")
    return RLEvaluationResult(
        metrics={f"val_{key}": float(value) for key, value in summary.items()},
        selected_metric=float(summary[checkpoint_metric]),
        num_episodes=len(trajectories),
    )
