"""从完整 trajectory 采样连续训练窗口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nimloth.agent import AgentPrompt, PolicyReplayInput
from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.validation import validate_rollout_trajectory


@dataclass(frozen=True)
class TrajectoryWindow:
    """同一条 trajectory 内连续的 H 个动作和 H+1 个状态。"""

    trajectory: RolloutTrajectory
    start_step: int
    history_size: int

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("trajectory window history_size must be positive")
        if not 0 <= self.start_step <= self.trajectory.num_steps - self.history_size:
            raise ValueError(
                f"invalid window [{self.start_step}, "
                f"{self.start_step + self.history_size}) for trajectory "
                f"{self.trajectory.record_id!r} with {self.trajectory.num_steps} steps"
            )

    @property
    def record_id(self) -> str:
        return self.trajectory.record_id

    def state_prompts(self) -> tuple[AgentPrompt, ...]:
        """返回窗口内真实的 H+1 个 policy-state prompt。"""

        return tuple(
            self.trajectory.build_policy_prompt(step_index)
            for step_index in range(
                self.start_step,
                self.start_step + self.history_size + 1,
            )
        )

    def policy_replay_inputs(self) -> tuple[PolicyReplayInput, ...]:
        """返回窗口内 H 个动作的 PPO 重放输入。"""

        latent_token_count = self.trajectory.resolved_latent_token_count()
        return tuple(
            PolicyReplayInput(
                prompt=self.trajectory.build_policy_prompt(step_index),
                action_index=self.trajectory.action_indices[step_index],
                sampling_temperature=self.trajectory.sampling_temperature,
                sampling_top_p=self.trajectory.sampling_top_p,
                latent_token_count=latent_token_count,
            )
            for step_index in range(
                self.start_step,
                self.start_step + self.history_size,
            )
        )


def count_trajectory_windows(
    trajectories: Sequence[RolloutTrajectory],
    *,
    history_size: int,
) -> int:
    """统计所有不会跨越 episode 边界的固定长度窗口。"""

    if history_size < 1:
        raise ValueError(f"history_size must be positive, got {history_size}")
    return sum(
        max(0, trajectory.num_steps - history_size + 1)
        for trajectory in trajectories
    )


def sample_trajectory_windows(
    trajectories: Sequence[RolloutTrajectory],
    *,
    history_size: int,
    batch_size: int,
    seed: int,
) -> tuple[TrajectoryWindow, ...]:
    """使用独立 CPU generator 均匀采样连续窗口。"""

    if history_size < 1:
        raise ValueError(f"history_size must be positive, got {history_size}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for trajectory in trajectories:
        validate_rollout_trajectory(trajectory)

    candidates = [
        TrajectoryWindow(
            trajectory=trajectory,
            start_step=start,
            history_size=history_size,
        )
        for trajectory in trajectories
        for start in range(trajectory.num_steps - history_size + 1)
    ]
    if len(candidates) < batch_size:
        raise ValueError(
            f"only {len(candidates)} sequence windows are available, need {batch_size}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(candidates), generator=generator)[:batch_size]
    return tuple(candidates[int(index)] for index in indices)


__all__ = [
    "TrajectoryWindow",
    "count_trajectory_windows",
    "sample_trajectory_windows",
]
