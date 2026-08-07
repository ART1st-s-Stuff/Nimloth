"""Planner episode training data at real environment-transition granularity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nimloth.agent import AgentPrompt
from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory
from nimloth.rollout.transitions import discounted_action_value_targets


@dataclass(frozen=True)
class ExecutedTransition:
    """One executed action between two consecutive real Qwen states."""

    trajectory: RolloutTrajectory
    step_index: int

    def __post_init__(self) -> None:
        if not 0 <= self.step_index < self.trajectory.num_steps:
            raise ValueError("transition step is outside the trajectory")
        expected_anchors = list(range(self.trajectory.num_steps + 1))
        if self.trajectory.state_anchor_steps != expected_anchors:
            raise ValueError("planner transition requires a real state at every step")
        trace = self.trajectory.planner_policy_trace(self.step_index)
        if trace is None:
            raise ValueError("planner transition has no search trace")
        trace.validate_executed_action(self.action_index)

    @property
    def action_index(self) -> int:
        return int(self.trajectory.action_indices[self.step_index])

    @property
    def state_prompt(self) -> AgentPrompt:
        """The complete real prefix used to recompute the current Qwen state."""

        return self.trajectory.build_state_prompt(self.step_index)

    def state_history(self, history_size: int) -> torch.Tensor:
        """Return detached real states ending at the current environment state."""

        if history_size < 1:
            raise ValueError("history_size must be positive")
        context_start = max(0, self.step_index - history_size + 1)
        return torch.tensor(
            self.trajectory.world_model_states[
                context_start : self.step_index + 1
            ],
            dtype=torch.float32,
        )

    def previous_actions(self, history_size: int) -> torch.Tensor:
        """Return actions preceding the current state in the same WM context."""

        if history_size < 1:
            raise ValueError("history_size must be positive")
        context_start = max(0, self.step_index - history_size + 1)
        return torch.tensor(
            self.trajectory.action_indices[context_start : self.step_index],
            dtype=torch.long,
        )

    def actual_next_state(self) -> torch.Tensor:
        """Return the fixed real-Qwen target after the executed action."""

        return torch.tensor(
            self.trajectory.world_model_states[self.step_index + 1],
            dtype=torch.float32,
        )

    def rollout_decision_state(self) -> torch.Tensor:
        """Return the frozen rollout state on which the executed action was chosen."""

        return torch.tensor(
            self.trajectory.world_model_states[self.step_index],
            dtype=torch.float32,
        )

    def behavior_action_log_probs(self) -> torch.Tensor:
        """Return the complete frozen planner behavior distribution."""

        return torch.tensor(
            self.trajectory.action_log_probs[self.step_index],
            dtype=torch.float32,
        )

    @property
    def next_image_path(self) -> str:
        return self.trajectory.image_paths[self.step_index + 1]


@dataclass(frozen=True)
class EpisodeTrainingBatch:
    """One complete planner episode with one MC target per real transition."""

    trajectory: RolloutTrajectory
    transitions: tuple[ExecutedTransition, ...]
    return_targets: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.transitions) != self.trajectory.num_steps:
            raise ValueError("transitions must cover the complete episode")
        if tuple(step.step_index for step in self.transitions) != tuple(
            range(self.trajectory.num_steps)
        ):
            raise ValueError("transitions must be ordered by environment step")
        if self.return_targets.shape != (self.trajectory.num_steps,):
            raise ValueError("MC return targets must align with executed actions")


def build_episode_training_batches(
    trajectories: Sequence[RolloutTrajectory],
    *,
    gamma: float,
    truncated_bootstrap: float | None,
) -> tuple[EpisodeTrainingBatch, ...]:
    """Build dense real-transition supervision for receding-horizon planning."""

    batches: list[EpisodeTrainingBatch] = []
    for trajectory in trajectories:
        validate_rollout_trajectory(trajectory)
        if not trajectory.planner_policy_traces:
            raise ValueError("episode transition training requires planner trajectories")
        targets = discounted_action_value_targets(
            trajectory.to_record(),
            gamma=gamma,
            truncated_bootstrap=truncated_bootstrap,
        )
        batches.append(
            EpisodeTrainingBatch(
                trajectory=trajectory,
                transitions=tuple(
                    ExecutedTransition(trajectory, step_index)
                    for step_index in range(trajectory.num_steps)
                ),
                return_targets=torch.tensor(targets, dtype=torch.float32),
            )
        )
    return tuple(batches)


__all__ = [
    "EpisodeTrainingBatch",
    "ExecutedTransition",
    "build_episode_training_batches",
]
