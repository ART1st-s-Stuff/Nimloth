"""保留 WM 预测 state 的 episode 级训练数据。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from nimloth.agent import PolicyReplayInput
from nimloth.rollout import RolloutTrajectory, validate_rollout_trajectory
from nimloth.rollout.transitions import discounted_action_value_targets


@dataclass(frozen=True)
class TemporalDifferenceStep:
    """由两个真实 Qwen anchor 限定的一段已执行 WM 动作序列。"""

    trajectory: RolloutTrajectory
    start_step: int
    end_step: int

    def __post_init__(self) -> None:
        anchors = self.trajectory.state_anchor_steps
        if (
            not anchors
            or self.start_step not in anchors
            or self.end_step not in anchors
        ):
            raise ValueError("TD step endpoints must both be Qwen anchors")
        start_index = anchors.index(self.start_step)
        if (
            start_index + 1 >= len(anchors)
            or anchors[start_index + 1] != self.end_step
        ):
            raise ValueError("TD step must connect consecutive Qwen anchors")
        if self.start_step >= self.end_step:
            raise ValueError("TD step must contain at least one executed action")
        trace = self.trajectory.planner_policy_trace(self.start_step)
        if trace is None:
            raise ValueError("TD step start has no planner action trace")
        trace.validate_executed_prefix(self.action_indices)

    @property
    def action_indices(self) -> tuple[int, ...]:
        return tuple(
            self.trajectory.action_indices[self.start_step : self.end_step]
        )

    def retained_state_context(self, history_size: int) -> torch.Tensor:
        """返回以当前 anchor 结尾、长度不超过 ``history_size`` 的 state。"""

        if history_size < 1:
            raise ValueError("history_size must be positive")
        context_start = max(0, self.start_step - history_size + 1)
        return torch.tensor(
            self.trajectory.world_model_states[
                context_start : self.start_step + 1
            ],
            dtype=torch.float32,
        )

    def previous_actions(self, history_size: int) -> torch.Tensor:
        context_start = max(0, self.start_step - history_size + 1)
        return torch.tensor(
            self.trajectory.action_indices[context_start : self.start_step],
            dtype=torch.long,
        )

    def llm_hidden_at_step(self, step: int) -> torch.Tensor:
        try:
            anchor_index = self.trajectory.state_anchor_steps.index(step)
        except ValueError as error:
            raise ValueError(f"step {step} is not a Qwen anchor") from error
        return torch.tensor(
            self.trajectory.state_latent_hiddens[anchor_index],
            dtype=torch.float32,
        )

    def action_replay_input(self) -> PolicyReplayInput:
        token_trace = self.trajectory.policy_token_trace(self.start_step)
        planner_trace = self.trajectory.planner_policy_trace(self.start_step)
        if token_trace is None or planner_trace is None:
            raise ValueError("TD step lacks Qwen action-training provenance")
        return PolicyReplayInput(
            prompt=self.trajectory.build_policy_prompt(self.start_step),
            action_index=self.action_indices[0],
            sampling_temperature=self.trajectory.sampling_temperature,
            sampling_top_p=self.trajectory.sampling_top_p,
            latent_token_count=self.trajectory.resolved_latent_token_count(),
            credit_assignment="action",
            token_trace=token_trace,
            assistant_response=self.trajectory.assistant_responses[self.start_step],
            planner_trace=planner_trace,
        )


@dataclass(frozen=True)
class EpisodeTrainingBatch:
    """一个完整 episode 的 TD segments 与全程 MC targets。"""

    trajectory: RolloutTrajectory
    td_steps: tuple[TemporalDifferenceStep, ...]
    return_targets: torch.Tensor

    def __post_init__(self) -> None:
        if not self.td_steps:
            raise ValueError("planner episode contains no TD steps")
        if self.return_targets.shape != (self.trajectory.num_steps,):
            raise ValueError("MC return targets must align with executed actions")
        covered_actions = sum(
            step.end_step - step.start_step for step in self.td_steps
        )
        if covered_actions != self.trajectory.num_steps:
            raise ValueError("TD steps do not cover the complete episode")

    @property
    def action_states(self) -> torch.Tensor:
        return torch.tensor(
            self.trajectory.world_model_states[:-1],
            dtype=torch.float32,
        )

    @property
    def action_indices(self) -> torch.Tensor:
        return torch.tensor(self.trajectory.action_indices, dtype=torch.long)


def build_episode_training_batches(
    trajectories: Sequence[RolloutTrajectory],
    *,
    wm_prediction_steps: int,
    gamma: float,
    truncated_bootstrap: float | None,
) -> tuple[EpisodeTrainingBatch, ...]:
    """构造完整 episode 监督，不采样 latent window。"""

    batches: list[EpisodeTrainingBatch] = []
    for trajectory in trajectories:
        validate_rollout_trajectory(trajectory)
        if not trajectory.planner_policy_traces:
            raise ValueError("episode TD training requires planner trajectories")
        anchors = trajectory.state_anchor_steps
        td_steps: list[TemporalDifferenceStep] = []
        for start, end in zip(anchors, anchors[1:]):
            expected_end = min(
                start + wm_prediction_steps,
                trajectory.num_steps,
            )
            if end != expected_end:
                raise ValueError(
                    "Qwen state steps do not match configured WM prediction steps: "
                    f"start={start}, end={end}, expected_end={expected_end}"
                )
            td_steps.append(
                TemporalDifferenceStep(
                    trajectory=trajectory,
                    start_step=start,
                    end_step=end,
                )
            )
        targets = discounted_action_value_targets(
            trajectory.to_record(),
            gamma=gamma,
            truncated_bootstrap=truncated_bootstrap,
        )
        batches.append(
            EpisodeTrainingBatch(
                trajectory=trajectory,
                td_steps=tuple(td_steps),
                return_targets=torch.tensor(targets, dtype=torch.float32),
            )
        )
    return tuple(batches)


__all__ = [
    "EpisodeTrainingBatch",
    "TemporalDifferenceStep",
    "build_episode_training_batches",
]
