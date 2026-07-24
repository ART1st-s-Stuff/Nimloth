"""Agent 的 Qwen 慢路径、WM 快速规划与 environment 执行边界测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nimloth.agent import Agent, AgentRuntime, EpisodeRunner, NimlothPromptTemplate
from nimloth.agent.planning import PlanningPolicy, WorldModelPlanner
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.environment import EnvironmentObservation, EnvironmentStep
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.wm import WorldModel


class _CountingBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = torch.nn.Identity()
        self.forward_count = 0

    @property
    def model(self) -> torch.nn.Module:
        return self.language_model

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        self.forward_count += 1
        return BackboneOutput(batch.tensors["hidden"])

    def with_model(self, model: torch.nn.Module) -> "_CountingBackbone":
        view = _CountingBackbone()
        view.language_model = model
        return view

    def save_pretrained(self, output_dir: Path, **_kwargs) -> None:
        raise NotImplementedError


class _InputBuilder:
    processor = None

    def build(self, messages, images, *, include_labels: bool) -> BackboneBatch:
        assert len(messages) == len(images) == 1
        assert not include_labels
        return BackboneBatch({"hidden": torch.tensor([[0.25, -0.25]])})

    def collate_encoded(self, rows, *, include_labels: bool) -> BackboneBatch:
        raise NotImplementedError

    def cache_key(self, messages, images) -> str:
        raise NotImplementedError


class _RecordingPredictor(torch.nn.Module):
    def __init__(self, *, history_size: int = 2) -> None:
        super().__init__()
        self.config = type("PredictorConfig", (), {"history_size": history_size})()
        self.action_sequences: list[torch.Tensor] = []
        self.real_history_lengths: list[int] = []

    def forward(self, state, actions):
        raise AssertionError("planner must use rollout_states()")

    def rollout_from_history(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        self.action_sequences.append(action_sequences.detach().clone())
        self.real_history_lengths.append(state_history.shape[1])
        assert previous_actions.shape[1] == state_history.shape[1] - 1
        current = state_history[:, -1]
        predicted: list[torch.Tensor] = []
        for step in range(action_sequences.shape[1]):
            delta = action_sequences[:, step].to(current.dtype).unsqueeze(-1) + 1.0
            current = current + torch.cat((delta, -delta), dim=-1)
            predicted.append(current)
        return torch.stack(predicted, dim=1)


class _ActionValueHead(torch.nn.Module):
    def __init__(self, action_count: int) -> None:
        super().__init__()
        self.action_count = action_count

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        targets = torch.arange(
            self.action_count,
            device=state.device,
            dtype=state.dtype,
        )
        return -(state[:, :1] - targets.unsqueeze(0)).square()


class _TwoStepSession:
    action_space = NAVIGATION_ACTION_SPACE
    system_prompt = "Choose a navigation action."

    def __init__(self) -> None:
        self.step_count = 0
        self.closed = False

    def reset(self, *, seed: int) -> EnvironmentObservation:
        assert seed == 9
        return EnvironmentObservation("initial observation <image>", "image-0")

    def step(self, *, action_index: int, response: str) -> EnvironmentStep:
        assert 0 <= action_index < len(NAVIGATION_ACTION_SPACE)
        assert response
        self.step_count += 1
        return EnvironmentStep(
            observation=EnvironmentObservation(
                f"observation {self.step_count} <image>",
                f"image-{self.step_count}",
            ),
            reward=1.0,
            done=self.step_count == 2,
            success=self.step_count == 2,
        )

    def close(self) -> None:
        self.closed = True


def _planning_agent() -> tuple[Agent, _CountingBackbone, _RecordingPredictor]:
    backbone = _CountingBackbone()
    predictor = _RecordingPredictor()
    return (
        Agent(
            backbone=backbone,
            wm=WorldModel(
                state_proj=torch.nn.Identity(),
                wm_predictor=predictor,
                value_head=_ActionValueHead(len(NAVIGATION_ACTION_SPACE)),
            ),
        ),
        backbone,
        predictor,
    )


def test_planner_replays_complete_candidate_history_at_every_depth() -> None:
    agent, _, predictor = _planning_agent()
    planner = WorldModelPlanner(agent.wm, horizon=3, beam_width=8)

    plan = planner.plan(
        torch.tensor([[[0.25, -0.25]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    assert [actions.shape[1] for actions in predictor.action_sequences] == [1, 2, 3]
    assert plan.candidate_sequences.shape == (8, 3)
    assert plan.candidate_scores.shape == (8,)
    assert plan.root_action_scores.shape == (len(NAVIGATION_ACTION_SPACE),)
    assert torch.isfinite(plan.root_action_scores).any()


def test_planning_policy_fails_until_real_cot_generation_is_implemented() -> None:
    agent, backbone, predictor = _planning_agent()
    with pytest.raises(NotImplementedError, match="real CoT"):
        PlanningPolicy(
            agent=agent,
            input_builder=_InputBuilder(),
            horizon=3,
            beam_width=8,
            temperature=0.0,
            top_p=1.0,
        )
    assert backbone.forward_count == 0
    assert predictor.action_sequences == []
