from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from nimloth.agent import (
    ActionTrainingTrace,
    Agent,
    AgentTranscript,
    NimlothPromptTemplate,
    PlannerPolicyTrace,
    PolicyReplayOutput,
)
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.rollout import RolloutTrajectory
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.episodes import (
    TemporalDifferenceStep,
    build_episode_training_batches,
)
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.wm import WorldModel


class _UnusedBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Linear(1, 1)

    @property
    def model(self) -> torch.nn.Module:
        return self.module

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        return BackboneOutput(self.module(batch.tensors["hidden"]))

    def with_model(self, model: torch.nn.Module) -> "_UnusedBackbone":
        result = _UnusedBackbone()
        result.module = model
        return result

    def save_pretrained(self, output_dir, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _UnusedInputBuilder:
    processor = None

    def build(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("cached planner episode must not rebuild Qwen states")


class _StateProjector(torch.nn.Linear):
    qwen_hidden_dim = 3
    latent_token_count = 1

    def __init__(self) -> None:
        super().__init__(3, 2, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden[:, 0] if hidden.ndim == 3 else hidden)


class _SequencePredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"history_size": 4})()
        self.state = torch.nn.Linear(2, 2, bias=False)
        self.action = torch.nn.Embedding(8, 2)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.state(states) + self.action(actions)


class _ActionReplay(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 8))

    def forward(self, samples) -> PolicyReplayOutput:  # type: ignore[no-untyped-def]
        assert len(samples) == 1
        return PolicyReplayOutput(
            selected_log_probs=torch.empty(0),
            entropies=torch.empty(0),
            action_log_probs=self.logits.log_softmax(dim=-1),
        )


def _deterministic(action: int) -> tuple[float, ...]:
    return tuple(0.0 if index == action else float("-inf") for index in range(8))


def _planner_trajectory() -> RolloutTrajectory:
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    actions = [0, 1, 2, 3]
    action_tokens = list(range(100, 108))
    responses = [
        prompt.assistant_response(0, thought="anchor zero"),
        "<|action_start|><|action_(1)|><|action_end|>",
        prompt.assistant_response(2, thought="anchor two"),
        "<|action_start|><|action_(3)|><|action_end|>",
    ]
    observations = [f"Observation {step}.\n<image>" for step in range(5)]
    images = [f"step_{step}.png" for step in range(5)]
    transcript = AgentTranscript(
        system_prompt="Navigate.",
        observation_texts=tuple(observations),
        observation_images=tuple(images),
        action_indices=tuple(actions),
        assistant_responses=tuple(responses),
    )
    traces = []
    for first_action, sequence in ((0, (0, 1)), (2, (2, 3))):
        traces.append(
            PlannerPolicyTrace(
                qwen_action_log_probs=tuple([-math.log(8.0)] * 8),
                candidate_sequences=(sequence,),
                candidate_scores=(1.0,),
                root_action_scores=tuple(
                    1.0 if index == first_action else float("-inf")
                    for index in range(8)
                ),
                action_training=ActionTrainingTrace(
                    objective="distillation",
                    behavior_owner="world_model",
                    executed_action_index=first_action,
                    behavior_action_log_probs=_deterministic(first_action),
                    teacher_action_log_probs=_deterministic(first_action),
                ),
                horizon=2,
                search_mode="greedy",
                qwen_sampled_action_index=7,
            )
        )
    return RolloutTrajectory(
        record_id="planner_episode",
        image_paths=images,
        action_indices=actions,
        action_names=[NAVIGATION_ACTION_SPACE.key_for(index) for index in actions],
        action_log_probs=[list(_deterministic(action)) for action in actions],
        instruction="test",
        reward=1.0,
        rewards=[0.0, 0.0, 0.0, 1.0],
        terminated=True,
        messages=prompt.build_supervised_prompt(transcript).unbound_messages(),
        system_prompt="Navigate.",
        observation_texts=observations,
        policy_messages=[
            prompt.build_response_policy_prompt(
                transcript.policy_prefix(step)
            ).unbound_messages()
            for step in range(4)
        ],
        assistant_responses=responses,
        terminal_assistant_prefix=prompt.assistant_prefix(thought="terminal"),
        state_anchor_steps=[0, 2, 4],
        state_latent_hiddens=[
            [[0.0, 1.0, 2.0]],
            [[2.0, 3.0, 4.0]],
            [[4.0, 5.0, 6.0]],
        ],
        world_model_states=[
            [0.0, 0.5],
            [0.2, 0.7],
            [1.0, 1.5],
            [1.2, 1.7],
            [2.0, 2.5],
        ],
        policy_credit_assignment="action",
        policy_step_indices=[0, 2],
        policy_token_ids=[
            [50, action_tokens[0], 200],
            [52, action_tokens[2], 200],
        ],
        policy_token_log_probs=[[None, None, None], [None, None, None]],
        policy_loss_masks=[[False, False, False], [False, False, False]],
        policy_token_roles=[
            ["reasoning", "action", "injected"],
            ["reasoning", "action", "injected"],
        ],
        policy_action_token_ids=[action_tokens, action_tokens],
        policy_reasoning_texts=["anchor zero", "anchor two"],
        policy_finish_reasons=["stop", "stop"],
        policy_reasoning_truncated=[False, False],
        planner_policy_traces=traces,
        prompt_template_spec=prompt.spec,
    )


def test_planner_trajectory_rejects_a_tampered_segment_tail() -> None:
    trajectory = _planner_trajectory()
    trace = trajectory.planner_policy_traces[0]
    trajectory.planner_policy_traces[0] = replace(
        trace,
        candidate_sequences=((0, 7),),
    )

    with pytest.raises(ValueError, match="selected candidate prefix"):
        build_episode_training_batches(
            [trajectory],
            gamma=1.0,
            truncated_bootstrap=0.0,
        )

    with pytest.raises(ValueError, match="selected candidate prefix"):
        TemporalDifferenceStep(trajectory=trajectory, start_step=0, end_step=2)


def test_planner_trace_accepts_a_short_terminal_prefix() -> None:
    trace = _planner_trajectory().planner_policy_traces[0]

    trace.validate_executed_prefix((trace.selected_candidate_sequence[0],))


def test_planner_trajectory_rejects_a_segment_longer_than_its_horizon() -> None:
    trajectory = _planner_trajectory()
    trace = trajectory.planner_policy_traces[0]
    trajectory.planner_policy_traces[0] = replace(
        trace,
        candidate_sequences=((0,),),
        horizon=1,
    )

    with pytest.raises(ValueError, match="exceeds its horizon"):
        build_episode_training_batches(
            [trajectory],
            gamma=1.0,
            truncated_bootstrap=0.0,
        )


def test_episode_td_replays_mixed_history_then_mc_only_updates_value_head() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]
    assert len(episode.td_steps) == 2
    assert torch.allclose(
        episode.td_steps[1].retained_state_context(4),
        torch.tensor([[0.0, 0.5], [0.2, 0.7], [1.0, 1.5]]),
    )

    projector = _StateProjector()
    predictor = _SequencePredictor()
    value_head = torch.nn.Linear(2, 8)
    replay = _ActionReplay()
    runtime = RLModelRuntime(
        agent=Agent(
            backbone=_UnusedBackbone(),
            wm=WorldModel(
                state_proj=projector,
                wm_predictor=predictor,
                value_head=value_head,
            ),
        ),
        input_builder=_UnusedInputBuilder(),  # type: ignore[arg-type]
        representation_to_backbone=False,
        policy_replay=replay,
    )
    algorithm = RLAlgorithm(
        history_size=4,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=0.1,
        value_rank_weight=0.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
        action_objective="distillation",
        credit_assignment="action",
        planner_distillation_weight=1.0,
    )

    for td_step in episode.td_steps:
        output = algorithm.temporal_difference_step(
            runtime,
            td_step,
            total_td_steps=2,
        )
        assert math.isfinite(output.metrics["action_distillation_kl"])
        output.loss.backward()

    assert projector.weight.grad is not None
    assert predictor.state.weight.grad is not None
    assert replay.logits.grad is not None
    assert value_head.weight.grad is None

    projector.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    replay.zero_grad(set_to_none=True)
    mc_output = algorithm.monte_carlo_step(runtime, (episode,))
    mc_output.loss.backward()

    assert value_head.weight.grad is not None
    assert projector.weight.grad is None
    assert predictor.state.weight.grad is None
    assert replay.logits.grad is None
