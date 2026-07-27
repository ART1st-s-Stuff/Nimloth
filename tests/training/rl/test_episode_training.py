from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nimloth.agent import (
    Agent,
    AgentTranscript,
    NimlothPromptTemplate,
    PlannerPolicyTrace,
)
from nimloth.backbone import Backbone, BackboneBatch, BackboneOutput
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.rollout import RolloutTrajectory
from nimloth.rollout.record_format import STEP_REWARD_PROVENANCE
from nimloth.training.rl.algorithm import RLAlgorithm
from nimloth.training.rl.episodes import build_episode_training_batches
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.wm import WorldModel


def _deterministic(action: int) -> tuple[float, ...]:
    return tuple(0.0 if index == action else float("-inf") for index in range(8))


def _planner_trajectory() -> RolloutTrajectory:
    prompt = NimlothPromptTemplate(latent_token_count=1, action_count=8)
    actions = [0, 1, 2, 3]
    action_tokens = list(range(100, 108))
    responses = [
        prompt.assistant_response(action, thought=f"step {step}")
        for step, action in enumerate(actions)
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
    traces = [
        PlannerPolicyTrace(
            candidate_sequences=((action, (action + 5) % 8),),
            candidate_scores=(1.0,),
            root_action_scores=tuple(
                1.0 if index == action else float("-inf")
                for index in range(8)
            ),
            executed_action_index=action,
            horizon=2,
            search_mode="greedy",
        )
        for action in actions
    ]
    return RolloutTrajectory(
        record_id="planner_episode",
        reward_provenance=STEP_REWARD_PROVENANCE,
        image_paths=images,
        action_indices=actions,
        action_names=[NAVIGATION_ACTION_SPACE.key_for(index) for index in actions],
        action_log_probs=[list(_deterministic(action)) for action in actions],
        instruction="test",
        reward=1.0,
        rewards=[0.0, 0.0, 0.0, 1.0],
        terminated=True,
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
        state_anchor_steps=[0, 1, 2, 3, 4],
        state_latent_hiddens=[
            [[float(step), float(step + 1), float(step + 2)]]
            for step in range(5)
        ],
        world_model_states=[
            [0.0, 0.5],
            [0.2, 0.7],
            [1.0, 1.5],
            [1.2, 1.7],
            [2.0, 2.5],
        ],
        policy_credit_assignment="none",
        policy_step_indices=[0, 1, 2, 3],
        policy_token_ids=[
            [50 + step, action_tokens[action], 200]
            for step, action in enumerate(actions)
        ],
        policy_token_log_probs=[[None, None, None] for _ in actions],
        policy_loss_masks=[[False, False, False] for _ in actions],
        policy_token_roles=[
            ["reasoning", "action", "injected"] for _ in actions
        ],
        policy_action_token_ids=[action_tokens for _ in actions],
        policy_reasoning_texts=[f"step {step}" for step in range(4)],
        policy_finish_reasons=["stop" for _ in actions],
        policy_reasoning_truncated=[False for _ in actions],
        planner_policy_traces=traces,
        prompt_template_spec=prompt.spec,
    )


class _PrefixInputBuilder:
    processor = None

    def __init__(self) -> None:
        self.assistant_counts: list[int] = []

    def build(self, messages, _images, *, include_labels):  # type: ignore[no-untyped-def]
        assert include_labels is False
        rows = []
        for prompt_messages in messages:
            assistant_count = sum(
                message["role"] == "assistant" for message in prompt_messages
            )
            self.assistant_counts.append(assistant_count)
            rows.append([float(assistant_count - 1), 1.0, 1.0])
        return BackboneBatch(tensors={"features": torch.tensor(rows)})


class _PrefixBackbone(Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.module = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.module.weight.copy_(torch.eye(3))

    @property
    def model(self) -> torch.nn.Module:
        return self.module

    def forward(self, batch: BackboneBatch, **_kwargs) -> BackboneOutput:
        hidden = self.module(batch.tensors["features"]).unsqueeze(1)
        return BackboneOutput(hidden)

    def with_model(self, model: torch.nn.Module) -> "_PrefixBackbone":
        result = _PrefixBackbone()
        result.module = model  # type: ignore[assignment]
        return result

    def save_pretrained(self, output_dir, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _StateProjector(torch.nn.Linear):
    qwen_hidden_dim = 3
    latent_token_count = 1

    def __init__(self) -> None:
        super().__init__(3, 2, bias=False)
        with torch.no_grad():
            self.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return super().forward(hidden[:, 0] if hidden.ndim == 3 else hidden)


class _SequencePredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"history_size": 4})()
        self.state = torch.nn.Linear(2, 2, bias=False)
        self.action = torch.nn.Embedding(8, 2)
        with torch.no_grad():
            self.state.weight.copy_(torch.eye(2))
            self.action.weight.zero_()

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.state(states) + self.action(actions)


class _RecordingValueHead(torch.nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 8, bias=False)
        self.last_state: torch.Tensor | None = None
        with torch.no_grad():
            self.weight.fill_(0.5)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        self.last_state = state
        return super().forward(state)


def _runtime() -> tuple[
    RLModelRuntime,
    _PrefixBackbone,
    _PrefixInputBuilder,
    _StateProjector,
    _SequencePredictor,
    _RecordingValueHead,
]:
    backbone = _PrefixBackbone()
    builder = _PrefixInputBuilder()
    projector = _StateProjector()
    predictor = _SequencePredictor()
    value_head = _RecordingValueHead()
    runtime = RLModelRuntime(
        agent=Agent(
            backbone=backbone,
            wm=WorldModel(
                state_proj=projector,
                wm_predictor=predictor,
                value_head=value_head,
            ),
        ),
        input_builder=builder,  # type: ignore[arg-type]
        state_source="recompute",
        representation_to_backbone=True,
        policy_replay=None,
    )
    return runtime, backbone, builder, projector, predictor, value_head


def _algorithm(*, train_world_model: bool = False, dino_weight: float = 0.0) -> RLAlgorithm:
    return RLAlgorithm(
        history_size=4,
        sigreg=None,
        sigreg_weight=0.0,
        value_rank_margin=0.1,
        value_rank_weight=0.0,
        ppo_clip_ratio=0.2,
        entropy_weight=0.0,
        train_world_model=train_world_model,
        world_model_weight=0.75,
        dino_grid_weight=dino_weight,
    )


def test_episode_builds_one_training_transition_per_executed_action() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]

    assert [transition.step_index for transition in episode.transitions] == [0, 1, 2, 3]
    assert torch.allclose(
        episode.transitions[2].state_history(4),
        torch.tensor([[0.0, 0.5], [0.2, 0.7], [1.0, 1.5]]),
    )
    assert torch.allclose(
        episode.transitions[2].actual_next_state(),
        torch.tensor([1.2, 1.7]),
    )
    assert [transition.next_image_path for transition in episode.transitions] == [
        "step_1.png",
        "step_2.png",
        "step_3.png",
        "step_4.png",
    ]


def test_planned_tail_is_diagnostic_and_is_not_bound_to_later_actions() -> None:
    trajectory = _planner_trajectory()
    trace = trajectory.planner_policy_traces[0]
    trajectory.planner_policy_traces[0] = replace(
        trace,
        candidate_sequences=((0, 7),),
    )

    # Step 1 really executed action 1.  The previous search's tail action 7 is
    # deliberately ignored because a fresh search owns step 1.
    build_episode_training_batches(
        [trajectory],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )


def test_value_loss_on_predicted_next_state_reaches_full_prefix_qwen_and_wm() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]
    runtime, backbone, builder, projector, predictor, value_head = _runtime()
    transition = episode.transitions[2]

    output = _algorithm().actor_transition_step(
        runtime,
        transition,
        return_target=torch.tensor(5.0),
        total_transitions=1,
    )
    output.loss.backward()

    assert runtime.policy_replay is None
    assert builder.assistant_counts == [3]
    assert value_head.last_state is not None
    assert value_head.last_state.grad_fn is not None
    assert projector.weight.grad is not None
    assert predictor.state.weight.grad is not None
    assert backbone.module.weight.grad is not None
    # Column 0 is multiplied by the number of previous assistant responses in
    # the current full-prefix forward.  Its gradient proves that previous-history
    # activations were not detached from this step's Qwen graph.
    assert torch.count_nonzero(backbone.module.weight.grad[:, 0]) > 0

    executed_row = transition.action_index
    assert value_head.weight.grad is not None
    assert torch.count_nonzero(value_head.weight.grad[executed_row]) > 0
    other_rows = torch.cat(
        (
            value_head.weight.grad[:executed_row],
            value_head.weight.grad[executed_row + 1 :],
        )
    )
    assert torch.count_nonzero(other_rows) == 0


def test_transition_wm_target_is_saved_next_state_not_a_second_qwen_forward() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]
    runtime, _backbone, builder, _projector, _predictor, _value_head = _runtime()

    output = _algorithm(train_world_model=True).actor_transition_step(
        runtime,
        episode.transitions[1],
        return_target=episode.return_targets[1],
        total_transitions=1,
    )

    assert output.losses["wm"] is not None
    assert builder.assistant_counts == [2]


def test_transition_adds_dino_loss_for_each_real_next_observation() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]
    runtime, *_rest = _runtime()

    outputs = [
        _algorithm(train_world_model=True, dino_weight=0.25).actor_transition_step(
            runtime,
            transition,
            return_target=episode.return_targets[index],
            total_transitions=len(episode.transitions),
            dino_grid_target=torch.full((1, 2), float(index + 1)),
        )
        for index, transition in enumerate(episode.transitions)
    ]

    assert all(output.losses["dino"] is not None for output in outputs)
    assert all(output.metrics["lambda_wm"] == 0.75 for output in outputs)
    assert all(output.metrics["lambda_dino"] == 0.25 for output in outputs)


def test_transition_rejects_detached_rollout_qwen_mode() -> None:
    episode = build_episode_training_batches(
        [_planner_trajectory()],
        gamma=1.0,
        truncated_bootstrap=0.0,
    )[0]
    runtime, *_rest = _runtime()
    detached_runtime = replace(
        runtime,
        state_source="rollout",
        representation_to_backbone=False,
    )

    with pytest.raises(RuntimeError, match="full-prefix Qwen recomputation"):
        _algorithm().actor_transition_step(
            detached_runtime,
            episode.transitions[0],
            return_target=episode.return_targets[0],
            total_transitions=1,
        )
