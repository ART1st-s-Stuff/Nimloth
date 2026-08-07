"""Agent 的 Qwen 慢路径、WM 快速规划与 environment 执行边界测试。"""

from __future__ import annotations

import pytest
import torch

from nimloth.agent import (
    AgentPrompt,
    PolicyDecision,
    PolicyState,
    PolicyTokenTrace,
    PromptTemplateSpec,
)
from nimloth.agent.planning import PlanningPolicy, WorldModelPlanner
from nimloth.environment.navigation import NAVIGATION_ACTION_SPACE
from nimloth.backbone.qwen25vl.vllm_hidden import VLLMPolicyState
from nimloth.backbone.qwen25vl.vllm_policy import QwenTurnGeneration
from nimloth.wm import WorldModel


class _RecordingPredictor(torch.nn.Module):
    def __init__(self, *, history_size: int = 2) -> None:
        super().__init__()
        self.config = type("PredictorConfig", (), {"history_size": history_size})()
        self.action_contexts: list[torch.Tensor] = []
        self.state_contexts: list[torch.Tensor] = []

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self.action_contexts.append(actions.detach().clone())
        self.state_contexts.append(states.detach().clone())
        delta = actions.to(states.dtype).unsqueeze(-1) + 1.0
        return states + torch.cat((delta, -delta), dim=-1)


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


class _FirstTokenProjector(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != 1:
            raise ValueError(f"expected one latent token, got {tuple(hidden.shape)}")
        return hidden[:, 0]


def _planning_world_model() -> tuple[WorldModel, _RecordingPredictor]:
    predictor = _RecordingPredictor()
    return (
        WorldModel(
            state_proj=_FirstTokenProjector(),
            wm_predictor=predictor,
            value_head=_ActionValueHead(len(NAVIGATION_ACTION_SPACE)),
        ),
        predictor,
    )


def test_planner_replays_one_greedy_prefix_at_every_depth() -> None:
    world_model, predictor = _planning_world_model()
    planner = WorldModelPlanner(
        world_model,
        horizon=3,
        search_mode="greedy",
    )

    plan = planner.plan(
        torch.tensor([[[0.25, -0.25]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    assert [actions.shape[1] for actions in predictor.action_contexts] == [
        1,
        1,
        2,
        1,
        2,
        2,
    ]
    assert plan.candidate_sequences.tolist() == [[0, 1, 3]]
    assert plan.candidate_scores.shape == (1,)
    assert plan.root_action_scores.shape == (len(NAVIGATION_ACTION_SPACE),)
    assert torch.isfinite(plan.root_action_scores[0])
    assert torch.isneginf(plan.root_action_scores[1:]).all()


def test_exhaustive_planner_simulates_all_action_sequences_as_one_batch() -> None:
    world_model, predictor = _planning_world_model()
    planner = WorldModelPlanner(
        world_model,
        horizon=2,
        search_mode="exhaustive",
    )

    plan = planner.plan(
        torch.tensor([[[0.25, -0.25]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    action_count = len(NAVIGATION_ACTION_SPACE)
    assert predictor.action_contexts[0].shape == (action_count**2, 1)
    assert predictor.action_contexts[1].shape == (action_count**2, 2)
    assert plan.candidate_sequences.shape == (action_count**2, 2)
    assert plan.candidate_sequences[0].tolist() == [0, 0]
    assert plan.candidate_sequences[-1].tolist() == [action_count - 1] * 2
    assert torch.isfinite(plan.candidate_scores).all()
    assert torch.isfinite(plan.root_action_scores).all()


class _FirstActionPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("PredictorConfig", (), {"history_size": 1})()

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        first_action = torch.where(
            states[..., 0] == 0,
            actions.to(states.dtype),
            states[..., 1],
        )
        return torch.stack((states[..., 0] + 1, first_action), dim=-1)


class _LookaheadValueHead(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action_count = len(NAVIGATION_ACTION_SPACE)
        root_values = -torch.arange(
            action_count,
            device=state.device,
            dtype=state.dtype,
        ).expand(state.shape[0], -1)
        leaf_values = state[:, 1:2].expand(-1, action_count)
        return torch.where(state[:, :1] == 0, root_values, leaf_values)


def test_exhaustive_lookahead_can_reverse_the_root_greedy_action() -> None:
    world_model = WorldModel(
        state_proj=torch.nn.Identity(),
        wm_predictor=_FirstActionPredictor(),
        value_head=_LookaheadValueHead(),
    )
    state_history = torch.tensor([[[0.0, 0.0]]])
    previous_actions = torch.empty((1, 0), dtype=torch.long)

    greedy = WorldModelPlanner(
        world_model,
        horizon=2,
        search_mode="greedy",
    ).plan(state_history, previous_actions)
    exhaustive = WorldModelPlanner(
        world_model,
        horizon=2,
        search_mode="exhaustive",
    ).plan(state_history, previous_actions)

    greedy_action = int(greedy.candidate_sequences[0, 0])
    best_candidate = int(exhaustive.candidate_scores.argmax())
    exhaustive_action = int(exhaustive.candidate_sequences[best_candidate, 0])
    assert greedy_action == 0
    assert exhaustive_action == len(NAVIGATION_ACTION_SPACE) - 1


class _MCTSIncomingActionPredictor(torch.nn.Module):
    def __init__(self, *, history_size: int = 1) -> None:
        super().__init__()
        self.config = type(
            "PredictorConfig", (), {"history_size": history_size}
        )()

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        action = actions.to(states.dtype)
        first_action = torch.where(
            states[..., 0] == 0,
            action,
            states[..., 1],
        )
        return torch.stack(
            (states[..., 0] + 1, first_action, action),
            dim=-1,
        )


class _MCTSOutgoingActionValueHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_depths: list[int] = []

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action_count = len(NAVIGATION_ACTION_SPACE)
        self.state_depths.extend(int(value) for value in state[:, 0].tolist())
        # The synthetic decision state remembers the root action.  Every outgoing
        # slot has that value so the test isolates which temporal state was read.
        return state[:, 1:2].expand(-1, action_count)


class _ZeroMCTSProjector(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.new_zeros((hidden.shape[0], 3))


def test_mcts_uses_h1_k_step_final_outgoing_action_value() -> None:
    value_head = _MCTSOutgoingActionValueHead()
    world_model = WorldModel(
        state_proj=torch.nn.Identity(),
        wm_predictor=_MCTSIncomingActionPredictor(),
        value_head=value_head,
    )
    action_count = len(NAVIGATION_ACTION_SPACE)
    plan = WorldModelPlanner(
        world_model,
        horizon=4,
        search_mode="mcts",
        mcts_num_simulations=64,
        mcts_exploration_constant=1.0,
    ).plan(
        torch.tensor([[[0.0, 0.0, 0.0]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    assert plan.selected_action_index == action_count - 1
    assert plan.candidate_sequences.shape[1] == 4
    assert plan.candidate_visit_counts is not None
    assert int(plan.candidate_visit_counts.sum()) == 64
    assert plan.root_visit_counts is not None
    assert int(plan.root_visit_counts.sum()) == 64
    torch.testing.assert_close(
        plan.root_action_scores,
        torch.arange(action_count, dtype=torch.float32),
    )
    assert set(value_head.state_depths) == {0, 3}


def test_mcts_rejects_non_h1_predictor() -> None:
    world_model = WorldModel(
        state_proj=torch.nn.Identity(),
        wm_predictor=_MCTSIncomingActionPredictor(history_size=2),
        value_head=_MCTSOutgoingActionValueHead(),
    )
    planner = WorldModelPlanner(
        world_model,
        horizon=4,
        search_mode="mcts",
        mcts_num_simulations=8,
        mcts_exploration_constant=1.0,
    )

    with pytest.raises(ValueError, match="history_size=1"):
        planner.plan(
            torch.tensor([[[0.0, 0.0, 0.0]]]),
            torch.empty((1, 0), dtype=torch.long),
        )


def test_beam_planner_expands_multiple_candidates_at_each_depth() -> None:
    world_model, predictor = _planning_world_model()
    planner = WorldModelPlanner(
        world_model,
        horizon=3,
        search_mode="beam",
        beam_width=4,
    )

    plan = planner.plan(
        torch.tensor([[[0.25, -0.25]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    action_count = len(NAVIGATION_ACTION_SPACE)
    assert [tuple(actions.shape) for actions in predictor.action_contexts] == [
        (action_count, 1),
        (4 * action_count, 1),
        (4 * action_count, 2),
        (4 * action_count, 1),
        (4 * action_count, 2),
        (4 * action_count, 2),
    ]
    assert plan.candidate_sequences.shape == (4, 3)
    assert plan.candidate_scores.shape == (4,)


class _TurnPolicy:
    credit_assignment = "token"

    def __init__(self) -> None:
        self.action_calls = 0
        self.terminal_calls = 0

    def reset_episode(self) -> None:
        pass

    def select_response_with_state(self, _prompt):
        self.action_calls += 1
        return QwenTurnGeneration(
            qwen_decision=PolicyDecision(
                action_index=2,
                action_log_probs=tuple([-torch.log(torch.tensor(8.0)).item()] * 8),
                response=(
                    "<think>real cot</think><|latent_state|><|action_start|>"
                    "<|action_(2)|><|action_end|>"
                ),
                token_trace=PolicyTokenTrace(
                    token_ids=(5, 20, 25, 22),
                    old_log_probs=(-0.2, None, -torch.log(torch.tensor(8.0)).item(), None),
                    loss_mask=(True, False, True, False),
                    token_roles=("reasoning", "injected", "action", "injected"),
                    action_token_ids=tuple(range(23, 31)),
                    reasoning_text="real cot",
                    finish_reason="stop",
                ),
            ),
            policy_state=VLLMPolicyState(
                latent_hidden=torch.tensor(
                    [[0.25 * self.action_calls, -0.25 * self.action_calls]]
                ),
                action_logits=torch.arange(8, dtype=torch.float32),
            ),
        )

    def generate_state(self, _prompt):
        self.terminal_calls += 1
        return PolicyState(
            assistant_prefix=(
                "<think>terminal</think><|latent_state|><|action_start|>"
            ),
            latent_hidden=torch.tensor([[0.5, -0.5]]),
        )


def test_planning_policy_uses_batched_lookahead_and_excludes_action_from_ppo() -> None:
    world_model, predictor = _planning_world_model()
    stages: list[str] = []
    turn_policy = _TurnPolicy()
    policy = PlanningPolicy(
        turn_policy=turn_policy,
        world_model=world_model,
        horizon=2,
        search_mode="exhaustive",
        planner_device=torch.device("cpu"),
        progress_callback=stages.append,
    )
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )

    decision = policy.select_action(prompt)

    assert decision.planner_trace is not None
    assert decision.state_latent_hidden.tolist() == [[0.25, -0.25]]
    assert len(decision.planner_trace.candidate_sequences) == 64
    assert decision.planner_trace.search_mode == "exhaustive"
    assert decision.action_log_probs[0] == 0.0
    assert all(value == float("-inf") for value in decision.action_log_probs[1:])
    action_position = decision.token_trace.token_roles.index("action")
    assert decision.token_trace.loss_mask[action_position] is False
    assert decision.token_trace.old_log_probs[action_position] is None
    assert f"<|action_({decision.action_index})|>" in decision.response
    assert predictor.state_contexts[0].shape[1] == 1
    assert stages == ["planner_start", "planner_done"]


class _FixedPolicyHead(torch.nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = torch.arange(
            len(NAVIGATION_ACTION_SPACE),
            dtype=state.dtype,
            device=state.device,
        )
        return logits.unsqueeze(0).expand(state.shape[0], -1)


def test_planner_policy_head_samples_train_and_uses_argmax_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world_model, _predictor = _planning_world_model()
    world_model.planner_policy_head = _FixedPolicyHead()
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )
    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda *_args, **_kwargs: torch.tensor([1]),
    )
    training_policy = PlanningPolicy(
        turn_policy=_TurnPolicy(),
        world_model=world_model,
        horizon=1,
        search_mode="policy",
        planner_device=torch.device("cpu"),
        policy_temperature=0.5,
        sample_policy=True,
        policy_generator=torch.Generator().manual_seed(3),
    )
    evaluation_policy = PlanningPolicy(
        turn_policy=_TurnPolicy(),
        world_model=world_model,
        horizon=1,
        search_mode="policy",
        planner_device=torch.device("cpu"),
        policy_temperature=0.5,
        sample_policy=False,
    )

    sampled = training_policy.select_action(prompt)
    greedy = evaluation_policy.select_action(prompt)

    assert sampled.action_index == 1
    assert sampled.planner_trace.selection_mode == "policy_sample"
    assert sampled.action_log_probs == sampled.planner_trace.policy_action_log_probs
    assert greedy.action_index == len(NAVIGATION_ACTION_SPACE) - 1
    assert greedy.planner_trace.selection_mode == "policy_argmax"


def test_planning_policy_records_mcts_visits_and_executes_selected_root() -> None:
    world_model = WorldModel(
        state_proj=_ZeroMCTSProjector(),
        wm_predictor=_MCTSIncomingActionPredictor(),
        value_head=_MCTSOutgoingActionValueHead(),
    )
    policy = PlanningPolicy(
        turn_policy=_TurnPolicy(),
        world_model=world_model,
        horizon=2,
        search_mode="mcts",
        mcts_num_simulations=16,
        mcts_exploration_constant=1.0,
        planner_device=torch.device("cpu"),
    )
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )

    decision = policy.select_action(prompt)

    trace = decision.planner_trace
    assert trace is not None
    assert trace.search_mode == "mcts"
    assert trace.num_simulations == 16
    assert trace.root_visit_counts is not None
    assert sum(trace.root_visit_counts) == 16
    assert trace.candidate_visit_counts is not None
    assert sum(trace.candidate_visit_counts) == 16
    assert decision.action_index == len(NAVIGATION_ACTION_SPACE) - 1


def test_planning_policy_replans_every_step_from_real_qwen_state() -> None:
    world_model, predictor = _planning_world_model()
    turn_policy = _TurnPolicy()
    policy = PlanningPolicy(
        turn_policy=turn_policy,
        world_model=world_model,
        horizon=2,
        search_mode="greedy",
        planner_device=torch.device("cpu"),
    )
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )

    first_action = policy.select_action(prompt)
    second_action = policy.select_action(prompt)
    third_action = policy.select_action(prompt)

    assert turn_policy.action_calls == 3
    assert all(
        decision.token_trace is not None and decision.planner_trace is not None
        for decision in (first_action, second_action, third_action)
    )
    assert first_action.world_model_state.tolist() == [0.25, -0.25]
    assert second_action.world_model_state.tolist() == [0.5, -0.5]
    assert third_action.world_model_state.tolist() == [0.75, -0.75]
    assert all(
        decision.response.startswith("<think>real cot</think>")
        for decision in (first_action, second_action, third_action)
    )
    # history_size=2: the third search starts from the two latest real Qwen
    # states.  No predicted tail from either previous search is retained.
    assert predictor.state_contexts[-2].tolist() == [
        [[0.5, -0.5], [0.75, -0.75]]
    ]
