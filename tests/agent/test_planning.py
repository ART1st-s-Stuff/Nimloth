"""Agent 的 Qwen 慢路径、WM 快速规划与 environment 执行边界测试。"""

from __future__ import annotations

import torch

from nimloth.agent import (
    AgentPrompt,
    PolicyDecision,
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


def _planning_world_model() -> tuple[WorldModel, _RecordingPredictor]:
    predictor = _RecordingPredictor()
    return (
        WorldModel(
            state_proj=torch.nn.Identity(),
            wm_predictor=predictor,
            value_head=_ActionValueHead(len(NAVIGATION_ACTION_SPACE)),
        ),
        predictor,
    )


def test_planner_replays_complete_candidate_history_at_every_depth() -> None:
    world_model, predictor = _planning_world_model()
    planner = WorldModelPlanner(
        world_model,
        horizon=3,
        search_mode="exhaustive",
    )

    plan = planner.plan(
        torch.tensor([[[0.25, -0.25]]]),
        torch.empty((1, 0), dtype=torch.long),
    )

    assert [actions.shape[1] for actions in predictor.action_sequences] == [1, 2, 3]
    assert plan.candidate_sequences.shape == (8**3, 3)
    assert plan.candidate_scores.shape == (8**3,)
    assert plan.root_action_scores.shape == (len(NAVIGATION_ACTION_SPACE),)
    assert torch.isfinite(plan.root_action_scores).all()


class _TurnPolicy:
    credit_assignment = "token"

    def reset_episode(self) -> None:
        pass

    def select_response_with_state(self, _prompt):
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
                latent_hidden=torch.tensor([[0.25, -0.25]]),
                action_logits=torch.arange(8, dtype=torch.float32),
            ),
        )

    def generate_state_prefix(self, _prompt):
        return "<think>terminal</think><|latent_state|><|action_start|>"


def test_planning_policy_uses_same_turn_hidden_and_excludes_action_from_ppo() -> None:
    world_model, predictor = _planning_world_model()
    policy = PlanningPolicy(
        turn_policy=_TurnPolicy(),
        world_model=world_model,
        horizon=2,
        search_mode="exhaustive",
        teacher_temperature=1.0,
        planner_device=torch.device("cpu"),
    )
    prompt = AgentPrompt(
        messages=({"role": "assistant", "content": "<think>"},),
        images=(),
        template=PromptTemplateSpec("test", "v1"),
    )

    decision = policy.select_action(prompt)

    assert decision.planner_trace is not None
    assert len(decision.planner_trace.candidate_sequences) == 64
    action_position = decision.token_trace.token_roles.index("action")
    assert decision.token_trace.loss_mask[action_position] is False
    assert decision.token_trace.old_log_probs[action_position] is None
    assert f"<|action_({decision.action_index})|>" in decision.response
    assert predictor.real_history_lengths == [1, 1]
