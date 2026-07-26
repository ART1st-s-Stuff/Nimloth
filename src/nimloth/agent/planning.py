"""Use a real vLLM CoT state for batched latent World Model planning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Protocol

import torch

from nimloth.agent.policy import (
    PlannerPolicyTrace,
    PolicyDecision,
    PolicyState,
    PolicyTokenTrace,
)
from nimloth.agent.template import AgentPrompt
from nimloth.latent import LatentActionTokens
from nimloth.util.module import evaluating
from nimloth.wm.model import WorldModel


class VLLMTurnStatePolicy(Protocol):
    """Narrow interface supplied by ``QwenVLLMAgentPolicy``."""

    credit_assignment: str

    def reset_episode(self) -> None: ...

    def select_response_with_state(self, prompt: AgentPrompt) -> Any: ...

    def generate_state(self, prompt: AgentPrompt) -> PolicyState: ...


@dataclass(frozen=True)
class WorldModelPlan:
    """One search result with scored latent action-sequence candidates."""

    candidate_sequences: torch.Tensor
    candidate_scores: torch.Tensor
    root_action_scores: torch.Tensor


class WorldModelPlanner:
    """在 latent 空间搜索动作，不拥有也不调用 environment。

    当前模型没有 reward/done head，因此搜索只使用叶节点的最大 action-value 作为
    启发式 score。它不是环境 return 的替代品，也不会把各步 Q-value 错误累加。
    """

    def __init__(
        self,
        world_model: WorldModel,
        *,
        horizon: int,
        search_mode: str,
        beam_width: int | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"planning horizon must be positive, got {horizon}")
        if search_mode not in {"greedy", "exhaustive", "beam"}:
            raise ValueError(
                "planning search_mode must be greedy, exhaustive, or beam"
            )
        if search_mode == "beam" and (beam_width is None or beam_width < 1):
            raise ValueError("beam search requires a positive beam_width")
        if search_mode != "beam" and beam_width is not None:
            raise ValueError("beam_width is only valid for beam search")
        self.world_model = world_model
        self.horizon = int(horizon)
        self.search_mode = search_mode
        self.beam_width = beam_width

    def _score_sequences(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        sequences: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = sequences.shape[0]
        expanded_history = state_history.expand(
            candidate_count,
            *state_history.shape[1:],
        )
        expanded_previous = previous_actions.expand(candidate_count, -1)
        predicted_states = self.world_model.simulate_action_sequences(
            expanded_history,
            expanded_previous,
            sequences,
        )
        leaf_action_values = self.world_model.predict_action_values(
            predicted_states[:, -1]
        )
        if (
            leaf_action_values.ndim != 2
            or leaf_action_values.shape[0] != candidate_count
        ):
            raise ValueError(
                "value head must return one action row per planning candidate, "
                f"got {tuple(leaf_action_values.shape)}"
            )
        scores = leaf_action_values.max(dim=-1).values
        if not torch.isfinite(scores).all():
            raise ValueError("planning produced a non-finite candidate score")
        return scores

    @staticmethod
    def _root_action_scores(
        sequences: torch.Tensor,
        scores: torch.Tensor,
        *,
        action_count: int,
    ) -> torch.Tensor:
        root_scores = scores.new_full((action_count,), float("-inf"))
        for action_index in range(action_count):
            selected = scores[sequences[:, 0] == action_index]
            if selected.numel() > 0:
                root_scores[action_index] = selected.max()
        return root_scores

    def plan(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> WorldModelPlan:
        """从最近的真实 state/action 上下文搜索未来动作。"""

        if state_history.ndim not in (3, 4) or state_history.shape[0] != 1:
            raise ValueError(
                "online planning requires state_history with shape "
                "(1,L,D) or (1,L,N,D), "
                f"got {tuple(state_history.shape)}"
            )
        expected_actions = (1, state_history.shape[1] - 1)
        if previous_actions.shape != expected_actions:
            raise ValueError(
                "previous_actions must align with state_history, "
                f"got {tuple(previous_actions.shape)}, expected {expected_actions}"
            )

        decision_state = state_history[:, -1]
        root_values = self.world_model.predict_action_values(decision_state)
        if root_values.ndim != 2 or root_values.shape[0] != 1:
            raise ValueError(
                "value head must return one root action row for online planning, "
                f"got {tuple(root_values.shape)}"
            )
        if not torch.isfinite(root_values).all():
            raise ValueError("planning produced non-finite root action values")
        action_count = root_values.shape[-1]

        if self.search_mode == "greedy":
            sequences = torch.empty(
                (1, 0),
                device=state_history.device,
                dtype=torch.long,
            )
            for _depth in range(self.horizon):
                action_values = self.world_model.predict_action_values(decision_state)
                if action_values.shape != (1, action_count):
                    raise ValueError(
                        "value head action count changed during greedy planning"
                    )
                if not torch.isfinite(action_values).all():
                    raise ValueError("planning produced non-finite action values")
                chosen_action = action_values.argmax(dim=-1, keepdim=True)
                sequences = torch.cat((sequences, chosen_action), dim=1)
                predicted_states = self.world_model.simulate_action_sequences(
                    state_history,
                    previous_actions,
                    sequences,
                )
                decision_state = predicted_states[:, -1]
            leaf_action_values = self.world_model.predict_action_values(decision_state)
            if leaf_action_values.shape != (1, action_count):
                raise ValueError("value head action count changed at the greedy leaf")
            scores = leaf_action_values.max(dim=-1).values
            if not torch.isfinite(scores).all():
                raise ValueError("planning produced a non-finite candidate score")
        elif self.search_mode == "exhaustive":
            sequences = torch.tensor(
                list(product(range(action_count), repeat=self.horizon)),
                dtype=torch.long,
                device=state_history.device,
            )
            scores = self._score_sequences(
                state_history,
                previous_actions,
                sequences,
            )
        else:
            assert self.beam_width is not None
            sequences = torch.empty(
                (1, 0),
                device=state_history.device,
                dtype=torch.long,
            )
            action_column = torch.arange(
                action_count,
                dtype=torch.long,
                device=state_history.device,
            )
            for depth in range(self.horizon):
                sequences = torch.cat(
                    (
                        sequences.repeat_interleave(action_count, dim=0),
                        action_column.repeat(sequences.shape[0]).unsqueeze(1),
                    ),
                    dim=1,
                )
                scores = self._score_sequences(
                    state_history,
                    previous_actions,
                    sequences,
                )
                if depth + 1 < self.horizon and len(scores) > self.beam_width:
                    selected = scores.topk(self.beam_width).indices
                    sequences = sequences[selected]
            if len(scores) > self.beam_width:
                selected = scores.topk(self.beam_width).indices
                sequences = sequences[selected]
                scores = scores[selected]

        return WorldModelPlan(
            candidate_sequences=sequences,
            candidate_scores=scores,
            root_action_scores=self._root_action_scores(
                sequences,
                scores,
                action_count=action_count,
            ),
        )


class PlanningPolicy:
    """Sample Qwen CoT, plan from its hidden state, then execute planner action."""

    prompt_mode = "response"
    credit_assignment = "token"

    def __init__(
        self,
        *,
        turn_policy: VLLMTurnStatePolicy,
        world_model: WorldModel,
        horizon: int,
        search_mode: str,
        beam_width: int | None = None,
        planner_device: torch.device,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if turn_policy.credit_assignment != "token":
            raise ValueError("planner policy requires token-level Qwen CoT credit")
        self.turn_policy = turn_policy
        self.world_model = world_model
        self.planner = WorldModelPlanner(
            world_model,
            horizon=horizon,
            search_mode=search_mode,
            beam_width=beam_width,
        )
        self.planner_device = planner_device
        self._progress_callback = progress_callback
        predictor = world_model.wm_predictor
        predictor_config = getattr(predictor, "config", None)
        self.history_size = int(getattr(predictor_config, "history_size", 0))
        if self.history_size < 1:
            raise ValueError(
                "PlanningPolicy requires wm_predictor.config.history_size"
            )
        self._state_history: list[torch.Tensor] = []
        self._action_history: list[int] = []

    def reset_episode(self) -> None:
        """清除上一个 episode 的真实 latent 历史。"""

        self.turn_policy.reset_episode()
        self._state_history.clear()
        self._action_history.clear()

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        """Generate one CoT, enumerate the latent plan, and replace its action."""

        generated = self.turn_policy.select_response_with_state(prompt)
        qwen_decision = generated.qwen_decision
        if qwen_decision.token_trace is None or qwen_decision.response is None:
            raise RuntimeError("vLLM planning turn lacks token/response provenance")
        if self._progress_callback is not None:
            self._progress_callback("planner_start")
        with evaluating(self.world_model), torch.no_grad():
            latent_hidden = generated.policy_state.latent_hidden.to(
                self.planner_device
            ).unsqueeze(0)
            state = self.world_model.project_state(latent_hidden)
            if state.ndim not in (2, 3) or state.shape[0] != 1:
                raise ValueError(
                    "planning state projector must return shape (1,D) or (1,N,D), "
                    f"got {tuple(state.shape)}"
                )
            self._state_history.append(state.detach())
            if len(self._state_history) > self.history_size:
                self._state_history.pop(0)
                self._action_history.pop(0)
            state_history = torch.stack(self._state_history, dim=1)
            previous_actions = torch.tensor(
                [self._action_history],
                dtype=torch.long,
                device=state.device,
            )
            plan = self.planner.plan(state_history, previous_actions)
            best_candidate = int(plan.candidate_scores.argmax().item())
            action_index = int(plan.candidate_sequences[best_candidate, 0].item())
            action_count = plan.root_action_scores.shape[0]
            behavior_log_probs = torch.full(
                (action_count,),
                float("-inf"),
                dtype=plan.candidate_scores.dtype,
                device=plan.candidate_scores.device,
            )
            behavior_log_probs[action_index] = 0.0
        if self._progress_callback is not None:
            self._progress_callback("planner_done")

        qwen_log_probs = torch.log_softmax(
            generated.policy_state.action_logits,
            dim=-1,
        )
        trace = qwen_decision.token_trace
        action_position = trace.token_roles.index("action")
        new_token_ids = list(trace.token_ids)
        new_token_ids[action_position] = trace.action_token_ids[action_index]
        new_old_log_probs = list(trace.old_log_probs)
        new_old_log_probs[action_position] = None
        new_loss_mask = list(trace.loss_mask)
        new_loss_mask[action_position] = False

        tokens = LatentActionTokens()
        old_suffix = (
            f"{tokens.action_start}{tokens.action_tokens[qwen_decision.action_index]}"
            f"{tokens.action_end}"
        )
        if not qwen_decision.response.endswith(old_suffix):
            raise RuntimeError("Qwen response action suffix does not match its trace")
        response = (
            qwen_decision.response[: -len(old_suffix)]
            + f"{tokens.action_start}{tokens.action_tokens[action_index]}"
            + tokens.action_end
        )
        planner_trace = PlannerPolicyTrace(
            qwen_action_log_probs=tuple(
                float(value) for value in qwen_log_probs.cpu().tolist()
            ),
            candidate_sequences=tuple(
                tuple(int(value) for value in row)
                for row in plan.candidate_sequences.cpu().tolist()
            ),
            candidate_scores=tuple(
                float(value) for value in plan.candidate_scores.cpu().tolist()
            ),
            root_action_scores=tuple(
                float(value) for value in plan.root_action_scores.cpu().tolist()
            ),
            teacher_action_log_probs=tuple(
                float(value) for value in behavior_log_probs.cpu().tolist()
            ),
            behavior_action_log_probs=tuple(
                float(value) for value in behavior_log_probs.cpu().tolist()
            ),
            horizon=self.planner.horizon,
            search_mode=self.planner.search_mode,
            beam_width=self.planner.beam_width,
        )
        self._action_history.append(action_index)
        return PolicyDecision(
            action_index=action_index,
            action_log_probs=planner_trace.behavior_action_log_probs,
            response=response,
            token_trace=PolicyTokenTrace(
                token_ids=tuple(new_token_ids),
                old_log_probs=tuple(new_old_log_probs),
                loss_mask=tuple(new_loss_mask),
                token_roles=trace.token_roles,
                action_token_ids=trace.action_token_ids,
                reasoning_text=trace.reasoning_text,
                finish_reason=trace.finish_reason,
                reasoning_truncated=trace.reasoning_truncated,
            ),
            planner_trace=planner_trace,
            state_latent_hidden=(
                generated.policy_state.latent_hidden.detach().cpu().clone()
            ),
        )

    def generate_state(self, prompt: AgentPrompt) -> PolicyState:
        """Terminal state 生成真实 CoT/hidden，但不运行 planner 或 environment。"""

        state = self.turn_policy.generate_state(prompt)
        if state.latent_hidden is None:
            raise RuntimeError("planner terminal state has no captured Qwen hidden")
        return state


__all__ = ["PlanningPolicy", "WorldModelPlan", "WorldModelPlanner"]
