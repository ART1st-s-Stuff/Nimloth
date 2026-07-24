"""Agent 的慢速 observation 编码与快速 World Model 规划。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nimloth.agent.model import Agent
from nimloth.agent.policy import PolicyDecision, sample_policy_decision
from nimloth.agent.template import AgentPrompt
from nimloth.backbone.base import BackboneInputBuilder
from nimloth.util.module import evaluating
from nimloth.wm.model import WorldModel


@dataclass(frozen=True)
class WorldModelPlan:
    """一次搜索保留的候选序列以及每个根动作的最终 score。"""

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
        beam_width: int,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"planning horizon must be positive, got {horizon}")
        if beam_width < 1:
            raise ValueError(f"planning beam_width must be positive, got {beam_width}")
        self.world_model = world_model
        self.horizon = int(horizon)
        self.beam_width = int(beam_width)

    def plan(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> WorldModelPlan:
        """从最近的真实 state/action 上下文搜索未来动作。"""

        if state_history.ndim != 3 or state_history.shape[0] != 1:
            raise ValueError(
                "online planning requires state_history with shape (1,L,D), "
                f"got {tuple(state_history.shape)}"
            )
        expected_actions = (1, state_history.shape[1] - 1)
        if previous_actions.shape != expected_actions:
            raise ValueError(
                "previous_actions must align with state_history, "
                f"got {tuple(previous_actions.shape)}, expected {expected_actions}"
            )

        current_state = state_history[:, -1]
        action_values = self.world_model.predict_action_values(current_state)
        if action_values.ndim != 2 or action_values.shape[0] != 1:
            raise ValueError(
                "value head must return one action row for online planning, "
                f"got {tuple(action_values.shape)}"
            )
        action_count = action_values.shape[-1]
        actions = torch.arange(
            action_count,
            device=state_history.device,
            dtype=torch.long,
        )
        sequences = torch.empty(
            (1, 0),
            device=state_history.device,
            dtype=torch.long,
        )

        for _depth in range(self.horizon):
            parent_count = sequences.shape[0]
            expanded_sequences = torch.cat(
                (
                    sequences.repeat_interleave(action_count, dim=0),
                    actions.repeat(parent_count).unsqueeze(1),
                ),
                dim=1,
            )
            # 每一层都从相同的真实 history 重放完整候选序列。
            candidate_count = expanded_sequences.shape[0]
            candidate_states = state_history.expand(candidate_count, -1, -1)
            candidate_previous_actions = previous_actions.expand(
                candidate_count,
                -1,
            )
            predicted_states = self.world_model.simulate_action_sequences(
                candidate_states,
                candidate_previous_actions,
                expanded_sequences,
            )
            leaf_states = predicted_states[:, -1]
            leaf_action_values = self.world_model.predict_action_values(leaf_states)
            expanded_scores = leaf_action_values.max(dim=-1).values
            if not torch.isfinite(expanded_scores).all():
                raise ValueError("planning produced non-finite leaf action values")

            keep = min(self.beam_width, expanded_sequences.shape[0])
            scores, indices = expanded_scores.topk(keep)
            sequences = expanded_sequences.index_select(0, indices)

        # PPO 尚未支持 planner replay，但 rollout 仍记录真实 behavior distribution。
        # 已被 beam 剪枝的根动作保留为 -inf，不会被后续采样选中。
        root_scores = torch.full(
            (action_count,),
            float("-inf"),
            device=state_history.device,
            dtype=scores.dtype,
        )
        for action_index in range(action_count):
            action_scores = scores[sequences[:, 0] == action_index]
            if action_scores.numel() > 0:
                root_scores[action_index] = action_scores.max()
        return WorldModelPlan(
            candidate_sequences=sequences,
            candidate_scores=scores,
            root_action_scores=root_scores,
        )


class PlanningPolicy:
    """每个真实 environment step 执行一次 Qwen，再用 WM 搜索首动作。"""

    prompt_mode = "action"

    def __init__(
        self,
        *,
        agent: Agent,
        input_builder: BackboneInputBuilder,
        horizon: int,
        beam_width: int,
        temperature: float,
        top_p: float,
    ) -> None:
        self.agent = agent
        self.input_builder = input_builder
        self.planner = WorldModelPlanner(
            agent.wm,
            horizon=horizon,
            beam_width=beam_width,
        )
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        predictor = agent.wm.wm_predictor
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

        self._state_history.clear()
        self._action_history.clear()

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        """编码当前真实 observation，模拟候选序列并只返回要执行的首动作。"""

        with evaluating(self.agent), torch.no_grad():
            batch = self.input_builder.build(
                [prompt.unbound_messages()],
                [prompt.images],
                include_labels=False,
            )
            encoded = self.agent.encode_state(
                batch,
                include_lm_loss=False,
            )
            state = encoded.state
            if state.ndim != 2 or state.shape[0] != 1:
                raise ValueError(
                    "planning state encoder must return shape (1,D), "
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
            decision = sample_policy_decision(
                plan.root_action_scores,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            self._action_history.append(decision.action_index)
            return decision


__all__ = ["PlanningPolicy", "WorldModelPlan", "WorldModelPlanner"]
