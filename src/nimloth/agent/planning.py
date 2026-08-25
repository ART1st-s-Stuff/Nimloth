"""Use a real vLLM CoT state for batched latent World Model planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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

    def select_responses_with_state(
        self,
        prompts: tuple[AgentPrompt, ...],
    ) -> tuple[Any, ...]: ...

    def generate_state(self, prompt: AgentPrompt) -> PolicyState: ...

    def generate_states(
        self,
        prompts: tuple[AgentPrompt, ...],
    ) -> tuple[PolicyState, ...]: ...


@dataclass(frozen=True)
class WorldModelPlan:
    """One search result with scored latent action-sequence candidates."""

    candidate_sequences: torch.Tensor
    candidate_scores: torch.Tensor
    root_action_scores: torch.Tensor
    selected_action_index: int
    candidate_visit_counts: torch.Tensor | None = None
    root_visit_counts: torch.Tensor | None = None
    mcts_trace: dict[str, Any] | None = None


@dataclass
class _MCTSNode:
    """One deterministic latent state in the UCT tree."""

    state: torch.Tensor
    sequence: tuple[int, ...]
    children: dict[int, "_MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count < 1:
            raise RuntimeError("MCTS node has no backed-up value")
        return self.value_sum / self.visit_count


@dataclass
class _MCTSLeafStats:
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visit_count < 1:
            raise RuntimeError("MCTS leaf has no evaluation")
        return self.value_sum / self.visit_count


class WorldModelPlanner:
    """在 latent 空间搜索动作，不拥有也不调用 environment。

    当前模型没有 reward/done head，因此搜索只使用候选路径最后一条 simulated
    edge 的标准 outgoing action-value 作为启发式 score。所有搜索模式都不会把
    各步 MC return 错误累加。
    """

    def __init__(
        self,
        world_model: WorldModel,
        *,
        horizon: int,
        search_mode: str,
        beam_width: int | None = None,
        mcts_num_simulations: int | None = None,
        mcts_exploration_constant: float | None = None,
        capture_mcts_trace: bool = False,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"planning horizon must be positive, got {horizon}")
        if search_mode not in {"greedy", "exhaustive", "beam", "mcts"}:
            raise ValueError(
                "planning search_mode must be greedy, exhaustive, beam, or mcts"
            )
        if search_mode == "beam" and (beam_width is None or beam_width < 1):
            raise ValueError("beam search requires a positive beam_width")
        if search_mode != "beam" and beam_width is not None:
            raise ValueError("beam_width is only valid for beam search")
        if search_mode == "mcts":
            if mcts_num_simulations is None or mcts_num_simulations < 1:
                raise ValueError("MCTS requires a positive num_simulations")
            if (
                mcts_exploration_constant is None
                or not math.isfinite(mcts_exploration_constant)
                or mcts_exploration_constant < 0.0
            ):
                raise ValueError(
                    "MCTS requires a finite non-negative exploration_constant"
                )
        elif (
            mcts_num_simulations is not None
            or mcts_exploration_constant is not None
            or capture_mcts_trace
        ):
            raise ValueError("MCTS parameters and trace capture are only valid for mcts search")
        self.world_model = world_model
        self.horizon = int(horizon)
        self.search_mode = search_mode
        self.beam_width = beam_width
        self.mcts_num_simulations = mcts_num_simulations
        self.mcts_exploration_constant = mcts_exploration_constant
        self.capture_mcts_trace = bool(capture_mcts_trace)

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
        final_decision_state = (
            expanded_history[:, -1]
            if sequences.shape[1] == 1
            else predicted_states[:, -2]
        )
        final_action_values = self.world_model.predict_action_values(
            final_decision_state
        )
        if (
            final_action_values.ndim != 2
            or final_action_values.shape[0] != candidate_count
        ):
            raise ValueError(
                "value head must return one action row per planning candidate, "
                f"got {tuple(final_action_values.shape)}"
            )
        scores = final_action_values.gather(
            dim=-1,
            index=sequences[:, -1:].to(device=final_action_values.device),
        ).squeeze(-1)
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

    def _mcts_child_state(
        self,
        state: torch.Tensor,
        action_index: int,
    ) -> torch.Tensor:
        """Advance one predicted step under the required H=1 SFT2 contract."""

        action = torch.tensor(
            [[action_index]],
            dtype=torch.long,
            device=state.device,
        )
        predicted = self.world_model.simulate_action_sequences(
            state.unsqueeze(1),
            torch.empty((1, 0), dtype=torch.long, device=state.device),
            action,
        )
        return predicted[:, -1]

    def _select_uct_child(self, node: _MCTSNode) -> _MCTSNode:
        assert self.mcts_exploration_constant is not None
        if node.visit_count < 1:
            raise RuntimeError("cannot select from an unvisited MCTS node")
        log_parent_visits = math.log(node.visit_count)

        def uct_score(item: tuple[int, _MCTSNode]) -> tuple[float, int]:
            action_index, child = item
            if child.visit_count < 1:
                raise RuntimeError("expanded MCTS child has no visit")
            exploration = self.mcts_exploration_constant * math.sqrt(
                log_parent_visits / child.visit_count
            )
            # Stable tie-break: the lower navigation action index wins.
            return child.mean_value + exploration, -action_index

        return max(node.children.items(), key=uct_score)[1]

    def _plan_mcts(
        self,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        *,
        action_count: int,
    ) -> WorldModelPlan:
        """Run deterministic UCT and back up the final simulated edge value.

        Every simulation reaches exactly ``horizon`` predicted transitions.  The
        K-action path is scored as ``Q(predicted_state_{K-1}, action_K)``: the
        final action is outgoing from its decision state, and the state it produces
        is never mislabeled as that action's source.  MC-return predictions are not
        accumulated across depth.
        """

        predictor = getattr(self.world_model.wm_predictor, "module", None)
        predictor = predictor or self.world_model.wm_predictor
        history_size = int(
            getattr(getattr(predictor, "config", None), "history_size", 0)
        )
        if history_size != 1 or state_history.shape[1] != 1:
            raise ValueError(
                "MCTS evaluation requires SFT2 history_size=1 and one real state; "
                f"checkpoint H={history_size}, input L={state_history.shape[1]}"
            )
        if previous_actions.shape != (1, 0):
            raise ValueError("H=1 MCTS must not receive previous actions")
        assert self.mcts_num_simulations is not None
        if self.mcts_num_simulations < action_count:
            raise ValueError(
                "MCTS num_simulations must visit every root action at least once: "
                f"simulations={self.mcts_num_simulations}, actions={action_count}"
            )

        root = _MCTSNode(state=state_history[:, -1], sequence=())
        leaf_stats: dict[tuple[int, ...], _MCTSLeafStats] = {}
        simulations: list[dict[str, Any]] = []
        for simulation_index in range(self.mcts_num_simulations):
            node = root
            path = [root]
            selection_steps: list[dict[str, Any]] = []
            while len(node.sequence) < self.horizon:
                parent = node
                if len(parent.children) < action_count:
                    action_index = next(
                        action
                        for action in range(action_count)
                        if action not in parent.children
                    )
                    child = _MCTSNode(
                        state=self._mcts_child_state(parent.state, action_index),
                        sequence=(*parent.sequence, action_index),
                    )
                    parent.children[action_index] = child
                    operation = "expand"
                    uct_candidates: list[dict[str, Any]] = []
                else:
                    if parent.visit_count < 1:
                        raise RuntimeError("cannot select from an unvisited MCTS node")
                    log_parent_visits = math.log(parent.visit_count)
                    assert self.mcts_exploration_constant is not None
                    uct_candidates = []
                    for candidate_action, candidate_child in sorted(parent.children.items()):
                        exploration = self.mcts_exploration_constant * math.sqrt(
                            log_parent_visits / candidate_child.visit_count
                        )
                        uct_candidates.append(
                            {
                                "action_id": candidate_action,
                                "child_sequence": candidate_child.sequence,
                                "visit_count": candidate_child.visit_count,
                                "mean_value": candidate_child.mean_value,
                                "exploration_bonus": exploration,
                                "uct_score": candidate_child.mean_value + exploration,
                            }
                        )
                    child = self._select_uct_child(parent)
                    action_index = child.sequence[-1]
                    operation = "select"
                if self.capture_mcts_trace:
                    selection_steps.append(
                        {
                            "depth": len(parent.sequence),
                            "parent_sequence": parent.sequence,
                            "parent_visit_count": parent.visit_count,
                            "operation": operation,
                            "action_id": action_index,
                            "child_sequence": child.sequence,
                            "uct_candidates": uct_candidates,
                        }
                    )
                node = child
                path.append(node)

            final_action = node.sequence[-1]
            final_decision_state = path[-2].state
            leaf_action_values = self.world_model.predict_action_values(
                final_decision_state
            )
            if leaf_action_values.shape != (1, action_count):
                raise ValueError(
                    "value head action count changed at the MCTS leaf"
                )
            leaf_value = float(leaf_action_values[0, final_action].item())
            if not math.isfinite(leaf_value):
                raise ValueError("MCTS leaf evaluation produced a non-finite value")

            stats = leaf_stats.setdefault(node.sequence, _MCTSLeafStats())
            stats.visit_count += 1
            stats.value_sum += leaf_value
            backups: list[dict[str, Any]] = []
            for visited in path:
                before_visits = visited.visit_count
                before_sum = visited.value_sum
                visited.visit_count += 1
                visited.value_sum += leaf_value
                if self.capture_mcts_trace:
                    backups.append(
                        {
                            "sequence": visited.sequence,
                            "visit_count_before": before_visits,
                            "value_sum_before": before_sum,
                            "visit_count_after": visited.visit_count,
                            "value_sum_after": visited.value_sum,
                            "mean_value_after": visited.mean_value,
                        }
                    )
            if self.capture_mcts_trace:
                simulations.append(
                    {
                        "simulation_index": simulation_index,
                        "selection_steps": selection_steps,
                        "leaf": {
                            "sequence": node.sequence,
                            "decision_sequence": path[-2].sequence,
                            "action_id": final_action,
                            "action_values": leaf_action_values[0].detach().clone(),
                            "value": leaf_value,
                        },
                        "backups": backups,
                    }
                )

        if len(root.children) != action_count:
            raise RuntimeError("MCTS did not expand every root action")
        root_visit_counts = torch.tensor(
            [root.children[action].visit_count for action in range(action_count)],
            dtype=torch.long,
            device=state_history.device,
        )
        root_action_scores = torch.tensor(
            [root.children[action].mean_value for action in range(action_count)],
            dtype=torch.float32,
            device=state_history.device,
        )
        selected_action_index = max(
            range(action_count),
            key=lambda action: (
                int(root_visit_counts[action].item()),
                float(root_action_scores[action].item()),
                -action,
            ),
        )

        sequences = tuple(sorted(leaf_stats))
        candidate_sequences = torch.tensor(
            sequences,
            dtype=torch.long,
            device=state_history.device,
        )
        candidate_scores = torch.tensor(
            [leaf_stats[sequence].mean_value for sequence in sequences],
            dtype=torch.float32,
            device=state_history.device,
        )
        candidate_visit_counts = torch.tensor(
            [leaf_stats[sequence].visit_count for sequence in sequences],
            dtype=torch.long,
            device=state_history.device,
        )
        mcts_trace = None
        if self.capture_mcts_trace:
            tree_nodes: list[dict[str, Any]] = []

            def visit_tree(node: _MCTSNode) -> None:
                tree_nodes.append(
                    {
                        "sequence": node.sequence,
                        "depth": len(node.sequence),
                        "predicted_state": (
                            None if not node.sequence else node.state[0].detach().clone()
                        ),
                        "visit_count": node.visit_count,
                        "value_sum": node.value_sum,
                        "mean_value": node.mean_value,
                    }
                )
                for action_id in sorted(node.children):
                    visit_tree(node.children[action_id])

            visit_tree(root)
            mcts_trace = {
                "schema": "nimloth_k4_mcts_process_v1",
                "num_simulations": self.mcts_num_simulations,
                "horizon": self.horizon,
                "exploration_constant": self.mcts_exploration_constant,
                "tree_nodes": tree_nodes,
                "simulations": simulations,
            }
        return WorldModelPlan(
            candidate_sequences=candidate_sequences,
            candidate_scores=candidate_scores,
            root_action_scores=root_action_scores,
            selected_action_index=selected_action_index,
            candidate_visit_counts=candidate_visit_counts,
            root_visit_counts=root_visit_counts,
            mcts_trace=mcts_trace,
        )

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
            scores = action_values.gather(dim=-1, index=chosen_action).squeeze(-1)
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
        elif self.search_mode == "beam":
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

        else:
            return self._plan_mcts(
                state_history,
                previous_actions,
                action_count=action_count,
            )

        selected_candidate = int(scores.argmax().item())
        return WorldModelPlan(
            candidate_sequences=sequences,
            candidate_scores=scores,
            root_action_scores=self._root_action_scores(
                sequences,
                scores,
                action_count=action_count,
            ),
            selected_action_index=int(
                sequences[selected_candidate, 0].item()
            ),
        )


class PlanningPolicy:
    """Replan from a real Qwen state and execute only the selected root action."""

    prompt_mode = "response"
    credit_assignment = "none"

    def __init__(
        self,
        *,
        turn_policy: VLLMTurnStatePolicy,
        world_model: WorldModel,
        horizon: int,
        search_mode: str,
        beam_width: int | None = None,
        mcts_num_simulations: int | None = None,
        mcts_exploration_constant: float | None = None,
        planner_device: torch.device,
        policy_temperature: float | None = None,
        sample_policy: bool = False,
        policy_generator: torch.Generator | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> None:
        if turn_policy.credit_assignment not in {"turn", "token"}:
            raise ValueError("planner policy requires real Qwen response generation")
        self.turn_policy = turn_policy
        self.world_model = world_model
        self.horizon = int(horizon)
        self.search_mode = search_mode
        self.policy_temperature = policy_temperature
        self.sample_policy = bool(sample_policy)
        self.policy_generator = policy_generator
        if search_mode == "policy":
            if horizon != 1:
                raise ValueError("PlannerPolicyHead action selection requires horizon=1")
            if (
                beam_width is not None
                or mcts_num_simulations is not None
                or mcts_exploration_constant is not None
            ):
                raise ValueError("PlannerPolicyHead action selection has no search parameters")
            if policy_temperature is None or policy_temperature <= 0.0:
                raise ValueError("PlannerPolicyHead requires positive policy_temperature")
            if world_model.planner_policy_head is None:
                raise ValueError("policy search requires a PlannerPolicyHead")
            self.planner: WorldModelPlanner | None = None
        else:
            if policy_temperature is not None or sample_policy or policy_generator is not None:
                raise ValueError("policy sampling arguments require search_mode=policy")
            self.planner = WorldModelPlanner(
                world_model,
                horizon=horizon,
                search_mode=search_mode,
                beam_width=beam_width,
                mcts_num_simulations=mcts_num_simulations,
                mcts_exploration_constant=mcts_exploration_constant,
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
        """Clear the previous episode's real state/action history."""

        self.turn_policy.reset_episode()
        self._state_history.clear()
        self._action_history.clear()

    def select_action(self, prompt: AgentPrompt) -> PolicyDecision:
        """Search k latent steps, execute only the best candidate's first action."""

        generated = self.turn_policy.select_response_with_state(prompt)
        qwen_decision = generated.qwen_decision
        if qwen_decision.token_trace is None or qwen_decision.response is None:
            raise RuntimeError("vLLM planning turn lacks token/response provenance")
        if self._progress_callback is not None:
            self._progress_callback("planner_start")
        with evaluating(self.world_model), torch.no_grad():
            state = self._project_hidden(generated.policy_state.latent_hidden)
            self._append_actual_state(state)
            context_start = max(0, len(self._state_history) - self.history_size)
            state_history = torch.stack(
                self._state_history[context_start:],
                dim=1,
            )
            previous_actions = torch.tensor(
                [self._action_history[context_start:]],
                dtype=torch.long,
                device=state.device,
            )
            if self.search_mode == "policy":
                plan, behavior_log_probs = self._policy_plan(state)
            else:
                assert self.planner is not None
                plan = self.planner.plan(state_history, previous_actions)
                behavior_log_probs = None
            action_index = int(plan.selected_action_index)
            self._action_history.append(action_index)
        if self._progress_callback is not None:
            self._progress_callback("planner_done")

        return self._decision_from_plan(
            generated,
            plan,
            action_index,
            state,
            behavior_log_probs=behavior_log_probs,
        )

    def select_actions(
        self,
        prompts: tuple[AgentPrompt, ...],
    ) -> tuple[PolicyDecision, ...]:
        """Select H=1 PlannerPolicyHead actions for active envs in one Qwen call."""

        if self.search_mode != "policy":
            raise RuntimeError(
                "batched planning currently requires search_mode=policy"
            )
        if not prompts:
            raise ValueError("batched planning requires at least one prompt")
        generated_rows = self.turn_policy.select_responses_with_state(prompts)
        if len(generated_rows) != len(prompts):
            raise RuntimeError(
                "batched Qwen states do not align with planner prompts: "
                f"{len(generated_rows)} != {len(prompts)}"
            )
        if self._progress_callback is not None:
            self._progress_callback("planner_start")
        decisions: list[PolicyDecision] = []
        with evaluating(self.world_model), torch.no_grad():
            for generated in generated_rows:
                qwen_decision = generated.qwen_decision
                if qwen_decision.token_trace is None or qwen_decision.response is None:
                    raise RuntimeError(
                        "batched vLLM planning turn lacks token/response provenance"
                    )
                state = self._project_hidden(generated.policy_state.latent_hidden)
                plan, behavior_log_probs = self._policy_plan(state)
                decisions.append(
                    self._decision_from_plan(
                        generated,
                        plan,
                        int(plan.selected_action_index),
                        state,
                        behavior_log_probs=behavior_log_probs,
                    )
                )
        if self._progress_callback is not None:
            self._progress_callback("planner_done")
        return tuple(decisions)

    def _policy_plan(
        self,
        state: torch.Tensor,
    ) -> tuple[WorldModelPlan, tuple[float, ...]]:
        assert self.policy_temperature is not None
        logits = self.world_model.predict_action_logits(state)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                "PlannerPolicyHead must return one action-logit row, "
                f"got {tuple(logits.shape)}"
            )
        if not torch.isfinite(logits).all():
            raise ValueError("PlannerPolicyHead produced non-finite logits")
        scaled_logits = logits / self.policy_temperature
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        if self.sample_policy:
            probabilities = log_probs.exp().squeeze(0).detach().cpu()
            action_index = int(
                torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self.policy_generator,
                ).item()
            )
        else:
            action_index = int(log_probs.argmax(dim=-1).item())
        action_count = int(logits.shape[-1])
        sequences = torch.arange(
            action_count,
            dtype=torch.long,
            device=logits.device,
        ).unsqueeze(-1)
        scores = logits.squeeze(0)
        return (
            WorldModelPlan(
                candidate_sequences=sequences,
                candidate_scores=scores,
                root_action_scores=scores,
                selected_action_index=action_index,
            ),
            tuple(float(value) for value in log_probs.squeeze(0).cpu().tolist()),
        )

    def _project_hidden(self, latent_hidden: torch.Tensor) -> torch.Tensor:
        state = self.world_model.project_state(
            latent_hidden.to(self.planner_device).unsqueeze(0)
        )
        if state.ndim not in (2, 3) or state.shape[0] != 1:
            raise ValueError(
                "planning state projector must return shape (1,D) or (1,N,D), "
                f"got {tuple(state.shape)}"
            )
        return state.detach()

    def _append_actual_state(self, actual_state: torch.Tensor) -> None:
        if len(self._state_history) != len(self._action_history):
            raise RuntimeError("planner state/action history is misaligned")
        self._state_history.append(actual_state)

    def _decision_from_plan(
        self,
        generated: Any,
        plan: WorldModelPlan,
        action_index: int,
        state: torch.Tensor,
        *,
        behavior_log_probs: tuple[float, ...] | None = None,
    ) -> PolicyDecision:
        action_count = int(plan.root_action_scores.shape[0])
        if behavior_log_probs is None:
            behavior_log_probs = tuple(
                0.0 if index == action_index else float("-inf")
                for index in range(action_count)
            )
        world_model_state = state.squeeze(0).detach().cpu().float().clone()
        qwen_decision = generated.qwen_decision

        trace = qwen_decision.token_trace
        action_position = trace.token_roles.index("action")
        new_token_ids = list(trace.token_ids)
        new_token_ids[action_position] = trace.action_token_ids[action_index]
        new_old_log_probs = [None] * len(trace.old_log_probs)
        new_loss_mask = [False] * len(trace.loss_mask)

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
            executed_action_index=action_index,
            horizon=self.horizon,
            search_mode=self.search_mode,
            beam_width=self.planner.beam_width if self.planner is not None else None,
            candidate_visit_counts=(
                tuple(
                    int(value)
                    for value in plan.candidate_visit_counts.cpu().tolist()
                )
                if plan.candidate_visit_counts is not None
                else None
            ),
            root_visit_counts=(
                tuple(
                    int(value)
                    for value in plan.root_visit_counts.cpu().tolist()
                )
                if plan.root_visit_counts is not None
                else None
            ),
            num_simulations=(
                self.planner.mcts_num_simulations if self.planner is not None else None
            ),
            exploration_constant=(
                self.planner.mcts_exploration_constant
                if self.planner is not None
                else None
            ),
            selection_mode=(
                "policy_sample"
                if self.search_mode == "policy" and self.sample_policy
                else "policy_argmax"
                if self.search_mode == "policy"
                else "value_argmax"
            ),
            policy_action_log_probs=(
                behavior_log_probs if self.search_mode == "policy" else None
            ),
        )
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
            world_model_state=world_model_state,
        )

    def generate_state(self, prompt: AgentPrompt) -> PolicyState:
        """Terminal state 生成真实 CoT/hidden，但不运行 planner 或 environment。"""

        qwen_state = self.turn_policy.generate_state(prompt)
        if qwen_state.latent_hidden is None:
            raise RuntimeError("planner terminal state has no captured Qwen hidden")
        with evaluating(self.world_model), torch.no_grad():
            actual_state = self._project_hidden(qwen_state.latent_hidden)
            self._append_actual_state(actual_state)
        return PolicyState(
            assistant_prefix=qwen_state.assistant_prefix,
            latent_hidden=qwen_state.latent_hidden,
            world_model_state=(
                actual_state.squeeze(0).detach().cpu().float().clone()
            ),
        )

    def generate_states(
        self,
        prompts: tuple[AgentPrompt, ...],
    ) -> tuple[PolicyState, ...]:
        """Batch terminal CoT/state generation for independent H=1 envs."""

        if self.search_mode != "policy":
            raise RuntimeError(
                "batched terminal planning currently requires search_mode=policy"
            )
        qwen_states = self.turn_policy.generate_states(prompts)
        if len(qwen_states) != len(prompts):
            raise RuntimeError(
                "batched terminal states do not align with prompts: "
                f"{len(qwen_states)} != {len(prompts)}"
            )
        results: list[PolicyState] = []
        with evaluating(self.world_model), torch.no_grad():
            for qwen_state in qwen_states:
                if qwen_state.latent_hidden is None:
                    raise RuntimeError(
                        "batched planner terminal state has no captured Qwen hidden"
                    )
                actual_state = self._project_hidden(qwen_state.latent_hidden)
                results.append(
                    PolicyState(
                        assistant_prefix=qwen_state.assistant_prefix,
                        latent_hidden=qwen_state.latent_hidden,
                        world_model_state=(
                            actual_state.squeeze(0).detach().cpu().float().clone()
                        ),
                    )
                )
        return tuple(results)


__all__ = ["PlanningPolicy", "WorldModelPlan", "WorldModelPlanner"]
