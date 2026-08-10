"""RL 的连续序列采样、模型前向与目标函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as F

from nimloth.agent import AgentPrompt, PolicyReplayInput, PolicyReplayOutput
from nimloth.rollout import TrajectoryWindow
from nimloth.rollout.transitions import discounted_action_value_targets
from nimloth.training.common import (
    ActionValueLoss,
    action_value_loss,
    world_model_loss,
)
from nimloth.training.rl.credit import expand_step_advantages, token_level_gae
from nimloth.training.rl.episodes import ExecutedTransition
from nimloth.training.rl.policy import (
    ppo_action_policy_loss,
    ppo_clipped_policy_loss,
)
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.training.rl.value import ppo_action_value_loss
from nimloth.util.module import move_to_device
from nimloth.wm import SequenceSIGReg


PLANNER_TRAINING_OBJECTIVE = "receding_horizon_decision_state_ppo_value_v1"
PLANNER_POLICY_TRAINING_OBJECTIVE = "receding_horizon_planner_policy_ppo_v1"


@dataclass(frozen=True)
class PlannerOldPolicyStatistics:
    """Frozen behavior-policy and critic statistics for one real transition."""

    selected_action_value: torch.Tensor
    selected_log_prob: torch.Tensor
    state_value: torch.Tensor


@dataclass(frozen=True)
class RLBatch:
    """一次 RL 更新消费的原始连续窗口与逐步监督。

    action/return 张量为 ``(B,H)``；``old_log_probs`` 按 window-major、time-minor
    展开后，再按每步 loss-mask token 展开。这个顺序必须与 policy replay 一致。
    可选 DINO target 已在训练 loop 中与 ``(B,H)`` next observations 对齐；algorithm
    不负责读取图像或调用 frozen teacher。
    """

    windows: tuple[TrajectoryWindow, ...]
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor
    dino_grid_target: torch.Tensor | None = None

    @property
    def state_prompts(self) -> tuple[AgentPrompt, ...]:
        """按 batch/time 顺序展开 H+1 个状态 prompt。"""

        return tuple(
            prompt
            for window in self.windows
            for prompt in window.state_prompts()
        )

    @property
    def policy_replay_inputs(self) -> tuple[PolicyReplayInput, ...]:
        """按 batch/time 顺序展开 H 个动作重放输入。"""

        return tuple(
            sample
            for window in self.windows
            for sample in window.policy_replay_inputs()
        )

    @property
    def rollout_state_hiddens(self) -> torch.Tensor:
        """堆叠 rollout 时保存的 ``(B,H+1,K,D)`` Qwen state hidden。"""

        return torch.stack(
            [window.rollout_state_hiddens() for window in self.windows],
            dim=0,
        )

    @property
    def next_image_paths(self) -> tuple[str, ...]:
        """按 batch/time 顺序展开 H 个 next observation 图像。"""

        return tuple(
            path
            for window in self.windows
            for path in window.trajectory.image_paths[
                window.start_step + 1 : window.start_step + 1 + window.history_size
            ]
        )


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


@dataclass(frozen=True)
class _PlannerTransitionContext:
    """一个真实 planner transition 的当前可微 state 与 WM 输入。"""

    current_state: torch.Tensor
    state_context: torch.Tensor
    action_context: torch.Tensor


@dataclass(frozen=True)
class _PlannerTransitionLosses:
    """Planner transition 在归一化前的 objective 分项。"""

    weighted_wm_loss: torch.Tensor
    wm_mse: torch.Tensor
    dino_grid_mse: torch.Tensor | None
    value_loss: torch.Tensor
    selected_action_values: torch.Tensor
    value_clipped_mse: torch.Tensor
    value_clip_fraction: torch.Tensor
    policy_loss: torch.Tensor | None
    policy_entropy: torch.Tensor | None
    policy_clip_fraction: torch.Tensor | None
    policy_probability_ratio: torch.Tensor | None


@dataclass(frozen=True)
class _PlannerValueAndPolicyLosses:
    """互斥的 planner critic / PlannerPolicyHead PPO 分项。"""

    value_loss: torch.Tensor
    value_clipped_mse: torch.Tensor
    value_clip_fraction: torch.Tensor
    policy_loss: torch.Tensor | None
    policy_entropy: torch.Tensor | None
    policy_clip_fraction: torch.Tensor | None
    policy_probability_ratio: torch.Tensor | None


@dataclass(frozen=True)
class _PlannerBatchRow:
    """共享 Qwen micro-batch 中一个 transition 的监督与权重。"""

    transition: ExecutedTransition
    return_target: torch.Tensor
    old_action_value: torch.Tensor
    old_policy_log_prob: torch.Tensor | None
    policy_advantage: torch.Tensor | None
    dino_grid_target: torch.Tensor | None
    loss_weight: float


def low_variance_kl(log_ratio: torch.Tensor) -> torch.Tensor:
    """计算 VAGEN 截断后的低方差 KL，并避免 ``exp`` 溢出。

    最终惩罚在该输入区间之外已经饱和到 10，因此先截断指数输入不会改变
    输出值及其零梯度区间。
    """

    safe_log_ratio = log_ratio.clamp(min=-11.0, max=3.0)
    return (safe_log_ratio.exp() - safe_log_ratio - 1.0).clamp(-10.0, 10.0)


def build_rl_batch(
    windows: tuple[TrajectoryWindow, ...],
    *,
    gamma: float,
    truncated_bootstrap: float | None = None,
    device: torch.device,
) -> RLBatch:
    """把已采样窗口的动作、return 和行为概率整理为 ``(B,H)`` 张量。

    return 先在完整 episode 上计算再切片，避免把训练窗口末端误当作 episode 终点；
    old log-prob 取自 rollout 时保存的行为分布中实际执行的动作。
    """

    if not windows:
        raise ValueError("RL batch requires at least one trajectory window")
    history_sizes = {window.history_size for window in windows}
    if len(history_sizes) != 1:
        raise ValueError("one RL batch cannot mix trajectory window lengths")
    history_size = history_sizes.pop()
    if any(window.trajectory.planner_policy_traces for window in windows):
        raise ValueError(
            "planner trajectories use complete-episode transition training, not RLBatch"
        )
    replay_inputs = tuple(
        sample
        for window in windows
        for sample in window.policy_replay_inputs()
    )
    return RLBatch(
        windows=windows,
        action_indices=torch.tensor(
            [
                window.trajectory.action_indices[
                    window.start_step : window.start_step + history_size
                ]
                for window in windows
            ],
            dtype=torch.long,
            device=device,
        ),
        return_targets=torch.tensor(
            [
                discounted_action_value_targets(
                    window.trajectory.to_record(),
                    gamma=gamma,
                    truncated_bootstrap=truncated_bootstrap,
                )[window.start_step : window.start_step + history_size]
                for window in windows
            ],
            dtype=torch.float32,
            device=device,
        ),
        old_log_probs=torch.tensor(
            [
                old_log_prob
                for sample in replay_inputs
                for old_log_prob in sample.selected_old_log_probs
            ],
            dtype=torch.float32,
            device=device,
        ),
    )


def normalized_monte_carlo_advantages(
    *,
    return_targets: torch.Tensor,
    predicted_values: torch.Tensor,
) -> torch.Tensor:
    """在 ``B*H`` 个动作位置上，用当前 value baseline 归一化 Monte Carlo return。

    baseline 在这里 detach；PPO 不通过 advantage 更新 ValueHead，ValueHead 由自己的
    监督目标更新。
    """

    advantages = return_targets - predicted_values.detach()
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


class RLAlgorithm:
    """定义 RL 一个 batch 的完整计算图。

    WM 的下一状态监督值保持固定。StateProjector 只从 current/start state 路径训练；
    Backbone 是否接收 WM/value/SIGReg 梯度由 ``RLModelRuntime`` 的显式模式决定。
    policy replay、optimizer、rollout 生命周期和 checkpoint 由运行期管理。
    """

    def __init__(
        self,
        *,
        history_size: int,
        sigreg: SequenceSIGReg | None,
        sigreg_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
        ppo_clip_ratio: float,
        entropy_weight: float,
        value_ppo_clip_range: float | None = None,
        credit_assignment: Literal["action", "turn", "token"] = "action",
        token_gamma: float | None = None,
        token_gae_lambda: float | None = None,
        token_value_loss_weight: float | None = None,
        reference_kl_loss_weight: float = 0.0,
        train_world_model: bool = True,
        world_model_weight: float = 1.0,
        dino_grid_weight: float = 0.0,
        planner_policy_enabled: bool = False,
        planner_policy_clip_ratio: float = 0.2,
        planner_policy_entropy_weight: float = 0.0,
        planner_policy_temperature: float = 1.0,
    ) -> None:
        """消费已经由 ``RLConfig`` 校验过的算法参数。"""

        self.history_size = int(history_size)
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.value_ppo_clip_range = value_ppo_clip_range
        self.ppo_clip_ratio = float(ppo_clip_ratio)
        self.entropy_weight = float(entropy_weight)
        self.credit_assignment = credit_assignment
        self.token_gamma = token_gamma
        self.token_gae_lambda = token_gae_lambda
        self.token_value_loss_weight = token_value_loss_weight
        self.reference_kl_loss_weight = float(reference_kl_loss_weight)
        self.train_world_model = bool(train_world_model)
        self.world_model_weight = float(world_model_weight)
        self.dino_grid_weight = float(dino_grid_weight)
        self.planner_policy_enabled = bool(planner_policy_enabled)
        self.planner_policy_clip_ratio = float(planner_policy_clip_ratio)
        self.planner_policy_entropy_weight = float(planner_policy_entropy_weight)
        self.planner_policy_temperature = float(planner_policy_temperature)

    @torch.no_grad()
    def planner_old_action_value(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
    ) -> torch.Tensor:
        """Evaluate frozen rollout `Q_old(s_t,a_t)` without recomputing Qwen.

        Planner trajectories persist the actual projected decision state produced by
        the behavior checkpoint.  Fresh-policy validation guarantees the trainer uses
        the same ValueHead checkpoint, so evaluating that saved state before any update
        gives the old critic value without confusing MCTS root scores or action-token
        probabilities with direct outgoing `Q(s_t,a_t)`.
        """

        rollout_state = move_to_device(
            transition.rollout_decision_state().unsqueeze(0),
            runtime.agent.wm.value_head,
        )
        action_values = runtime.agent.wm.predict_action_values(rollout_state)
        action = torch.tensor(
            [[transition.action_index]],
            dtype=torch.long,
            device=action_values.device,
        )
        return action_values.gather(-1, action).reshape(()).detach().cpu()

    @torch.no_grad()
    def planner_old_policy_statistics(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
    ) -> PlannerOldPolicyStatistics:
        """Evaluate the frozen behavior actor and state-only critic baseline."""

        if not self.planner_policy_enabled:
            raise RuntimeError("PlannerPolicyHead statistics require policy PPO")
        policy_head = runtime.agent.wm.planner_policy_head
        if policy_head is None:
            raise RuntimeError("policy PPO runtime has no PlannerPolicyHead")
        rollout_state = move_to_device(
            transition.rollout_decision_state().unsqueeze(0),
            policy_head,
        )
        action_values = runtime.agent.wm.predict_action_values(rollout_state)
        logits = runtime.agent.wm.predict_action_logits(rollout_state)
        old_log_probs = torch.log_softmax(
            logits / self.planner_policy_temperature,
            dim=-1,
        ).squeeze(0)
        stored_log_probs = transition.behavior_action_log_probs().to(
            device=old_log_probs.device,
            dtype=old_log_probs.dtype,
        )
        if not torch.allclose(
            old_log_probs,
            stored_log_probs,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError(
                "fresh rollout PlannerPolicyHead log-probs do not match the "
                "training checkpoint"
            )
        selected_action = torch.tensor(
            transition.action_index,
            dtype=torch.long,
            device=old_log_probs.device,
        )
        state_value = (old_log_probs.exp() * action_values.squeeze(0)).sum()
        return PlannerOldPolicyStatistics(
            selected_action_value=action_values.squeeze(0)[selected_action]
            .detach()
            .cpu(),
            selected_log_prob=old_log_probs[selected_action].detach().cpu(),
            state_value=state_value.detach().cpu(),
        )

    def actor_transition_step(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        *,
        return_target: torch.Tensor,
        old_action_value: torch.Tensor,
        old_policy_log_prob: torch.Tensor | None = None,
        policy_advantage: torch.Tensor | None = None,
        total_transitions: int,
        dino_grid_target: torch.Tensor | None = None,
        include_world_model: bool = True,
        precomputed_hidden: torch.Tensor | None = None,
    ) -> RLStepOutput:
        """计算一个真实 planner transition 的完整训练 objective。

        当前 state 必须由完整、可微的 Qwen prefix forward 产生。WM/DINO 使用
        rollout 持久化的 next-state target；value 与 PlannerPolicyHead 则只监督本次
        实际执行的 environment action。
        """

        self._validate_planner_transition_runtime(runtime, total_transitions)
        hidden = self._planner_transition_hidden(
            runtime,
            transition,
            precomputed_hidden=precomputed_hidden,
        )
        context = self._planner_transition_context(runtime, transition, hidden)
        losses = self._planner_transition_losses(
            runtime,
            transition,
            context,
            return_target=return_target,
            old_action_value=old_action_value,
            old_policy_log_prob=old_policy_log_prob,
            policy_advantage=policy_advantage,
            dino_grid_target=dino_grid_target,
            include_world_model=include_world_model,
        )
        return self._planner_step_output(
            losses,
            old_action_value=old_action_value,
            policy_advantage=policy_advantage,
            total_transitions=total_transitions,
        )

    def _validate_planner_transition_runtime(
        self,
        runtime: RLModelRuntime,
        total_transitions: int,
    ) -> None:
        """拒绝无法将 planner objective 反传到完整 Qwen prefix 的 runtime。"""

        if total_transitions < 1:
            raise ValueError("total_transitions must be positive")
        if runtime.state_source != "recompute" or not runtime.representation_to_backbone:
            raise RuntimeError(
                "planner transition training requires differentiable full-prefix "
                "Qwen recomputation"
            )

    def _planner_transition_hidden(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        *,
        precomputed_hidden: torch.Tensor | None,
    ) -> torch.Tensor:
        """取得恰好一个 transition 的 Qwen hidden，并校验 batch 对齐。"""

        hidden = (
            runtime.encode_state_prompts((transition.state_prompt,))
            if precomputed_hidden is None
            else precomputed_hidden
        )
        if hidden.ndim not in (2, 3) or hidden.shape[0] != 1:
            raise ValueError(
                "planner transition hidden must have one batch row, "
                f"got {tuple(hidden.shape)}"
            )
        return move_to_device(hidden, runtime.agent.wm.state_proj)

    def _planner_transition_context(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        hidden: torch.Tensor,
    ) -> _PlannerTransitionContext:
        """拼接持久化历史与当前可微 state，供 WM 预测最后一个 next state。"""

        current_state = runtime.agent.wm.project_state(hidden)
        stored_history = move_to_device(
            transition.state_history(self.history_size),
            runtime.agent.wm.wm_predictor,
        )
        current_state = current_state.to(device=stored_history.device)
        state_context = torch.cat(
            (stored_history[:-1].unsqueeze(0), current_state.unsqueeze(1)),
            dim=1,
        )
        previous_actions = transition.previous_actions(self.history_size).to(
            device=state_context.device
        )
        executed_action = torch.tensor(
            [transition.action_index],
            dtype=torch.long,
            device=state_context.device,
        )
        return _PlannerTransitionContext(
            current_state=current_state,
            state_context=state_context,
            action_context=torch.cat((previous_actions, executed_action)).unsqueeze(0),
        )

    def _planner_transition_losses(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        context: _PlannerTransitionContext,
        *,
        return_target: torch.Tensor,
        old_action_value: torch.Tensor,
        old_policy_log_prob: torch.Tensor | None,
        policy_advantage: torch.Tensor | None,
        dino_grid_target: torch.Tensor | None,
        include_world_model: bool,
    ) -> _PlannerTransitionLosses:
        """计算未归一化的 WM/DINO、executed-action value 与可选 policy loss。"""

        weighted_wm_loss, wm_mse, dino_grid_mse = self._planner_wm_losses(
            runtime,
            transition,
            context,
            dino_grid_target=dino_grid_target,
            include_world_model=include_world_model,
        )
        action_values = runtime.agent.wm.predict_action_values(context.current_state)
        executed_action = torch.tensor(
            [transition.action_index],
            dtype=torch.long,
            device=action_values.device,
        )
        target = return_target.reshape(1).to(
            device=action_values.device,
            dtype=action_values.dtype,
        )
        selected_action_values = action_values.gather(
            -1,
            executed_action.unsqueeze(-1),
        ).squeeze(-1)
        value_and_policy = self._planner_value_and_policy_losses(
            runtime,
            current_state=context.current_state,
            action_values=action_values,
            executed_action=executed_action,
            target=target,
            selected_action_values=selected_action_values,
            old_action_value=old_action_value,
            old_policy_log_prob=old_policy_log_prob,
            policy_advantage=policy_advantage,
        )
        return _PlannerTransitionLosses(
            weighted_wm_loss=weighted_wm_loss,
            wm_mse=wm_mse,
            dino_grid_mse=dino_grid_mse,
            value_loss=value_and_policy.value_loss,
            selected_action_values=selected_action_values,
            value_clipped_mse=value_and_policy.value_clipped_mse,
            value_clip_fraction=value_and_policy.value_clip_fraction,
            policy_loss=value_and_policy.policy_loss,
            policy_entropy=value_and_policy.policy_entropy,
            policy_clip_fraction=value_and_policy.policy_clip_fraction,
            policy_probability_ratio=value_and_policy.policy_probability_ratio,
        )

    def _planner_wm_losses(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        context: _PlannerTransitionContext,
        *,
        dino_grid_target: torch.Tensor | None,
        include_world_model: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """计算 WM/DINO 监督；保存的 next state 永远是 detach 后的 target。"""

        if not self.train_world_model or not include_world_model:
            zero = context.current_state.new_zeros(())
            return zero, zero, None
        predicted_next_state = runtime.agent.wm.predict_state_sequence(
            context.state_context,
            context.action_context,
        )[:, -1]
        expected_next_state = move_to_device(
            transition.actual_next_state(),
            predicted_next_state,
        ).unsqueeze(0).detach()
        current_dino_target = (
            dino_grid_target.to(
                device=predicted_next_state.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if dino_grid_target is not None
            else None
        )
        objective = world_model_loss(
            predicted_next_state,
            expected_next_state,
            state_weight=self.world_model_weight,
            dino_grid_target=current_dino_target,
            dino_grid_weight=self.dino_grid_weight,
        )
        return objective.loss, objective.state_mse, objective.dino_grid_mse

    def _planner_value_and_policy_losses(
        self,
        runtime: RLModelRuntime,
        *,
        current_state: torch.Tensor,
        action_values: torch.Tensor,
        executed_action: torch.Tensor,
        target: torch.Tensor,
        selected_action_values: torch.Tensor,
        old_action_value: torch.Tensor,
        old_policy_log_prob: torch.Tensor | None,
        policy_advantage: torch.Tensor | None,
    ) -> _PlannerValueAndPolicyLosses:
        """选择互斥的 planner critic 或 PlannerPolicyHead PPO objective。"""

        if self.planner_policy_enabled:
            if old_policy_log_prob is None or policy_advantage is None:
                raise RuntimeError(
                    "PlannerPolicyHead PPO requires old log-prob and advantage"
                )
            policy_objective = ppo_action_policy_loss(
                action_logits=runtime.agent.wm.predict_action_logits(current_state),
                executed_actions=executed_action,
                old_log_probs=old_policy_log_prob.reshape(1),
                advantages=policy_advantage.reshape(1),
                temperature=self.planner_policy_temperature,
                clip_ratio=self.planner_policy_clip_ratio,
            )
            return _PlannerValueAndPolicyLosses(
                value_loss=F.mse_loss(selected_action_values, target),
                value_clipped_mse=selected_action_values.new_zeros(()),
                value_clip_fraction=selected_action_values.new_zeros(()),
                policy_loss=policy_objective.loss,
                policy_entropy=policy_objective.entropy,
                policy_clip_fraction=policy_objective.clip_fraction,
                policy_probability_ratio=policy_objective.probability_ratio,
            )
        if old_policy_log_prob is not None or policy_advantage is not None:
            raise RuntimeError("planner policy statistics require PlannerPolicyHead PPO")
        if self.value_ppo_clip_range is None:
            raise RuntimeError("planner critic clipping requires value_ppo_clip_range")
        value_objective = ppo_action_value_loss(
            action_values,
            executed_action,
            target,
            old_action_value.reshape(1),
            clip_range=self.value_ppo_clip_range,
        )
        return _PlannerValueAndPolicyLosses(
            value_loss=value_objective.loss,
            value_clipped_mse=value_objective.clipped_mse,
            value_clip_fraction=value_objective.clip_fraction,
            policy_loss=None,
            policy_entropy=None,
            policy_clip_fraction=None,
            policy_probability_ratio=None,
        )

    def _planner_step_output(
        self,
        losses: _PlannerTransitionLosses,
        *,
        old_action_value: torch.Tensor,
        policy_advantage: torch.Tensor | None,
        total_transitions: int,
    ) -> RLStepOutput:
        """归一化一条 transition 的 loss，并在图外构造日志指标。"""

        normalized_wm_loss = losses.weighted_wm_loss / total_transitions
        normalized_wm_mse = losses.wm_mse / total_transitions
        normalized_dino_mse = (
            losses.dino_grid_mse / total_transitions
            if losses.dino_grid_mse is not None
            else None
        )
        normalized_value_loss = losses.value_loss / total_transitions
        zero = losses.selected_action_values.new_zeros(())
        normalized_policy_loss = (
            losses.policy_loss / total_transitions
            if losses.policy_loss is not None
            else zero
        )
        normalized_policy_entropy = (
            losses.policy_entropy / total_transitions
            if losses.policy_entropy is not None
            else zero
        )
        total = normalized_wm_loss + normalized_value_loss.to(
            device=normalized_wm_loss.device
        )
        total = total + normalized_policy_loss.to(device=total.device)
        total = total - self.planner_policy_entropy_weight * (
            normalized_policy_entropy.to(device=total.device)
        )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": normalized_wm_mse,
                "dino": normalized_dino_mse,
                "sigreg": None,
                "value": normalized_value_loss,
                "policy": normalized_policy_loss if losses.policy_loss is not None else None,
                "token_value": None,
                "reference_kl": None,
            },
            metrics=self._planner_step_metrics(
                total=total,
                losses=losses,
                old_action_value=old_action_value,
                policy_advantage=policy_advantage,
                total_transitions=total_transitions,
                normalized_wm_mse=normalized_wm_mse,
                normalized_dino_mse=normalized_dino_mse,
                normalized_value_loss=normalized_value_loss,
                normalized_policy_loss=normalized_policy_loss,
                normalized_policy_entropy=normalized_policy_entropy,
            ),
        )

    def _planner_step_metrics(
        self,
        *,
        total: torch.Tensor,
        losses: _PlannerTransitionLosses,
        old_action_value: torch.Tensor,
        policy_advantage: torch.Tensor | None,
        total_transitions: int,
        normalized_wm_mse: torch.Tensor,
        normalized_dino_mse: torch.Tensor | None,
        normalized_value_loss: torch.Tensor,
        normalized_policy_loss: torch.Tensor,
        normalized_policy_entropy: torch.Tensor,
    ) -> dict[str, float]:
        """将 planner loss 分项转为与 loop/reporting 兼容的标量指标。"""

        policy_enabled = losses.policy_loss is not None
        return {
            "wm_mse": float(normalized_wm_mse.detach().item()),
            "dino_grid_mse": (
                float(normalized_dino_mse.detach().item())
                if normalized_dino_mse is not None
                else 0.0
            ),
            "lambda_wm": self.world_model_weight,
            "lambda_dino": self.dino_grid_weight,
            "sigreg_loss": 0.0,
            "value_loss": float(normalized_value_loss.detach().item()),
            "value_mc_mse": float(normalized_value_loss.detach().item()),
            "value_clipped_mse": float(
                (losses.value_clipped_mse / total_transitions).detach().item()
            ),
            "value_clip_fraction": float(
                (losses.value_clip_fraction / total_transitions).detach().item()
            ),
            "value_old_mean": float(
                (old_action_value / total_transitions).detach().item()
            ),
            "value_delta_abs_mean": float(
                (
                    (
                        losses.selected_action_values.detach()
                        - old_action_value.to(
                            device=losses.selected_action_values.device,
                            dtype=losses.selected_action_values.dtype,
                        )
                    ).abs().mean()
                    / total_transitions
                ).item()
            ),
            "value_rank": 0.0,
            "total_loss": float(total.detach().item()),
            "actor_loss": 0.0,
            "planner_policy_loss": float(normalized_policy_loss.detach().item()),
            "planner_policy_entropy": float(
                normalized_policy_entropy.detach().item()
            ),
            "planner_policy_clip_fraction": (
                float((losses.policy_clip_fraction / total_transitions).item())
                if losses.policy_clip_fraction is not None
                else 0.0
            ),
            "planner_policy_mean_ratio": (
                float(
                    (
                        losses.policy_probability_ratio.mean() / total_transitions
                    ).detach().item()
                )
                if losses.policy_probability_ratio is not None
                else 0.0
            ),
            "planner_policy_mean_advantage": (
                float((policy_advantage / total_transitions).detach().item())
                if policy_advantage is not None
                else 0.0
            ),
            "planner_policy_actions": 1.0 / total_transitions if policy_enabled else 0.0,
            "token_value_loss": 0.0,
            "reference_kl_loss": 0.0,
            "policy_tokens": 0.0,
        }

    def actor_transition_batch_step(
        self,
        runtime: RLModelRuntime,
        transitions: tuple[ExecutedTransition, ...],
        *,
        return_targets: tuple[torch.Tensor, ...],
        old_action_values: tuple[torch.Tensor, ...],
        old_policy_log_probs: tuple[torch.Tensor | None, ...] | None = None,
        policy_advantages: tuple[torch.Tensor | None, ...] | None = None,
        total_transitions: int,
        dino_grid_targets: tuple[torch.Tensor | None, ...] | None = None,
        loss_weights: tuple[float, ...] | None = None,
        include_world_model: bool = True,
    ) -> RLStepOutput:
        """共享一次 padded Qwen forward，逐 row 计算原有 planner transition objective。"""

        rows = self._planner_batch_rows(
            transitions,
            return_targets=return_targets,
            old_action_values=old_action_values,
            old_policy_log_probs=old_policy_log_probs,
            policy_advantages=policy_advantages,
            dino_grid_targets=dino_grid_targets,
            loss_weights=loss_weights,
        )
        hidden_batch = runtime.encode_state_prompts(
            tuple(row.transition.state_prompt for row in rows)
        )
        if hidden_batch.ndim not in (2, 3) or hidden_batch.shape[0] != len(rows):
            raise ValueError(
                "planner Qwen batch output does not align with transitions: "
                f"hidden={tuple(hidden_batch.shape)}, transitions={len(rows)}"
            )
        outputs = tuple(
            self.actor_transition_step(
                runtime,
                row.transition,
                return_target=row.return_target,
                old_action_value=row.old_action_value,
                old_policy_log_prob=row.old_policy_log_prob,
                policy_advantage=row.policy_advantage,
                total_transitions=total_transitions,
                dino_grid_target=row.dino_grid_target,
                include_world_model=include_world_model,
                precomputed_hidden=hidden_batch[index : index + 1],
            )
            for index, row in enumerate(rows)
        )
        return self._aggregate_planner_outputs(outputs, rows)

    def _planner_batch_rows(
        self,
        transitions: tuple[ExecutedTransition, ...],
        *,
        return_targets: tuple[torch.Tensor, ...],
        old_action_values: tuple[torch.Tensor, ...],
        old_policy_log_probs: tuple[torch.Tensor | None, ...] | None,
        policy_advantages: tuple[torch.Tensor | None, ...] | None,
        dino_grid_targets: tuple[torch.Tensor | None, ...] | None,
        loss_weights: tuple[float, ...] | None,
    ) -> tuple[_PlannerBatchRow, ...]:
        """验证并对齐 planner micro-batch 的每个逐 transition 输入字段。"""

        if not transitions:
            raise ValueError("planner transition batch must not be empty")
        batch_size = len(transitions)
        fields = {
            "return_targets": return_targets,
            "old_action_values": old_action_values,
            "old_policy_log_probs": (
                old_policy_log_probs
                if old_policy_log_probs is not None
                else (None,) * batch_size
            ),
            "policy_advantages": (
                policy_advantages
                if policy_advantages is not None
                else (None,) * batch_size
            ),
            "dino_grid_targets": (
                dino_grid_targets
                if dino_grid_targets is not None
                else (None,) * batch_size
            ),
            "loss_weights": (
                loss_weights if loss_weights is not None else (1.0,) * batch_size
            ),
        }
        for name, values in fields.items():
            if len(values) != batch_size:
                raise ValueError(
                    f"planner batch {name} must have {batch_size} rows, "
                    f"got {len(values)}"
                )
        return tuple(
            _PlannerBatchRow(
                transition=transition,
                return_target=return_target,
                old_action_value=old_action_value,
                old_policy_log_prob=old_policy_log_prob,
                policy_advantage=policy_advantage,
                dino_grid_target=dino_grid_target,
                loss_weight=float(loss_weight),
            )
            for (
                transition,
                return_target,
                old_action_value,
                old_policy_log_prob,
                policy_advantage,
                dino_grid_target,
                loss_weight,
            ) in zip(
                transitions,
                fields["return_targets"],
                fields["old_action_values"],
                fields["old_policy_log_probs"],
                fields["policy_advantages"],
                fields["dino_grid_targets"],
                fields["loss_weights"],
                strict=True,
            )
        )

    def _aggregate_planner_outputs(
        self,
        outputs: tuple[RLStepOutput, ...],
        rows: tuple[_PlannerBatchRow, ...],
    ) -> RLStepOutput:
        """按 loss weight 合并 row objective；零权重 padding 不进入 metrics。"""

        total_loss = torch.stack(
            tuple(
                output.loss * row.loss_weight
                for output, row in zip(outputs, rows, strict=True)
            )
        ).sum()
        combined_losses: dict[str, torch.Tensor | None] = {}
        for name in outputs[0].losses:
            weighted_values = tuple(
                output.losses[name].to(device=total_loss.device) * row.loss_weight
                for output, row in zip(outputs, rows, strict=True)
                if output.losses[name] is not None
            )
            combined_losses[name] = (
                torch.stack(weighted_values).sum() if weighted_values else None
            )
        return RLStepOutput(
            loss=total_loss,
            losses=combined_losses,
            metrics={
                name: sum(
                    output.metrics[name]
                    for output, row in zip(outputs, rows, strict=True)
                    if row.loss_weight != 0.0
                )
                for name in outputs[0].metrics
            },
        )

    def sequence_step(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> RLStepOutput:
        """构造 RL 计算图并计算 WM、value 与可选 PPO 目标。"""

        hidden_states = self._state_hidden_sequence(runtime, batch)

        state_sequence = runtime.agent.wm.project_state_sequence(hidden_states)
        state_context = state_sequence[:, :-1]
        action_values = runtime.agent.wm.predict_action_values(state_context)

        # WM 与 value 共享 state_context，但监督目标和梯度边界各自独立。
        wm_objective = None
        if self.train_world_model:
            predicted_next_states = runtime.agent.wm.predict_state_sequence(
                state_context,
                batch.action_indices,
            )
            # 下一状态只作为固定监督值；同一状态在它作为 current state 时训练
            # StateProjector，不能让监督值反向靠近当前预测。
            expected_next_states = state_sequence[:, 1:].detach()
            dino_grid_target = (
                batch.dino_grid_target.to(
                    device=predicted_next_states.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if batch.dino_grid_target is not None
                else None
            )
            wm_objective = world_model_loss(
                predicted_next_states,
                expected_next_states,
                state_weight=self.world_model_weight,
                dino_grid_target=dino_grid_target,
                dino_grid_weight=self.dino_grid_weight,
            )
        value_objective = action_value_loss(
            action_values,
            batch.action_indices,
            batch.return_targets,
            ranking_margin=self.value_rank_margin,
            ranking_weight=self.value_rank_weight,
        )
        total = (
            value_objective.loss
            if wm_objective is None
            else wm_objective.loss + value_objective.loss
        )

        # 各 WM variant 明确选择 SIGReg 的统计单位；grid 与 SFT2 一致，对 slot
        # mean pooling 后再把 (B,T,D) 交给 SequenceSIGReg。
        sigreg_states = runtime.agent.wm.sigreg_state_sequence(state_sequence)
        sigreg_loss = (
            self.sigreg(sigreg_states)
            if self.sigreg is not None and self.sigreg_weight > 0.0
            else None
        )
        if sigreg_loss is not None:
            total = total + self.sigreg_weight * sigreg_loss

        policy, token_value_loss, reference_kl_loss = self._policy_replay_losses(
            runtime,
            batch,
            value_objective,
        )
        if policy is not None:
            # policy["loss"] 已取 clipped surrogate 的负号；entropy 作为奖励项减去。
            # Accelerate 会把device-mapped Qwen的输出复制回输入GPU，而WM/value
            # 位于model output GPU。只复制两个标量到监督loss设备；CopyBackward
            # 保留PPO到Qwen logits的完整梯度，避免搬运selected vocabulary logits。
            policy_loss = policy["loss"].to(device=total.device)
            policy_entropy = policy["entropy"].to(device=total.device)
            total = total + policy_loss - self.entropy_weight * policy_entropy
            if token_value_loss is not None:
                token_value_weight = cast(float, self.token_value_loss_weight)
                total = total + token_value_weight * token_value_loss.to(
                    device=total.device
                )
            if reference_kl_loss is not None:
                total = total + self.reference_kl_loss_weight * (
                    reference_kl_loss.to(device=total.device)
                )

        metrics = {
            "wm_mse": (
                float(wm_objective.state_mse.detach().item())
                if wm_objective is not None
                else 0.0
            ),
            "dino_grid_mse": (
                float(wm_objective.dino_grid_mse.detach().item())
                if wm_objective is not None
                and wm_objective.dino_grid_mse is not None
                else 0.0
            ),
            "lambda_wm": self.world_model_weight,
            "lambda_dino": self.dino_grid_weight,
            "sigreg_loss": (
                float(sigreg_loss.detach().item()) if sigreg_loss is not None else 0.0
            ),
            "value_loss": float(value_objective.loss.detach().item()),
            "value_mc_mse": float(
                value_objective.monte_carlo_mse.detach().item()
            ),
            "value_rank": float(value_objective.ranking.detach().item()),
            "total_loss": float(total.detach().item()),
            "actor_loss": float(policy["loss"].detach().item()) if policy else 0.0,
            "token_value_loss": (
                float(token_value_loss.detach().item())
                if token_value_loss is not None
                else 0.0
            ),
            "reference_kl_loss": (
                float(reference_kl_loss.detach().item())
                if reference_kl_loss is not None
                else 0.0
            ),
        }
        if policy is not None:
            metrics.update(
                {
                    "entropy": float(policy["entropy"].detach().item()),
                    "mean_advantage": float(policy["advantages"].mean().item()),
                    "clip_fraction": float(policy["clip_fraction"].item()),
                    "mean_ratio": float(policy["probability_ratio"].mean().item()),
                    "policy_tokens": float(policy["advantages"].numel()),
                }
            )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": (
                    wm_objective.state_mse if wm_objective is not None else None
                ),
                "dino": (
                    wm_objective.dino_grid_mse
                    if wm_objective is not None
                    else None
                ),
                "sigreg": sigreg_loss,
                "value": value_objective.loss,
                "policy": policy["loss"] if policy else None,
                "token_value": token_value_loss,
                "reference_kl": reference_kl_loss,
            },
            metrics=metrics,
        )

    def _state_hidden_sequence(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> torch.Tensor:
        """按显式 state source 读取 rollout hidden 或重新执行 Qwen。"""

        state_steps = self.history_size + 1
        if runtime.state_source == "rollout":
            hidden_states = batch.rollout_state_hiddens
            runtime.validate_rollout_state_hiddens(
                hidden_states,
                batch_size=len(batch.windows),
                state_steps=state_steps,
            )
            return move_to_device(
                hidden_states.detach(),
                runtime.agent.wm.state_proj,
            )
        return runtime.encode_state_sequence(
            batch.state_prompts,
            batch_size=len(batch.windows),
            state_steps=state_steps,
        )

    def _policy_replay_losses(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
        value_objective: ActionValueLoss,
    ) -> tuple[
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """按 rollout 的 token 顺序重放 policy，并返回 PPO/value/KL 分项。"""

        if runtime.policy_replay is None:
            return None, None, None
        replay_inputs = batch.policy_replay_inputs
        replay_output = runtime.policy_replay(replay_inputs)
        reference_kl_loss = self._reference_kl_loss(replay_output, replay_inputs)

        token_value_loss: torch.Tensor | None = None
        if self.credit_assignment == "token":
            token_values = replay_output.token_values
            if token_values is None:
                raise RuntimeError("token credit replay returned no token values")
            token_credit = token_level_gae(
                batch.return_targets.flatten(),
                token_values,
                replay_inputs,
                gamma=cast(float, self.token_gamma),
                gae_lambda=cast(float, self.token_gae_lambda),
            )
            advantages = token_credit.advantages.to(
                device=replay_output.selected_log_probs.device,
                dtype=replay_output.selected_log_probs.dtype,
            )
            token_value_loss = F.mse_loss(
                token_values,
                token_credit.returns.to(
                    device=token_values.device,
                    dtype=token_values.dtype,
                ),
            )
        else:
            step_advantages = normalized_monte_carlo_advantages(
                return_targets=batch.return_targets.flatten().to(
                    device=value_objective.selected_action_values.device,
                    dtype=value_objective.selected_action_values.dtype,
                ),
                predicted_values=value_objective.selected_action_values.flatten(),
            ).to(
                device=replay_output.selected_log_probs.device,
                dtype=replay_output.selected_log_probs.dtype,
            )
            advantages = expand_step_advantages(
                step_advantages,
                replay_inputs,
                credit_assignment=self.credit_assignment,
            )

        policy = self._policy_loss(
            new_log_probs=replay_output.selected_log_probs,
            old_log_probs=batch.old_log_probs.to(
                device=replay_output.selected_log_probs.device,
                dtype=replay_output.selected_log_probs.dtype,
            ),
            entropies=replay_output.entropies,
            advantages=advantages,
        )
        return policy, token_value_loss, reference_kl_loss

    def _reference_kl_loss(
        self,
        replay_output: PolicyReplayOutput,
        replay_inputs: tuple[PolicyReplayInput, ...],
    ) -> torch.Tensor | None:
        """校验并消费预先持久化的 frozen-reference token log-prob。"""

        if self.reference_kl_loss_weight <= 0.0:
            return None
        current_log_probs = replay_output.selected_full_log_probs
        if current_log_probs is None:
            raise RuntimeError(
                "reference KL requires current full-vocabulary log-probs"
            )
        reference_rows = [
            sample.selected_reference_log_probs for sample in replay_inputs
        ]
        if any(row is None for row in reference_rows):
            raise RuntimeError(
                "reference KL requires frozen reference log-probs on every step"
            )
        reference_log_probs = torch.tensor(
            [
                value
                for row in reference_rows
                if row is not None
                for value in row
            ],
            dtype=current_log_probs.dtype,
            device=current_log_probs.device,
        )
        if reference_log_probs.shape != current_log_probs.shape:
            raise RuntimeError("reference/current CoT log-prob counts do not align")
        # VAGEN 低方差 KL：exp(ref-logp) - (ref-logp) - 1。
        return low_variance_kl(reference_log_probs - current_log_probs).mean()

    def _policy_loss(
        self,
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        entropies: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """计算逐 loss-mask token PPO clipped surrogate 与实际 behavior entropy。"""

        objective = ppo_clipped_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            entropies=entropies,
            advantages=advantages,
            clip_ratio=self.ppo_clip_ratio,
        )
        return {
            "loss": objective.loss,
            "entropy": objective.entropy,
            "advantages": advantages,
            "probability_ratio": objective.probability_ratio,
            "clip_fraction": objective.clip_fraction,
        }


__all__ = [
    "PLANNER_TRAINING_OBJECTIVE",
    "PLANNER_POLICY_TRAINING_OBJECTIVE",
    "PlannerOldPolicyStatistics",
    "RLAlgorithm",
    "RLBatch",
    "RLStepOutput",
    "build_rl_batch",
    "low_variance_kl",
    "normalized_monte_carlo_advantages",
]
