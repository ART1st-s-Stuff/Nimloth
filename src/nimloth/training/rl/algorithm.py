"""RL 的连续序列采样、模型前向与目标函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from nimloth.agent import AgentPrompt, PolicyReplayInput, PolicyReplayOutput
from nimloth.config.rl import RLConfig
from nimloth.rollout import TrajectoryWindow
from nimloth.rollout.transitions import discounted_action_value_targets
from nimloth.training.common import (
    ActionValueLoss,
    action_value_loss,
    world_model_loss,
)
from nimloth.training.common.value_semantics import (
    PLANNER_POLICY_TRAINING_OBJECTIVE,
    PLANNER_TRAINING_OBJECTIVE,
)
from nimloth.training.rl.credit import expand_step_advantages, token_level_gae
from nimloth.training.rl.episodes import ExecutedTransition
from nimloth.training.rl.policy import (
    ppo_action_policy_loss,
    ppo_clipped_policy_loss,
)
from nimloth.training.rl.reporting import planner_step_metrics
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.training.rl.value import ppo_action_value_loss
from nimloth.util.module import move_to_device
from nimloth.wm import SequenceSIGReg


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
    可选 DINO target 已在训练 loop 中与 ``(B,H)`` current observations 对齐；algorithm
    不负责读取图像或调用 frozen teacher。
    """

    windows: tuple[TrajectoryWindow, ...]
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor
    dino_grid_target: torch.Tensor | None = None
    action_success_targets: torch.Tensor | None = None
    action_success_mask: torch.Tensor | None = None
    policy_step_advantages: torch.Tensor | None = None

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
    def current_image_paths(self) -> tuple[str, ...]:
        """按 batch/time 顺序展开 H 个 current observation 图像。"""

        return tuple(
            path
            for window in self.windows
            for path in window.trajectory.image_paths[
                window.start_step : window.start_step + window.history_size
            ]
        )


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


@dataclass(frozen=True)
class SequenceLossNormalization:
    """Full-update denominators used by one sequence micro-batch.

    Every micro-batch returns its contribution to the same effective-batch mean.
    This keeps gradient accumulation equivalent to one full-batch objective even
    when policy token counts or available Outcome labels differ by window.
    """

    action_positions: int
    outcome_labels: int
    policy_tokens: int

    def __post_init__(self) -> None:
        if self.action_positions < 1:
            raise ValueError("sequence normalization requires action positions")
        if self.outcome_labels < 0 or self.policy_tokens < 0:
            raise ValueError("sequence normalization counts must be non-negative")


def slice_rl_batch(batch: RLBatch, start: int, stop: int) -> RLBatch:
    """Slice complete windows and their variable-length policy-token stream."""

    batch_size = len(batch.windows)
    if not 0 <= start < stop <= batch_size:
        raise ValueError(
            f"invalid RLBatch slice [{start}:{stop}] for batch size {batch_size}"
        )
    token_offsets = [0]
    for window in batch.windows:
        token_offsets.append(
            token_offsets[-1]
            + sum(
                len(sample.selected_old_log_probs)
                for sample in window.policy_replay_inputs()
            )
        )
    token_start = token_offsets[start]
    token_stop = token_offsets[stop]

    def _slice_optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value[start:stop]

    return RLBatch(
        windows=batch.windows[start:stop],
        action_indices=batch.action_indices[start:stop],
        return_targets=batch.return_targets[start:stop],
        old_log_probs=batch.old_log_probs[token_start:token_stop],
        dino_grid_target=_slice_optional(batch.dino_grid_target),
        action_success_targets=_slice_optional(batch.action_success_targets),
        action_success_mask=_slice_optional(batch.action_success_mask),
        policy_step_advantages=_slice_optional(batch.policy_step_advantages),
    )


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
    action_success_rows: list[list[float]] = []
    action_success_mask_rows: list[list[bool]] = []
    for window in windows:
        labels = window.trajectory.action_successes
        selected = (
            labels[window.start_step : window.start_step + history_size]
            if labels is not None
            else [None] * history_size
        )
        action_success_rows.append(
            [float(value) if value is not None else 0.0 for value in selected]
        )
        action_success_mask_rows.append([value is not None for value in selected])
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
        action_success_targets=torch.tensor(
            action_success_rows,
            dtype=torch.float32,
            device=device,
        ),
        action_success_mask=torch.tensor(
            action_success_mask_rows,
            dtype=torch.bool,
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
        config: RLConfig,
        sigreg: SequenceSIGReg | None,
    ) -> None:
        """保存已校验的 RL 配置与可选 SIGReg；不重复定义配置接口。"""

        self.config = config
        self.sigreg = sigreg

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

        if not self.config.planner_policy.enabled:
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
            logits / self.config.planner_policy.temperature,
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

    def planner_transition_step(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        *,
        return_target: torch.Tensor,
        old_action_value: torch.Tensor,
        old_policy_log_prob: torch.Tensor | None = None,
        policy_advantage: torch.Tensor | None = None,
        total_transitions: int,
        total_outcomes: int | None = None,
        dino_grid_target: torch.Tensor | None = None,
        include_world_model: bool = True,
        precomputed_hidden: torch.Tensor | None = None,
    ) -> RLStepOutput:
        """计算一个真实 planner transition 的完整训练目标。

        顺序固定为：完整 Qwen prefix 得到当前 state；WM/DINO 预测真实 successor；
        ValueHead 监督 executed action；可选 PlannerPolicyHead 对同一 action 做 PPO。
        除masked Outcome BCE按有效标签数归一化外，其余objective按完整batch的
        真实transition数归一化。
        """

        if runtime.state_source != "recompute":
            raise RuntimeError("planner transition training requires full-prefix Qwen recomputation")

        # 1. 完整 Qwen prefix只执行一次。DINO需要时保留Qwen图；WM/value/outcome
        # 是否回传Qwen由representation_to_backbone显式控制。
        hidden = (
            runtime.encode_state_prompts(
                (transition.state_prompt,),
                retain_backbone_graph=self.config.predictor.lambda_dino > 0.0,
            )
            if precomputed_hidden is None
            else precomputed_hidden
        )
        hidden = move_to_device(hidden, runtime.agent.wm.state_proj)
        downstream_hidden = (
            hidden if runtime.representation_to_backbone else hidden.detach()
        )
        current_state = runtime.agent.wm.project_state(downstream_hidden)

        # Stage3的DINO锚定作用于真实current observation state，而不是WM预测。
        dino_mse = None
        if dino_grid_target is not None:
            observed_state = (
                current_state
                if runtime.representation_to_backbone
                else runtime.agent.wm.project_state(hidden)
            )
            current_dino_target = dino_grid_target.to(
                device=observed_state.device,
                dtype=torch.float32,
                non_blocking=True,
            ).detach()
            if observed_state.shape != current_dino_target.shape:
                raise ValueError(
                    "current-state DINO target shape mismatch: "
                    f"state={tuple(observed_state.shape)}, "
                    f"target={tuple(current_dino_target.shape)}"
                )
            dino_mse = F.mse_loss(observed_state.float(), current_dino_target)
        elif self.config.predictor.lambda_dino != 0.0:
            raise ValueError("positive DINO-grid weight requires a current-state target")

        # 2. WM context：只有 current_state 可微，持久化历史和 successor target 都固定。
        stored_history = move_to_device(
            transition.state_history(self.config.predictor.history_size),
            runtime.agent.wm.wm_predictor,
        )
        current_state = current_state.to(device=stored_history.device)
        state_context = torch.cat(
            (stored_history[:-1].unsqueeze(0), current_state.unsqueeze(1)),
            dim=1,
        )
        previous_actions = transition.previous_actions(self.config.predictor.history_size).to(
            device=state_context.device
        )
        action_context = torch.cat(
            (
                previous_actions,
                torch.tensor(
                    [transition.action_index],
                    dtype=torch.long,
                    device=state_context.device,
                ),
            )
        ).unsqueeze(0)

        # 3. WM/DINO：预测这一次实际执行 action 后的真实 successor。
        wm_objective = None
        if self.config.predictor.train_wm and include_world_model:
            predicted_next_state = runtime.agent.wm.predict_state_sequence(
                state_context,
                action_context,
            )[:, -1]
            expected_next_state = move_to_device(
                transition.actual_next_state(),
                predicted_next_state,
            ).unsqueeze(0).detach()
            wm_objective = world_model_loss(
                predicted_next_state,
                expected_next_state,
                state_weight=self.config.predictor.lambda_wm,
            )
            weighted_wm_loss = wm_objective.loss
            wm_mse = wm_objective.state_mse
        else:
            weighted_wm_loss = current_state.new_zeros(())
            wm_mse = weighted_wm_loss

        # 4. Outcome读取同一个WM successor。标签缺失时保留同步forward但mask loss。
        outcome_loss = current_state.new_zeros(())
        outcome_count = 0
        outcome_correct = 0
        if self.config.outcome_head.enabled:
            outcome_head = runtime.agent.wm.outcome_head
            if outcome_head is None:
                raise RuntimeError("OutcomeHead objective is enabled but the runtime has no head")
            if wm_objective is None:
                raise RuntimeError("OutcomeHead requires an enabled WM successor prediction")
            outcome_logits = outcome_head(predicted_next_state)
            if transition.action_success is not None:
                outcome_target = torch.tensor(
                    [float(transition.action_success)],
                    device=outcome_logits.device,
                    dtype=outcome_logits.dtype,
                )
                outcome_loss = F.binary_cross_entropy_with_logits(
                    outcome_logits.reshape(1), outcome_target
                )
                outcome_count = 1
                outcome_correct = int(
                    (outcome_logits.detach().reshape(1) >= 0).item()
                    == transition.action_success
                )
            else:
                outcome_loss = outcome_logits.sum() * 0.0

        # 5. Value/Policy：两者均只读取 current_state 和实际执行 action。
        action_values = runtime.agent.wm.predict_action_values(current_state)
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
        planner_policy_objective = None
        if self.config.planner_policy.enabled:
            if old_policy_log_prob is None or policy_advantage is None:
                raise RuntimeError(
                    "PlannerPolicyHead PPO requires old log-prob and advantage"
                )
            value_loss = F.mse_loss(selected_action_values, target)
            action_logits = runtime.agent.wm.predict_action_logits(current_state)
            planner_policy_objective = ppo_action_policy_loss(
                action_logits=action_logits,
                executed_actions=executed_action,
                old_log_probs=old_policy_log_prob.reshape(1),
                advantages=policy_advantage.reshape(1),
                temperature=self.config.planner_policy.temperature,
                clip_ratio=self.config.planner_policy.clip_ratio,
            )
        else:
            if old_policy_log_prob is not None or policy_advantage is not None:
                raise RuntimeError(
                    "planner policy statistics require PlannerPolicyHead PPO"
                )
            if self.config.value_head.ppo_clip_range is None:
                raise RuntimeError("planner critic clipping requires value_ppo_clip_range")
            value_objective = ppo_action_value_loss(
                action_values,
                executed_action,
                target,
                old_action_value.reshape(1),
                clip_range=self.config.value_head.ppo_clip_range,
            )
            value_loss = value_objective.loss

        # 6. 合并 objective
        normalized_wm_loss = weighted_wm_loss / total_transitions
        normalized_wm_mse = wm_mse / total_transitions
        normalized_dino_mse = (
            dino_mse / total_transitions if dino_mse is not None else None
        )
        outcome_denominator = (
            total_transitions if total_outcomes is None else max(total_outcomes, 1)
        )
        normalized_outcome_loss = outcome_loss / outcome_denominator
        normalized_value_loss = value_loss / total_transitions
        normalized_policy_loss = 0
        normalized_policy_entropy = 0
        if planner_policy_objective is not None:
            normalized_policy_loss = planner_policy_objective.loss / total_transitions
            normalized_policy_entropy = (
                planner_policy_objective.entropy / total_transitions
            )
        total = normalized_wm_loss + normalized_value_loss.to(
            device=normalized_wm_loss.device
        )
        if normalized_dino_mse is not None:
            total = total + self.config.predictor.lambda_dino * normalized_dino_mse
        total = total + self.config.outcome_head.lambda_bce * normalized_outcome_loss
        total = total + normalized_policy_loss
        total = total - (
            self.config.planner_policy.entropy_coeff * normalized_policy_entropy
        )
        if not torch.isfinite(total):
            raise FloatingPointError("planner transition produced a non-finite total loss")
        losses = {
            "wm": normalized_wm_mse,
            "dino": normalized_dino_mse,
            "outcome": normalized_outcome_loss,
            "sigreg": None,
            "value": normalized_value_loss,
            "policy": (
                normalized_policy_loss
                if planner_policy_objective is not None
                else None
            ),
            "token_value": None,
            "reference_kl": None,
        }
        return RLStepOutput(
            loss=total,
            losses=losses,
            metrics=planner_step_metrics(
                losses=losses,
                total_loss=total,
                old_action_value=old_action_value,
                selected_action_values=selected_action_values,
                value_objective=(
                    None
                    if self.config.planner_policy.enabled
                    else value_objective
                ),
                policy_objective=planner_policy_objective,
                policy_advantage=policy_advantage,
                total_transitions=total_transitions,
                world_model_weight=self.config.predictor.lambda_wm,
                dino_grid_weight=self.config.predictor.lambda_dino,
                outcome_weight=self.config.outcome_head.lambda_bce,
                outcome_count=outcome_count,
                outcome_correct=outcome_correct,
            ),
        )

    def planner_transition_batch_step(
        self,
        runtime: RLModelRuntime,
        transitions: tuple[ExecutedTransition, ...],
        *,
        return_targets: tuple[torch.Tensor, ...],
        old_action_values: tuple[torch.Tensor, ...],
        old_policy_log_probs: tuple[torch.Tensor | None, ...] | None = None,
        policy_advantages: tuple[torch.Tensor | None, ...] | None = None,
        total_transitions: int,
        total_outcomes: int | None = None,
        dino_grid_targets: tuple[torch.Tensor | None, ...] | None = None,
        loss_weights: tuple[float, ...] | None = None,
        include_world_model: bool = True,
    ) -> RLStepOutput:
        """Share one padded Qwen forward across a planner micro-batch.

        Downstream WM/value/policy objectives retain the proven scalar transition
        path.  Summing those normalized outputs gives exact loss/gradient parity
        while removing repeated full-prefix Qwen calls inside each micro-batch.
        """

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

        hidden_batch = runtime.encode_state_prompts(
            tuple(transition.state_prompt for transition in transitions),
            retain_backbone_graph=self.config.predictor.lambda_dino > 0.0,
        )
        if hidden_batch.ndim not in (2, 3) or hidden_batch.shape[0] != batch_size:
            raise ValueError(
                "planner Qwen batch output does not align with transitions: "
                f"hidden={tuple(hidden_batch.shape)}, transitions={batch_size}"
            )
        outputs = tuple(
            self.planner_transition_step(
                runtime,
                transition,
                return_target=return_target,
                old_action_value=old_action_value,
                old_policy_log_prob=old_policy_log_prob,
                policy_advantage=policy_advantage,
                total_transitions=total_transitions,
                total_outcomes=total_outcomes,
                dino_grid_target=dino_grid_target,
                include_world_model=include_world_model,
                precomputed_hidden=hidden_batch[index : index + 1],
            )
            for index, (
                transition,
                return_target,
                old_action_value,
                old_policy_log_prob,
                policy_advantage,
                dino_grid_target,
            ) in enumerate(
                zip(
                    transitions,
                    fields["return_targets"],
                    fields["old_action_values"],
                    fields["old_policy_log_probs"],
                    fields["policy_advantages"],
                    fields["dino_grid_targets"],
                    strict=True,
                )
            )
        )
        weights = tuple(float(value) for value in fields["loss_weights"])
        total_loss = torch.stack(
            tuple(
                output.loss * weight
                for output, weight in zip(outputs, weights, strict=True)
            )
        ).sum()
        combined_losses: dict[str, torch.Tensor | None] = {}
        for name in outputs[0].losses:
            weighted_values = tuple(
                output.losses[name].to(device=total_loss.device) * weight
                for output, weight in zip(outputs, weights, strict=True)
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
                    for output, weight in zip(outputs, weights, strict=True)
                    if weight != 0.0
                )
                for name in outputs[0].metrics
            },
        )

    def sequence_step(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
        *,
        normalization: SequenceLossNormalization | None = None,
        include_policy: bool = True,
    ) -> RLStepOutput:
        """构造表示计算图，并可选地在同一图中计算 PPO 目标。

        ``include_policy=False`` 供 sequence micro-batching 使用：调用者先对
        表示目标反向并释放这张 Qwen 图，再调用 ``sequence_policy_step``。
        这样仍然累积到同一次 optimizer update，但不会让两张 Qwen 计算图
        同时驻留显存。
        """

        local_action_positions = batch.action_indices.numel()
        action_scale = (
            1.0
            if normalization is None
            else local_action_positions / normalization.action_positions
        )

        hidden_states = self._state_hidden_sequence(runtime, batch)
        downstream_hidden = (
            hidden_states
            if runtime.representation_to_backbone
            else hidden_states.detach()
        )
        state_sequence = runtime.agent.wm.project_state_sequence(downstream_hidden)
        state_context = state_sequence[:, :-1]
        action_values = runtime.agent.wm.predict_action_values(state_context)

        # WM 与 value 共享 state_context，但监督目标和梯度边界各自独立。
        wm_objective = None
        if self.config.predictor.train_wm:
            predicted_next_states = runtime.agent.wm.predict_state_sequence(
                state_context,
                batch.action_indices,
            )
            # 下一状态只作为固定监督值；同一状态在它作为 current state 时训练
            # StateProjector，不能让监督值反向靠近当前预测。
            expected_next_states = state_sequence[:, 1:].detach()
            wm_objective = world_model_loss(
                predicted_next_states,
                expected_next_states,
                state_weight=self.config.predictor.lambda_wm,
            )

        # DINO anchors the real current observation state. Auxiliary objectives
        # may stop at Qwen hidden while this path still trains Qwen and projector.
        dino_mse = None
        if batch.dino_grid_target is not None:
            observed_state_sequence = (
                state_sequence
                if runtime.representation_to_backbone
                else runtime.agent.wm.project_state_sequence(hidden_states)
            )
            observed_current_states = observed_state_sequence[:, :-1]
            dino_grid_target = batch.dino_grid_target.to(
                device=observed_current_states.device,
                dtype=torch.float32,
                non_blocking=True,
            ).detach()
            if observed_current_states.shape != dino_grid_target.shape:
                raise ValueError(
                    "current-state DINO target shape mismatch: "
                    f"state={tuple(observed_current_states.shape)}, "
                    f"target={tuple(dino_grid_target.shape)}"
                )
            dino_mse = F.mse_loss(observed_current_states.float(), dino_grid_target)
        elif self.config.predictor.lambda_dino != 0.0:
            raise ValueError("positive DINO-grid weight requires current-state targets")

        outcome_loss = None
        outcome_count = 0
        outcome_correct = 0
        if self.config.outcome_head.enabled:
            outcome_head = runtime.agent.wm.outcome_head
            if outcome_head is None:
                raise RuntimeError(
                    "OutcomeHead objective is enabled but the runtime has no head"
                )
            if wm_objective is None:
                raise RuntimeError(
                    "OutcomeHead requires an enabled WM successor prediction"
                )
            if (
                batch.action_success_targets is None
                or batch.action_success_mask is None
            ):
                raise ValueError(
                    "OutcomeHead training requires action_success labels and mask"
                )
            outcome_logits = outcome_head(predicted_next_states)
            outcome_targets = batch.action_success_targets.to(
                device=outcome_logits.device,
                dtype=outcome_logits.dtype,
            )
            outcome_mask = batch.action_success_mask.to(device=outcome_logits.device)
            if (
                outcome_logits.shape != outcome_targets.shape
                or outcome_mask.shape != outcome_logits.shape
            ):
                raise ValueError(
                    "OutcomeHead labels must align with sequence action positions"
                )
            outcome_count = int(outcome_mask.sum().item())
            if outcome_count == 0 and normalization is None:
                raise ValueError(
                    "OutcomeHead training requires at least one fresh action_success label"
                )
            if outcome_count > 0:
                outcome_loss = F.binary_cross_entropy_with_logits(
                    outcome_logits[outcome_mask],
                    outcome_targets[outcome_mask],
                )
                outcome_correct = int(
                    (
                        (outcome_logits.detach()[outcome_mask] >= 0)
                        == (outcome_targets[outcome_mask] >= 0.5)
                    ).sum().item()
                )
            elif normalization is not None:
                # OutcomeHead is a separate DDP module with
                # find_unused_parameters=False. Keep its graph in every micro
                # backward even when this chunk has no valid labels; the full
                # logical batch precheck guarantees a positive denominator.
                outcome_loss = outcome_logits.sum() * 0.0
        value_objective = action_value_loss(
            action_values,
            batch.action_indices,
            batch.return_targets,
            ranking_margin=self.config.value_head.rank_margin,
            ranking_weight=self.config.value_head.lambda_rank,
        )
        total = action_scale * value_objective.loss
        if wm_objective is not None:
            total = total + action_scale * wm_objective.loss
        if dino_mse is not None:
            total = total + (
                action_scale * self.config.predictor.lambda_dino * dino_mse
            )
        if outcome_loss is not None:
            outcome_scale = (
                1.0
                if normalization is None
                else outcome_count / normalization.outcome_labels
            )
            total = total + (
                outcome_scale * self.config.outcome_head.lambda_bce * outcome_loss
            )

        # 各 WM variant 明确选择 SIGReg 的统计单位；grid 与 SFT2 一致，对 slot
        # mean pooling 后再把 (B,T,D) 交给 SequenceSIGReg。
        sigreg_states = runtime.agent.wm.sigreg_state_sequence(state_sequence)
        sigreg_loss = (
            self.sigreg(sigreg_states)
            if self.sigreg is not None and self.config.predictor.lambda_sigreg > 0.0
            else None
        )
        if sigreg_loss is not None:
            total = total + self.config.predictor.lambda_sigreg * sigreg_loss

        policy, token_value_loss, reference_kl_loss = (
            self._policy_replay_losses(runtime, batch, value_objective)
            if include_policy
            else (None, None, None)
        )
        if policy is not None:
            # policy["loss"] 已取 clipped surrogate 的负号；entropy 作为奖励项减去。
            # Accelerate 会把device-mapped Qwen的输出复制回输入GPU，而WM/value
            # 位于model output GPU。只复制两个标量到监督loss设备；CopyBackward
            # 保留PPO到Qwen logits的完整梯度，避免搬运selected vocabulary logits。
            policy_loss = policy["loss"].to(device=total.device)
            policy_entropy = policy["entropy"].to(device=total.device)
            policy_tokens = int(policy["advantages"].numel())
            policy_scale = (
                1.0
                if normalization is None
                else policy_tokens / normalization.policy_tokens
            )
            total = total + policy_scale * policy_loss
            if self.config.actor.entropy_coeff > 0.0:
                total = total - (
                    policy_scale
                    * self.config.actor.entropy_coeff
                    * policy_entropy
                )
            if token_value_loss is not None:
                token_value_weight = cast(
                    float,
                    self.config.token_credit.value_loss_weight,
                )
                total = total + token_value_weight * token_value_loss.to(
                    device=total.device
                )
            if reference_kl_loss is not None:
                total = total + policy_scale * (
                    self.config.actor.reference_kl_loss_weight
                    * reference_kl_loss.to(device=total.device)
                )

        outcome_scale = (
            0.0
            if outcome_loss is None
            else (
                1.0
                if normalization is None
                else outcome_count / normalization.outcome_labels
            )
        )
        policy_scale = (
            0.0
            if policy is None
            else (
                1.0
                if normalization is None
                else int(policy["advantages"].numel())
                / normalization.policy_tokens
            )
        )

        metrics = {
            "wm_mse": (
                action_scale * float(wm_objective.state_mse.detach().item())
                if wm_objective is not None
                else 0.0
            ),
            "dino_grid_mse": (
                action_scale * float(dino_mse.detach().item())
                if dino_mse is not None
                else 0.0
            ),
            "lambda_wm": self.config.predictor.lambda_wm,
            "lambda_dino": self.config.predictor.lambda_dino,
            "outcome_bce": (
                outcome_scale * float(outcome_loss.detach().item())
                if outcome_loss is not None
                else 0.0
            ),
            "lambda_outcome": self.config.outcome_head.lambda_bce,
            "outcome_count": float(outcome_count),
            "outcome_correct": float(outcome_correct),
            "sigreg_loss": (
                float(sigreg_loss.detach().item()) if sigreg_loss is not None else 0.0
            ),
            "value_loss": action_scale * float(value_objective.loss.detach().item()),
            "value_mc_mse": action_scale * float(
                value_objective.monte_carlo_mse.detach().item()
            ),
            "value_rank": action_scale
            * float(value_objective.ranking.detach().item()),
            "total_loss": float(total.detach().item()),
            "actor_loss": (
                policy_scale * float(policy["loss"].detach().item())
                if policy
                else 0.0
            ),
            "token_value_loss": (
                float(token_value_loss.detach().item())
                if token_value_loss is not None
                else 0.0
            ),
            "reference_kl_loss": (
                policy_scale * float(reference_kl_loss.detach().item())
                if reference_kl_loss is not None
                else 0.0
            ),
        }
        if policy is not None:
            metrics.update(
                {
                    "entropy": policy_scale
                    * float(policy["entropy"].detach().item()),
                    "mean_advantage": policy_scale
                    * float(policy["advantages"].mean().item()),
                    "mean_abs_advantage": policy_scale
                    * float(
                        policy["advantages"].abs().mean().item()
                    ),
                    "clip_fraction": policy_scale
                    * float(policy["clip_fraction"].item()),
                    "mean_ratio": policy_scale
                    * float(policy["probability_ratio"].mean().item()),
                    "policy_tokens": float(policy["advantages"].numel()),
                }
            )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": (
                    wm_objective.state_mse if wm_objective is not None else None
                ),
                "dino": dino_mse,
                "outcome": outcome_loss,
                "sigreg": sigreg_loss,
                "value": value_objective.loss,
                "policy": policy["loss"] if policy else None,
                "token_value": token_value_loss,
                "reference_kl": reference_kl_loss,
            },
            metrics=metrics,
        )

    def sequence_policy_step(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
        *,
        normalization: SequenceLossNormalization,
    ) -> RLStepOutput:
        """单独构造 direct PPO policy replay 图。

        只允许在 full-batch critic advantage 已预计算后调用，保证拆分前后
        使用同一组全局归一化 advantage。该 loss 随表示 loss 一起累积梯度，
        由训练 loop 统一执行一次 optimizer step。
        """

        if runtime.policy_replay is None:
            raise RuntimeError("sequence policy step requires policy replay")
        if batch.policy_step_advantages is None:
            raise ValueError(
                "split sequence policy step requires precomputed step advantages"
            )
        policy, token_value_loss, reference_kl_loss = self._policy_replay_losses(
            runtime,
            batch,
            None,
        )
        if policy is None:
            raise RuntimeError("sequence policy step produced no PPO objective")
        policy_tokens = int(policy["advantages"].numel())
        policy_scale = policy_tokens / normalization.policy_tokens
        total = policy_scale * policy["loss"]
        if self.config.actor.entropy_coeff > 0.0:
            total = total - (
                policy_scale
                * self.config.actor.entropy_coeff
                * policy["entropy"]
            )
        if token_value_loss is not None:
            total = total + (
                cast(float, self.config.token_credit.value_loss_weight)
                * token_value_loss
            )
        if reference_kl_loss is not None:
            total = total + policy_scale * (
                self.config.actor.reference_kl_loss_weight * reference_kl_loss
            )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": None,
                "dino": None,
                "outcome": None,
                "sigreg": None,
                "value": None,
                "policy": policy["loss"],
                "token_value": token_value_loss,
                "reference_kl": reference_kl_loss,
            },
            metrics={
                "total_loss": float(total.detach().item()),
                "actor_loss": policy_scale
                * float(policy["loss"].detach().item()),
                "token_value_loss": (
                    float(token_value_loss.detach().item())
                    if token_value_loss is not None
                    else 0.0
                ),
                "reference_kl_loss": (
                    policy_scale * float(reference_kl_loss.detach().item())
                    if reference_kl_loss is not None
                    else 0.0
                ),
                "entropy": policy_scale
                * float(policy["entropy"].detach().item()),
                "mean_advantage": policy_scale
                * float(policy["advantages"].mean().item()),
                "mean_abs_advantage": policy_scale
                * float(policy["advantages"].abs().mean().item()),
                "clip_fraction": policy_scale
                * float(policy["clip_fraction"].item()),
                "mean_ratio": policy_scale
                * float(policy["probability_ratio"].mean().item()),
                "policy_tokens": float(policy_tokens),
            },
        )

    @torch.no_grad()
    def sequence_old_action_values(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> torch.Tensor:
        """Compute the frozen pre-update critic baseline for advantage whitening."""

        hidden_states = self._state_hidden_sequence(runtime, batch)
        downstream_hidden = (
            hidden_states
            if runtime.representation_to_backbone
            else hidden_states.detach()
        )
        state_sequence = runtime.agent.wm.project_state_sequence(downstream_hidden)
        action_values = runtime.agent.wm.predict_action_values(
            state_sequence[:, :-1]
        )
        return action_values.gather(
            -1,
            batch.action_indices.to(device=action_values.device).unsqueeze(-1),
        ).squeeze(-1).detach()

    def _state_hidden_sequence(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> torch.Tensor:
        """按显式 state source 读取 rollout hidden 或重新执行 Qwen。"""

        state_steps = self.config.predictor.history_size + 1
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
            retain_backbone_graph=self.config.predictor.lambda_dino > 0.0,
        )

    def _policy_replay_losses(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
        value_objective: ActionValueLoss | None,
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
        if self.config.actor.credit_assignment == "token":
            token_values = replay_output.token_values
            if token_values is None:
                raise RuntimeError("token credit replay returned no token values")
            token_credit = token_level_gae(
                batch.return_targets.flatten(),
                token_values,
                replay_inputs,
                gamma=cast(float, self.config.token_credit.gamma),
                gae_lambda=cast(float, self.config.token_credit.gae_lambda),
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
            if batch.policy_step_advantages is None:
                if value_objective is None:
                    raise ValueError(
                        "policy replay requires critic values or precomputed advantages"
                    )
                step_advantages = normalized_monte_carlo_advantages(
                    return_targets=batch.return_targets.flatten().to(
                        device=value_objective.selected_action_values.device,
                        dtype=value_objective.selected_action_values.dtype,
                    ),
                    predicted_values=value_objective.selected_action_values.flatten(),
                )
            else:
                step_advantages = batch.policy_step_advantages.flatten()
            step_advantages = step_advantages.to(
                device=replay_output.selected_log_probs.device,
                dtype=replay_output.selected_log_probs.dtype,
            )
            if step_advantages.numel() != batch.return_targets.numel():
                raise ValueError(
                    "policy step advantages must align with sequence action positions"
                )
            advantages = expand_step_advantages(
                step_advantages,
                replay_inputs,
                credit_assignment=self.config.actor.credit_assignment,
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

        if self.config.actor.reference_kl_loss_weight <= 0.0:
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
            clip_ratio=self.config.actor.clip_ratio,
        )
        return {
            "loss": objective.loss,
            "entropy": objective.entropy,
            "advantages": advantages,
            "probability_ratio": objective.probability_ratio,
            "clip_fraction": objective.clip_fraction,
        }


__all__ = [
    "PLANNER_POLICY_TRAINING_OBJECTIVE",
    "PLANNER_TRAINING_OBJECTIVE",
    "PlannerOldPolicyStatistics",
    "RLAlgorithm",
    "RLBatch",
    "RLStepOutput",
    "SequenceLossNormalization",
    "build_rl_batch",
    "low_variance_kl",
    "normalized_monte_carlo_advantages",
    "slice_rl_batch",
]
