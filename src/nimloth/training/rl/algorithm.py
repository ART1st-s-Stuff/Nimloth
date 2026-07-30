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
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.util.module import move_to_device
from nimloth.wm import SequenceSIGReg


PLANNER_TRAINING_OBJECTIVE = "receding_horizon_decision_state_mc_v2"


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
        credit_assignment: Literal["action", "turn", "token"] = "action",
        token_gamma: float | None = None,
        token_gae_lambda: float | None = None,
        token_value_loss_weight: float | None = None,
        reference_kl_loss_weight: float = 0.0,
        train_world_model: bool = True,
        world_model_weight: float = 1.0,
        dino_grid_weight: float = 0.0,
    ) -> None:
        """消费已经由 ``RLConfig`` 校验过的算法参数。"""

        self.history_size = int(history_size)
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
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

    def actor_transition_step(
        self,
        runtime: RLModelRuntime,
        transition: ExecutedTransition,
        *,
        return_target: torch.Tensor,
        total_transitions: int,
        dino_grid_target: torch.Tensor | None = None,
    ) -> RLStepOutput:
        """Train one executed transition through Qwen, WM, and ValueHead.

        Qwen is recomputed on the complete persisted prefix for this environment
        step.  The previous-history tokens are fixed inputs, while every activation
        in this current forward remains in the graph.  The value target is applied
        to the executed-action slot on the current decision state.  Its gradient
        reaches ValueHead -> StateProjector -> the complete Qwen prefix; WM and DINO
        losses independently supervise the predicted successor state.
        """

        if total_transitions < 1:
            raise ValueError("total_transitions must be positive")
        if runtime.state_source != "recompute" or not runtime.representation_to_backbone:
            raise RuntimeError(
                "planner transition training requires differentiable full-prefix "
                "Qwen recomputation"
            )

        hidden = runtime.encode_state_prompts((transition.state_prompt,))
        hidden = move_to_device(hidden, runtime.agent.wm.state_proj)
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
        predicted_next_state = runtime.agent.wm.predict_state_sequence(
            state_context,
            action_context,
        )[:, -1]

        expected_next_state = move_to_device(
            transition.actual_next_state(),
            predicted_next_state,
        ).unsqueeze(0).detach()
        wm_objective = None
        if self.train_world_model:
            current_dino_target = (
                dino_grid_target.to(
                    device=predicted_next_state.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                if dino_grid_target is not None
                else None
            )
            wm_objective = world_model_loss(
                predicted_next_state,
                expected_next_state,
                state_weight=self.world_model_weight,
                dino_grid_target=current_dino_target,
                dino_grid_weight=self.dino_grid_weight,
            )
            weighted_wm_loss = wm_objective.loss
            wm_mse = wm_objective.state_mse
        else:
            weighted_wm_loss = predicted_next_state.sum() * 0.0
            wm_mse = weighted_wm_loss

        action_values = runtime.agent.wm.predict_action_values(current_state)
        executed_action = torch.tensor(
            [transition.action_index],
            dtype=torch.long,
            device=action_values.device,
        )
        value_objective = action_value_loss(
            action_values,
            executed_action,
            return_target.reshape(1).to(device=action_values.device),
            ranking_margin=self.value_rank_margin,
            ranking_weight=self.value_rank_weight,
        )

        normalized_wm_loss = weighted_wm_loss / total_transitions
        normalized_wm_mse = wm_mse / total_transitions
        normalized_dino_mse = (
            wm_objective.dino_grid_mse / total_transitions
            if wm_objective is not None and wm_objective.dino_grid_mse is not None
            else None
        )
        normalized_value_loss = value_objective.loss / total_transitions
        total = normalized_wm_loss + normalized_value_loss.to(
            device=normalized_wm_loss.device
        )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": normalized_wm_mse,
                "dino": normalized_dino_mse,
                "sigreg": None,
                "value": normalized_value_loss,
                "policy": None,
                "token_value": None,
                "reference_kl": None,
            },
            metrics={
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
                "value_mc_mse": float(
                    (value_objective.monte_carlo_mse / total_transitions)
                    .detach()
                    .item()
                ),
                "value_rank": float(
                    (value_objective.ranking / total_transitions).detach().item()
                ),
                "total_loss": float(total.detach().item()),
                "actor_loss": 0.0,
                "token_value_loss": 0.0,
                "reference_kl_loss": 0.0,
                "policy_tokens": 0.0,
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

        shapes = {
            tuple(new_log_probs.shape),
            tuple(old_log_probs.shape),
            tuple(entropies.shape),
            tuple(advantages.shape),
        }
        if len(shapes) != 1 or new_log_probs.ndim != 1:
            raise ValueError("PPO token log-probs, entropy and advantages must align")

        probability_ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(
            probability_ratio,
            1.0 - self.ppo_clip_ratio,
            1.0 + self.ppo_clip_ratio,
        )
        loss = -torch.min(
            probability_ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        with torch.no_grad():
            clip_fraction = (
                (probability_ratio - 1.0)
                .abs()
                .gt(self.ppo_clip_ratio)
                .float()
                .mean()
            )
        return {
            "loss": loss,
            "entropy": entropies.mean(),
            "advantages": advantages,
            "probability_ratio": probability_ratio,
            "clip_fraction": clip_fraction,
        }


__all__ = [
    "PLANNER_TRAINING_OBJECTIVE",
    "RLAlgorithm",
    "RLBatch",
    "RLStepOutput",
    "build_rl_batch",
    "low_variance_kl",
    "normalized_monte_carlo_advantages",
]
