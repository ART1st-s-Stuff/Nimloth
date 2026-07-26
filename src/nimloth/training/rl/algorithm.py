"""RL 的连续序列采样、模型前向与目标函数。"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from nimloth.agent import AgentPrompt, PolicyReplayInput
from nimloth.rollout import TrajectoryWindow
from nimloth.rollout.transitions import discounted_action_value_targets
from nimloth.training.rl.credit import expand_step_advantages, token_level_gae
from nimloth.training.rl.episodes import (
    EpisodeTrainingBatch,
    TemporalDifferenceStep,
)
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.wm import SequenceSIGReg


@dataclass(frozen=True)
class RLBatch:
    """一次 RL 更新消费的原始连续窗口与逐步监督。

    action/return 张量为 ``(B,H)``；``old_log_probs`` 按 window-major、time-minor
    展开后，再按每步 loss-mask token 展开。这个顺序必须与 policy replay 一致。
    """

    windows: tuple[TrajectoryWindow, ...]
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor

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
    def cached_state_latent_hiddens(self) -> torch.Tensor | None:
        """返回 ``(B,H+1,K,D)`` rollout hidden；一个 batch 禁止混用来源。"""

        rows = [window.cached_state_latent_hiddens() for window in self.windows]
        populated = [row is not None for row in rows]
        if any(populated) and not all(populated):
            raise ValueError("one RL batch cannot mix cached and uncached Qwen states")
        if not any(populated):
            return None
        return torch.stack([row for row in rows if row is not None], dim=0)


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


def low_variance_kl(log_ratio: torch.Tensor) -> torch.Tensor:
    """Evaluate VAGEN's clamped low-variance KL without overflowing exp().

    The final penalty is already saturated at 10 outside this input interval,
    so clamping the exponent input preserves the value and zero-gradient region.
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

    下一状态保持 stop-gradient；Backbone 是否接收 WM/value/SIGReg 梯度由
    ``RLModelRuntime`` 的显式模式决定，StateProjector 是否训练只由参数冻结状态
    决定。policy replay、optimizer、rollout 生命周期和 checkpoint 由运行期管理。
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
        action_objective: str = "ppo",
        credit_assignment: str = "action",
        token_gamma: float | None = None,
        token_gae_lambda: float | None = None,
        token_value_loss_weight: float | None = None,
        planner_distillation_weight: float | None = None,
        reference_kl_loss_weight: float = 0.0,
        reference_kl_loss_type: str | None = None,
        train_world_model: bool = True,
    ) -> None:
        self.history_size = int(history_size)
        if self.history_size < 1:
            raise ValueError(
                f"history_size must be positive, got {self.history_size}"
            )
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.ppo_clip_ratio = float(ppo_clip_ratio)
        self.entropy_weight = float(entropy_weight)
        if action_objective not in {"distillation", "ppo"}:
            raise ValueError("action objective must be distillation or ppo")
        self.action_objective = action_objective
        if credit_assignment not in {"action", "turn", "token"}:
            raise ValueError(
                f"unsupported PPO credit assignment: {credit_assignment!r}"
            )
        self.credit_assignment = credit_assignment
        self.token_gamma = token_gamma
        self.token_gae_lambda = token_gae_lambda
        self.token_value_loss_weight = token_value_loss_weight
        self.planner_distillation_weight = planner_distillation_weight
        self.reference_kl_loss_weight = float(reference_kl_loss_weight)
        self.reference_kl_loss_type = reference_kl_loss_type
        if self.reference_kl_loss_weight < 0.0:
            raise ValueError("reference KL loss weight must be non-negative")
        if self.reference_kl_loss_weight > 0.0:
            if self.reference_kl_loss_type != "low_var_kl":
                raise ValueError("reference KL loss requires low_var_kl")
            if self.credit_assignment != "token":
                raise ValueError("reference KL loss requires token credit")
        elif self.reference_kl_loss_type is not None:
            raise ValueError("reference KL type requires a positive loss weight")
        self.train_world_model = bool(train_world_model)
        if self.credit_assignment == "token" and any(
            value is None
            for value in (
                self.token_gamma,
                self.token_gae_lambda,
                self.token_value_loss_weight,
            )
        ):
            raise ValueError("token credit requires explicit token GAE parameters")

    def _predict_executed_segment(
        self,
        runtime: RLModelRuntime,
        state_history: torch.Tensor,
        previous_actions: torch.Tensor,
        segment_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Replay executed actions autoregressively through the wrapped WM."""

        all_states = state_history
        all_actions = torch.cat((previous_actions, segment_actions), dim=1)
        history_steps = state_history.shape[1]
        predictions: list[torch.Tensor] = []
        for segment_index in range(segment_actions.shape[1]):
            state_index = history_steps - 1 + segment_index
            context_start = max(0, state_index - self.history_size + 1)
            state_context = all_states[:, context_start : state_index + 1]
            action_context = all_actions[:, context_start : state_index + 1]
            predicted_sequence = runtime.agent.wm.predict_state_sequence(
                state_context,
                action_context,
            )
            next_state = predicted_sequence[:, -1]
            predictions.append(next_state)
            all_states = torch.cat((all_states, next_state.unsqueeze(1)), dim=1)
        return torch.stack(predictions, dim=1)

    def temporal_difference_step(
        self,
        runtime: RLModelRuntime,
        step: TemporalDifferenceStep,
        *,
        total_td_steps: int,
    ) -> RLStepOutput:
        """Replay one Qwen anchor and its executed WM segment, then form TD loss."""

        if total_td_steps < 1:
            raise ValueError("total_td_steps must be positive")
        if self.sigreg is not None and self.sigreg_weight > 0.0:
            raise ValueError("episode TD training does not yet define SIGReg units")
        if self.action_objective != "distillation":
            raise ValueError(
                "greedy planner TD steps require action distillation; action PPO "
                "requires Qwen-owned sampled behavior"
            )
        if self.planner_distillation_weight is None:
            raise ValueError("planner TD step requires a distillation weight")
        if runtime.policy_replay is None:
            raise RuntimeError("planner TD step requires Qwen action replay")

        saved_context = runtime.prepare_world_model_states(
            step.retained_state_context(self.history_size)
        )
        start_hidden = runtime.prepare_anchor_hidden(
            step.anchor_hidden(step.start_step)
        )
        start_state = runtime.agent.wm.project_state(start_hidden.unsqueeze(0))
        prior_states = saved_context[:-1].unsqueeze(0)
        state_history = torch.cat(
            (prior_states, start_state.unsqueeze(1)),
            dim=1,
        )
        previous_actions = step.previous_actions(self.history_size).to(
            device=state_history.device
        ).unsqueeze(0)
        segment_actions = torch.tensor(
            [step.action_indices],
            dtype=torch.long,
            device=state_history.device,
        )
        predicted_states = self._predict_executed_segment(
            runtime,
            state_history,
            previous_actions,
            segment_actions,
        )
        with torch.no_grad():
            target_hidden = runtime.prepare_anchor_hidden(
                step.anchor_hidden(step.end_step)
            )
            target_state = runtime.agent.wm.project_target_state(
                target_hidden.unsqueeze(0)
            ).detach()
        wm_loss = (
            F.mse_loss(predicted_states[:, -1], target_state)
            if self.train_world_model
            else predicted_states[:, -1].sum() * 0.0
        )

        replay_input = step.action_replay_input()
        planner_trace = replay_input.planner_trace
        assert planner_trace is not None
        if planner_trace.action_training.objective != self.action_objective:
            raise ValueError("rollout action objective does not match training config")
        replay_output = runtime.policy_replay((replay_input,))
        if replay_output.action_log_probs is None:
            raise RuntimeError("distillation replay returned no Qwen action logits")
        teacher = planner_trace.action_training.teacher_action_log_probs
        if teacher is None:
            raise ValueError("distillation trace has no teacher distribution")
        teacher_log_probs = torch.tensor(
            teacher,
            dtype=replay_output.action_log_probs.dtype,
            device=replay_output.action_log_probs.device,
        )
        teacher_probs = teacher_log_probs.exp().detach()
        action_distillation_loss = -(
            teacher_probs * replay_output.action_log_probs[0]
        ).sum()
        safe_teacher_log_probs = torch.where(
            torch.isfinite(teacher_log_probs),
            teacher_log_probs,
            torch.zeros_like(teacher_log_probs),
        )
        action_distillation_kl = (
            teacher_probs
            * (safe_teacher_log_probs - replay_output.action_log_probs[0])
        ).sum()

        normalized_wm_loss = wm_loss / total_td_steps
        normalized_action_loss = action_distillation_loss / total_td_steps
        total = normalized_wm_loss + self.planner_distillation_weight * (
            normalized_action_loss.to(device=normalized_wm_loss.device)
        )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": normalized_wm_loss,
                "sigreg": None,
                "value": None,
                "policy": None,
                "token_value": None,
                "action_distillation": normalized_action_loss,
                "reference_kl": None,
            },
            metrics={
                "wm_mse": float(normalized_wm_loss.detach().item()),
                "sigreg_loss": 0.0,
                "value_loss": 0.0,
                "total_loss": float(total.detach().item()),
                "actor_loss": 0.0,
                "token_value_loss": 0.0,
                "action_distillation_loss": float(
                    normalized_action_loss.detach().item()
                ),
                "action_distillation_kl": float(
                    (action_distillation_kl / total_td_steps).detach().item()
                ),
                "reference_kl_loss": 0.0,
                "policy_tokens": 0.0,
            },
        )

    def monte_carlo_step(
        self,
        runtime: RLModelRuntime,
        episodes: tuple[EpisodeTrainingBatch, ...],
    ) -> RLStepOutput:
        """Fit ValueHead on full-episode returns using only detached WM states."""

        if not episodes:
            raise ValueError("MC step requires at least one episode")
        states = runtime.prepare_value_states(
            torch.cat([episode.action_states for episode in episodes], dim=0)
        )
        actions = torch.cat([episode.action_indices for episode in episodes], dim=0)
        actions = actions.to(device=states.device)
        returns = torch.cat([episode.return_targets for episode in episodes], dim=0)
        returns = returns.to(device=states.device)
        action_values = runtime.agent.wm.predict_action_values(states.detach())
        value_loss, _chosen_values = self._value_loss(
            action_values,
            actions,
            returns,
        )
        return RLStepOutput(
            loss=value_loss,
            losses={
                "wm": None,
                "sigreg": None,
                "value": value_loss,
                "policy": None,
                "token_value": None,
                "action_distillation": None,
                "reference_kl": None,
            },
            metrics={
                "wm_mse": 0.0,
                "sigreg_loss": 0.0,
                "value_loss": float(value_loss.detach().item()),
                "total_loss": float(value_loss.detach().item()),
                "actor_loss": 0.0,
                "token_value_loss": 0.0,
                "action_distillation_loss": 0.0,
                "action_distillation_kl": 0.0,
                "reference_kl_loss": 0.0,
                "policy_tokens": 0.0,
            },
        )

    def training_step(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> RLStepOutput:
        """构造 RL 计算图并计算 WM、value 与可选 PPO 目标。"""

        expected_state_steps = self.history_size + 1
        cached_hidden_states = batch.cached_state_latent_hiddens
        if cached_hidden_states is None:
            # 旧离线 trajectory 没有 rollout hidden 时，按时间位置逐次执行 Qwen；
            # 每次 forward 只有 B 个 prompt，不把 history_size 展平进 Qwen batch。
            hidden_states = runtime.encode_state_sequence(
                batch.state_prompts,
                batch_size=len(batch.windows),
                state_steps=expected_state_steps,
            )
        else:
            hidden_states = runtime.prepare_cached_state_sequence(
                cached_hidden_states,
                batch_size=len(batch.windows),
                state_steps=expected_state_steps,
            )
        if (
            hidden_states.ndim not in (3, 4)
            or hidden_states.shape[0] != len(batch.windows)
            or hidden_states.shape[1] != expected_state_steps
        ):
            raise ValueError(
                "RL hidden_states must have shape (B, history_size + 1, D) "
                "or (B, history_size + 1, k, D), "
                f"got {tuple(hidden_states.shape)} for history_size={self.history_size}"
            )
        expected_actions = (
            hidden_states.shape[0],
            self.history_size,
        )
        if batch.action_indices.shape != expected_actions:
            raise ValueError(
                "RLBatch action_indices must have shape (B, history_size), "
                f"got {tuple(batch.action_indices.shape)}, expected {expected_actions}"
            )

        # 标准 latent WM 复用同一份在线 state 作为 target；grid WM 在这里通过
        # 自己的冻结 EMA encoder 生成 target，objective 不依赖具体 variant。
        if self.train_world_model:
            state_sequence, target_state_sequence = (
                runtime.agent.wm.project_training_state_sequences(hidden_states)
            )
        else:
            state_sequence = runtime.agent.wm.project_state_sequence(hidden_states)
            target_state_sequence = state_sequence
        state_context = state_sequence[:, :-1]
        action_values = runtime.agent.wm.predict_action_values(state_context)

        # WM 与 value 共享 state_context，但监督目标和梯度边界各自独立。
        wm_loss: torch.Tensor | None = None
        if self.train_world_model:
            predicted_next_states = runtime.agent.wm.predict_state_sequence(
                state_context,
                batch.action_indices,
            )
            target_next_states = target_state_sequence[:, 1:].detach()
            wm_loss = F.mse_loss(predicted_next_states, target_next_states)
        value_loss, chosen_values = self._value_loss(
            action_values,
            batch.action_indices,
            batch.return_targets,
        )
        total = value_loss if wm_loss is None else wm_loss + value_loss

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

        policy: dict[str, torch.Tensor] | None = None
        token_value_loss: torch.Tensor | None = None
        reference_kl_loss: torch.Tensor | None = None
        if runtime.policy_replay is not None:
            # replay 与上面的监督张量使用相同的 window/time 展开顺序；flatten 后每个
            # ratio 仍对应 rollout 中同一个动作位置。
            replay_output = runtime.policy_replay(
                batch.policy_replay_inputs
            )
            planner_traces = [
                sample.planner_trace for sample in batch.policy_replay_inputs
            ]
            planner_batch = all(trace is not None for trace in planner_traces)
            if any(trace is not None for trace in planner_traces) and not planner_batch:
                raise ValueError("one RL batch cannot mix planner and direct behavior")
            if planner_batch:
                raise ValueError(
                    "planner trajectories must use complete-episode TD/MC training"
                )
            if self.reference_kl_loss_weight > 0.0:
                if replay_output.selected_full_log_probs is None:
                    raise RuntimeError(
                        "reference KL requires current full-vocabulary log-probs"
                    )
                reference_rows = [
                    sample.selected_reference_log_probs
                    for sample in batch.policy_replay_inputs
                ]
                if any(row is None for row in reference_rows):
                    raise ValueError(
                        "reference KL requires frozen reference log-probs on every step"
                    )
                reference_log_probs = torch.tensor(
                    [
                        value
                        for row in reference_rows
                        if row is not None
                        for value in row
                    ],
                    dtype=replay_output.selected_full_log_probs.dtype,
                    device=replay_output.selected_full_log_probs.device,
                )
                if reference_log_probs.shape != (
                    replay_output.selected_full_log_probs.shape
                ):
                    raise ValueError(
                        "reference/current CoT log-prob counts do not align"
                    )
                # VAGEN low_var_kl: exp(ref-logp) - (ref-logp) - 1.
                # Reference values are constants persisted before actor updates.
                log_ratio = (
                    reference_log_probs
                    - replay_output.selected_full_log_probs
                )
                reference_kl_loss = low_variance_kl(log_ratio).mean()
            if self.credit_assignment == "token":
                if replay_output.token_values is None:
                    raise RuntimeError("token credit replay returned no token values")
                assert self.token_gamma is not None
                assert self.token_gae_lambda is not None
                assert self.token_value_loss_weight is not None
                token_credit = token_level_gae(
                    batch.return_targets.flatten(),
                    replay_output.token_values,
                    batch.policy_replay_inputs,
                    gamma=self.token_gamma,
                    gae_lambda=self.token_gae_lambda,
                )
                advantages = token_credit.advantages.to(
                    device=replay_output.selected_log_probs.device,
                    dtype=replay_output.selected_log_probs.dtype,
                )
                token_value_loss = F.mse_loss(
                    replay_output.token_values,
                    token_credit.returns.to(
                        device=replay_output.token_values.device,
                        dtype=replay_output.token_values.dtype,
                    ),
                )
            else:
                step_advantages = normalized_monte_carlo_advantages(
                    return_targets=batch.return_targets.flatten().to(
                        device=chosen_values.device,
                        dtype=chosen_values.dtype,
                    ),
                    predicted_values=chosen_values.flatten(),
                ).to(
                    device=replay_output.selected_log_probs.device,
                    dtype=replay_output.selected_log_probs.dtype,
                )
                advantages = expand_step_advantages(
                    step_advantages,
                    batch.policy_replay_inputs,
                    credit_assignment=self.credit_assignment,  # type: ignore[arg-type]
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
            # policy["loss"] 已取 clipped surrogate 的负号；entropy 作为奖励项减去。
            # Accelerate 会把device-mapped Qwen的输出复制回输入GPU，而WM/value
            # 位于model output GPU。只复制两个标量到监督loss设备；CopyBackward
            # 保留PPO到Qwen logits的完整梯度，避免搬运selected vocabulary logits。
            policy_loss = policy["loss"].to(device=total.device)
            policy_entropy = policy["entropy"].to(device=total.device)
            total = total + policy_loss - self.entropy_weight * policy_entropy
            if token_value_loss is not None:
                total = total + self.token_value_loss_weight * token_value_loss.to(
                    device=total.device
                )
            if reference_kl_loss is not None:
                total = total + self.reference_kl_loss_weight * (
                    reference_kl_loss.to(device=total.device)
                )

        metrics = {
            "wm_mse": float(wm_loss.detach().item()) if wm_loss is not None else 0.0,
            "sigreg_loss": (
                float(sigreg_loss.detach().item()) if sigreg_loss is not None else 0.0
            ),
            "value_loss": float(value_loss.detach().item()),
            "total_loss": float(total.detach().item()),
            "actor_loss": float(policy["loss"].detach().item()) if policy else 0.0,
            "token_value_loss": (
                float(token_value_loss.detach().item())
                if token_value_loss is not None
                else 0.0
            ),
            "action_distillation_loss": 0.0,
            "action_distillation_kl": 0.0,
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
                "wm": wm_loss,
                "sigreg": sigreg_loss,
                "value": value_loss,
                "policy": policy["loss"] if policy else None,
                "token_value": token_value_loss,
                "action_distillation": None,
                "reference_kl": reference_kl_loss,
            },
            metrics=metrics,
        )

    def _value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """回归实际动作的 return，并可选约束其高于当前最优未选动作。"""

        chosen_values = all_values.gather(
            -1,
            action_indices.unsqueeze(-1),
        ).squeeze(-1)
        targets = return_targets.to(device=all_values.device, dtype=all_values.dtype)
        regression = F.mse_loss(chosen_values, targets)
        chosen_mask = F.one_hot(
            action_indices,
            num_classes=all_values.shape[-1],
        ).bool()
        max_other = all_values.masked_fill(chosen_mask, float("-inf")).max(dim=-1).values
        ranking = F.relu(
            self.value_rank_margin + max_other - chosen_values
        ).mean()
        return regression + self.value_rank_weight * ranking, chosen_values

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
    "RLAlgorithm",
    "RLBatch",
    "RLStepOutput",
    "build_rl_batch",
    "low_variance_kl",
    "normalized_monte_carlo_advantages",
]
