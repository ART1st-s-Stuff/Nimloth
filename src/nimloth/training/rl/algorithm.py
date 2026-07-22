"""RL 的连续序列采样、模型前向与目标函数。"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from nimloth.agent import AgentPrompt, PolicyReplayInput
from nimloth.rollout import TrajectoryWindow
from nimloth.rollout.transitions import discounted_action_value_targets
from nimloth.training.rl.runtime import RLModelRuntime
from nimloth.wm import SequenceSIGReg


@dataclass(frozen=True)
class RLBatch:
    """一次 RL 更新消费的原始连续窗口与逐步监督。"""

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


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


def build_rl_batch(
    windows: tuple[TrajectoryWindow, ...],
    *,
    gamma: float,
    device: torch.device,
) -> RLBatch:
    """把已采样窗口的动作、return 和行为概率整理成张量。"""

    if not windows:
        raise ValueError("RL batch requires at least one trajectory window")
    history_sizes = {window.history_size for window in windows}
    if len(history_sizes) != 1:
        raise ValueError("one RL batch cannot mix trajectory window lengths")
    history_size = history_sizes.pop()
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
                )[window.start_step : window.start_step + history_size]
                for window in windows
            ],
            dtype=torch.float32,
            device=device,
        ),
        old_log_probs=torch.tensor(
            [
                [
                    window.trajectory.action_log_probs[step_index][
                        window.trajectory.action_indices[step_index]
                    ]
                    for step_index in range(
                        window.start_step,
                        window.start_step + history_size,
                    )
                ]
                for window in windows
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
    """用当前 value baseline 归一化 Monte Carlo return。"""

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

    def training_step(
        self,
        runtime: RLModelRuntime,
        batch: RLBatch,
    ) -> RLStepOutput:
        """构造 RL 计算图并计算 WM、value 与可选 PPO 目标。"""

        hidden_states = runtime.encode_state_sequence(
            batch.state_prompts,
            batch_size=len(batch.windows),
            state_steps=self.history_size + 1,
        )
        expected_state_steps = self.history_size + 1
        if (
            hidden_states.ndim != 3
            or hidden_states.shape[1] != expected_state_steps
        ):
            raise ValueError(
                "RL hidden_states must have shape (B, history_size + 1, D), "
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

        state_sequence = runtime.agent.wm.project_state(hidden_states)
        state_context = state_sequence[:, :-1]
        predicted_next_states = runtime.agent.wm.predict_state_sequence(
            state_context,
            batch.action_indices,
        )

        # 一个状态可以在当前位置更新 projector，同时在前一位置作为 stop-gradient
        # target；因此先统一投影，再只 detach 右移后的 target 视图。
        target_next_states = state_sequence[:, 1:].detach()
        action_values = runtime.agent.wm.predict_action_values(state_context)

        # 三个目标在这里直接组合，避免把同一批 tensor 再转发给一层 objective。
        wm_loss = F.mse_loss(predicted_next_states, target_next_states)
        value_loss, chosen_values = self._value_loss(
            action_values,
            batch.action_indices,
            batch.return_targets,
        )
        total = wm_loss + value_loss

        sigreg_loss = (
            self.sigreg(state_sequence)
            if self.sigreg is not None and self.sigreg_weight > 0.0
            else None
        )
        if sigreg_loss is not None:
            total = total + self.sigreg_weight * sigreg_loss

        policy: dict[str, torch.Tensor] | None = None
        if runtime.policy_replay is not None:
            new_log_probs, action_log_probs = runtime.policy_replay(
                batch.policy_replay_inputs
            )
            advantages = normalized_monte_carlo_advantages(
                return_targets=batch.return_targets.flatten().to(
                    device=chosen_values.device,
                    dtype=chosen_values.dtype,
                ),
                predicted_values=chosen_values.flatten(),
            ).to(device=new_log_probs.device, dtype=new_log_probs.dtype)
            policy = self._policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch.old_log_probs.flatten().to(
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                ),
                action_log_probs=action_log_probs,
                advantages=advantages,
            )
            total = total + policy["loss"] - self.entropy_weight * policy["entropy"]

        metrics = {
            "wm_mse": float(wm_loss.detach().item()),
            "sigreg_loss": (
                float(sigreg_loss.detach().item()) if sigreg_loss is not None else 0.0
            ),
            "value_loss": float(value_loss.detach().item()),
            "total_loss": float(total.detach().item()),
            "actor_loss": float(policy["loss"].detach().item()) if policy else 0.0,
        }
        if policy is not None:
            metrics.update(
                {
                    "entropy": float(policy["entropy"].detach().item()),
                    "mean_advantage": float(policy["advantages"].mean().item()),
                    "clip_fraction": float(policy["clip_fraction"].item()),
                    "mean_ratio": float(policy["probability_ratio"].mean().item()),
                }
            )
        return RLStepOutput(
            loss=total,
            losses={
                "wm": wm_loss,
                "sigreg": sigreg_loss,
                "value": value_loss,
                "policy": policy["loss"] if policy else None,
            },
            metrics=metrics,
        )

    def _value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        probabilities = action_log_probs.exp()
        entropy_terms = torch.where(
            probabilities > 0,
            probabilities * action_log_probs,
            torch.zeros_like(action_log_probs),
        )
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
            "entropy": -entropy_terms.sum(dim=-1).mean(),
            "advantages": advantages,
            "probability_ratio": probability_ratio,
            "clip_fraction": clip_fraction,
        }


__all__ = [
    "RLAlgorithm",
    "RLBatch",
    "RLStepOutput",
    "build_rl_batch",
    "normalized_monte_carlo_advantages",
]
