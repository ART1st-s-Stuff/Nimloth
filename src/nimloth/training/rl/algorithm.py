"""RL 的连续序列采样、模型前向、loss、backward 与 optimizer step。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from nimloth.agent import ActionLogProbReplay, Agent
from nimloth.backbone import BackboneEMA
from nimloth.rollout import EncodedTrajectory, EncodedTransition
from nimloth.wm import SequenceSIGReg


@dataclass(frozen=True)
class RLBatch:
    """一次 RL 更新消费的连续 WM window 与逐步 RL 监督。"""

    windows: tuple[tuple[EncodedTransition, ...], ...]
    hidden_states: torch.Tensor
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor

    @property
    def transitions(self) -> tuple[EncodedTransition, ...]:
        """按 batch/time 顺序展开，供 policy replay 使用。"""

        return tuple(transition for window in self.windows for transition in window)


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


def count_sequence_windows(
    trajectories: Sequence[EncodedTrajectory],
    *,
    history_size: int,
) -> int:
    """统计不会跨越 trajectory 边界的固定长度训练窗口。"""

    if history_size < 1:
        raise ValueError(f"history_size must be positive, got {history_size}")
    return sum(
        max(0, trajectory.num_steps - history_size + 1)
        for trajectory in trajectories
    )


def select_sequence_batch(
    trajectories: Sequence[EncodedTrajectory],
    *,
    history_size: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> RLBatch:
    """使用独立 CPU generator 采样同一 trajectory 内的连续窗口。"""

    if history_size < 1:
        raise ValueError(f"history_size must be positive, got {history_size}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    candidates = [
        (trajectory, start)
        for trajectory in trajectories
        for start in range(trajectory.num_steps - history_size + 1)
    ]
    if len(candidates) < batch_size:
        raise ValueError(
            f"only {len(candidates)} sequence windows are available, need {batch_size}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(candidates), generator=generator)[:batch_size]
    windows = tuple(
        tuple(
            candidates[int(index)][0].transitions[
                candidates[int(index)][1] : candidates[int(index)][1] + history_size
            ]
        )
        for index in indices
    )

    # 每个 H-step window 由 s0 和 H 个 transition 的 next state 组成 H+1 个状态。
    hidden_states = torch.stack(
        [
            torch.stack(
                [window[0].current_hidden]
                + [transition.next_hidden for transition in window]
            )
            for window in windows
        ]
    ).to(device)
    return RLBatch(
        windows=windows,
        hidden_states=hidden_states,
        action_indices=torch.tensor(
            [
                [transition.action_index for transition in window]
                for window in windows
            ],
            dtype=torch.long,
            device=device,
        ),
        return_targets=torch.tensor(
            [
                [transition.value_target for transition in window]
                for window in windows
            ],
            dtype=torch.float32,
            device=device,
        ),
        old_log_probs=torch.tensor(
            [
                [transition.old_log_prob for transition in window]
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
    """定义一次 RL optimizer update 的完整执行顺序。

    本类保留 RL 特有的梯度边界：WM current 更新 StateProjector/WMPredictor，
    next state 是固定 target，value 输入 detach。rollout 生命周期和 checkpoint
    不属于单批算法，继续由 loop 管理。
    """

    def __init__(
        self,
        *,
        agent: Agent,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        vision_ema: BackboneEMA | None,
        policy_replay: ActionLogProbReplay | None,
        history_size: int,
        sigreg: SequenceSIGReg | None,
        sigreg_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
        ppo_clip_ratio: float,
        entropy_weight: float,
    ) -> None:
        self.agent = agent
        self.optimizer = optimizer
        self.device = device
        self.vision_ema = vision_ema
        self.policy_replay = policy_replay
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

    def update(
        self,
        trajectories: Sequence[EncodedTrajectory],
        *,
        batch_size: int,
        batch_seed: int,
    ) -> dict[str, float]:
        """采样一个 minibatch，完成 backward、裁剪、optimizer 与 EMA。"""

        batch = select_sequence_batch(
            trajectories,
            history_size=self.history_size,
            batch_size=batch_size,
            seed=batch_seed,
            device=self.device,
        )
        if self.policy_replay is not None:
            torch.cuda.empty_cache()
            gc.collect()

        output = self.training_step(batch)
        self.optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            ],
            1.0,
        )
        self.optimizer.step()
        if self.vision_ema is not None:
            self.vision_ema.update(self.agent.backbone.model)
        return output.metrics

    def training_step(self, batch: RLBatch) -> RLStepOutput:
        """构造 RL 计算图并计算 WM、value 与可选 PPO 目标。"""

        expected_state_steps = self.history_size + 1
        if batch.hidden_states.ndim != 3 or batch.hidden_states.shape[1] != expected_state_steps:
            raise ValueError(
                "RLBatch hidden_states must have shape (B, history_size + 1, D), "
                f"got {tuple(batch.hidden_states.shape)} for history_size={self.history_size}"
            )
        expected_actions = (
            batch.hidden_states.shape[0],
            self.history_size,
        )
        if batch.action_indices.shape != expected_actions:
            raise ValueError(
                "RLBatch action_indices must have shape (B, history_size), "
                f"got {tuple(batch.action_indices.shape)}, expected {expected_actions}"
            )

        state_sequence = self.agent.wm.project_state(batch.hidden_states)
        state_context = state_sequence[:, :-1]
        predicted_next_states = self.agent.wm.predict_state_sequence(
            state_context,
            batch.action_indices,
        )

        # 一个状态可以在当前位置更新 projector，同时在前一位置作为 stop-gradient
        # target；因此先统一投影，再只 detach 右移后的 target 视图。
        target_next_states = state_sequence[:, 1:].detach()
        action_values = self.agent.wm.predict_action_values(state_context.detach())

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
        if self.policy_replay is not None:
            new_log_probs, action_log_probs = self.policy_replay(batch.transitions)
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
    "count_sequence_windows",
    "normalized_monte_carlo_advantages",
    "select_sequence_batch",
]
