"""RL 的完整单批算法：采样、模型前向、loss、backward 与 optimizer step。"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from nimloth.agent import ActionLogProbReplay, Agent
from nimloth.backbone import BackboneEMA
from nimloth.rollout import EncodedTransition


@dataclass(frozen=True)
class RLBatch:
    """一次 RL 更新消费的 hidden、return 与 policy provenance。"""

    transitions: tuple[EncodedTransition, ...]
    current_hidden: torch.Tensor
    next_hidden: torch.Tensor
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor


@dataclass(frozen=True)
class RLStepOutput:
    """一次 RL 前向产生的总 loss、分项张量与指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]


def select_transition_batch(
    transitions: Sequence[EncodedTransition],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> RLBatch:
    """使用独立 CPU generator 做可复现的 transition 子采样。"""

    if len(transitions) < batch_size:
        raise ValueError(
            f"only {len(transitions)} transitions are available, need {batch_size}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(transitions), generator=generator)[:batch_size]
    selected = tuple(transitions[int(index)] for index in indices)
    return RLBatch(
        transitions=selected,
        current_hidden=torch.stack(
            [transition.current_hidden for transition in selected]
        ).to(device),
        next_hidden=torch.stack(
            [transition.next_hidden for transition in selected]
        ).to(device),
        action_indices=torch.tensor(
            [transition.action_index for transition in selected],
            dtype=torch.long,
            device=device,
        ),
        return_targets=torch.tensor(
            [transition.value_target for transition in selected],
            dtype=torch.float32,
            device=device,
        ),
        old_log_probs=torch.tensor(
            [transition.old_log_prob for transition in selected],
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
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.ppo_clip_ratio = float(ppo_clip_ratio)
        self.entropy_weight = float(entropy_weight)

    def update(
        self,
        transitions: Sequence[EncodedTransition],
        *,
        batch_size: int,
        batch_seed: int,
    ) -> dict[str, float]:
        """采样一个 minibatch，完成 backward、裁剪、optimizer 与 EMA。"""

        batch = select_transition_batch(
            transitions,
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

        current_state = self.agent.wm.project_state(batch.current_hidden)
        predicted_next_state = self.agent.wm.predict_next_state(
            current_state,
            batch.action_indices,
        )

        # 下一状态仅提供 WM 监督；value 继续只更新 ValueHead。
        with torch.no_grad():
            target_next_state = self.agent.wm.project_state(batch.next_hidden)
        action_values = self.agent.wm.predict_action_values(current_state.detach())

        new_log_probs: torch.Tensor | None = None
        action_log_probs: torch.Tensor | None = None
        if self.policy_replay is not None:
            new_log_probs, action_log_probs = self.policy_replay(batch.transitions)

        # 三个目标在这里直接组合，避免把同一批 tensor 再转发给一层 objective。
        wm_loss = F.mse_loss(predicted_next_state, target_next_state)
        value_loss, chosen_values = self._value_loss(
            action_values,
            batch.action_indices,
            batch.return_targets,
        )
        total = wm_loss + value_loss

        policy: dict[str, torch.Tensor] | None = None
        if new_log_probs is not None:
            if action_log_probs is None:
                raise ValueError("action_log_probs are required with new_log_probs")
            advantages = normalized_monte_carlo_advantages(
                return_targets=batch.return_targets.to(
                    device=chosen_values.device,
                    dtype=chosen_values.dtype,
                ),
                predicted_values=chosen_values,
            ).to(device=new_log_probs.device, dtype=new_log_probs.dtype)
            policy = self._policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch.old_log_probs.to(
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                ),
                action_log_probs=action_log_probs,
                advantages=advantages,
            )
            total = total + policy["loss"] - self.entropy_weight * policy["entropy"]

        metrics = {
            "wm_mse": float(wm_loss.detach().item()),
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
        chosen_values = all_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        targets = return_targets.to(device=all_values.device, dtype=all_values.dtype)
        regression = F.mse_loss(chosen_values, targets)
        chosen_mask = F.one_hot(
            action_indices,
            num_classes=all_values.shape[1],
        ).bool()
        max_other = all_values.masked_fill(chosen_mask, float("-inf")).max(dim=1).values
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
    "normalized_monte_carlo_advantages",
    "select_transition_batch",
]
