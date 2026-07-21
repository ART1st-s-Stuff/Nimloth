"""RL 单个 transition batch 的完整更新算法。

当前梯度契约（本次重构保持原行为）：

- rollout latent 在采集后以 ``no_grad`` 编码，因此 dynamics/value 不更新 Qwen；
- dynamics 只通过当前状态更新 StateProjector，下一状态 projector 是 target；
- value 输入显式 detach，因此 value supervision 只更新 ValueHead；
- actor 开启时，用保存的完整 prompt 和采样参数重放 Qwen，并计算 PPO clipped loss。

公共 dynamics/value 数学由完整 ``NimlothModel.wm`` 模块负责；Qwen prompt
replay 位于 ``nimloth.backbone.qwen25vl.policy``。
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from nimloth.backbone.qwen25vl.policy import (
    categorical_entropy_from_log_probs,
    replay_rollout_action_log_probs,
)
from nimloth.backbone.qwen25vl.rollout import EncodedRolloutTransition
from nimloth.config.rl import RLConfig
from nimloth.training.rl.components import RLComponents
from nimloth.wm.model import ActionValueLoss, DynamicsLoss


@dataclass(frozen=True)
class RLBatch:
    """RL objective 直接消费的 latent、return 和 policy provenance。"""

    transitions: tuple[EncodedRolloutTransition, ...]
    current_hidden: torch.Tensor
    next_hidden: torch.Tensor
    action_indices: torch.Tensor
    return_targets: torch.Tensor
    old_log_probs: torch.Tensor


@dataclass(frozen=True)
class PolicyLoss:
    """PPO actor 目标及其诊断量。"""

    loss: torch.Tensor
    entropy: torch.Tensor
    advantages: torch.Tensor
    probability_ratio: torch.Tensor
    clip_fraction: torch.Tensor


@dataclass(frozen=True)
class RLLosses:
    """一次 RL 更新中的三类目标。"""

    dynamics: DynamicsLoss
    value: ActionValueLoss
    policy: PolicyLoss | None
    total: torch.Tensor


def select_transition_batch(
    transitions: list[EncodedRolloutTransition],
    *,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> RLBatch:
    """用独立 CPU generator 选择跨 rank 一致的 transition batch。"""

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
            [transition.qwen_hidden_current for transition in selected]
        ).to(device),
        next_hidden=torch.stack(
            [transition.qwen_hidden_next for transition in selected]
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
    """计算并标准化 ``G_t - V(s_t, a_t)``；返回值不携带 value 梯度。"""

    advantages = return_targets - predicted_values.detach()
    return (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )


@dataclass(frozen=True)
class RLAlgorithm:
    """把 latent dynamics、value 与可选 PPO actor 组合成一次 optimizer 更新。"""

    components: RLComponents
    config: RLConfig
    actor_enabled: bool
    device: torch.device

    def compute_losses(self, batch: RLBatch) -> RLLosses:
        """只构造 autograd 图；optimizer 生命周期由 ``update`` 统一执行。"""

        model = self.components.nimloth_model
        wm = model.wm
        current_state = wm.project_state(batch.current_hidden)
        with torch.no_grad():
            target_next_state = wm.project_state(batch.next_hidden)
        dynamics = wm.compute_dynamics_loss(
            current_state=current_state,
            target_next_state=target_next_state,
            action_indices=batch.action_indices,
        )

        # 保留既有 ownership：value 只更新 ValueHead，不更新 StateProjector。
        value_state = self._unwrapped(wm.state_proj)(batch.current_hidden).float().detach()
        value = wm.compute_action_value_loss(
            state=value_state,
            action_indices=batch.action_indices,
            return_targets=batch.return_targets,
            rank_margin=self.config.value_head.rank_margin,
            rank_weight=self.config.value_head.lambda_rank,
        )

        policy: PolicyLoss | None = None
        total = dynamics.loss + value.loss
        if self.actor_enabled:
            advantages = normalized_monte_carlo_advantages(
                return_targets=batch.return_targets.to(
                    device=value.chosen_values.device,
                    dtype=value.chosen_values.dtype,
                ),
                predicted_values=value.chosen_values,
            )
            new_log_probs, action_log_probs = replay_rollout_action_log_probs(
                transitions=batch.transitions,
                model=model.llm,
                processor=self.components.processor,
                token_id_map=self.components.token_id_map,
                device=self.device,
            )
            policy = self._compute_policy_loss(
                new_log_probs=new_log_probs,
                old_log_probs=batch.old_log_probs.to(
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                ),
                action_log_probs=action_log_probs,
                advantages=advantages.to(
                    device=new_log_probs.device,
                    dtype=new_log_probs.dtype,
                ),
            )
            total = (
                total
                + policy.loss
                - self.config.actor.entropy_coeff * policy.entropy
            )
        return RLLosses(
            dynamics=dynamics,
            value=value,
            policy=policy,
            total=total,
        )

    def _compute_policy_loss(
        self,
        *,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> PolicyLoss:
        """计算离散动作的 PPO clipped surrogate 和采样后分布 entropy。"""

        probability_ratio = torch.exp(new_log_probs - old_log_probs)
        clip_ratio = self.config.actor.clip_ratio
        clipped_ratio = torch.clamp(
            probability_ratio,
            1.0 - clip_ratio,
            1.0 + clip_ratio,
        )
        loss = -torch.min(
            probability_ratio * advantages,
            clipped_ratio * advantages,
        ).mean()
        with torch.no_grad():
            clip_fraction = (
                (probability_ratio - 1.0).abs().gt(clip_ratio).float().mean()
            )
        entropy = categorical_entropy_from_log_probs(action_log_probs)
        return PolicyLoss(
            loss=loss,
            entropy=entropy,
            advantages=advantages,
            probability_ratio=probability_ratio,
            clip_fraction=clip_fraction,
        )

    def update(
        self,
        transitions: list[EncodedRolloutTransition],
        *,
        batch_size: int,
        batch_seed: int,
    ) -> dict[str, float]:
        """选择一个 batch、反向传播并更新所有启用的 RL 组件。"""

        batch = select_transition_batch(
            transitions,
            batch_size=batch_size,
            seed=batch_seed,
            device=self.device,
        )
        if self.actor_enabled:
            torch.cuda.empty_cache()
            gc.collect()
        losses = self.compute_losses(batch)

        optimizer = self.components.optimizer
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ],
            1.0,
        )
        optimizer.step()
        if self.components.vision_ema is not None:
            self.components.vision_ema.update(self.components.nimloth_model.llm)
        return self._metrics(losses)

    @staticmethod
    def _metrics(losses: RLLosses) -> dict[str, float]:
        policy = losses.policy
        metrics = {
            "wm_mse": float(losses.dynamics.loss.detach().item()),
            "value_loss": float(losses.value.loss.detach().item()),
            "total_loss": float(losses.total.detach().item()),
            "actor_loss": float(policy.loss.detach().item()) if policy else 0.0,
        }
        if policy is not None:
            metrics.update(
                {
                    "entropy": float(policy.entropy.detach().item()),
                    "mean_advantage": float(policy.advantages.mean().item()),
                    "clip_fraction": float(policy.clip_fraction.item()),
                    "mean_ratio": float(policy.probability_ratio.mean().item()),
                }
            )
        return metrics

    @staticmethod
    def _unwrapped(module: torch.nn.Module) -> torch.nn.Module:
        return module.module if hasattr(module, "module") else module


__all__ = [
    "RLAlgorithm",
    "RLBatch",
    "RLLosses",
    "normalized_monte_carlo_advantages",
    "select_transition_batch",
]
