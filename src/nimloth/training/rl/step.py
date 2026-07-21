"""RL 阶段单次 optimizer step 的 batch、loss 与参数更新。"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA
from nimloth.config.rl import RLConfig
from nimloth.rollout.encoding import EncodedRolloutTransition
from nimloth.training.rl.actor import compute_current_policy_log_probs
from nimloth.training.rl.loss import (
    compute_action_entropy_from_log_probs,
    compute_actor_loss,
    compute_predictor_loss,
    compute_value_loss,
)


@dataclass(frozen=True)
class RLStepResult:
    metrics: dict[str, float]


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def select_transition_batch(
    transitions: list[EncodedRolloutTransition],
    *,
    batch_size: int,
    seed: int,
) -> list[EncodedRolloutTransition]:
    """用独立 CPU generator 选择跨 rank 一致的 transition batch。"""

    if len(transitions) < batch_size:
        raise ValueError(
            f"only {len(transitions)} transitions are available, need {batch_size}"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(transitions), generator=generator)[:batch_size]
    return [transitions[int(index)] for index in indices]


def run_rl_optimizer_step(
    *,
    transitions: list[EncodedRolloutTransition],
    batch_size: int,
    batch_seed: int,
    model: torch.nn.Module,
    processor,
    token_id_map: dict[str, int],
    state_proj: torch.nn.Module,
    wm_predictor: torch.nn.Module,
    value_head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    vision_ema: VisionEncoderEMA | None,
    config: RLConfig,
    actor_enabled: bool,
    device: torch.device,
) -> RLStepResult:
    """执行 predictor、value 与可选 PPO actor 的一次联合更新。"""

    batch = select_transition_batch(
        transitions,
        batch_size=batch_size,
        seed=batch_seed,
    )
    hidden_current = torch.stack(
        [transition.qwen_hidden_current for transition in batch]
    ).to(device)
    hidden_next = torch.stack(
        [transition.qwen_hidden_next for transition in batch]
    ).to(device)
    actions = torch.tensor(
        [transition.action_index for transition in batch],
        dtype=torch.long,
        device=device,
    )
    value_targets = torch.tensor(
        [transition.value_target for transition in batch],
        dtype=torch.float32,
        device=device,
    )

    predictor_loss, predictor_metrics = compute_predictor_loss(
        qwen_hidden_current=hidden_current,
        qwen_hidden_next=hidden_next,
        action_indices=actions,
        state_proj=state_proj,
        wm_predictor=wm_predictor,
    )

    # 保留现有梯度 ownership：StateProjector 只接收 dynamics loss，value
    # supervision 只更新 ValueHead。若要改变该语义，应新增显式配置与梯度测试。
    value_state = _unwrap(state_proj)(hidden_current).float().detach()
    value_loss, value_metrics = compute_value_loss(
        state_emb=value_state,
        action_indices=actions,
        action_value_targets=value_targets,
        value_head=value_head,
        rank_margin=config.value_head.rank_margin,
        lambda_rank=config.value_head.lambda_rank,
    )

    actor_metrics: dict[str, float] = {}
    if actor_enabled:
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            all_values = value_head(value_state).float()
            chosen_values = all_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        advantages = value_targets.to(
            device=chosen_values.device,
            dtype=chosen_values.dtype,
        ) - chosen_values.detach()
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        new_log_probs, action_log_probs = compute_current_policy_log_probs(
            batch,
            model,
            processor,
            token_id_map,
            device,
        )
        old_log_probs = torch.tensor(
            [transition.old_log_prob for transition in batch],
            device=new_log_probs.device,
            dtype=new_log_probs.dtype,
        )
        actor_loss, actor_metrics = compute_actor_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages.to(
                device=new_log_probs.device,
                dtype=new_log_probs.dtype,
            ),
            clip_ratio=config.actor.clip_ratio,
        )
        entropy = compute_action_entropy_from_log_probs(action_log_probs)
        total_loss = (
            predictor_loss
            + value_loss
            + actor_loss
            - config.actor.entropy_coeff * entropy
        )
        actor_metrics["entropy"] = float(entropy.detach().item())
        actor_metrics["mean_advantage"] = float(advantages.mean().item())
    else:
        total_loss = predictor_loss + value_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ],
        1.0,
    )
    optimizer.step()
    if vision_ema is not None:
        vision_ema.update(model)

    metrics = {
        "wm_mse": float(predictor_metrics.get("wm_mse", 0.0)),
        "value_loss": float(
            value_metrics.get(
                "value_loss",
                value_metrics.get("value_total", 0.0),
            )
        ),
        "total_loss": float(total_loss.detach().item()),
        "actor_loss": float(actor_metrics.get("actor_loss", 0.0)),
    }
    metrics.update(
        {key: value for key, value in actor_metrics.items() if key != "actor_loss"}
    )
    return RLStepResult(metrics=metrics)
