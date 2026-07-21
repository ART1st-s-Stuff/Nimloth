"""SFT2 单个 batch 的完整算法。

核心路径：

1. Qwen 编码当前 prefix 得到 ``h_t``，训练时同时产生 CE loss。
2. Qwen 以无梯度 target forward 编码下一 prefix 得到 ``h_{t+1}``。
3. ``StateProjector`` 同时投影两侧；SFT2 保留两侧 projector 梯度。
4. ``WMPredictor(s_t, a_t)`` 拟合 ``s_{t+1}``，ValueHead 拟合 MC return。
5. 训练循环只负责按步调度权重、反向传播和 optimizer 生命周期。

Qwen cache、prompt 去重和 Vision EMA 由 ``QwenTransitionEncoder`` 负责；
公共 dynamics/value 数学由 ``nimloth.wm.objectives`` 负责。
"""

from __future__ import annotations

import contextlib
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

import torch

from nimloth.backbone.qwen25vl.transition import QwenTransitionEncoder
from nimloth.training.sft2.data.batch import SFT2Batch, prepare_sft2_batch
from nimloth.training.sft2.utils import preserve_module_modes, unwrap_module
from nimloth.wm import SIGReg
from nimloth.wm.objectives import (
    ActionValueLoss,
    DynamicsLoss,
    compute_action_value_loss,
    compute_dynamics_loss,
)


class SFT2Mode(Enum):
    """显式区分训练目标和验证目标，避免一个布尔值承载多项策略。"""

    TRAIN = "train"
    VALIDATE = "validate"


@dataclass(frozen=True)
class SFT2Losses:
    """一个 batch 的未加权目标及可记录指标。"""

    lm: torch.Tensor | None
    dynamics: torch.Tensor | None
    sigreg: torch.Tensor | None
    value: torch.Tensor
    metrics: dict[str, float]


@dataclass(frozen=True)
class WeightedSFT2Loss:
    """按当前训练步加权后的总目标。"""

    loss: torch.Tensor
    wm_weight: float
    metrics: dict[str, float]


def build_trajectory_sigreg_inputs(
    trajectory_steps: Sequence[tuple[str, int]],
    current_states: torch.Tensor,
    next_states: torch.Tensor,
) -> list[torch.Tensor]:
    """把同一 record 的 transition 还原为 SIGReg 所需的 ``(T, 1, D)``。"""

    if not trajectory_steps:
        return []
    groups: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    all_record_ids_empty = True
    for (record_id, step_index), current, target in zip(
        trajectory_steps,
        current_states,
        next_states,
        strict=True,
    ):
        if record_id:
            all_record_ids_empty = False
        groups[record_id].append((step_index, current, target))

    # 旧 cache 缺少 record_id 时不能把整个 batch 误当作同一条 trajectory。
    if all_record_ids_empty and len(groups) <= 1:
        return []

    inputs: list[torch.Tensor] = []
    for entries in groups.values():
        entries.sort(key=lambda entry: entry[0])
        sequence = [entries[0][1], *(entry[2] for entry in entries)]
        inputs.append(torch.stack(sequence, dim=0).unsqueeze(1))
    return inputs


def compute_sft2_dynamics(
    *,
    current_hidden: torch.Tensor,
    next_hidden: torch.Tensor,
    action_indices: torch.Tensor,
    trajectory_steps: Sequence[tuple[str, int]] | None,
    state_proj: torch.nn.Module,
    wm_predictor: torch.nn.Module,
    sigreg: SIGReg | None,
) -> tuple[DynamicsLoss, torch.Tensor | None]:
    """执行 SFT2 特有的双侧 projector 梯度和 trajectory SIGReg。"""

    # 合并 forward 可避免 SafeBatchNorm1d 在同一 autograd 图内两次修改 running buffer。
    projected = state_proj(torch.cat([current_hidden, next_hidden], dim=0)).float()
    batch_size = current_hidden.shape[0]
    current_state = projected[:batch_size]
    target_next_state = projected[batch_size:]
    dynamics = compute_dynamics_loss(
        current_state=current_state,
        target_next_state=target_next_state,
        action_indices=action_indices,
        predictor=wm_predictor,
    )

    sigreg_loss: torch.Tensor | None = None
    if sigreg is not None:
        inputs = (
            build_trajectory_sigreg_inputs(
                trajectory_steps,
                current_state,
                target_next_state,
            )
            if trajectory_steps is not None
            else []
        )
        if not inputs:
            inputs = [torch.stack([current_state, target_next_state], dim=0)]
        sigreg_loss = torch.stack([sigreg(value) for value in inputs]).mean()
    return dynamics, sigreg_loss


def wm_loss_weight_schedule(
    global_step: int,
    total_steps: int,
    *,
    start: float = 0.1,
    end: float = 1.0,
    warmup_fraction: float = 0.3,
) -> float:
    """在训练前段用 cosine ramp 增加 dynamics loss 权重。"""

    if total_steps <= 0:
        return end
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if global_step >= warmup_steps:
        return end
    progress = global_step / warmup_steps
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


def combine_sft2_losses(
    losses: SFT2Losses,
    *,
    wm_weight: float,
    sigreg_weight: float,
    value_weight: float,
    ce_weight: float,
) -> WeightedSFT2Loss:
    """显式组合 SFT2 的四个训练目标。"""

    tensors = (losses.lm, losses.dynamics, losses.value, losses.sigreg)
    device = next(value.device for value in tensors if value is not None)
    total = torch.zeros((), device=device)
    metrics = dict(losses.metrics)
    metrics.update(
        {
            "lambda_wm": float(wm_weight),
            "lambda_sigreg": float(sigreg_weight),
            "lambda_value": float(value_weight),
            "lambda_ce": float(ce_weight),
        }
    )
    if losses.dynamics is not None:
        total = total + wm_weight * losses.dynamics.to(device)
    if losses.sigreg is not None and sigreg_weight > 0.0:
        total = total + sigreg_weight * losses.sigreg.to(device)
    total = total + value_weight * losses.value.to(device)
    if losses.lm is not None:
        total = total + ce_weight * losses.lm.to(device)
    metrics["total_loss"] = float(total.detach().item())
    return WeightedSFT2Loss(loss=total, wm_weight=wm_weight, metrics=metrics)


@dataclass(frozen=True)
class SFT2Algorithm:
    """SFT2 train/validation 共用的单 batch 算法入口。"""

    qwen: QwenTransitionEncoder
    state_proj: torch.nn.Module
    wm_predictor: torch.nn.Module
    value_head: torch.nn.Module
    sigreg: SIGReg | None
    value_rank_margin: float = 0.1
    value_rank_weight: float = 1.0

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        return self.qwen.model, self.state_proj, self.wm_predictor, self.value_head

    def unwrapped(self) -> SFT2Algorithm:
        model = unwrap_module(self.qwen.model)
        return replace(
            self,
            qwen=replace(self.qwen, model=model),
            state_proj=unwrap_module(self.state_proj),
            wm_predictor=unwrap_module(self.wm_predictor),
            value_head=unwrap_module(self.value_head),
        )

    @contextlib.contextmanager
    def validation_context(self):
        ema_context = (
            self.qwen.vision_ema.use_ema_weights(self.qwen.model)
            if self.qwen.vision_ema is not None
            else contextlib.nullcontext()
        )
        with preserve_module_modes(self.modules, training=False), ema_context:
            yield

    def compute(self, raw_batch: Any, *, mode: SFT2Mode) -> SFT2Losses:
        """规范化 DataLoader 输出后执行一次完整 SFT2 前向。"""

        batch = prepare_sft2_batch(
            raw_batch,
            self.qwen.processor,
            self.qwen.max_length,
            latent_token_count=self.qwen.latent_token_count,
            mask_latent_query_labels=self.qwen.mask_latent_query_labels,
        )
        current_hidden, lm_loss = self.qwen.encode_current(
            batch.current_encoding,
            include_lm_loss=mode is SFT2Mode.TRAIN,
        )
        dynamics_loss, sigreg_loss, dynamics_metrics = self._compute_dynamics(
            batch,
            current_hidden,
            mode=mode,
        )
        value = self._compute_value(batch, current_hidden, mode=mode)
        metrics = {
            **dynamics_metrics,
            "value_reg": float(value.regression.detach().item()),
            "value_rank": float(value.ranking.detach().item()),
            "value_total": float(value.loss.detach().item()),
        }
        if lm_loss is not None:
            metrics["lm_ce"] = float(lm_loss.detach().item())
        return SFT2Losses(
            lm=lm_loss,
            dynamics=dynamics_loss,
            sigreg=sigreg_loss,
            value=value.loss,
            metrics=metrics,
        )

    def _compute_dynamics(
        self,
        batch: SFT2Batch,
        current_hidden: torch.Tensor,
        *,
        mode: SFT2Mode,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, float]]:
        indices = [
            index
            for index, transition in enumerate(batch.transitions)
            if transition.qwen.next is not None
        ]
        if not indices:
            return self._ddp_aligned_zero_dynamics(batch, current_hidden, mode=mode)

        next_hidden = self.qwen.encode_next(
            [transition.qwen for transition in batch.transitions],
            indices,
            cached=batch.cached_next,
            use_vision_ema=mode is SFT2Mode.TRAIN,
        )
        actions = torch.tensor(
            [batch.transitions[index].action_index for index in indices],
            dtype=torch.long,
            device=current_hidden.device,
        )
        trajectory_steps = [
            batch.transitions[index].trajectory_step
            for index in indices
        ]
        dynamics, sigreg_loss = compute_sft2_dynamics(
            current_hidden=current_hidden[indices],
            next_hidden=next_hidden,
            action_indices=actions,
            trajectory_steps=trajectory_steps,
            state_proj=self.state_proj,
            wm_predictor=self.wm_predictor,
            sigreg=self.sigreg,
        )
        metrics = {"wm_mse": float(dynamics.loss.detach().item())}
        if sigreg_loss is not None:
            metrics["sigreg_loss"] = float(sigreg_loss.detach().item())
        return dynamics.loss, sigreg_loss, metrics

    def _ddp_aligned_zero_dynamics(
        self,
        batch: SFT2Batch,
        current_hidden: torch.Tensor,
        *,
        mode: SFT2Mode,
    ) -> tuple[torch.Tensor, None, dict[str, float]]:
        """terminal-only rank 调用相同模块，以免其他 rank 等待 gradient all-reduce。"""

        dummy_target = self.qwen.encode_ddp_dummy_target(
            batch.transitions[0].qwen.current,
            use_vision_ema=mode is SFT2Mode.TRAIN,
        )
        current_state = self.state_proj(current_hidden[:1])
        with torch.no_grad():
            target_state = self.state_proj(dummy_target[:1])
        actions = torch.zeros(
            (current_state.shape[0],),
            dtype=torch.long,
            device=current_state.device,
        )
        prediction = self.wm_predictor(current_state, actions)
        zero = (current_state.sum() + target_state.sum() + prediction.sum()) * 0.0
        return zero, None, {}

    def _compute_value(
        self,
        batch: SFT2Batch,
        current_hidden: torch.Tensor,
        *,
        mode: SFT2Mode,
    ) -> ActionValueLoss:
        actions = torch.tensor(
            [transition.action_index for transition in batch.transitions],
            dtype=torch.long,
            device=current_hidden.device,
        )
        targets = torch.tensor(
            [transition.value_target for transition in batch.transitions],
            dtype=torch.float32,
            device=current_hidden.device,
        )
        state = self.state_proj(current_hidden)
        return compute_action_value_loss(
            state=state,
            action_indices=actions,
            return_targets=targets,
            value_head=self.value_head,
            rank_margin=self.value_rank_margin if mode is SFT2Mode.TRAIN else 0.0,
            rank_weight=self.value_rank_weight if mode is SFT2Mode.TRAIN else 0.0,
        )

__all__ = [
    "SFT2Algorithm",
    "SFT2Losses",
    "SFT2Mode",
    "WeightedSFT2Loss",
    "build_trajectory_sigreg_inputs",
    "combine_sft2_losses",
    "compute_sft2_dynamics",
    "wm_loss_weight_schedule",
]
