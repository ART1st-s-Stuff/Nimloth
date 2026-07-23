"""SFT2 的完整单批算法：模型前向、目标函数和权重策略。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from nimloth.training.sft2.batch import SFT2Batch
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.wm import (
    LatentWMPredictor,
    SequenceSIGReg,
)


def require_sft2_wm_history(
    wm_predictor: LatentWMPredictor,
    *,
    history_size: int,
    source: Path,
) -> None:
    """拒绝加载与当前 SFT2 LeWM 上下文长度不一致的 predictor。"""

    actual = int(wm_predictor.config.history_size)
    expected = int(history_size)
    if actual != expected:
        raise ValueError(
            "SFT2 WM checkpoint history_size does not match config: "
            f"checkpoint={actual}, config={expected}, source={source}"
        )


@dataclass(frozen=True)
class SFT2StepOutput:
    """一次 SFT2 主前向产生的 loss、当前 state 和日志指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    current_state: torch.Tensor
    sample_count: int


@dataclass(frozen=True)
class SFT2SIGRegStepOutput:
    """主 loss 反传完成后，单独执行的 SIGReg 阶段结果。"""

    loss: torch.Tensor
    raw_loss: torch.Tensor | None
    metrics: dict[str, float]


class SFT2Algorithm:
    """定义 SFT2 一个 batch 的目标函数与两阶段计算图。

    主阶段只计算当前 step 的 CE/WM/value；它反传并释放 Qwen 图后，SIGReg 阶段
    才以 detached ``s_t`` 和在线 ``s_{t+1}`` 计算正则。这样 SIGReg 数值上仍看见
    两个连续状态，但梯度只进入新状态侧，也不会同时保留两份 Qwen 激活。
    """

    def __init__(
        self,
        *,
        history_size: int,
        sigreg: SequenceSIGReg | None,
        sigreg_weight: float,
        value_weight: float,
        ce_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
        wm_weight_start: float = 0.1,
        wm_weight_end: float = 1.0,
        wm_warmup_fraction: float = 0.3,
    ) -> None:
        self.history_size = int(history_size)
        if self.history_size < 1:
            raise ValueError(
                f"history_size must be positive, got {self.history_size}"
            )
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_weight = float(value_weight)
        self.ce_weight = float(ce_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)
        self.wm_weight_start = float(wm_weight_start)
        self.wm_weight_end = float(wm_weight_end)
        self.wm_warmup_fraction = float(wm_warmup_fraction)

    def wm_weight(self, global_step: int, total_steps: int) -> float:
        """在训练前段用 cosine ramp 增加 WM loss 权重。"""

        if total_steps <= 0:
            return self.wm_weight_end
        warmup_steps = max(1, int(total_steps * self.wm_warmup_fraction))
        if global_step >= warmup_steps:
            return self.wm_weight_end
        progress = global_step / warmup_steps
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.wm_weight_start + (
            self.wm_weight_end - self.wm_weight_start
        ) * cosine

    @property
    def has_sigreg_stage(self) -> bool:
        """训练是否需要在主 loss 反传后执行独立 SIGReg 阶段。"""

        return self.sigreg is not None and self.sigreg_weight > 0.0

    def training_primary_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        wm_weight: float,
    ) -> SFT2StepOutput:
        return self._step(
            runtime,
            batch,
            wm_weight=wm_weight,
            include_lm_loss=True,
            include_value_ranking=True,
        )

    def training_sigreg_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        detached_current_state: torch.Tensor,
    ) -> SFT2SIGRegStepOutput:
        """只让在线 ``s_{t+1}`` 接收 SIGReg 梯度。

        调用者必须先完成主 loss backward，再调用本方法。小于两个样本的 rank
        无法估计 SIGReg 分布，但仍用依赖在线 state 的零 loss 参与 DDP backward。
        """

        if not self.has_sigreg_stage:
            raise RuntimeError("SFT2 SIGReg stage is disabled")
        if detached_current_state.requires_grad:
            raise ValueError("SFT2 SIGReg current_state must be detached")

        next_state = runtime.agent.encode_state(
            batch.online_tail,
            include_lm_loss=False,
        ).state
        sigreg_loss = self._sigreg_loss(detached_current_state, next_state)
        if sigreg_loss is None:
            # 保留在线 Qwen/StateProjector 的 DDP 图参与，但不伪造 B<2 的统计量。
            backward_loss = next_state.sum() * 0.0
            metrics = {"sigreg_skipped_small_batch": 1.0}
        else:
            backward_loss = self.sigreg_weight * sigreg_loss
            metrics = {"sigreg_loss": float(sigreg_loss.detach().item())}
        if batch.is_padding:
            backward_loss = backward_loss * 0.0
        return SFT2SIGRegStepOutput(
            loss=backward_loss,
            raw_loss=sigreg_loss,
            metrics=metrics,
        )

    def merge_training_metrics(
        self,
        primary_metrics: dict[str, float],
        sigreg: SFT2SIGRegStepOutput | None,
    ) -> dict[str, float]:
        """把两个显式反传阶段合并为一个 optimizer-step 日志视图。"""

        metrics = dict(primary_metrics)
        metrics["lambda_sigreg"] = self.sigreg_weight if sigreg is not None else 0.0
        if sigreg is None:
            return metrics
        metrics.update(sigreg.metrics)
        metrics["total_loss"] += float(sigreg.loss.detach().item())
        return metrics

    def evaluation_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
    ) -> SFT2StepOutput:
        return self._step(
            runtime,
            batch,
            wm_weight=1.0,
            include_lm_loss=False,
            include_value_ranking=False,
        )

    def _step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        wm_weight: float,
        include_lm_loss: bool,
        include_value_ranking: bool,
    ) -> SFT2StepOutput:
        """按照 current forward → next target → CE/WM/value 完成主阶段。"""

        if not 1 <= batch.history_size <= self.history_size:
            raise ValueError(
                "SFT2 batch context length exceeds algorithm history_size: "
                f"batch={batch.history_size}, algorithm={self.history_size}"
            )
        # 当前 Qwen 只执行一次；更老 state 来自它们先前 current forward 的 cache。
        current_encoded = runtime.agent.encode_state(
            batch.current,
            include_lm_loss=include_lm_loss,
        )
        cached_history = runtime.history_cache.history(
            batch.history_keys,
            reference=current_encoded.state,
        )
        model_output = runtime.agent.forward_step_from_history(
            batch.action_indices,
            cached_history,
            encoded_current=current_encoded,
        )
        runtime.history_cache.store(
            batch.current_keys,
            model_output.state[:, -1],
            enabled=not batch.is_padding,
        )
        target_states = runtime.target_state(batch.next)
        aligned_targets = target_states[batch.current_next_indices]

        wm_loss = F.mse_loss(
            model_output.predicted_next_state,
            aligned_targets,
        )
        value = self._value_loss(
            model_output.action_values,
            batch.current_action_indices,
            batch.current_value_targets,
            include_ranking=include_value_ranking,
        )
        total = wm_weight * wm_loss + self.value_weight * value["loss"]
        if model_output.lm_loss is not None:
            total = total + self.ce_weight * model_output.lm_loss
        sample_count = 0 if batch.is_padding else batch.batch_size
        if batch.is_padding:
            total = total * 0.0

        metrics = {
            "value_reg": float(value["regression"].detach().item()),
            "value_rank": float(value["ranking"].detach().item()),
            "value_total": float(value["loss"].detach().item()),
            "lambda_wm": float(wm_weight),
            "lambda_sigreg": 0.0,
            "lambda_value": self.value_weight,
            "lambda_ce": self.ce_weight,
            "context_length": float(batch.history_size),
            "current_batch_size": float(sample_count),
            "history_cache_entries": float(runtime.history_cache.count),
            "total_loss": float(total.detach().item()),
        }
        metrics["wm_mse"] = float(wm_loss.detach().item())
        if model_output.lm_loss is not None:
            metrics["lm_ce"] = float(model_output.lm_loss.detach().item())

        return SFT2StepOutput(
            loss=total,
            losses={
                "lm": model_output.lm_loss,
                "wm": wm_loss,
                "value": value["loss"],
            },
            metrics=metrics,
            current_state=model_output.state[:, -1],
            sample_count=sample_count,
        )

    def _value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
        *,
        include_ranking: bool,
    ) -> dict[str, torch.Tensor]:
        """Value head loss"""

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
        max_other = all_values.masked_fill(
            chosen_mask,
            float("-inf"),
        ).max(dim=-1).values
        ranking = F.relu(
            self.value_rank_margin + max_other - chosen_values
        ).mean()
        rank_weight = self.value_rank_weight if include_ranking else 0.0
        return {
            "loss": regression + rank_weight * ranking,
            "regression": regression,
            "ranking": ranking,
            "chosen_values": chosen_values,
        }

    def _sigreg_loss(
        self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor | None:
        """为当前 transition 的在线 ``(s_t,s_{t+1})`` 计算一次 SIGReg。"""

        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        if current_state.ndim != 2 or next_state.ndim != 2:
            raise ValueError(
                "SFT2 SIGReg expects current_state/next_state=(B,D), "
                f"got {tuple(current_state.shape)} and {tuple(next_state.shape)}"
            )
        if current_state.shape != next_state.shape:
            raise ValueError("SFT2 SIGReg state batch sizes do not match")
        return self.sigreg(torch.stack((current_state, next_state), dim=1))

__all__ = [
    "SFT2Algorithm",
    "SFT2SIGRegStepOutput",
    "SFT2StepOutput",
    "require_sft2_wm_history",
]
