"""SFT2 的完整单批算法：模型前向、目标函数和权重策略。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from nimloth.agent import AgentOutput
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
    """一次 SFT2 前向产生的总 loss、分项 loss 和日志指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    model_output: AgentOutput


class SFT2Algorithm:
    """定义 SFT2 一个 batch 的完整计算图。

    训练循环只负责取 batch、backward 和 optimizer step。本类负责当前状态、目标
    状态、WM/value/SIGReg/CE loss 以及这些 loss 的组合，不依赖 processor、cache、
    DDP、EMA 的具体实现或 checkpoint。
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
        backbone_rows_per_forward: int | None = None,
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
        self.backbone_rows_per_forward = (
            None if backbone_rows_per_forward is None else int(backbone_rows_per_forward)
        )
        if self.backbone_rows_per_forward is not None and self.backbone_rows_per_forward < 1:
            raise ValueError("backbone_rows_per_forward must be positive")

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

    def training_step(
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
            include_sigreg=True,
        )

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
            include_sigreg=False,
        )

    def _step(
        self,
        runtime: SFT2ModelRuntime,
        batch: SFT2Batch,
        *,
        wm_weight: float,
        include_lm_loss: bool,
        include_value_ranking: bool,
        include_sigreg: bool,
    ) -> SFT2StepOutput:
        """按照 current forward → next target → loss 的顺序完成一次计算。"""

        if batch.history_size != self.history_size:
            raise ValueError(
                "SFT2 batch history_size does not match algorithm: "
                f"batch={batch.history_size}, algorithm={self.history_size}"
            )
        model_output = runtime.agent.forward_sequence(
            batch.current,
            batch.action_indices,
            include_lm_loss=include_lm_loss,
            backbone_rows_per_forward=self.backbone_rows_per_forward,
        )
        target_states = runtime.target_state(batch.next)
        aligned_targets = target_states[
            batch.next_indices.flatten()
        ].reshape(
            batch.batch_size,
            self.history_size,
            *target_states.shape[1:],
        )

        wm_loss = F.mse_loss(
            model_output.predicted_next_state,
            aligned_targets,
        )
        value = self._value_loss(
            model_output.action_values,
            batch.action_indices,
            batch.value_targets,
            include_ranking=include_value_ranking,
        )
        sigreg_loss = None
        if (
            include_sigreg
            and self.sigreg is not None
            and self.sigreg_weight > 0.0
        ):
            sigreg_loss = self._sigreg_loss(
                model_output.state,
                runtime.agent.encode_state(
                    batch.online_tail,
                    include_lm_loss=False,
                ).state,
            )

        total = wm_weight * wm_loss + self.value_weight * value["loss"]
        if sigreg_loss is not None and self.sigreg_weight > 0.0:
            total = total + self.sigreg_weight * sigreg_loss
        if model_output.lm_loss is not None:
            total = total + self.ce_weight * model_output.lm_loss

        metrics = {
            "value_reg": float(value["regression"].detach().item()),
            "value_rank": float(value["ranking"].detach().item()),
            "value_total": float(value["loss"].detach().item()),
            "lambda_wm": float(wm_weight),
            "lambda_sigreg": self.sigreg_weight if include_sigreg else 0.0,
            "lambda_value": self.value_weight,
            "lambda_ce": self.ce_weight,
            "total_loss": float(total.detach().item()),
        }
        metrics["wm_mse"] = float(wm_loss.detach().item())
        if sigreg_loss is not None:
            metrics["sigreg_loss"] = float(sigreg_loss.detach().item())
        elif include_sigreg and self.sigreg is not None and self.sigreg_weight > 0.0:
            metrics["sigreg_skipped_small_batch"] = 1.0
        if model_output.lm_loss is not None:
            metrics["lm_ce"] = float(model_output.lm_loss.detach().item())

        return SFT2StepOutput(
            loss=total,
            losses={
                "lm": model_output.lm_loss,
                "wm": wm_loss,
                "sigreg": sigreg_loss,
                "value": value["loss"],
            },
            metrics=metrics,
            model_output=model_output,
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
        current_states: torch.Tensor,
        final_state: torch.Tensor,
    ) -> torch.Tensor | None:
        """用在线 encoder 的 ``H+1`` 个真实状态构造 LeWM SIGReg 输入。"""

        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        if current_states.ndim != 3 or final_state.ndim != 2:
            raise ValueError(
                "SFT2 SIGReg expects current_states=(B,H,D) and final_state=(B,D), "
                f"got {tuple(current_states.shape)} and {tuple(final_state.shape)}"
            )
        if current_states.shape[0] != final_state.shape[0]:
            raise ValueError("SFT2 SIGReg state batch sizes do not match")
        return self.sigreg(
            torch.cat((current_states, final_state.unsqueeze(1)), dim=1)
        )

__all__ = [
    "SFT2Algorithm",
    "SFT2StepOutput",
    "require_sft2_wm_history",
]
