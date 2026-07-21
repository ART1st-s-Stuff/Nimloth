"""SFT2 的完整单批算法：模型前向、目标函数和权重策略。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.agent import Agent, AgentBatch, AgentOutput, AgentTarget
from nimloth.wm import (
    ONE_STEP_WM_HISTORY_SIZE,
    LatentWMPredictor,
    OneStepSIGReg,
    require_one_step_wm_predictor,
)


SFT2_WM_HISTORY_SIZE = ONE_STEP_WM_HISTORY_SIZE


def require_sft2_wm_history(
    wm_predictor: LatentWMPredictor,
    source: Path,
) -> None:
    """SFT2 当前只支持一步 world-model predictor。"""

    require_one_step_wm_predictor(wm_predictor, source=source)


@dataclass(frozen=True)
class SFT2StepOutput:
    """一次 SFT2 前向产生的总 loss、分项 loss 和日志指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    model_output: AgentOutput


class SFT2Algorithm(nn.Module):
    """定义 SFT2 一个 batch 的完整计算图。

    训练循环只负责取 batch、backward 和 optimizer step。本类负责当前状态、目标
    状态、WM/value/SIGReg/CE loss 以及这些 loss 的组合，不依赖 processor、cache、
    DDP、EMA 的具体实现或 checkpoint。
    """

    def __init__(
        self,
        *,
        agent: Agent,
        target: AgentTarget,
        sigreg: OneStepSIGReg | None,
        sigreg_weight: float,
        value_weight: float,
        ce_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
        wm_weight_start: float = 0.1,
        wm_weight_end: float = 1.0,
        wm_warmup_fraction: float = 0.3,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.target = target
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

    def training_step(
        self,
        batch: AgentBatch,
        *,
        wm_weight: float,
    ) -> SFT2StepOutput:
        return self._step(
            batch,
            wm_weight=wm_weight,
            include_lm_loss=True,
            include_value_ranking=True,
            include_sigreg=True,
        )

    def evaluation_step(self, batch: AgentBatch) -> SFT2StepOutput:
        return self._step(
            batch,
            wm_weight=1.0,
            include_lm_loss=False,
            include_value_ranking=False,
            include_sigreg=False,
        )

    def _step(
        self,
        batch: AgentBatch,
        *,
        wm_weight: float,
        include_lm_loss: bool,
        include_value_ranking: bool,
        include_sigreg: bool,
    ) -> SFT2StepOutput:
        """按照 current forward → next target → loss 的顺序完成一次计算。"""

        model_output = self.agent(
            batch.current,
            batch.action_indices,
            include_lm_loss=include_lm_loss,
        )
        target_states = self.target(batch.next)
        aligned_targets = target_states[batch.next_indices]

        wm_loss = self._wm_loss(
            model_output.predicted_next_state,
            aligned_targets,
            batch.non_terminal_mask,
        )
        value = self.value_loss(
            model_output.action_values,
            batch.action_indices,
            batch.value_targets,
            include_ranking=include_value_ranking,
        )
        sigreg_loss = (
            self._sigreg_loss(model_output.state, aligned_targets, batch)
            if include_sigreg
            else None
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
        if bool(batch.non_terminal_mask.any()):
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

    @staticmethod
    def _wm_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """用 mask 统一处理普通 batch 与 terminal-only batch。"""

        per_row = (prediction - target).pow(2).flatten(start_dim=1).mean(dim=1)
        weights = mask.to(device=per_row.device, dtype=per_row.dtype)
        return (per_row * weights).sum() / weights.sum().clamp_min(1.0)

    def value_loss(
        self,
        all_values: torch.Tensor,
        action_indices: torch.Tensor,
        return_targets: torch.Tensor,
        *,
        include_ranking: bool,
    ) -> dict[str, torch.Tensor]:
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
        next_states: torch.Tensor,
        batch: AgentBatch,
    ) -> torch.Tensor | None:
        """把有效单步 transition 交给公共的固定 ``T=2`` SIGReg 模块。"""

        if self.sigreg is None or self.sigreg_weight <= 0.0:
            return None
        indices = batch.non_terminal_mask.nonzero(as_tuple=False).flatten()
        if indices.numel() < 2:
            return None
        return self.sigreg(
            current_states.index_select(0, indices),
            next_states.index_select(0, indices),
        )

    def unwrapped(self) -> "SFT2Algorithm":
        agent = self.agent.unwrapped()
        return SFT2Algorithm(
            agent=agent,
            target=self.target.with_agent(agent),
            sigreg=self.sigreg,
            sigreg_weight=self.sigreg_weight,
            value_weight=self.value_weight,
            ce_weight=self.ce_weight,
            value_rank_margin=self.value_rank_margin,
            value_rank_weight=self.value_rank_weight,
            wm_weight_start=self.wm_weight_start,
            wm_weight_end=self.wm_weight_end,
            wm_warmup_fraction=self.wm_warmup_fraction,
        )


__all__ = [
    "SFT2_WM_HISTORY_SIZE",
    "SFT2Algorithm",
    "SFT2StepOutput",
    "require_sft2_wm_history",
]
