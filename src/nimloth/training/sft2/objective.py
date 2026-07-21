"""SFT2 的目标函数与指标计算。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.agent import AgentBatch, AgentOutput


@dataclass(frozen=True)
class SFT2StepOutput:
    """一次 SFT2 前向产生的总 loss、分项 loss 和日志指标。"""

    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    model_output: AgentOutput


def build_trajectory_sigreg_inputs(
    trajectory_steps: Sequence[tuple[str, int]],
    current_states: torch.Tensor,
    next_states: torch.Tensor,
) -> list[torch.Tensor]:
    """把同一 record 的 transition 还原为 SIGReg 使用的状态序列。"""

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
    if all_record_ids_empty and len(groups) <= 1:
        return []

    inputs: list[torch.Tensor] = []
    for entries in groups.values():
        entries.sort(key=lambda entry: entry[0])
        sequence = [entries[0][1], *(entry[2] for entry in entries)]
        inputs.append(torch.stack(sequence, dim=0).unsqueeze(1))
    return inputs


class SFT2Objective(nn.Module):
    """根据结构化 Agent 输出计算 SFT2 的全部训练目标。"""

    def __init__(
        self,
        *,
        sigreg: nn.Module | None,
        sigreg_weight: float,
        value_weight: float,
        ce_weight: float,
        value_rank_margin: float,
        value_rank_weight: float,
    ) -> None:
        super().__init__()
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_weight = float(value_weight)
        self.ce_weight = float(ce_weight)
        self.value_rank_margin = float(value_rank_margin)
        self.value_rank_weight = float(value_rank_weight)

    def forward(
        self,
        *,
        model_output: AgentOutput,
        target_states: torch.Tensor,
        batch: AgentBatch,
        wm_weight: float,
        training: bool,
    ) -> SFT2StepOutput:
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
            training=training,
        )
        sigreg_loss = self._sigreg_loss(
            model_output.state,
            aligned_targets,
            batch,
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
            "lambda_sigreg": self.sigreg_weight,
            "lambda_value": self.value_weight,
            "lambda_ce": self.ce_weight,
            "total_loss": float(total.detach().item()),
        }
        if bool(batch.non_terminal_mask.any()):
            metrics["wm_mse"] = float(wm_loss.detach().item())
        if sigreg_loss is not None:
            metrics["sigreg_loss"] = float(sigreg_loss.detach().item())
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
        training: bool,
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
        rank_weight = self.value_rank_weight if training else 0.0
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
        if self.sigreg is None or not bool(batch.non_terminal_mask.any()):
            return None
        indices = batch.non_terminal_mask.nonzero(as_tuple=False).flatten()
        selected_current = current_states[indices]
        selected_next = next_states[indices]
        trajectory_steps = [
            batch.trajectory_steps[index]
            for index in indices.detach().cpu().tolist()
        ]
        inputs = build_trajectory_sigreg_inputs(
            trajectory_steps,
            selected_current,
            selected_next,
        )
        if not inputs:
            inputs = [torch.stack([selected_current, selected_next], dim=0)]
        return torch.stack([self.sigreg(value) for value in inputs]).mean()


__all__ = [
    "SFT2Objective",
    "SFT2StepOutput",
    "build_trajectory_sigreg_inputs",
]
