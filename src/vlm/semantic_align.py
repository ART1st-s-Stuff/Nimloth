"""Phase 3 语义对齐模型接口。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.core.interfaces import Model
from src.vlm.losses import info_nce_loss, temporal_consistency_loss
from src.vlm.semantic_state import SemanticStateGenerator


class DeltaProjector(nn.Module):
    """h(z_t, z_t+k) 轻量投影器。"""

    def __init__(self, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp], dim=-1))


class SemanticAlignModel(Model):
    """封装 DeltaProjector + SemanticStateGenerator 的统一语义对齐训练接口。"""

    def __init__(
        self,
        *,
        projector: DeltaProjector,
        semantic_generator: SemanticStateGenerator,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        temporal_weight: float = 0.1,
        grad_clip_norm: float = 1.0,
    ) -> None:
        super().__init__()
        self.projector = projector.to(device)
        self.semantic_generator = semantic_generator
        self.optimizer = optimizer
        self.device = device
        self.temporal_weight = temporal_weight
        self.grad_clip_norm = grad_clip_norm
        self.projector.train()

    def train_step(self, batch: Any) -> dict[str, Any]:
        z_t = batch["z_t"].to(self.device)
        z_t_pos = batch["z_t_pos"].to(self.device)
        z_t_neg = batch["z_t_neg"].to(self.device)
        s_t_list: list[torch.Tensor] = []
        s_tp1_list: list[torch.Tensor] = []
        for i in range(z_t.size(0)):
            output = self.semantic_generator.infer(
                image_path=batch["image_path"][i],
                history_image_paths=[batch["image_path"][i]],
                task_text=batch["task_text"][i],
                env_context=batch["env_context"][i],
            )
            next_output = self.semantic_generator.infer(
                image_path=batch["pos_image_path"][i],
                history_image_paths=[batch["image_path"][i], batch["pos_image_path"][i]],
                task_text=batch["task_text"][i],
                env_context=batch["env_context"][i],
            )
            s_t_list.append(output.s_t)
            s_tp1_list.append(next_output.s_t)
        s_t = torch.stack(s_t_list, dim=0).to(self.device)
        s_tp1 = torch.stack(s_tp1_list, dim=0).to(self.device)
        pred_positive = self.projector(z_t=z_t, z_tp=z_t_pos)
        pred_negative = self.projector(z_t=z_t, z_tp=z_t_neg)
        loss_nce = info_nce_loss(
            anchor=s_t,
            positive=pred_positive,
            negatives=pred_negative,
            temperature=float(getattr(self, "_temperature", 0.1)),
        )
        loss_temporal = temporal_consistency_loss(s_t=s_t, s_tp1=s_tp1)
        loss = loss_nce + self.temporal_weight * loss_temporal
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.projector.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        return {
            "loss": float(loss.item()),
            "loss_nce": float(loss_nce.item()),
            "loss_temporal": float(loss_temporal.item()),
        }

    def eval_step(self, batch: Any) -> dict[str, Any]:
        with torch.no_grad():
            z_t = batch["z_t"].to(self.device)
            z_t_pos = batch["z_t_pos"].to(self.device)
            z_t_neg = batch["z_t_neg"].to(self.device)
            s_t_list: list[torch.Tensor] = []
            s_tp1_list: list[torch.Tensor] = []
            for i in range(z_t.size(0)):
                output = self.semantic_generator.infer(
                    image_path=batch["image_path"][i],
                    history_image_paths=[batch["image_path"][i]],
                    task_text=batch["task_text"][i],
                    env_context=batch["env_context"][i],
                )
                next_output = self.semantic_generator.infer(
                    image_path=batch["pos_image_path"][i],
                    history_image_paths=[batch["image_path"][i], batch["pos_image_path"][i]],
                    task_text=batch["task_text"][i],
                    env_context=batch["env_context"][i],
                )
                s_t_list.append(output.s_t)
                s_tp1_list.append(next_output.s_t)
            s_t = torch.stack(s_t_list, dim=0).to(self.device)
            s_tp1 = torch.stack(s_tp1_list, dim=0).to(self.device)
            pred_positive = self.projector(z_t=z_t, z_tp=z_t_pos)
            pred_negative = self.projector(z_t=z_t, z_tp=z_t_neg)
            loss_nce = info_nce_loss(
                anchor=s_t,
                positive=pred_positive,
                negatives=pred_negative,
                temperature=float(getattr(self, "_temperature", 0.1)),
            )
            loss_temporal = temporal_consistency_loss(s_t=s_t, s_tp1=s_tp1)
            loss = loss_nce + self.temporal_weight * loss_temporal
        return {
            "loss": float(loss.item()),
            "loss_nce": float(loss_nce.item()),
            "loss_temporal": float(loss_temporal.item()),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "projector_state_dict": self.projector.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if "projector_state_dict" in state:
            self.projector.load_state_dict(state["projector_state_dict"])
        if "optimizer_state_dict" in state:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
