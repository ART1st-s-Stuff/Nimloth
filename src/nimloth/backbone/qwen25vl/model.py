"""Qwen2.5-VL 的可训练 Backbone 实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from nimloth.backbone.base import Backbone, BackboneBatch, BackboneOutput
from nimloth.backbone.qwen25vl.checkpoint import save_full_vision_state
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.backbone.qwen25vl.state_training import (
    QwenStateTrainingBatch,
    QwenStateTrainingOutput,
    forward_qwen_state_training,
)
from nimloth.latent import materialize_query_embedding_adapter


class Qwen25VLBackbone(Backbone):
    """封装 Qwen 模型及 latent query 提取协议。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        token_id_map: dict[str, int],
        device: torch.device,
        latent_token_count: int,
        lora: bool,
        vision_tune: str,
    ) -> None:
        super().__init__()
        self.language_model = model
        self.token_id_map = dict(token_id_map)
        self.device = device
        self.latent_token_count = int(latent_token_count)
        self.lora = bool(lora)
        self.vision_tune = str(vision_tune)

    @property
    def model(self) -> nn.Module:
        return self.language_model

    def forward(
        self,
        batch: BackboneBatch,
        *,
        include_lm_loss: bool = False,
    ) -> BackboneOutput:
        model_inputs = dict(batch.tensors)
        if not include_lm_loss:
            model_inputs.pop("labels", None)
        hidden, lm_loss = extract_qwen_latents(
            self.model,
            model_inputs,
            self.token_id_map,
            self.device,
            latent_token_count=self.latent_token_count,
        )
        return BackboneOutput(
            hidden=hidden,
            lm_loss=lm_loss if include_lm_loss else None,
        )

    def forward_state_training(
        self,
        batch: QwenStateTrainingBatch,
    ) -> QwenStateTrainingOutput:
        """Expose v2 K16 hidden and action logits from one explicit Qwen call."""

        return forward_qwen_state_training(
            self.model,
            batch,
            self.token_id_map,
            self.device,
            latent_token_count=self.latent_token_count,
        )

    def with_model(self, model: nn.Module) -> "Qwen25VLBackbone":
        return Qwen25VLBackbone(
            model,
            token_id_map=self.token_id_map,
            device=self.device,
            latent_token_count=self.latent_token_count,
            lora=self.lora,
            vision_tune=self.vision_tune,
        )

    def save_pretrained(
        self,
        output_dir: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(model, "save_pretrained"):
            raise TypeError("Qwen25VLBackbone.model must implement save_pretrained()")
        for key, value in (metadata or {}).items():
            setattr(model.config, key, value)
        with materialize_query_embedding_adapter(model) as materialized_state:
            save_state = materialized_state if materialized_state is not None else state_dict
            kwargs: dict[str, Any] = {"safe_serialization": True}
            if save_state is not None:
                kwargs["state_dict"] = save_state
            model.save_pretrained(output_dir, **kwargs)
        if self.lora and self.vision_tune == "full":
            save_full_vision_state(model, output_dir / "vision_full_state.pt")


__all__ = ["Qwen25VLBackbone"]
