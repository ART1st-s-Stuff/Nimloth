"""Qwen2.5-VL 的可训练 Backbone 实现。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from nimloth.backbone.base import Backbone, BackboneBatch, BackboneOutput
from nimloth.backbone.qwen25vl.batch import split_qwen_batch_rows
from nimloth.backbone.qwen25vl.checkpoint import save_full_vision_state
from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
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

    def forward_chunked(
        self,
        batch: BackboneBatch,
        *,
        max_rows: int,
        include_lm_loss: bool = False,
        offload_saved_tensors: bool = False,
    ) -> BackboneOutput:
        root = self.model.module if hasattr(self.model, "module") else self.model
        config = root.config
        image_token_id = int(config.image_token_id)
        vision_config = getattr(config, "vision_config", None)
        spatial_merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
        chunks = split_qwen_batch_rows(
            dict(batch.tensors),
            max_rows=max_rows,
            image_token_id=image_token_id,
            spatial_merge_size=spatial_merge_size,
        )
        def pack_saved_tensor(tensor: torch.Tensor):
            if not offload_saved_tensors or not tensor.is_cuda or tensor.is_leaf:
                return False, tensor
            return True, tensor.device, tensor.detach().to("cpu")

        def unpack_saved_tensor(packed):
            if not packed[0]:
                return packed[1]
            _, device, tensor = packed
            return tensor.to(device, non_blocking=True)

        outputs: list[BackboneOutput] = []
        for chunk in chunks:
            saved_tensor_context = (
                torch.autograd.graph.saved_tensors_hooks(
                    pack_saved_tensor,
                    unpack_saved_tensor,
                )
                if offload_saved_tensors
                else contextlib.nullcontext()
            )
            with saved_tensor_context:
                outputs.append(
                    self.forward(
                        BackboneBatch(chunk),
                        include_lm_loss=include_lm_loss,
                    )
                )
        hidden = torch.cat([output.hidden for output in outputs], dim=0)
        if not include_lm_loss:
            return BackboneOutput(hidden=hidden)

        weighted_losses: list[torch.Tensor] = []
        valid_label_counts: list[int] = []
        for chunk, output in zip(chunks, outputs, strict=True):
            if output.lm_loss is None:
                raise RuntimeError("Qwen chunk did not return LM loss")
            labels = chunk.get("labels")
            if labels is None:
                raise ValueError("include_lm_loss requires labels")
            count = int((labels[:, 1:] != -100).sum().item())
            if count > 0:
                weighted_losses.append(output.lm_loss * count)
                valid_label_counts.append(count)
        if not valid_label_counts:
            raise ValueError("Qwen batch contains no shifted CE supervision labels")
        lm_loss = torch.stack(weighted_losses).sum() / sum(valid_label_counts)
        return BackboneOutput(hidden=hidden, lm_loss=lm_loss)

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
