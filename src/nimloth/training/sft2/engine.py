"""Shared SFT2 forward path used by both training and validation."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import Any

import torch

from nimloth.backbone.qwen25vl.latent import extract_qwen_latents
from nimloth.backbone.qwen25vl.vision_ema import VisionEncoderEMA
from nimloth.training.sft2.data.batch import unpack_transition_batch
from nimloth.training.sft2.step import compute_step_value_loss, compute_step_wm_loss
from nimloth.training.sft2.types import SFT2StepOutput
from nimloth.training.sft2.utils import preserve_module_modes, unwrap_module
from nimloth.wm import LatentWMPredictor, SIGReg, StateProjector, ValueHead


@dataclass(frozen=True)
class SFT2StepRunner:
    model: torch.nn.Module
    state_proj: StateProjector
    wm_predictor: LatentWMPredictor
    value_head: ValueHead
    processor: Any
    token_id_map: dict[str, int]
    device: torch.device
    max_length: int
    pad_token_id: int
    latent_token_count: int = 1
    mask_latent_query_labels: bool = True
    vision_ema: VisionEncoderEMA | None = None
    sigreg_module: SIGReg | None = None
    value_rank_margin: float = 0.1
    value_rank_lambda: float = 1.0

    @property
    def modules(self) -> tuple[torch.nn.Module, ...]:
        return self.model, self.state_proj, self.wm_predictor, self.value_head

    def unwrapped(self) -> SFT2StepRunner:
        return replace(
            self,
            model=unwrap_module(self.model),
            state_proj=unwrap_module(self.state_proj),
            wm_predictor=unwrap_module(self.wm_predictor),
            value_head=unwrap_module(self.value_head),
        )

    @contextlib.contextmanager
    def validation_context(self):
        ema_context = (
            self.vision_ema.use_ema_weights(self.model)
            if self.vision_ema is not None
            else contextlib.nullcontext()
        )
        with preserve_module_modes(self.modules, training=False), ema_context:
            yield

    def forward(self, batch: Any, *, training: bool) -> SFT2StepOutput:
        items, encoding, next_encoding_rows = unpack_transition_batch(
            batch,
            self.processor,
            self.max_length,
            pad_token_id=self.pad_token_id,
            latent_token_count=self.latent_token_count,
            mask_latent_query_labels=self.mask_latent_query_labels,
        )
        if not training:
            encoding.pop("labels", None)
        current_latent, lm_loss = extract_qwen_latents(
            self.model,
            encoding,
            self.token_id_map,
            self.device,
            latent_token_count=self.latent_token_count,
        )
        wm_loss, sigreg_loss, wm_metrics = compute_step_wm_loss(
            self.model,
            items,
            current_latent,
            self.processor,
            self.token_id_map,
            self.device,
            self.state_proj,
            self.wm_predictor,
            self.max_length,
            vision_ema=self.vision_ema if training else None,
            next_enc_rows=next_encoding_rows,
            pad_token_id=self.pad_token_id,
            sigreg_module=self.sigreg_module,
            latent_token_count=self.latent_token_count,
        )
        value_loss, value_metrics = compute_step_value_loss(
            current_latent,
            items,
            self.state_proj,
            self.value_head,
            rank_margin=self.value_rank_margin if training else 0.0,
            lambda_rank=self.value_rank_lambda if training else 0.0,
        )
        return SFT2StepOutput(
            lm_loss=lm_loss if training else None,
            wm_loss=wm_loss,
            sigreg_loss=sigreg_loss,
            value_loss=value_loss,
            metrics={**wm_metrics, **value_metrics},
        )
