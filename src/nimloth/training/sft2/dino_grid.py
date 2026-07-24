"""DINO-grid SFT2 的 batch target 装配与显式 objective。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.backbone.dino_grid import (
    CachedDINOGridTargets,
    DINOV2_LARGE_IDENTITY,
)
from nimloth.training.sft2.algorithm import (
    SFT2Algorithm,
    SFT2SIGRegStepOutput,
    SFT2StepOutput,
    gather_global_sigreg_states,
    shared_sigreg_rng,
)
from nimloth.training.sft2.batch import SFT2Batch, SFT2BatchAssembler
from nimloth.training.sft2.runtime import SFT2ModelRuntime
from nimloth.training.sft2.variant import SFT2VariantBuildContext
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridPredictorConfig,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    TemporalSpatialGridPredictor,
    load_sft1_slot_projector,
    warm_start_legacy_grid_components,
)
from nimloth.wm.value_head import ValueHead


@dataclass(frozen=True)
class DINOGridSFT2Batch:
    """公共 SFT2 batch 加当前 transition 的 next-image DINO grid。"""

    base: SFT2Batch
    target_grid: torch.Tensor

    def __post_init__(self) -> None:
        if self.target_grid.ndim != 3 or self.target_grid.shape[0] != self.base.batch_size:
            raise ValueError(
                "DINO grid target must have shape (B,grid_tokens,hidden), "
                f"got {tuple(self.target_grid.shape)} for B={self.base.batch_size}"
            )


class DINOGridBatchAssembler:
    """在公共 transition batch 之外装配独立 DINO target sidecar。"""

    def __init__(
        self,
        base: SFT2BatchAssembler,
        targets: CachedDINOGridTargets,
    ) -> None:
        if targets.grid_size != 4 or targets.identity.hidden_size != 1024:
            raise ValueError(
                "SFT2 DINO supervision requires a 4x4 grid with hidden size 1024"
            )
        self.base = base
        self.targets = targets

    @property
    def processor(self) -> Any:
        return self.base.processor

    def collate_transition_samples(self, batch: list[Any]) -> Any:
        return self.base.collate_transition_samples(batch)

    def collate_cached_transition_batch(
        self,
        batch: list[dict[str, Any]],
    ) -> Any:
        return self.base.collate_cached_transition_batch(batch)

    def prepare(self, raw_batch: Any) -> DINOGridSFT2Batch:
        base = self.base.prepare(raw_batch)
        if len(base.next_image_paths) != base.batch_size or any(
            not path for path in base.next_image_paths
        ):
            raise ValueError(
                "DINO grid supervision requires one next_image_path per current step"
            )
        return DINOGridSFT2Batch(
            base=base,
            target_grid=self.targets.load(
                base.next_image_paths,
                device=base.sample_weights.device,
            ),
        )


class DINOGridSFT2Algorithm(SFT2Algorithm):
    """当前 step 一次 CE + latent WM + decoded DINO + value。

    历史 state 只来自 detached cache。SIGReg 仍在主阶段 backward 之后单独
    执行，只让 online next-state 一侧收到梯度。DINO target 不参与 query
    representation 对齐，只监督 WM prediction 经过 decoder 后的 16 tokens。
    """

    def __init__(self, *, dino_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if float(dino_weight) != 0.5:
            raise ValueError(
                "authoritative decoded-DINO grid loss weight must be exactly 0.5"
            )
        self.dino_weight = float(dino_weight)

    def training_primary_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: DINOGridSFT2Batch,
        *,
        wm_weight: float,
    ) -> SFT2StepOutput:
        return self._grid_step(
            runtime,
            batch,
            wm_weight=wm_weight,
            include_lm_loss=True,
            include_value_ranking=True,
        )

    def evaluation_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: DINOGridSFT2Batch,
    ) -> SFT2StepOutput:
        return self._grid_step(
            runtime,
            batch,
            wm_weight=1.0,
            include_lm_loss=False,
            include_value_ranking=False,
        )

    def training_sigreg_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: DINOGridSFT2Batch,
        *,
        detached_current_state: torch.Tensor,
        sigreg_seed: int,
    ) -> SFT2SIGRegStepOutput:
        """对 mean-pooled grid state 使用当前全局-batch SIGReg 语义。"""

        if not self.has_sigreg_stage:
            raise RuntimeError("SFT2 SIGReg stage is disabled")
        if detached_current_state.ndim != 3:
            raise ValueError(
                "DINO-grid current state must have shape (B,16,D), "
                f"got {tuple(detached_current_state.shape)}"
            )
        next_grid = runtime.agent.encode_state(
            batch.base.online_tail,
            include_lm_loss=False,
        ).state
        if next_grid.ndim != 3:
            raise ValueError(
                "DINO-grid next state must have shape (B,16,D), "
                f"got {tuple(next_grid.shape)}"
            )
        global_current, global_next, global_batch_size = gather_global_sigreg_states(
            detached_current_state.mean(dim=1),
            next_grid.mean(dim=1),
            batch.base.sample_weights > 0.0,
        )
        with shared_sigreg_rng(sigreg_seed, global_next.device):
            sigreg_loss = self._sigreg_loss(global_current, global_next)
        if sigreg_loss is None:
            backward_loss = global_next.sum() * 0.0
            metrics = {"sigreg_skipped_small_batch": 1.0}
        else:
            backward_loss = self.sigreg_weight * sigreg_loss
            metrics = {"sigreg_loss": float(sigreg_loss.detach().item())}
        metrics["sigreg_global_batch_size"] = float(global_batch_size)
        return SFT2SIGRegStepOutput(
            loss=backward_loss,
            raw_loss=sigreg_loss,
            metrics=metrics,
        )

    def _grid_step(
        self,
        runtime: SFT2ModelRuntime,
        batch: DINOGridSFT2Batch,
        *,
        wm_weight: float,
        include_lm_loss: bool,
        include_value_ranking: bool,
    ) -> SFT2StepOutput:
        base = batch.base
        if not 1 <= base.history_size <= self.history_size:
            raise ValueError(
                "SFT2 batch context exceeds configured H: "
                f"batch={base.history_size}, H={self.history_size}"
            )
        if not isinstance(runtime.agent.wm, GridWorldModel):
            raise TypeError("DINOGridSFT2Algorithm requires GridWorldModel")

        # 当前 Qwen 只执行这一次，并在这里产生本 step 唯一的 CE。
        current_encoded = runtime.agent.encode_state(
            base.current,
            include_lm_loss=include_lm_loss,
        )
        cached_history = runtime.history_cache.history(
            base.history_keys,
            reference=current_encoded.state,
        )
        model_output = runtime.agent.forward_step_from_history(
            base.action_indices,
            cached_history,
            encoded_current=current_encoded,
        )
        current_state = model_output.state[:, -1]
        runtime.history_cache.store(
            base.current_keys,
            current_state,
            enabled=not base.is_padding,
        )

        target_states = runtime.target_state(base.next)
        aligned_targets = target_states[base.current_next_indices]
        wm_loss = F.mse_loss(
            model_output.predicted_next_state.float(),
            aligned_targets.detach().float(),
        )
        decoded_grid = runtime.agent.wm.decode_prediction(
            model_output.predicted_next_state
        )
        if decoded_grid.shape != batch.target_grid.shape:
            raise ValueError(
                "decoded WM prediction and DINO grid target must match: "
                f"{tuple(decoded_grid.shape)} != {tuple(batch.target_grid.shape)}"
            )
        dino_loss = F.mse_loss(
            decoded_grid.float(),
            batch.target_grid.detach().float(),
        )
        value = self._value_loss(
            model_output.action_values,
            base.current_action_indices,
            base.current_value_targets,
            include_ranking=include_value_ranking,
        )
        total = (
            wm_weight * wm_loss
            + self.dino_weight * dino_loss
            + self.value_weight * value["loss"]
        )
        if model_output.lm_loss is not None:
            total = total + self.ce_weight * model_output.lm_loss
        sample_count = 0 if base.is_padding else base.batch_size
        if base.is_padding:
            total = total * 0.0

        metrics = {
            "wm_mse": float(wm_loss.detach().item()),
            "dino_grid_mse": float(dino_loss.detach().item()),
            "value_reg": float(value["regression"].detach().item()),
            "value_rank": float(value["ranking"].detach().item()),
            "value_total": float(value["loss"].detach().item()),
            "lambda_wm": float(wm_weight),
            "lambda_dino": self.dino_weight,
            "lambda_sigreg": 0.0,
            "lambda_value": self.value_weight,
            "lambda_ce": self.ce_weight,
            "context_length": float(base.history_size),
            "current_batch_size": float(sample_count),
            "history_cache_entries": float(runtime.history_cache.count),
            "total_loss": float(total.detach().item()),
        }
        if model_output.lm_loss is not None:
            metrics["lm_ce"] = float(model_output.lm_loss.detach().item())
        return SFT2StepOutput(
            loss=total,
            losses={
                "lm": model_output.lm_loss,
                "wm": wm_loss,
                "dino": dino_loss,
                "value": value["loss"],
            },
            metrics=metrics,
            current_state=current_state,
            sample_count=sample_count,
        )


class DINOGridSFT2Variant:
    """DINO-grid 自己拥有模型、数据 sidecar、objective 与 artifact 语义。"""

    name = "dino_grid"
    metric_fields = ("dino_grid_mse", "lambda_dino")

    def validate_args(self, args: Any) -> None:
        required = {
            "latent_token_count": (args.latent_token_count, 16),
            "history_size": (args.history_size, 4),
            "emb_dim": (args.emb_dim, 1024),
            "latent_query_mode": (args.latent_query_mode, "inject"),
            "lambda_dino": (args.lambda_dino, 0.5),
            "lambda_sigreg": (args.lambda_sigreg, 0.1),
            "grid_size": (args.grid_size, 4),
            "grid_ema_decay": (args.grid_ema_decay, 0.99),
        }
        mismatches = {
            name: values for name, values in required.items() if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                f"authoritative DINO-grid SFT2 invariants mismatch: {mismatches}"
            )
        if args.dino_grid_cache is None or args.grid_warmstart is None:
            raise ValueError(
                "DINO-grid SFT2 requires --dino-grid-cache and --grid-warmstart"
            )

    def build_world_model(self, context: SFT2VariantBuildContext) -> GridWorldModel:
        args = context.args
        slot_projector = load_sft1_slot_projector(
            args.model,
            qwen_hidden_dim=int(context.model.config.hidden_size),
            state_dim=args.emb_dim,
            grid_tokens=args.latent_token_count,
            map_location=context.aux_device,
            dtype=context.model_dtype,
        ).to(context.aux_device)
        online_encoder = LeWMGridEncoder(
            emb_dim=args.emb_dim,
            hidden_dim=args.grid_encoder_hidden_dim,
        ).to(device=context.aux_device, dtype=torch.float32)
        world_model = GridWorldModel(
            state_proj=GridStateProjector(slot_projector, online_encoder).to(
                context.aux_device
            ),
            target_encoder=EMATargetGridEncoder(
                online_encoder,
                decay=args.grid_ema_decay,
            ).to(context.aux_device),
            wm_predictor=TemporalSpatialGridPredictor(
                GridPredictorConfig(
                    grid_tokens=args.latent_token_count,
                    emb_dim=args.emb_dim,
                    history_size=args.history_size,
                    depth=args.grid_wm_depth,
                    heads=args.grid_wm_heads,
                    dim_head=args.grid_wm_dim_head,
                    mlp_dim=args.grid_wm_mlp_dim,
                    dropout=args.grid_wm_dropout,
                )
            ).to(device=context.aux_device, dtype=torch.float32),
            dino_decoder=LeWMGridDecoder(
                emb_dim=args.emb_dim,
                hidden_dim=args.grid_decoder_hidden_dim,
            ).to(device=context.aux_device, dtype=torch.float32),
            value_head=ValueHead(args.emb_dim).to(
                device=context.aux_device,
                dtype=torch.float32,
            ),
        )
        if not args.resume and args.grid_warmstart is not None:
            args.grid_warmstart_metadata = warm_start_legacy_grid_components(
                world_model,
                args.grid_warmstart,
            )
        return world_model

    def build_batch_builder(
        self,
        args: Any,
        base: SFT2BatchAssembler,
    ) -> DINOGridBatchAssembler:
        targets = CachedDINOGridTargets.from_cache_root(
            args.dino_grid_cache,
            identity=DINOV2_LARGE_IDENTITY,
            grid_size=args.grid_size,
        )
        args.dino_cache_fingerprint = targets.cache_fingerprint
        return DINOGridBatchAssembler(base, targets)

    def build_algorithm(
        self,
        args: Any,
        **common_kwargs: Any,
    ) -> DINOGridSFT2Algorithm:
        return DINOGridSFT2Algorithm(
            dino_weight=args.lambda_dino,
            **common_kwargs,
        )

    def checkpoint_invariants(self, args: Any) -> dict[str, Any]:
        return {
            "grid_tokens": 16,
            "grid_ordering": "row_major",
            "dino_grid_size": 4,
            "dino_identity": vars(DINOV2_LARGE_IDENTITY),
            "dino_cache_fingerprint": args.dino_cache_fingerprint,
            "dino_weight": 0.5,
            "grid_ema_decay": 0.99,
            "grid_warmstart": str(Path(args.grid_warmstart).resolve()),
            "grid_warmstart_mode": "id33_spatial_plus_zero_temporal_position",
        }

    def runtime_metadata(self, args: Any) -> dict[str, Any]:
        return {
            "supervision_cache": str(args.dino_grid_cache),
            "world_model_warmstart": str(args.grid_warmstart),
        }


__all__ = [
    "DINOGridBatchAssembler",
    "DINOGridSFT2Algorithm",
    "DINOGridSFT2Batch",
    "DINOGridSFT2Variant",
]
