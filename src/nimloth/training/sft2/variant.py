"""SFT2 objective variant registry and latent baseline implementation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch

from nimloth.training.sft2.algorithm import SFT2Algorithm, require_sft2_wm_history
from nimloth.training.sft2.batch import SFT2BatchAssembler
from nimloth.wm import (
    LeWMConfig,
    LatentWMPredictor,
    StateProjector,
    ValueHead,
    WorldModel,
)


@dataclass(frozen=True)
class SFT2VariantBuildContext:
    args: Any
    model: torch.nn.Module
    aux_device: torch.device
    model_dtype: torch.dtype


class SFT2Variant(Protocol):
    name: str

    def validate_args(self, args: Any) -> None: ...

    def build_world_model(self, context: SFT2VariantBuildContext) -> WorldModel: ...

    def build_batch_builder(
        self,
        args: Any,
        base: SFT2BatchAssembler,
    ) -> Any: ...

    def build_algorithm(self, args: Any, **common_kwargs: Any) -> SFT2Algorithm: ...

    def checkpoint_invariants(self, args: Any) -> dict[str, Any]: ...

    def runtime_metadata(self, args: Any) -> dict[str, Any]: ...

    @property
    def metric_fields(self) -> tuple[str, ...]: ...


_VARIANT_IMPORTS = {
    "dino_grid": "nimloth.training.sft2.dino_grid:DINOGridSFT2Variant",
    "latent": "nimloth.training.sft2.variant:LatentSFT2Variant",
}


def available_sft2_variants() -> tuple[str, ...]:
    return tuple(sorted(_VARIANT_IMPORTS))


def resolve_sft2_variant(name: str) -> SFT2Variant:
    spec = _VARIANT_IMPORTS.get(str(name))
    if spec is None:
        raise ValueError(
            f"unsupported SFT2 objective {name!r}; "
            f"available={available_sft2_variants()}"
        )
    module_name, class_name = spec.split(":", 1)
    variant_type = getattr(importlib.import_module(module_name), class_name)
    return variant_type()


class LatentSFT2Variant:
    name = "latent"
    metric_fields: tuple[str, ...] = ()

    def validate_args(self, args: Any) -> None:
        del args

    def build_world_model(self, context: SFT2VariantBuildContext) -> WorldModel:
        args = context.args
        if args.wm_predictor_checkpoint is not None:
            predictor = LatentWMPredictor.load_checkpoint(
                args.wm_predictor_checkpoint,
                map_location=context.aux_device,
            ).to(context.aux_device)
            require_sft2_wm_history(
                predictor,
                history_size=args.history_size,
                source=args.wm_predictor_checkpoint,
            )
        else:
            predictor = LatentWMPredictor.create(
                LeWMConfig(
                    emb_dim=args.emb_dim,
                    history_size=args.history_size,
                )
            ).to(context.aux_device)
        return WorldModel(
            state_proj=StateProjector(
                context.model.config.hidden_size,
                predictor.emb_dim,
                latent_token_count=args.latent_token_count,
            ).to(device=context.aux_device, dtype=context.model_dtype),
            wm_predictor=predictor,
            value_head=ValueHead(predictor.emb_dim).to(
                device=context.aux_device,
                dtype=context.model_dtype,
            ),
        )

    def build_batch_builder(
        self,
        args: Any,
        base: SFT2BatchAssembler,
    ) -> SFT2BatchAssembler:
        del args
        return base

    def build_algorithm(self, args: Any, **common_kwargs: Any) -> SFT2Algorithm:
        del args
        return SFT2Algorithm(**common_kwargs)

    def checkpoint_invariants(self, args: Any) -> dict[str, Any]:
        del args
        return {}

    def runtime_metadata(self, args: Any) -> dict[str, Any]:
        del args
        return {}


__all__ = [
    "SFT2Variant",
    "SFT2VariantBuildContext",
    "available_sft2_variants",
    "resolve_sft2_variant",
]
