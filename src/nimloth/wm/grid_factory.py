"""Spatial-grid WorldModel checkpoint loader plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from nimloth.wm.factory import WorldModelLoadRequest
from nimloth.wm.grid import (
    EMATargetGridEncoder,
    GridStateProjector,
    GridWorldModel,
    LeWMGridDecoder,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)
from nimloth.wm.value_head import ValueHead


def _mlp_hidden_dim(
    state: dict[str, torch.Tensor],
    *,
    first_weight: str,
    emb_dim: int,
) -> int:
    weight = state.get(first_weight)
    if weight is None or weight.ndim != 2 or weight.shape[1] != emb_dim:
        raise ValueError(
            "cannot infer grid MLP hidden_dim from checkpoint tensor "
            f"{first_weight!r}"
        )
    return int(weight.shape[0])


class GridWorldModelLoader:
    name = "spatial_grid"

    def matches(self, predictor_config: Mapping[str, object]) -> bool:
        return "grid_tokens" in predictor_config

    def load(self, request: WorldModelLoadRequest) -> GridWorldModel:
        if (
            request.predictor_checkpoint is None
            or request.state_proj_checkpoint is None
            or request.value_head_checkpoint is None
        ):
            raise ValueError(
                "spatial-grid WorldModel requires predictor, state projector, "
                "and value-head checkpoints"
            )
        checkpoint_root = request.predictor_checkpoint.parent
        predictor = TemporalSpatialGridPredictor.load_checkpoint(
            request.predictor_checkpoint,
            map_location="cpu",
        )
        state_proj_state = torch.load(
            request.state_proj_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        encoder_hidden_dim = _mlp_hidden_dim(
            state_proj_state,
            first_weight="online_encoder.net.net.0.weight",
            emb_dim=predictor.config.emb_dim,
        )
        slot_first = state_proj_state.get("slot_projector.net.0.weight")
        slot_last = state_proj_state.get("slot_projector.net.3.weight")
        if (
            slot_first is None
            or slot_last is None
            or slot_first.ndim != 2
            or slot_last.ndim != 2
            or slot_first.shape[1] != request.qwen_hidden_dim
            or slot_last.shape[0] != predictor.config.emb_dim
            or slot_last.shape[1] != slot_first.shape[0]
        ):
            raise ValueError(
                "spatial-grid state projector is incompatible with the "
                "Qwen/predictor dimensions"
            )
        slot_projector = SharedSlotProjector(
            input_dim=request.qwen_hidden_dim,
            output_dim=predictor.config.emb_dim,
            hidden_dim=int(slot_first.shape[0]),
            grid_tokens=predictor.config.grid_tokens,
        ).to(dtype=slot_first.dtype)
        state_proj = GridStateProjector(
            slot_projector,
            LeWMGridEncoder(
                emb_dim=predictor.config.emb_dim,
                hidden_dim=encoder_hidden_dim,
            ),
        )
        state_proj.load_state_dict(state_proj_state)

        metadata_path = checkpoint_root / "dino_grid_config.json"
        decoder_path = checkpoint_root / "dino_grid_decoder.pt"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        decoder_state = torch.load(
            decoder_path,
            map_location="cpu",
            weights_only=True,
        )
        decoder_hidden_dim = _mlp_hidden_dim(
            decoder_state,
            first_weight="net.net.0.weight",
            emb_dim=predictor.config.emb_dim,
        )
        world_model = GridWorldModel(
            state_proj=state_proj,
            target_encoder=EMATargetGridEncoder(
                state_proj.online_encoder,
                decay=float(metadata["ema_decay"]),
            ),
            wm_predictor=predictor,
            dino_decoder=LeWMGridDecoder(
                emb_dim=predictor.config.emb_dim,
                hidden_dim=decoder_hidden_dim,
            ),
            value_head=ValueHead.load_checkpoint(
                request.value_head_checkpoint,
                emb_dim=predictor.config.emb_dim,
                map_location="cpu",
            ),
        )
        world_model.load_checkpoint_extras(
            checkpoint_root,
            map_location=torch.device("cpu"),
        )
        return world_model

    def required_artifacts(self, checkpoint_root: Path) -> tuple[Path, ...]:
        return (
            checkpoint_root / "target_grid_encoder_ema.pt",
            checkpoint_root / "dino_grid_decoder.pt",
            checkpoint_root / "dino_grid_config.json",
        )


__all__ = ["GridWorldModelLoader"]
