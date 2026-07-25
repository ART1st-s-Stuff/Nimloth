"""Load only the SFT2 modules required by rollout-time WM planning."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.backbone import backbone_hidden_size
from nimloth.wm import LatentWMPredictor, StateProjector, ValueHead, WorldModel
from nimloth.wm.grid import (
    GridStateProjector,
    LeWMGridEncoder,
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)


class _PlanningGridWorldModel(WorldModel):
    """Rollout-time grid WM with the same mean-pooled ValueHead contract."""

    def predict_action_values(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim < 3:
            raise ValueError(
                "grid value input must have shape (...,N,D), "
                f"got {tuple(state.shape)}"
            )
        return self.value_head(state.mean(dim=-2)).float()


def _is_grid_predictor_checkpoint(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing WM predictor config: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return "grid_tokens" in raw


def _grid_hidden_dim(
    state: dict[str, torch.Tensor],
    *,
    key: str,
    emb_dim: int,
) -> int:
    weight = state.get(key)
    if weight is None or weight.ndim != 2 or weight.shape[1] != emb_dim:
        raise ValueError(f"cannot infer grid projector hidden dimension from {key}")
    return int(weight.shape[0])


def load_planning_world_model(
    *,
    qwen_config,
    wm_checkpoint: Path,
    state_proj_checkpoint: Path,
    value_head_checkpoint: Path,
    device: torch.device,
) -> WorldModel:
    """Load projector, predictor and action value head without DINO auxiliaries."""

    qwen_hidden_dim = backbone_hidden_size(qwen_config)
    if _is_grid_predictor_checkpoint(wm_checkpoint):
        predictor = TemporalSpatialGridPredictor.load_checkpoint(
            wm_checkpoint,
            map_location="cpu",
        )
        state = torch.load(
            state_proj_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise ValueError("grid state projector checkpoint must be a state dict")
        slot_first = state.get("slot_projector.net.0.weight")
        slot_last = state.get("slot_projector.net.3.weight")
        if (
            slot_first is None
            or slot_last is None
            or slot_first.ndim != 2
            or slot_last.ndim != 2
            or slot_first.shape[1] != qwen_hidden_dim
            or slot_last.shape[0] != predictor.config.emb_dim
            or slot_last.shape[1] != slot_first.shape[0]
        ):
            raise ValueError(
                "grid state projector is incompatible with Qwen/WM dimensions"
            )
        state_proj = GridStateProjector(
            SharedSlotProjector(
                input_dim=qwen_hidden_dim,
                output_dim=predictor.config.emb_dim,
                hidden_dim=int(slot_first.shape[0]),
                grid_tokens=predictor.config.grid_tokens,
            ).to(dtype=slot_first.dtype),
            LeWMGridEncoder(
                emb_dim=predictor.config.emb_dim,
                hidden_dim=_grid_hidden_dim(
                    state,
                    key="online_encoder.net.net.0.weight",
                    emb_dim=predictor.config.emb_dim,
                ),
            ),
        )
        state_proj.load_state_dict(state)
        emb_dim = predictor.config.emb_dim
    else:
        predictor = LatentWMPredictor.load_checkpoint(
            wm_checkpoint,
            map_location="cpu",
        )
        emb_dim = predictor.emb_dim
        state = torch.load(
            state_proj_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict):
            raise ValueError("state projector checkpoint must be a state dict")
        first = state.get("net.net.0.weight")
        if (
            first is None
            or first.ndim != 2
            or first.shape[1] % qwen_hidden_dim != 0
        ):
            raise ValueError(
                "state projector is incompatible with the Qwen hidden dimension"
            )
        state_proj = StateProjector(
            qwen_hidden_dim=qwen_hidden_dim,
            lewm_emb_dim=emb_dim,
            projector_hidden_dim=int(first.shape[0]),
            latent_token_count=int(first.shape[1] // qwen_hidden_dim),
        )
        state_proj.load_state_dict(state)

    value_head = ValueHead.load_checkpoint(
        value_head_checkpoint,
        emb_dim=emb_dim,
        map_location="cpu",
    )
    world_model_type = _PlanningGridWorldModel if _is_grid_predictor_checkpoint(
        wm_checkpoint
    ) else WorldModel
    world_model = world_model_type(
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
    ).to(device)
    world_model.requires_grad_(False).eval()
    return world_model


__all__ = ["load_planning_world_model"]
