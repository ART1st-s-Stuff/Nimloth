"""只加载 rollout 阶段 WM planning 所需的 SFT2 模块。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.backbone import backbone_hidden_size
from nimloth.wm import LatentWMPredictor, StateProjector, ValueHead, WorldModel
from nimloth.wm.grid import (
    SharedSlotProjector,
    TemporalSpatialGridPredictor,
)


class _PlanningGridWorldModel(WorldModel):
    """rollout 阶段使用、保持相同 mean-pool ValueHead 契约的 grid WM。"""

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


def load_planning_world_model(
    *,
    qwen_config,
    wm_checkpoint: Path,
    state_proj_checkpoint: Path,
    value_head_checkpoint: Path,
    device: torch.device,
) -> WorldModel:
    """加载 rollout planning 所需的 projector、predictor 和动作 ValueHead。"""

    qwen_hidden_dim = backbone_hidden_size(qwen_config)
    is_grid = _is_grid_predictor_checkpoint(wm_checkpoint)
    if is_grid:
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
        projector_first = state.get("net.0.weight")
        projector_last = state.get("net.3.weight")
        if (
            projector_first is None
            or projector_last is None
            or projector_first.ndim != 2
            or projector_last.ndim != 2
            or projector_first.shape[1] != qwen_hidden_dim
            or projector_last.shape[0] != predictor.config.emb_dim
            or projector_last.shape[1] != projector_first.shape[0]
        ):
            raise ValueError(
                "state_proj.pt is not the current trainable SFT1 projector format"
            )
        state_proj = SharedSlotProjector(
            input_dim=qwen_hidden_dim,
            output_dim=predictor.config.emb_dim,
            hidden_dim=int(projector_first.shape[0]),
            grid_tokens=predictor.config.grid_tokens,
        ).to(dtype=projector_first.dtype)
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
    world_model_type = _PlanningGridWorldModel if is_grid else WorldModel
    world_model = world_model_type(
        state_proj=state_proj,
        wm_predictor=predictor,
        value_head=value_head,
    ).to(device)
    world_model.requires_grad_(False).eval()
    return world_model


__all__ = ["load_planning_world_model"]
