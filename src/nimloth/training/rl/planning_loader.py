"""只加载 rollout 阶段 WM planning 所需的 SFT2 模块。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from nimloth.backbone import backbone_hidden_size
from nimloth.training.rl.algorithm import PLANNER_TRAINING_OBJECTIVE
from nimloth.training.sft2.algorithm import SFT2_VALUE_OBJECTIVE
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


def validate_planning_value_semantics(
    *,
    wm_checkpoint: Path,
    state_proj_checkpoint: Path,
    value_head_checkpoint: Path,
) -> Path:
    """Reject component sets whose action slots were trained on successor states."""

    wm_checkpoint = Path(wm_checkpoint).resolve()
    state_proj_checkpoint = Path(state_proj_checkpoint).resolve()
    value_head_checkpoint = Path(value_head_checkpoint).resolve()
    roots = {
        wm_checkpoint.parent,
        state_proj_checkpoint.parent,
        value_head_checkpoint.parent,
    }
    if len(roots) != 1:
        raise ValueError(
            "planning components must share one checkpoint root so value semantics "
            "can be verified"
        )
    root = roots.pop()
    sft2_state_path = root / "training_state.pt"
    rl_state_path = root / "rl_state.pt"
    if sft2_state_path.is_file():
        state = torch.load(
            sft2_state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        invariants = state.get("training_invariants") if isinstance(state, dict) else None
        actual = invariants.get("value_objective") if isinstance(invariants, dict) else None
        if actual != SFT2_VALUE_OBJECTIVE:
            raise ValueError(
                "planning checkpoint uses an incompatible SFT2 value objective: "
                f"expected={SFT2_VALUE_OBJECTIVE!r}, actual={actual!r}"
            )
        return root
    if rl_state_path.is_file():
        state = torch.load(
            rl_state_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        actual = state.get("planner_training_objective") if isinstance(state, dict) else None
        if actual != PLANNER_TRAINING_OBJECTIVE:
            raise ValueError(
                "planning checkpoint uses an incompatible RL value objective: "
                f"expected={PLANNER_TRAINING_OBJECTIVE!r}, actual={actual!r}"
            )
        return root
    raise FileNotFoundError(
        "planning component root has no training_state.pt or rl_state.pt; "
        f"cannot verify outgoing Q(s,a) semantics: {root}"
    )


def load_planning_world_model(
    *,
    qwen_config,
    wm_checkpoint: Path,
    state_proj_checkpoint: Path,
    value_head_checkpoint: Path,
    device: torch.device,
) -> WorldModel:
    """加载 rollout planning 所需的 projector、predictor 和动作 ValueHead。"""

    validate_planning_value_semantics(
        wm_checkpoint=wm_checkpoint,
        state_proj_checkpoint=state_proj_checkpoint,
        value_head_checkpoint=value_head_checkpoint,
    )
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
        projector_first = state["net.0.weight"]
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


__all__ = ["load_planning_world_model", "validate_planning_value_semantics"]
