"""RL checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from nimloth.training.common.dist import is_main
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def save_rl_checkpoint(
    out_dir: Path,
    *,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    optimizer: torch.optim.Optimizer | None = None,
    iteration: int = 0,
    global_step: int = 0,
    best_value_loss: float = float("inf"),
) -> None:
    if not is_main():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_unwrap(state_proj).state_dict(), out_dir / "state_proj.pt")
    _unwrap(wm_predictor).save_checkpoint(out_dir / "wm_predictor")
    _unwrap(value_head).save_checkpoint(out_dir / "value_head")

    state = {
        "iteration": iteration,
        "global_step": global_step,
        "best_value_loss": best_value_loss,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, out_dir / "rl_state.pt")


def load_rl_checkpoint(
    ckpt_dir: Path,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    value_head: ValueHead,
    device: torch.device,
) -> dict:
    """Load WM components from a checkpoint directory.  Returns the training state dict."""

    sp_path = ckpt_dir / "state_proj.pt"
    if sp_path.is_file():
        _unwrap(state_proj).load_state_dict(
            torch.load(sp_path, map_location=device, weights_only=True)
        )
    pred_dir = ckpt_dir / "wm_predictor"
    if pred_dir.is_dir():
        loaded_pred = LatentWMPredictor.load_checkpoint(pred_dir, map_location=device)
        _unwrap(wm_predictor).load_state_dict(loaded_pred.state_dict())
    head_dir = ckpt_dir / "value_head"
    if head_dir.is_dir():
        head = _unwrap(value_head)
        loaded_head = ValueHead.load_checkpoint(
            head_dir, emb_dim=head.net[0].in_features, map_location=device
        )
        head.load_state_dict(loaded_head.state_dict())

    state_path = ckpt_dir / "rl_state.pt"
    if state_path.is_file():
        return torch.load(state_path, map_location="cpu", weights_only=False)
    return {}
