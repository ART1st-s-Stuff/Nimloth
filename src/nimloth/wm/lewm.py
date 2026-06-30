"""Nimloth config and helpers around ``external/le-wm``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS

__all__ = [
    "ARPredictor",
    "Embedder",
    "LeWMConfig",
    "MLP",
    "action_one_hot",
]


@dataclass
class LeWMConfig:
    """Predictor hyper-parameters for Qwen-latent dynamics (no pixel encoder).

    Scaled to match Qwen-scale latent: the embed dimension is the bridge between
    Qwen hidden states (~3584 dim) and the WM predictor.  Predictor capacity
    follows LeWM paper defaults (ViT-S scale): 6 layers, 16 heads, 10M params.
    """

    emb_dim: int = 1024
    action_dim: int = NUM_NAVIGATION_ACTIONS
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_mlp_dim: int = 4096
    predictor_hidden_dim: int = 1024
    history_size: int = 4


def action_one_hot(indices: torch.Tensor, num_actions: int = NUM_NAVIGATION_ACTIONS) -> torch.Tensor:
    """indices: (B,) int64 -> (B, 1, num_actions) float."""

    one_hot = F.one_hot(indices.long(), num_classes=num_actions).float()
    return one_hot.unsqueeze(1)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
