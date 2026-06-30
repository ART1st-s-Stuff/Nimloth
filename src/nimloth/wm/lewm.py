"""Nimloth config and helpers around ``external/le-wm``."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nimloth.wm._vendor_lewm import ARPredictor, Embedder, MLP, SIGReg
from nimloth.wm.dataset import NUM_NAVIGATION_ACTIONS

__all__ = [
    "ARPredictor",
    "Embedder",
    "LeWMConfig",
    "MLP",
    "SIGReg",
    "SafeBatchNorm1d",
    "action_one_hot",
]


class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running stats for singleton batches.

    LeWM uses BatchNorm1d in projector MLPs.  Nimloth often runs small
    per-rank micro-batches; PyTorch BatchNorm1d raises in training mode when
    the batch has only one sample.  For that singleton case we use the stored
    running statistics without updating them, while preserving standard
    BatchNorm1d behavior for normal batches.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.ndim == 2 and input.shape[0] <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=self.momentum,
                eps=self.eps,
            )
        return super().forward(input)


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

    # SIGReg regularization (LeWM paper §3.3)
    lambda_sigreg: float = 0.1
    sigreg_num_proj: int = 1024
    sigreg_knots: int = 17


def action_one_hot(indices: torch.Tensor, num_actions: int = NUM_NAVIGATION_ACTIONS) -> torch.Tensor:
    """indices: (B,) int64 -> (B, 1, num_actions) float."""

    one_hot = F.one_hot(indices.long(), num_classes=num_actions).float()
    return one_hot.unsqueeze(1)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False
