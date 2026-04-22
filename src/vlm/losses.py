"""Phase 3 语义对齐损失。"""

from __future__ import annotations

import torch
from torch import nn


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    """单负样本或多负样本通用 InfoNCE。"""
    tau = max(float(temperature), 1e-6)
    anchor = nn.functional.normalize(anchor, dim=-1)
    positive = nn.functional.normalize(positive, dim=-1)
    negatives = nn.functional.normalize(negatives, dim=-1)
    pos_logits = torch.sum(anchor * positive, dim=-1, keepdim=True) / tau
    if negatives.dim() == 2:
        neg_logits = torch.sum(anchor * negatives, dim=-1, keepdim=True) / tau
    else:
        neg_logits = torch.einsum("bd,bnd->bn", anchor, negatives) / tau
    logits = torch.cat([pos_logits, neg_logits], dim=-1)
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return nn.functional.cross_entropy(logits, labels)


def temporal_consistency_loss(s_t: torch.Tensor, s_tp1: torch.Tensor) -> torch.Tensor:
    return nn.functional.mse_loss(s_t, s_tp1)
