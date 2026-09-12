"""回答 token 的因果 CE；动作编号与边界/EOS 独立加权，不改变标签缓存或参数精度。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nimloth.latent import LatentActionTokens

ACTION_TOKEN_LOSS_SCOPE = "action_number_tokens_v1"
BOUNDARY_TOKEN_LOSS_SCOPE = "action_boundaries_and_eos_v1"


def validate_action_weight(weight: float) -> float:
    weight = float(weight)
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("action token loss weight must be finite and >= 1")
    return weight


def resolve_action_number_token_ids(tokenizer) -> tuple[int, ...]:
    protocol = LatentActionTokens()
    tokens = protocol.action_tokens
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
    if any(not isinstance(i, int) or i < 0 or i == tokenizer.unk_token_id for i in ids):
        raise ValueError("all eight action-number tokens must exist in the tokenizer")
    if len(set(ids)) != 8 or any(
        tokenizer.encode(token, add_special_tokens=False) != [i]
        for token, i in zip(tokens, ids)
    ):
        raise ValueError(
            "all eight action-number tokens must have distinct atomic token IDs"
        )
    return ids


def resolve_boundary_token_ids(tokenizer) -> tuple[int, ...]:
    protocol = LatentActionTokens()
    tokens = (protocol.action_start, protocol.action_end)
    ids = tuple(tokenizer.convert_tokens_to_ids(token) for token in tokens)
    eos = tokenizer.eos_token_id
    if any(type(i) is not int or i < 0 or i == tokenizer.unk_token_id for i in (*ids, eos)):
        raise ValueError("action boundaries and EOS must have valid token IDs")
    if any(tokenizer.encode(token, add_special_tokens=False) != [i] for token, i in zip(tokens, ids)):
        raise ValueError("action boundaries must be atomic tokens")
    result = (*ids, eos)
    if len(set(result)) != 3 or set(result) & set(resolve_action_number_token_ids(tokenizer)):
        raise ValueError("boundary/EOS IDs must be distinct and disjoint from actions")
    return result


def weighted_answer_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    action_token_ids: Sequence[int],
    action_weight: float,
    *,
    chunk_size: int = 128,
    boundary_token_ids: Sequence[int] = (),
    boundary_weight: float = 1.0,
) -> torch.Tensor:
    """按每个微批次有效权重之和归一化，保留原梯度累积语义。"""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or chunk_size < 1:
        raise ValueError(
            "expected aligned [batch, sequence, vocabulary] logits and labels"
        )
    weight = validate_action_weight(action_weight)
    boundary_weight = validate_action_weight(boundary_weight)
    boundary_ids = tuple(boundary_token_ids)
    if boundary_ids or boundary_weight != 1:
        if len(boundary_ids) != 3 or len(set(boundary_ids)) != 3 or set(boundary_ids) & set(action_token_ids) or any(type(i) is not int or i < 0 or i >= logits.shape[-1] for i in boundary_ids):
            raise ValueError("expected three distinct boundary/EOS IDs disjoint from actions")
    ids = tuple(action_token_ids)
    if (
        len(ids) != 8
        or len(set(ids)) != 8
        or any(type(i) is not int or i < 0 or i >= logits.shape[-1] for i in ids)
    ):
        raise ValueError(
            "expected eight distinct in-vocabulary action-number token IDs"
        )
    targets = labels[:, 1:]
    positions = (targets != -100).nonzero(as_tuple=False)
    if positions.numel() == 0:
        raise ValueError("answer has no supervised next-token positions")
    action_ids = torch.tensor(ids, device=labels.device)
    boundary_id_tensor = torch.tensor(boundary_ids, device=labels.device, dtype=torch.long)
    numerator = logits.new_zeros((), dtype=torch.float32)
    denominator = logits.new_zeros((), dtype=torch.float32)

    def chunk_loss(scores, target, weights):
        return (
            F.cross_entropy(scores.float(), target, reduction="none") * weights
        ).sum()

    for pos in positions.split(chunk_size):
        target = targets[pos[:, 0], pos[:, 1]]
        weights = torch.where(torch.isin(target, action_ids), weight, 1.0).float()
        if boundary_ids:
            weights = torch.where(torch.isin(target, boundary_id_tensor), boundary_weight, weights)
        scores = logits[pos[:, 0], pos[:, 1]]
        # 重算分块 CE，避免反传前保留整段 FP32 softmax；参数/优化器不转精度。
        value = (
            checkpoint(chunk_loss, scores, target, weights, use_reentrant=False)
            if torch.is_grad_enabled() and scores.requires_grad
            else chunk_loss(scores, target, weights)
        )
        numerator = numerator + value
        denominator = denominator + weights.sum()
    return numerator / denominator


def training_loss(model, batch, *, action_token_ids=(), action_weight=1.0, boundary_token_ids=(), boundary_weight=1.0):
    """权重为 1 时保持原模型 loss 路径，包括 query stage2。"""
    weight = validate_action_weight(action_weight)
    boundary_weight = validate_action_weight(boundary_weight)
    if weight == 1 and boundary_weight == 1:
        return model(**batch).loss
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    return weighted_answer_loss(
        model(**inputs).logits, batch["labels"], action_token_ids, weight,
        boundary_token_ids=boundary_token_ids, boundary_weight=boundary_weight
    )
