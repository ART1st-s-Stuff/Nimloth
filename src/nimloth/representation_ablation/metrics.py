"""Offline metrics for representation ablation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass
class EncodedTransition:
    record_id: str
    step_index: int
    action_index: int
    action_value_target: float
    success: bool
    state: torch.Tensor
    next_state: torch.Tensor


def _safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def topk_action_accuracy(values: torch.Tensor, action_indices: torch.Tensor, *, ks: tuple[int, ...] = (1, 2)) -> dict[str, float]:
    """Return top-k accuracy of value-head scores against logged/VAGEN action indices."""

    if values.ndim != 2:
        raise ValueError(f"values must have shape (B, A), got {tuple(values.shape)}")
    if action_indices.ndim != 1:
        raise ValueError(f"action_indices must have shape (B,), got {tuple(action_indices.shape)}")
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, values.shape[1])
        topk = values.topk(kk, dim=-1).indices
        correct = topk.eq(action_indices.unsqueeze(1)).any(dim=1).float().mean()
        out[f"value_top{k}_action_acc"] = float(correct.item())
    return out


def value_head_metrics(
    values: torch.Tensor,
    action_indices: torch.Tensor,
    targets: torch.Tensor,
    successes: torch.Tensor,
) -> dict[str, float]:
    """Compute regression, action top-k, ranking and calibration metrics."""

    values = values.float()
    action_indices = action_indices.long()
    targets = targets.to(device=values.device, dtype=values.dtype)
    successes = successes.to(device=values.device, dtype=torch.bool)
    chosen = values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    mask = F.one_hot(action_indices, num_classes=values.shape[1]).bool()
    max_other = values.masked_fill(mask, float("-inf")).max(dim=1).values
    out = {
        **topk_action_accuracy(values, action_indices, ks=(1, 2)),
        "value_chosen_mse": float(F.mse_loss(chosen, targets).item()),
        "value_chosen_mae": float(F.l1_loss(chosen, targets).item()),
        "value_chosen_mean": float(chosen.mean().item()),
        "value_target_mean": float(targets.mean().item()),
        "value_chosen_gt_other_rate": float(chosen.gt(max_other).float().mean().item()),
        "value_success_ranking_auc": success_ranking_auc(chosen.detach().cpu(), successes.detach().cpu()),
    }
    out.update(calibration_by_value(chosen.detach().cpu(), successes.detach().cpu(), num_bins=5))
    return out


def success_ranking_auc(scores: torch.Tensor, successes: torch.Tensor) -> float:
    """Pairwise AUC: probability a success transition has higher score than a failure."""

    scores = scores.flatten().float()
    successes = successes.flatten().bool()
    pos = scores[successes]
    neg = scores[~successes]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    cmp = pos[:, None] - neg[None, :]
    wins = cmp.gt(0).float() + 0.5 * cmp.eq(0).float()
    return float(wins.mean().item())


def calibration_by_value(scores: torch.Tensor, successes: torch.Tensor, *, num_bins: int = 5) -> dict[str, float]:
    """Sort by predicted chosen value and report success rate per equal-count bin."""

    scores = scores.flatten().float()
    successes = successes.flatten().float()
    if scores.numel() == 0:
        return {}
    order = torch.argsort(scores)
    bins = torch.chunk(order, min(num_bins, scores.numel()))
    out: dict[str, float] = {}
    for idx, bin_idx in enumerate(bins):
        if bin_idx.numel() == 0:
            continue
        out[f"value_calib_bin{idx}_score_mean"] = float(scores[bin_idx].mean().item())
        out[f"value_calib_bin{idx}_success_rate"] = float(successes[bin_idx].mean().item())
    return out


def predictor_one_step_metrics(pred: torch.Tensor, target: torch.Tensor, *, prefix: str = "predictor_1step") -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    cosine = F.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
    return {
        f"{prefix}_mse": float(mse.item()),
        f"{prefix}_mae": float(mae.item()),
        f"{prefix}_cosine": float(cosine.item()),
    }


def predictor_multistep_metrics(
    predictor,
    rows: list[EncodedTransition],
    depths: list[int],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Autoregressively roll out from cached true states and compare to future true states."""

    by_record: dict[str, list[EncodedTransition]] = {}
    for row in rows:
        by_record.setdefault(row.record_id, []).append(row)
    for record_rows in by_record.values():
        record_rows.sort(key=lambda item: item.step_index)

    out: dict[str, float] = {}
    for depth in depths:
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for record_rows in by_record.values():
            if len(record_rows) < depth:
                continue
            by_step = {row.step_index: row for row in record_rows}
            for row in record_rows:
                seq = [by_step.get(row.step_index + offset) for offset in range(depth)]
                if any(item is None for item in seq):
                    continue
                typed_seq = [item for item in seq if item is not None]
                actions = torch.tensor(
                    [[item.action_index for item in typed_seq]],
                    dtype=torch.long,
                    device=device,
                )
                start = row.state.unsqueeze(0).to(device)
                rollout = predictor.rollout_states(start, actions)
                preds.append(rollout[:, -1].detach().cpu())
                targets.append(typed_seq[-1].next_state.unsqueeze(0).detach().cpu())
        if not preds:
            out[f"predictor_depth{depth}_count"] = 0.0
            out[f"predictor_depth{depth}_mse"] = float("nan")
            out[f"predictor_depth{depth}_cosine"] = float("nan")
            continue
        pred_t = torch.cat(preds, dim=0)
        target_t = torch.cat(targets, dim=0)
        out[f"predictor_depth{depth}_count"] = float(pred_t.shape[0])
        out.update(predictor_one_step_metrics(pred_t, target_t, prefix=f"predictor_depth{depth}"))
    return out


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row and not math.isnan(float(row[key]))]
        out[key] = _safe_mean(vals)
    return out
