"""Loss functions for SFT2 (WM latent MSE + SIGReg + value head + optional LM CE)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from nimloth.wm._vendor_lewm import SIGReg
from nimloth.wm.predictor import LatentWMPredictor
from nimloth.wm.state_proj import StateProjector
from nimloth.wm.value_head import ValueHead

__all__ = [
    "SIGReg",
    "StateProjector",
    "_build_trajectory_sigreg_inputs",
    "compute_combined_loss",
    "compute_value_loss",
    "compute_wm_latent_loss",
    "wm_loss_weight_schedule",
]


def _build_trajectory_sigreg_inputs(
    items: list[dict[str, Any]],
    state_emb: torch.Tensor,
    target_emb: torch.Tensor,
) -> list[torch.Tensor]:
    """Group projected embeddings by record into trajectory sequences for SIGReg.

    SIGReg expects dense ``(T, B, D)`` input without masks.  This helper builds
    one ``(T_i, 1, D)`` tensor per trajectory so each record's full temporal
    sequence contributes to the regularizer, matching LeWM's trajectory-level
    application.

    Sequence construction (per record):
      1. Sort transitions by ``step_index``.
      2. Build ordered states: state_emb of first step, then target_emb of every
         step in order.  For consecutive steps this yields
         ``s_t, s_{t+1}, s_{t+2}, …`` with no duplicate adjacent states.
      3. For a record appearing with a single step, the sequence is length 2
         (s_t, s_{t+1}), equivalent to the old pair-stack fallback.

    Returns:
        One ``(T_i, 1, D)`` tensor per distinct record.  An empty list if no
        records can be grouped.
    """
    if not items:
        return []

    # Resolve record_id: prefer explicit field, fall back only for the canonical
    # "record_id:step_index" id format.  Unknown legacy ids should not be
    # treated as a real trajectory.
    groups: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = defaultdict(list)
    all_record_ids_empty = True
    for item, s_emb, t_emb in zip(items, state_emb, target_emb):
        record_id = str(item.get("record_id") or "")
        if not record_id:
            item_id = str(item.get("id", ""))
            if ":" in item_id:
                record_id = item_id.split(":", 1)[0]
        if record_id:
            all_record_ids_empty = False
        step_index = int(item.get("step_index", 0))
        groups[record_id].append((step_index, s_emb, t_emb))

    # Old preprocess caches may lack record_id; fall through so the caller can
    # fall back to pair-stack SIGReg.
    if all_record_ids_empty and len(groups) <= 1:
        return []

    sigreg_inputs: list[torch.Tensor] = []
    for _record_id, entries in groups.items():
        entries.sort(key=lambda x: x[0])
        # Sequence: state_emb of the earliest step, then target_emb of each step in order.
        seq_parts: list[torch.Tensor] = [entries[0][1]]  # state_emb of first transition
        for _, _s_emb, t_emb in entries:
            seq_parts.append(t_emb)  # target_emb of each step
        # (T, D) → (T, 1, D) for SIGReg.
        sigreg_inputs.append(torch.stack(seq_parts, dim=0).unsqueeze(1))

    return sigreg_inputs


def compute_wm_latent_loss(
    *,
    qwen_hidden_at_latent: torch.Tensor,
    qwen_hidden_at_next_latent: torch.Tensor,
    action_indices: torch.Tensor,
    state_proj: StateProjector,
    wm_predictor: LatentWMPredictor,
    sigreg_module: SIGReg | None = None,
    items: list[dict[str, Any]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
    """WM predictor MSE plus an optional raw SIGReg loss.

    SIGReg (Sketch Isotropic Gaussian Regularizer) from LeWM encourages the
    projected embeddings to be approximately isotropic Gaussian.  This function
    returns the raw MSE and raw SIGReg separately so the training objective can
    combine them as ``lambda_wm * mse + lambda_sigreg * sigreg`` rather than
    accidentally scheduling SIGReg by ``lambda_wm``.

    Gradient design (no stop-gradient; matches LeWM paper):
    - ``target_emb`` is computed with gradient so ``state_proj`` receives
      gradient from both the predictor MSE and SIGReg paths.
    - The Qwen backbone at the next-latent side may or may not carry gradient
      depending on the caller (``compute_step_wm_loss`` always computes
      ``next_latent`` under ``torch.no_grad()``, so only the projector gets
      gradient on the target side).

    Trajectory SIGReg:
      When ``items`` is provided and contains ``record_id`` / ``step_index``
      metadata, this function groups transitions by trajectory and runs SIGReg
      on the full ordered sequence per record (averaging over trajectories).
      When ``items`` is ``None``, falls back to a simple ``(T=2, B, D)``
      pair-wise stack (backward compatible, used by tests and legacy callers).
    """

    # Concatenate current+next hidden to run state_proj once, avoiding
    # SafeBatchNorm1d running-buffer inplace conflict when called twice
    # within the same autograd context.
    cat_hidden = torch.cat([qwen_hidden_at_latent, qwen_hidden_at_next_latent], dim=0)
    cat_emb = state_proj(cat_hidden).float()
    B = qwen_hidden_at_latent.shape[0]
    state_emb = cat_emb[:B]
    target_emb = cat_emb[B:]

    pred = wm_predictor(state_emb, action_indices)
    mse = F.mse_loss(pred, target_emb)

    sigreg_loss: torch.Tensor | None = None
    metrics: dict[str, float] = {"wm_mse": float(mse.detach().item())}
    if sigreg_module is not None:
        if items is not None:
            # Trajectory-aware SIGReg: one (T_i, 1, D) input per trajectory.
            sigreg_inputs = _build_trajectory_sigreg_inputs(items, state_emb, target_emb)
            if sigreg_inputs:
                sigreg_losses = [sigreg_module(inp) for inp in sigreg_inputs]
                sigreg_loss = torch.stack(sigreg_losses).mean()
            else:
                # No trajectories to group; fall back to pair stack.
                sigreg_input = torch.stack([state_emb, target_emb], dim=0)
                sigreg_loss = sigreg_module(sigreg_input)
        else:
            # Legacy fallback: (T=2, B, D) pair-wise stack.
            sigreg_input = torch.stack([state_emb, target_emb], dim=0)
            sigreg_loss = sigreg_module(sigreg_input)
        metrics["sigreg_loss"] = float(sigreg_loss.detach().item())

    return mse, sigreg_loss, metrics


def compute_value_loss(
    *,
    state_emb: torch.Tensor,
    action_indices: torch.Tensor,
    action_value_targets: torch.Tensor,
    value_head: ValueHead,
    rank_margin: float = 0.1,
    lambda_rank: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regression on chosen-action value + margin ranking vs unchosen actions."""

    values = value_head(state_emb).float()
    chosen_values = values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    targets = action_value_targets.to(device=values.device, dtype=values.dtype)
    reg_loss = F.mse_loss(chosen_values, targets)

    mask = F.one_hot(action_indices, num_classes=values.shape[1]).bool()
    other_values = values.masked_fill(mask, float("-inf"))
    max_other = other_values.max(dim=1).values
    rank_loss = F.relu(rank_margin + max_other - chosen_values).mean()

    total = reg_loss + lambda_rank * rank_loss
    metrics = {
        "value_reg": float(reg_loss.detach().item()),
        "value_rank": float(rank_loss.detach().item()),
        "value_total": float(total.detach().item()),
    }
    return total, metrics


def wm_loss_weight_schedule(
    global_step: int,
    total_steps: int,
    start: float = 0.1,
    end: float = 1.0,
    warmup_fraction: float = 0.3,
) -> float:
    """Cosine ramp for λ_wm over the first warmup_fraction of training."""

    if total_steps <= 0:
        return end
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if global_step >= warmup_steps:
        return end
    progress = global_step / warmup_steps
    cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
    return start + (end - start) * cosine


def compute_combined_loss(
    *,
    wm_loss: torch.Tensor | None,
    value_loss: torch.Tensor | None,
    lm_loss: torch.Tensor | None,
    lambda_wm: float,
    sigreg_loss: torch.Tensor | None = None,
    lambda_sigreg: float = 0.0,
    lambda_value: float = 1.0,
    lambda_ce: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {
        "lambda_wm": float(lambda_wm),
        "lambda_sigreg": float(lambda_sigreg),
        "lambda_value": float(lambda_value),
        "lambda_ce": float(lambda_ce),
    }
    device = lm_loss.device if lm_loss is not None else None
    if device is None:
        for candidate in (wm_loss, value_loss):
            if candidate is not None:
                device = candidate.device
                break
    total = torch.zeros((), device=device or "cpu")

    if wm_loss is not None:
        total = total + lambda_wm * wm_loss.to(total.device)
        metrics["wm_mse"] = float(wm_loss.detach().item())
    if sigreg_loss is not None and lambda_sigreg > 0.0:
        total = total + lambda_sigreg * sigreg_loss.to(total.device)
        metrics["sigreg_loss"] = float(sigreg_loss.detach().item())
    if value_loss is not None:
        total = total + lambda_value * value_loss.to(total.device)
        metrics["value_total"] = float(value_loss.detach().item())
    if lm_loss is not None:
        total = total + lambda_ce * lm_loss.to(total.device)
        metrics["lm_ce"] = float(lm_loss.detach().item())
    metrics["total_loss"] = float(total.detach().item())
    return total, metrics
