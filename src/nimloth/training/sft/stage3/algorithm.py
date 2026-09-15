"""Trajectory-native Stage3: encode once, supervise every eligible WM window."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch

from nimloth.agent.model import AgentStateOutput
from nimloth.training.common.value_semantics import SFT2_VALUE_OBJECTIVE
from nimloth.training.sft.stage3.batch import Stage3TrajectoryBatch
from nimloth.training.sft.stage3.runtime import SFT2ModelRuntime
from nimloth.training.sft.stage3.sigreg import gather_global_sigreg_states, shared_sigreg_rng
from nimloth.wm import LatentWMPredictor, SequenceSIGReg


def require_sft2_wm_history(wm_predictor: LatentWMPredictor, *, history_size: int, source: Path) -> None:
    actual = int(wm_predictor.config.history_size)
    if actual != int(history_size):
        raise ValueError(f"SFT2 WM checkpoint history_size does not match config: checkpoint={actual}, "
                         f"config={history_size}, source={source}")


@dataclass(frozen=True)
class SFT2StepOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor | None]
    metrics: dict[str, float]
    current_state: torch.Tensor
    sample_count: int
    online_states: torch.Tensor
    diagnostics: dict[str, torch.Tensor] | None = None


@dataclass(frozen=True)
class SFT2SIGRegStepOutput:
    loss: torch.Tensor
    raw_loss: torch.Tensor | None
    metrics: dict[str, float]


def _window_mean(terms: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Each eligible start owns one mean, independently of trajectory length."""
    if terms.shape[0] != weights.numel():
        raise ValueError("window loss and weight counts disagree")
    per_window = terms.reshape(terms.shape[0], -1).mean(-1)
    return (per_window * weights).sum() / weights.sum().clamp_min(1)


class SFT2Algorithm:
    """One target encoder and one online encoder per complete-trajectory batch.

    All WM windows reuse the same differentiable online states. Target states
    retain their separate no-grad EMA/eval path. The caller combines primary
    and SIGReg objectives before one ordinary backward through this graph.
    """

    def __init__(self, *, history_size: int, sigreg: SequenceSIGReg | None,
                 sigreg_weight: float, value_weight: float, ce_weight: float,
                 wm_weight_start: float = .1, wm_weight_end: float = 1.,
                 wm_warmup_fraction: float = .3, dino_grid_weight: float = 0.,
                 prediction_horizon: int = 1, outcome_weight: float = 0.) -> None:
        if int(history_size) != 1:
            raise ValueError("trajectory-native Stage3 requires history_size=1")
        if int(prediction_horizon) < 1:
            raise ValueError("prediction_horizon must be positive")
        if not math.isfinite(outcome_weight) or outcome_weight < 0:
            raise ValueError("outcome_weight must be finite and nonnegative")
        self.history_size = 1
        self.prediction_horizon = int(prediction_horizon)
        self.sigreg = sigreg
        self.sigreg_weight = float(sigreg_weight)
        self.value_weight = float(value_weight)
        self.ce_weight = float(ce_weight)
        self.wm_weight_start = float(wm_weight_start)
        self.wm_weight_end = float(wm_weight_end)
        self.wm_warmup_fraction = float(wm_warmup_fraction)
        self.dino_grid_weight = float(dino_grid_weight)
        self.outcome_weight = float(outcome_weight)

    def wm_weight(self, global_step: int, total_steps: int) -> float:
        """在训练前段用 cosine ramp 增加 WM loss 权重。"""

        if total_steps <= 0:
            return self.wm_weight_end
        warmup_steps = max(1, int(total_steps * self.wm_warmup_fraction))
        if global_step >= warmup_steps:
            return self.wm_weight_end
        progress = global_step / warmup_steps
        cosine = 0.5 * (1.0 - math.cos(math.pi * progress))
        return self.wm_weight_start + (
            self.wm_weight_end - self.wm_weight_start
        ) * cosine

    @property
    def has_sigreg_stage(self) -> bool:
        return self.sigreg is not None and self.sigreg_weight > 0.

    def training_primary_step(self, runtime: SFT2ModelRuntime, batch: Stage3TrajectoryBatch,
                              *, wm_weight: float) -> SFT2StepOutput:
        return self._step(runtime, batch, wm_weight=wm_weight)

    def training_sigreg_step(self, runtime: SFT2ModelRuntime,
                             batch: Stage3TrajectoryBatch, *, online_states: torch.Tensor,
                             sigreg_seed: int) -> SFT2SIGRegStepOutput:
        """Regularize each real adjacent transition once using the existing graph.

        Window overlap never defines this population: a trajectory with L actions
        contributes L pairs, including its final transition. Padding contributes
        none. Only successor occurrences receive SIGReg gradients.
        """
        if not self.has_sigreg_stage:
            raise RuntimeError("Stage3 SIGReg is disabled")
        if online_states.shape[0] != len(batch.state_keys):
            raise ValueError("SIGReg online states must cover the complete trajectory batch")
        current, valid = [], []
        seen = set()
        for record, left, right, start, end in zip(
                batch.trajectory_ids, batch.state_offsets[:-1], batch.state_offsets[1:],
                batch.window_offsets[:-1], batch.window_offsets[1:], strict=True):
            weights = batch.sample_weights[start:end]
            if not torch.all(weights == weights[0]):
                raise ValueError("complete trajectory windows must share one padding weight")
            active = bool(weights[0])
            if active and record in seen:
                raise ValueError("duplicate valid trajectory in SIGReg batch")
            if active:
                seen.add(record)
            current.extend(range(left, right - 1))
            valid.extend([active] * (right - left - 1))
        indices = torch.tensor(current, dtype=torch.long, device=online_states.device)
        mask = torch.tensor(valid, dtype=torch.bool, device=online_states.device)
        current_state = runtime.agent.wm.sigreg_state(online_states[indices].detach())
        next_state = runtime.agent.wm.sigreg_state(online_states[indices + 1])
        global_current, global_next, count = gather_global_sigreg_states(
            current_state, next_state, mask)
        with shared_sigreg_rng(sigreg_seed, global_next.device):
            raw_loss = self._sigreg_loss(global_current, global_next)
        if raw_loss is None:
            # Keep gather backward collectives identical even on all-padding ranks.
            loss = global_next.sum() * 0.0
            metrics = {"sigreg_skipped_small_batch": 1.0}
        else:
            loss = self.sigreg_weight * raw_loss
            metrics = {"sigreg_loss": float(raw_loss.detach())}
        metrics["sigreg_global_batch_size"] = float(count)
        return SFT2SIGRegStepOutput(loss, raw_loss, metrics)

    def evaluation_step(self, runtime: SFT2ModelRuntime,
                        batch: Stage3TrajectoryBatch) -> SFT2StepOutput:
        return self._step(runtime, batch, wm_weight=1.)

    def _step(self, runtime: SFT2ModelRuntime, batch: Stage3TrajectoryBatch,
              *, wm_weight: float) -> SFT2StepOutput:
        if batch.prediction_horizon != self.prediction_horizon:
            raise ValueError("trajectory prediction horizon does not match algorithm")
        # EMA swaps must complete before creating any online autograd graph.
        target_states = runtime.encode_next_state(batch.inputs)
        encoded = runtime.agent.backbone(batch.inputs, include_lm_loss=True)
        online_states = runtime.agent.wm.project_state(encoded.hidden)
        if online_states.shape != target_states.shape:
            raise ValueError("online and EMA trajectory state shapes disagree")
        if encoded.lm_losses is None or encoded.lm_losses.shape != batch.lm_weights.shape:
            raise ValueError("trajectory forward requires one LM loss per eligible window")
        lm_loss = (encoded.lm_losses * batch.lm_weights).sum() / batch.lm_weights.sum().clamp_min(1)
        current = AgentStateOutput(hidden=encoded.hidden[batch.current_indices],
                                   state=online_states[batch.current_indices], lm_loss=lm_loss)
        rollout = runtime.agent.forward_action_rollout(batch.action_sequences, encoded_current=current)
        expected = target_states[batch.next_indices]
        predicted = rollout.predicted_states
        if predicted.shape != expected.shape:
            raise ValueError("WM predictions must exactly match future target states")
        weights = batch.sample_weights
        wm = _window_mean((predicted - expected).square(), weights)
        selected_values = rollout.action_values.gather(-1, batch.action_sequences.unsqueeze(-1)).squeeze(-1)
        if selected_values.shape != batch.value_targets.shape:
            raise ValueError("outgoing action values and MC targets must have identical shapes")
        value = _window_mean((selected_values - batch.value_targets.to(selected_values)).square(), weights)
        dino = None
        if batch.dino_grid_target is not None:
            if predicted.shape != batch.dino_grid_target.shape:
                raise ValueError("DINO target shape must exactly match predicted states")
            dino = _window_mean((predicted.float() - batch.dino_grid_target.detach().float()).square(), weights)
        elif self.dino_grid_weight:
            raise ValueError("positive DINO-grid weight requires a DINO-grid target")
        head = getattr(runtime.agent.wm, "outcome_head", None)
        logits = head(predicted) if head is not None else None
        outcome = self._outcome_loss(batch, logits)
        total = wm_weight * wm + self.value_weight * value + self.ce_weight * lm_loss
        if dino is not None:
            total = total + self.dino_grid_weight * dino
        if outcome is not None:
            total = total + self.outcome_weight * outcome
        losses = {"lm": lm_loss, "wm": wm, "value": value, "dino": dino, "outcome": outcome}
        count = int(weights.sum().item())
        metrics = {"wm_mse": float(wm.detach()), "value_mc_mse": float(value.detach()),
                   "value_total": float(value.detach()), "lm_ce": float(lm_loss.detach()),
                   "lambda_wm": float(wm_weight), "lambda_sigreg": 0.,
                   "lambda_value": self.value_weight, "lambda_ce": self.ce_weight,
                   "lambda_dino": self.dino_grid_weight, "context_length": 1.,
                   "prediction_horizon": float(self.prediction_horizon),
                   "current_batch_size": float(count), "total_loss": float(total.detach())}
        if dino is not None:
            metrics["dino_grid_mse"] = float(dino.detach())
        if outcome is not None:
            metrics["outcome_bce"] = float(outcome.detach())
            metrics["outcome_count"] = float((batch.outcome_mask & (weights[:, None] > 0)).sum())
        diagnostics = {"predicted_states": predicted.detach(), "target_states": expected.detach()}
        if batch.dino_grid_target is not None:
            diagnostics["dino_targets"] = batch.dino_grid_target.detach()
        if batch.current_dino_target is not None:
            diagnostics["current_dino_targets"] = batch.current_dino_target.detach()
        if logits is not None:
            diagnostics["outcome_logits"] = logits.detach()
        return SFT2StepOutput(total, losses, metrics, current.state, count, online_states, diagnostics)

    def _outcome_loss(self, batch: Stage3TrajectoryBatch, logits: torch.Tensor | None):
        if not self.outcome_weight:
            return None
        if logits is None:
            raise ValueError("outcome supervision requires an outcome head")
        targets = batch.outcome_targets
        mask = batch.outcome_mask & (batch.sample_weights[:, None] > 0)
        if logits.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("outcome predictions and labels must have identical shape")
        if not torch.isfinite(targets[mask]).all() or not ((targets[mask] == 0) | (targets[mask] == 1)).all():
            raise ValueError("outcome targets must be finite binary labels")
        safe = torch.where(mask, targets, torch.zeros_like(targets))
        terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe, reduction="none")
        return (terms * mask).sum() / mask.sum().clamp_min(1)

    def merge_training_metrics(self, primary_metrics, sigreg):
        metrics = dict(primary_metrics)
        metrics["lambda_sigreg"] = self.sigreg_weight if sigreg is not None else 0.
        if sigreg is not None:
            metrics.update(sigreg.metrics)
            metrics["total_loss"] += float(sigreg.loss.detach())
        return metrics

    def _sigreg_loss(self, current_state, next_state):
        if not self.has_sigreg_stage:
            return None
        if current_state.ndim != 2 or current_state.shape != next_state.shape:
            raise ValueError("SIGReg requires matching [B,D] current and next states")
        return self.sigreg(torch.stack((current_state, next_state), dim=1))


__all__ = ["SFT2_VALUE_OBJECTIVE", "SFT2Algorithm", "SFT2SIGRegStepOutput",
           "SFT2StepOutput", "require_sft2_wm_history"]
