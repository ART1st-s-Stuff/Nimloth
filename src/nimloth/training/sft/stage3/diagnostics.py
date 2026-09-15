"""No-update diagnostics of the production Stage3 predictor objective."""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone


class DINOFeatureWriter:
    """Save complete spatial grids from production evaluation batches."""

    schema = "stage3_dino_feature_batch_v1"

    def __init__(self, directory: Path, *, rank: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.batch_index = 0
        self.paths: list[Path] = []

    def __call__(self, batch, output) -> None:
        diagnostic = output.diagnostics or {}
        required = {
            "predicted_states",
            "target_states",
            "dino_targets",
            "current_dino_targets",
        }
        missing = sorted(required - diagnostic.keys())
        if missing:
            raise ValueError(f"DINO feature export missing diagnostics: {missing}")
        horizon = int(batch.prediction_horizon)
        predicted = diagnostic["predicted_states"].detach().float().cpu()
        dino = diagnostic["dino_targets"].detach().float().cpu()
        direct = diagnostic["target_states"].detach().float().cpu()
        if predicted.shape != dino.shape or direct.shape != dino.shape:
            raise ValueError("predicted, direct, and DINO grids must have identical shapes")
        shape = (batch.batch_size, horizon, *predicted.shape[-2:])
        valid = batch.sample_weights.detach().bool().cpu()
        payload = {
            "schema": self.schema,
            "rank": self.rank,
            "batch_index": self.batch_index,
            "keys": [tuple(key) for index, key in enumerate(batch.current_keys) if valid[index]],
            "actions": batch.action_sequences.detach().cpu()[valid],
            "predicted": predicted.reshape(shape)[valid],
            "direct": direct.reshape(shape)[valid],
            "dino": dino.reshape(shape)[valid],
            "current_dino": diagnostic["current_dino_targets"].detach().float().cpu()[valid],
        }
        path = self.directory / f"rank_{self.rank:03d}_batch_{self.batch_index:04d}.pt"
        torch.save(payload, path)
        self.paths.append(path)
        self.batch_index += 1


class _PredictorDiagnosticBackbone(Backbone):
    """Keep exact backbone values and FSDP gathers, without an unused Qwen graph."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    @property
    def model(self):
        return self.inner.model

    def forward(self, batch, *, include_lm_loss=False):
        with torch.no_grad():
            return self.inner(batch, include_lm_loss=include_lm_loss)

    def with_model(self, model):
        return _PredictorDiagnosticBackbone(self.inner.with_model(model))

    def save_pretrained(self, *args, **kwargs):
        raise RuntimeError("diagnostic backbone view is not an exportable artifact")



def outcome_gradient_diagnostic(algorithm, runtime, batch, *, wm_weight: float) -> dict:
    """Measure rank-local first-microbatch gradients; preserve RNG and .grad.

    This does not estimate the full accumulated/global update. In particular, neither
    manually synchronized gradients nor a ratio cutoff enter the training objective.
    """
    if algorithm.outcome_weight <= 0:
        raise ValueError("outcome gradient diagnostic requires an active BCE objective")
    isolated = runtime.unwrapped()
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(isolated.agent.backbone.model, FSDP):
        isolated = replace(isolated, agent=Agent(
            backbone=_PredictorDiagnosticBackbone(isolated.agent.backbone),
            wm=isolated.agent.wm,
        ))
    parameters = [p for p in isolated.agent.wm.wm_predictor.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("outcome gradient diagnostic requires trainable predictor parameters")
    devices = sorted({p.device.index for p in isolated.agent.parameters() if p.is_cuda})
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            output = algorithm.training_primary_step(isolated, batch, wm_weight=wm_weight)
            outcome = output.losses.get("outcome")
            dino = output.losses.get("dino")
            if outcome is None or dino is None:
                raise ValueError("gradient diagnostic requires outcome and DINO losses")
            primary = wm_weight * output.losses["wm"] + algorithm.dino_grid_weight * dino
            auxiliary = algorithm.outcome_weight * outcome
            if not torch.isfinite(primary) or not torch.isfinite(auxiliary):
                raise ValueError("non-finite gradient diagnostic loss")
            gradients = [
                torch.autograd.grad(loss, parameters, retain_graph=index == 0, allow_unused=True)
                for index, loss in enumerate((primary, auxiliary))
            ]
            norms = []
            for values in gradients:
                if any(not torch.isfinite(value).all() for value in values if value is not None):
                    raise ValueError("non-finite predictor diagnostic gradient")
                norms.append(sum(float(value.detach().double().square().sum().item())
                                 for value in values if value is not None) ** 0.5)
            # Hash token identity, not raw prompt content, for matching canary inputs.
            ids = batch.inputs.tensors["input_ids"].detach().cpu().contiguous()
            digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
            return {
                "diagnostic": "outcome_predictor_gradient_ratio_v1",
                "scope": "rank_local_first_microbatch",
                "input_ids_sha256": digest,
                "wm_weight": float(wm_weight),
                "dino_weight": float(algorithm.dino_grid_weight),
                "outcome_weight": float(algorithm.outcome_weight),
                "wm_dino_loss": float(primary.detach()),
                "weighted_outcome_loss": float(auxiliary.detach()),
                "wm_dino_gradient_norm": norms[0],
                "outcome_gradient_norm": norms[1],
                "outcome_to_wm_dino_gradient_ratio": norms[1] / norms[0] if norms[0] else None,
                "zero_reference_gradient": norms[0] == 0,
            }
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
