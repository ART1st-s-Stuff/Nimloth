"""No-update diagnostics of the production Stage3 predictor objective."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nimloth.agent import Agent
from nimloth.backbone import Backbone


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DINOFeatureWriter:
    """Save complete spatial grids from production evaluation batches."""

    schema = "stage3_dino_feature_batch_v1"

    def __init__(self, directory: Path, *, rank: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.batch_index = 0
        self.paths: list[Path] = []
        self.batch_identities: list[dict] = []

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
        online_direct = output.online_states[batch.next_indices].detach().float().cpu()
        online_current = output.online_states[batch.current_indices].detach().float().cpu()
        if predicted.shape != dino.shape or direct.shape != dino.shape or online_direct.shape != dino.shape:
            raise ValueError("predicted, direct, and DINO grids must have identical shapes")
        shape = (batch.batch_size, horizon, *predicted.shape[-2:])
        if predicted.numel() != int(np.prod(shape)):
            raise ValueError("feature grids do not match the window/horizon dimensions")
        valid = batch.sample_weights.detach().bool().cpu()
        current_dino = diagnostic["current_dino_targets"].detach().float().cpu()
        if current_dino.shape != (batch.batch_size, *predicted.shape[-2:]):
            raise ValueError("current DINO grids do not match the window dimensions")
        if any(not torch.isfinite(value).all() for value in
               (predicted, dino, direct, online_direct, online_current, current_dino)):
            raise ValueError("non-finite feature grids")
        payload = {
            "schema": self.schema,
            "rank": self.rank,
            "batch_index": self.batch_index,
            "keys": [tuple(key) for index, key in enumerate(batch.current_keys) if valid[index]],
            "actions": batch.action_sequences.detach().cpu()[valid],
            "predicted": predicted.reshape(shape)[valid],
            "direct": direct.reshape(shape)[valid],
            "online_direct": online_direct.reshape(shape)[valid],
            "online_current": online_current[valid],
            "dino": dino.reshape(shape)[valid],
            "current_dino": current_dino[valid],
        }
        path = self.directory / f"rank_{self.rank:03d}_batch_{self.batch_index:04d}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(payload, path)
        self.batch_identities.append({"keys": payload["keys"], "actions": payload["actions"].tolist()})
        self.paths.append(path)
        self.batch_index += 1


class FrozenWMTrajectoryWriter:
    """Export one ordered state/DINO sequence per trajectory for WM-only diagnostics.

    Unlike :class:`DINOFeatureWriter`, this format never materializes overlapping
    windows.  The offline trainer derives every T-step window from the stored
    observation and action sequences.
    """

    schema = "frozen_wm_trajectory_shard_v1"

    def __init__(self, directory: Path, *, rank: int, identity: dict | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.identity = dict(identity or {})
        self.batch_index = 0
        self.paths: list[Path] = []
        self.trajectory_count = 0
        self.window_count = 0
        self.complete_path = self.directory / f"rank_{self.rank:03d}_COMPLETE.json"
        if any(self.directory.glob(f"rank_{self.rank:03d}_batch_*.pt")) or self.complete_path.exists():
            raise FileExistsError(
                f"frozen-WM export already contains rank {self.rank} shards: {self.directory}"
            )

    @staticmethod
    def _trajectory_actions(batch, start: int, end: int, state_count: int) -> torch.Tensor:
        windows = batch.action_sequences[start:end].detach().cpu().long()
        horizon = int(batch.prediction_horizon)
        if windows.shape != (state_count - horizon, horizon):
            raise ValueError("trajectory windows do not cover the state/action sequence")
        actions = torch.cat((windows[:, 0], windows[-1, 1:]), dim=0)
        if actions.shape != (state_count - 1,):
            raise ValueError("reconstructed actions do not align with trajectory states")
        for offset, window in enumerate(windows):
            if not torch.equal(window, actions[offset : offset + horizon]):
                raise ValueError("overlapping action windows disagree")
        return actions

    def __call__(self, batch, output) -> None:
        if batch.observed_dino_target is None:
            raise ValueError("frozen-WM export requires DINO targets for every observation")
        states = output.online_states.detach().float().cpu()
        dino = batch.observed_dino_target.detach().float().cpu()
        if states.shape != dino.shape or states.ndim != 3:
            raise ValueError(
                "frozen Stage2 states and DINO grids must have identical [N,K,D] shape"
            )
        if len(batch.state_keys) != states.shape[0]:
            raise ValueError("state identities do not align with exported grids")
        if not torch.isfinite(states).all() or not torch.isfinite(dino).all():
            raise ValueError("frozen-WM export contains non-finite grids")

        records = []
        for trajectory, left, right, start, end in zip(
            batch.trajectory_ids,
            batch.state_offsets[:-1],
            batch.state_offsets[1:],
            batch.window_offsets[:-1],
            batch.window_offsets[1:],
            strict=True,
        ):
            weights = batch.sample_weights[start:end].detach().cpu()
            if not torch.all(weights == weights[0]):
                raise ValueError("trajectory windows disagree on padding identity")
            if not bool(weights[0]):
                continue
            state_count = right - left
            keys = tuple(batch.state_keys[left:right])
            if keys != tuple((trajectory, step) for step in range(state_count)):
                raise ValueError("trajectory state identities are not contiguous")
            actions = self._trajectory_actions(batch, start, end, state_count)
            records.append(
                {
                    "trajectory_id": str(trajectory),
                    "states": states[left:right].contiguous(),
                    "dino": dino[left:right].contiguous(),
                    "actions": actions.contiguous(),
                }
            )
        if not records:
            return
        payload = {
            "schema": self.schema,
            "rank": self.rank,
            "batch_index": self.batch_index,
            "dtype": "float32",
            "identity_json": json.dumps(self.identity, sort_keys=True, separators=(",", ":")),
            "prediction_horizon": int(batch.prediction_horizon),
            "records": records,
        }
        path = self.directory / f"rank_{self.rank:03d}_batch_{self.batch_index:06d}.pt"
        if path.exists():
            raise FileExistsError(path)
        torch.save(payload, path)
        self.paths.append(path)
        self.trajectory_count += len(records)
        self.window_count += sum(len(record["states"]) - int(batch.prediction_horizon) for record in records)
        self.batch_index += 1

    def finalize(self) -> Path:
        """Atomically mark this rank complete after the full loader returns."""
        if self.complete_path.exists():
            raise FileExistsError(self.complete_path)
        payload = {
            "schema": "frozen_wm_rank_complete_v1",
            "rank": self.rank,
            "identity_json": json.dumps(self.identity, sort_keys=True, separators=(",", ":")),
            "batch_count": self.batch_index,
            "trajectory_count": self.trajectory_count,
            "window_count": self.window_count,
            "shards": [
                {
                    "path": path.name,
                    "sha256": _file_sha256(path),
                }
                for path in self.paths
            ],
        }
        temporary = self.complete_path.with_name(self.complete_path.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.complete_path)
        return self.complete_path


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
