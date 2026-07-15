"""Evaluation and rendering for 8192-vs-2048 SFT2 dynamics embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from nimloth.eval.matched_wm_metrics import load_state_records
from nimloth.eval.matched_wm_render import adapt_states, matched_noise, sample_conditions
from nimloth.eval.matched_wm_turns import TurnBatch, write_turn_artifacts
from nimloth.rcdm.image_utils import image_to_diffusion_tensor
from nimloth.wm.dynamics_dim_heads import DynamicsDimWMHeads

DYNAMICS_DIM_COLUMNS = ("GT", "Qwen positive", "Frozen State GT", "Full dynamics8192 WM", "Factorized dynamics2048 WM")


def _window_at(steps: dict, start: int, horizon: int):
    required = range(start, start + horizon + 1)
    if not all(step in steps for step in required):
        return None
    actions = torch.tensor([int(steps[step]["action_index"]) for step in range(start, start + horizon)])
    return steps[start]["state_emb"], actions, steps[start + horizon]["state_emb"]


def _windows(records: dict, horizon: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    output = []
    for steps in records.values():
        for start in sorted(steps):
            window = _window_at(steps, start, horizon)
            if window is not None:
                output.append(window)
    return output


def _batch(windows: list, start: int, size: int, device: torch.device):
    items = windows[start : start + size]
    state = torch.stack([item[0].reshape(-1) for item in items]).to(device=device, dtype=torch.float32)
    actions = torch.stack([item[1] for item in items]).to(device)
    target = torch.stack([item[2].reshape(-1) for item in items]).to(device=device, dtype=torch.float32)
    return state, actions, target


def _values(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred, truth = prediction.float(), target.float()
    mse = torch.nn.functional.mse_loss(pred, truth, reduction="none").mean(1).sum()
    cosine = torch.nn.functional.cosine_similarity(pred, truth).sum()
    return float(mse.cpu()), float(cosine.cpu())


def _totals() -> dict[str, dict[str, float]]:
    return {name: {"mse": 0.0, "cosine": 0.0} for name in ("full", "factorized")}


def _add(totals: dict, predictions: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> None:
    for name, prediction in zip(("full", "factorized"), predictions, strict=True):
        mse, cosine = _values(prediction, target)
        totals[name]["mse"] += mse
        totals[name]["cosine"] += cosine


def _autocast(device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


def _add_rollout_control(wrong: dict, heads, state: torch.Tensor, actions: torch.Tensor, target: torch.Tensor) -> None:
    if actions.shape[1] != 1:
        return
    rollouts = heads.rollout(state, actions.roll(1, dims=0))
    _add(wrong, tuple(output[:, -1] for output in rollouts), target)


def _evaluate_horizon(heads, windows: list, device: torch.device, batch_size: int) -> dict[str, Any]:
    totals, wrong = _totals(), _totals()
    horizon = int(windows[0][1].shape[0])
    with torch.inference_mode(), _autocast(device):
        for start in range(0, len(windows), batch_size):
            state, actions, target = _batch(windows, start, batch_size, device)
            rollouts = heads.rollout(state, actions)
            _add(totals, tuple(output[:, -1] for output in rollouts), target)
            _add_rollout_control(wrong, heads, state, actions, target)
    result = {name: {key: value / len(windows) for key, value in metrics.items()} for name, metrics in totals.items()}
    if horizon == 1:
        for name in result:
            result[name].update({f"shuffled_{key}": value / len(windows) for key, value in wrong[name].items()})
    output = {"count": len(windows), "mode": "autoregressive_rollout", **result}
    if horizon == 1:
        output["shuffled_mode"] = "autoregressive_rollout"
    return output


def _evaluate_direct(heads, windows: list, device: torch.device, batch_size: int) -> dict[str, Any]:
    totals, wrong = _totals(), _totals()
    with torch.inference_mode(), _autocast(device):
        for start in range(0, len(windows), batch_size):
            state, actions, target = _batch(windows, start, batch_size, device)
            _add(totals, heads.predict_next(state, actions[:, 0]), target)
            _add(wrong, heads.predict_next(state, actions[:, 0].roll(1)), target)
    result = {name: {key: value / len(windows) for key, value in metrics.items()} for name, metrics in totals.items()}
    for name in result:
        result[name].update({f"shuffled_{key}": value / len(windows) for key, value in wrong[name].items()})
    return {"count": len(windows), "mode": "direct_predict_next", **result}


def evaluate_dynamics_dims(heads: DynamicsDimWMHeads, cache_dir: Path, device: torch.device, *, batch_size: int) -> dict[str, Any]:
    records, horizons = load_state_records(cache_dir), {}
    heads.to(device).eval()
    direct = _evaluate_direct(heads, _windows(records, 1), device, batch_size)
    for horizon in range(1, 6):
        windows = _windows(records, horizon)
        if not windows:
            raise ValueError(f"no dynamics-dimension windows for horizon {horizon}")
        horizons[str(horizon)] = _evaluate_horizon(heads, windows, device, batch_size)
    one_step = {name: direct[name] for name in ("full", "factorized")}
    return {"one_step_count": direct["count"], "one_step_mode": direct["mode"], "one_step_control_mode": direct["mode"], "one_step": one_step, "horizons": horizons}


def _wm_states(batch: TurnBatch, heads: DynamicsDimWMHeads, device: torch.device):
    initial = batch.initial_state.reshape(batch.initial_state.shape[0], -1).to(device=device, dtype=torch.float32)
    with torch.inference_mode(), _autocast(device):
        full, factorized = heads.to(device).eval().rollout(initial, batch.actions.to(device))
    shape = (-1, batch.initial_state.shape[1], batch.initial_state.shape[2])
    return full.reshape(shape).cpu(), factorized.reshape(shape).cpu()


def _conditions(batch: TurnBatch, heads, adapter, device, chunk_size: int) -> dict[str, torch.Tensor]:
    full, factorized = _wm_states(batch, heads, device)
    targets = batch.target_states.reshape(-1, *batch.target_states.shape[2:])
    return {
        "Qwen positive": batch.positive_tokens,
        "Frozen State GT": adapt_states(adapter, targets, device, chunk_size),
        "Full dynamics8192 WM": adapt_states(adapter, full, device, chunk_size),
        "Factorized dynamics2048 WM": adapt_states(adapter, factorized, device, chunk_size),
    }


def render_dynamics_dim_comparison(batch: TurnBatch, heads, adapter, cfm, device: torch.device, *, steps: int, cfg_scale: float, chunk_size: int, seed: int):
    conditions = _conditions(batch, heads, adapter, device, chunk_size)
    noise, fingerprint = matched_noise(len(batch.rows), seed=seed)
    generated = {name: sample_conditions(cfm, value, noise, device, steps, cfg_scale, chunk_size) for name, value in conditions.items()}
    gt = torch.stack([image_to_diffusion_tensor(row["gt_image_path"], image_size=128) for row in batch.rows])
    return {"GT": gt, **generated}, fingerprint


def write_dynamics_dim_artifacts(batch: TurnBatch, images: dict[str, torch.Tensor], output_dir: Path, *, seed: int, steps: int, cfg_scale: float, noise_fingerprint: str):
    return write_turn_artifacts(batch, images, output_dir, seed=seed, steps=steps, cfg_scale=cfg_scale, noise_fingerprint=noise_fingerprint, columns=DYNAMICS_DIM_COLUMNS)
