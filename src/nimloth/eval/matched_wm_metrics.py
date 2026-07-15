"""Full-cache one-step and autoregressive horizon metrics for matched WM heads."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from nimloth.rcdm.state_cache import RCDMStateCacheDataset
from nimloth.wm.frozen_query_state import StateViews
from nimloth.wm.matched_heads import MatchedWMHeads


def load_state_records(cache_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    dataset = RCDMStateCacheDataset(cache_dir)
    records: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for index in range(len(dataset)):
        item = dataset[index]
        record, step = str(item["record_id"]), int(item["step_index"])
        if step in records[record]:
            raise ValueError(f"duplicate frozen State row: {record} step{step}")
        records[record][step] = item
    return records


def _window_at(steps: dict, start: int, horizon: int):
    required = range(start, start + horizon + 1)
    if not all(step in steps for step in required):
        return None
    actions = torch.tensor([int(steps[step]["action_index"]) for step in range(start, start + horizon)])
    return steps[start]["state_emb"], actions, steps[start + horizon]["state_emb"]


def _windows(records: dict, horizon: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    windows = []
    for steps in records.values():
        for start in sorted(steps):
            window = _window_at(steps, start, horizon)
            if window is not None:
                windows.append(window)
    return windows


def _values(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    pred, truth = prediction.float().flatten(1), target.float().flatten(1)
    mse = torch.nn.functional.mse_loss(pred, truth, reduction="none").mean(1).sum()
    cosine = torch.nn.functional.cosine_similarity(pred, truth).sum()
    return float(mse.cpu()), float(cosine.cpu())


def _batch_windows(windows: list, start: int, size: int, device: torch.device) -> tuple[StateViews, torch.Tensor, torch.Tensor]:
    items = windows[start : start + size]
    state = torch.stack([item[0] for item in items]).to(device)
    actions = torch.stack([item[1] for item in items]).to(device)
    target = torch.stack([item[2] for item in items]).to(device)
    return StateViews.from_tokens(state.contiguous()), actions, target


def _branch_totals() -> dict[str, dict[str, float]]:
    return {name: {"mse": 0.0, "cosine": 0.0} for name in ("vector", "token")}


def _add_predictions(totals: dict, predictions: tuple[torch.Tensor, torch.Tensor], target: torch.Tensor) -> None:
    targets = (target.reshape(target.shape[0], 1, -1), target)
    for name, prediction, truth in zip(("vector", "token"), predictions, targets, strict=True):
        mse, cosine = _values(prediction, truth)
        totals[name]["mse"] += mse
        totals[name]["cosine"] += cosine


def _evaluate_horizon(heads: MatchedWMHeads, windows: list, device: torch.device, batch_size: int) -> dict[str, Any]:
    totals, wrong = _branch_totals(), _branch_totals()
    horizon = int(windows[0][1].shape[0])
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            views, actions, target = _batch_windows(windows, start, batch_size, device)
            rollouts = heads.rollout(views, actions)
            _add_predictions(totals, tuple(output[:, -1] for output in rollouts), target)
            if horizon == 1:
                shuffled = heads.predict_next(views, actions[:, 0].roll(1))
                _add_predictions(wrong, shuffled, target)
    result = {name: {key: value / len(windows) for key, value in metrics.items()} for name, metrics in totals.items()}
    if horizon == 1:
        for name in result:
            result[name].update({f"shuffled_{key}": value / len(windows) for key, value in wrong[name].items()})
    return {"count": len(windows), **result}


def evaluate_full_dynamics(heads: MatchedWMHeads, cache_dir: Path, device: torch.device, *, batch_size: int) -> dict[str, Any]:
    records = load_state_records(cache_dir)
    heads.to(device).eval()
    horizons = {}
    for horizon in range(1, 6):
        windows = _windows(records, horizon)
        if not windows:
            raise ValueError(f"no full-cache windows for horizon {horizon}")
        horizons[str(horizon)] = _evaluate_horizon(heads, windows, device, batch_size)
    one_step = {name: horizons["1"][name] for name in ("vector", "token")}
    return {"one_step_count": horizons["1"]["count"], "one_step": one_step, "horizons": horizons}
