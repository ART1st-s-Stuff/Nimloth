"""Read-only SFT1/ID74 backbone-projector checkpoint compatibility matrix.

The evaluator uses actual pre-RL validation observations and recorded CoT.  It
never runs an optimizer, backward pass, model generation, or parameter update.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_SOURCES = (
    "navigation_base_val_from_train",
    "navigation_common_val_from_train",
    "navigation_long_horizon_val_from_train",
)


def _stable_key(record: dict[str, Any]) -> str:
    return hashlib.sha256(str(record["id"]).encode("utf-8")).hexdigest()


def select_early_transition_records(
    records: Sequence[dict[str, Any]],
    *,
    per_source: int,
    max_step_index: int = 3,
    expected_sources: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic, source-balanced early transitions.

    Every selected transition has an exact following decision state/action.
    Within each source, candidates from steps 0..``max_step_index`` are
    consumed round-robin by executed action after stable hash ordering. At
    most one transition is selected per trajectory.
    """

    if per_source < 1:
        raise ValueError("per_source must be positive")
    if max_step_index < 0:
        raise ValueError("max_step_index must be nonnegative")
    eligible = [row for row in records if len(row.get("action_indices", ())) >= 2]
    if not eligible:
        raise ValueError("selection requires trajectories with at least two actions")
    sources = tuple(expected_sources or sorted({str(row["data_source"]) for row in eligible}))
    selected: list[dict[str, Any]] = []
    for source in sources:
        source_records = [row for row in eligible if str(row.get("data_source")) == source]
        if len(source_records) < per_source:
            raise ValueError(
                f"source {source!r} has fewer than {per_source} trajectories "
                "with at least two actions"
            )
        by_action: dict[int, list[tuple[dict[str, Any], int]]] = defaultdict(list)
        for row in source_records:
            actions = list(row["action_indices"])
            for step_index in range(min(max_step_index + 1, len(actions) - 1)):
                by_action[int(actions[step_index])].append((row, step_index))
        for candidates in by_action.values():
            candidates.sort(key=lambda item: (_stable_key(item[0]), item[1]))
        source_selected: list[dict[str, Any]] = []
        used_records: set[str] = set()
        action_ids = sorted(by_action)
        offsets = {action_id: 0 for action_id in action_ids}
        while len(source_selected) < per_source:
            progressed = False
            for action_id in action_ids:
                candidates = by_action[action_id]
                while offsets[action_id] < len(candidates):
                    row, step_index = candidates[offsets[action_id]]
                    offsets[action_id] += 1
                    record_id = str(row["id"])
                    if record_id in used_records:
                        continue
                    selected_row = dict(row)
                    selected_row["_selected_step_index"] = int(step_index)
                    source_selected.append(selected_row)
                    used_records.add(record_id)
                    progressed = True
                    break
                if len(source_selected) == per_source:
                    break
            if not progressed:  # pragma: no cover - protected by source count
                raise RuntimeError(f"selection stalled for source {source!r}")
        selected.extend(source_selected)
    return selected


def _finite(name: str, value: np.ndarray, *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return array


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.sum(left * right) / max(denominator, 1e-12))


def _skill(prediction: np.ndarray, baseline: np.ndarray, target: np.ndarray) -> float:
    prediction_mse = float(np.mean(np.square(prediction - target)))
    baseline_mse = float(np.mean(np.square(baseline - target)))
    return float(1.0 - prediction_mse / max(baseline_mse, 1e-12))


def _state_statistics(state: np.ndarray) -> dict[str, float]:
    slot_center = state.mean(axis=-2, keepdims=True)
    return {
        "mean": float(state.mean()),
        "std": float(state.std()),
        "rms": float(np.sqrt(np.mean(np.square(state)))),
        "slot_deviation_rms": float(np.sqrt(np.mean(np.square(state - slot_center)))),
    }


def _state_dino(prefix: str, state: np.ndarray, dino: np.ndarray) -> dict[str, float]:
    centered_state = state - state.mean(axis=-1, keepdims=True)
    centered_dino = dino - dino.mean(axis=-1, keepdims=True)
    token_denominator = np.linalg.norm(centered_state, axis=-1) * np.linalg.norm(
        centered_dino, axis=-1
    )
    token_cosine = np.sum(centered_state * centered_dino, axis=-1) / np.maximum(
        token_denominator, 1e-12
    )
    return {
        f"{prefix}_rmse": _rmse(state, dino),
        f"{prefix}_cosine": _cosine(state.reshape(-1), dino.reshape(-1)),
        f"{prefix}_token_centered_cosine": float(token_cosine.mean()),
    }


def _executed_values(q: np.ndarray, actions: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or actions.ndim != 1 or q.shape[0] != actions.shape[0]:
        raise ValueError("Q/actions batch shapes do not align")
    if np.any(actions < 0) or np.any(actions >= q.shape[1]):
        raise ValueError("action index is outside Q action dimension")
    return q[np.arange(q.shape[0]), actions]


def combination_metrics(
    *,
    current_state: np.ndarray,
    actual_next_state: np.ndarray,
    predicted_next_state: np.ndarray,
    current_dino: np.ndarray,
    next_dino: np.ndarray,
    current_q: np.ndarray,
    actual_next_q: np.ndarray,
    predicted_next_q: np.ndarray,
    current_actions: np.ndarray,
    next_actions: np.ndarray,
    current_returns: np.ndarray,
    next_returns: np.ndarray,
) -> dict[str, float]:
    """Compute one matrix cell's unified-state, WM, DINO, and Q metrics."""

    current = _finite("current_state", current_state, ndim=3)
    actual_next = _finite("actual_next_state", actual_next_state, ndim=3)
    predicted_next = _finite("predicted_next_state", predicted_next_state, ndim=3)
    dino_current = _finite("current_dino", current_dino, ndim=3)
    dino_next = _finite("next_dino", next_dino, ndim=3)
    expected_shape = current.shape
    for name, array in (
        ("actual_next_state", actual_next),
        ("predicted_next_state", predicted_next),
        ("current_dino", dino_current),
        ("next_dino", dino_next),
    ):
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} != {expected_shape}")

    current_q_array = _finite("current_q", current_q, ndim=2)
    actual_next_q_array = _finite("actual_next_q", actual_next_q, ndim=2)
    predicted_next_q_array = _finite("predicted_next_q", predicted_next_q, ndim=2)
    actions = np.asarray(current_actions, dtype=np.int64)
    following_actions = np.asarray(next_actions, dtype=np.int64)
    returns = _finite("current_returns", current_returns, ndim=1)
    following_returns = _finite("next_returns", next_returns, ndim=1)
    current_values = _executed_values(current_q_array, actions)
    actual_next_values = _executed_values(actual_next_q_array, following_actions)
    predicted_next_values = _executed_values(predicted_next_q_array, following_actions)

    result: dict[str, float] = {
        "behavior_copy_rmse": _rmse(current, actual_next),
        "behavior_predicted_rmse": _rmse(predicted_next, actual_next),
        "behavior_predicted_vs_copy_skill": _skill(predicted_next, current, actual_next),
        "dino_predicted_vs_copy_skill": _skill(predicted_next, current, dino_next),
        "dino_actual_next_vs_copy_skill": _skill(actual_next, current, dino_next),
        "value_current_executed_rmse": _rmse(current_values, returns),
        "value_actual_next_executed_rmse": _rmse(actual_next_values, following_returns),
        "value_predicted_next_executed_rmse": _rmse(predicted_next_values, following_returns),
        "value_current_executed_mean": float(current_values.mean()),
        "value_actual_next_executed_mean": float(actual_next_values.mean()),
        "value_predicted_next_executed_mean": float(predicted_next_values.mean()),
        "current_return_mean": float(returns.mean()),
        "next_return_mean": float(following_returns.mean()),
    }
    result.update(_state_dino("current_dino", current, dino_current))
    result.update(_state_dino("actual_next_dino", actual_next, dino_next))
    result.update(_state_dino("predicted_next_dino", predicted_next, dino_next))
    for prefix, state in (
        ("current_state", current),
        ("actual_next_state", actual_next),
        ("predicted_next_state", predicted_next),
    ):
        result.update({f"{prefix}_{key}": value for key, value in _state_statistics(state).items()})
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_files(root: Path) -> list[dict[str, Any]]:
    candidates = sorted(root.glob("model*.safetensors"))
    for relative in (
        "model.safetensors.index.json",
        "config.json",
        "grid_state_config.json",
        "slot_projector.pt",
        "state_proj.pt",
        "vision_ema.pt",
        "wm_predictor/config.json",
        "wm_predictor/predictor.pt",
        "value_head/value_head.pt",
        "training_state.pt",
    ):
        path = root / relative
        if path.is_file():
            candidates.append(path)
    unique = sorted(set(candidates))
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in unique
    ]


def _backbone_args(model: Path, *, resume: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=str(model),
        max_pixels=100352,
        gradient_checkpointing=False,
        attn_implementation="flash_attention_2",
        llm_tune="freeze",
        vision_tune="freeze",
        lora=False,
        query_tune="freeze",
        resume=bool(resume),
    )


def _chunks(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _build_batches(builder: Any, samples: Sequence[Any], *, next_state: bool, batch_size: int) -> list[Any]:
    batches: list[Any] = []
    for chunk in _chunks(samples, batch_size):
        if next_state:
            messages = [sample.next_prefix_messages for sample in chunk]
            images = [sample.next_prefix_image_paths for sample in chunk]
            if any(value is None for value in messages) or any(value is None for value in images):
                raise ValueError("selected transition is missing exact next-state prompt")
        else:
            messages = [sample.prefix_messages for sample in chunk]
            images = [sample.prefix_image_paths for sample in chunk]
        batches.append(builder.build(messages, images, include_labels=False))
    return batches


def _encode(backbone: Any, batches: Sequence[Any], *, label: str) -> np.ndarray:
    import torch

    backbone.eval()
    outputs: list[np.ndarray] = []
    started = time.monotonic()
    with torch.inference_mode():
        for index, batch in enumerate(batches, start=1):
            hidden = backbone(batch, include_lm_loss=False).hidden
            outputs.append(hidden.detach().float().cpu().numpy())
            print(json.dumps({"encode": label, "batch": index, "batches": len(batches)}), flush=True)
    result = np.concatenate(outputs, axis=0)
    if result.ndim != 3 or result.shape[1:] != (16, 2048) or not np.isfinite(result).all():
        raise ValueError(f"{label} hidden must be finite [B,16,2048], got {result.shape}")
    print(json.dumps({"encode_complete": label, "seconds": time.monotonic() - started}), flush=True)
    return result


def _project(hidden: np.ndarray, projector: Any, *, device: Any, batch_size: int = 32) -> np.ndarray:
    import torch

    projector.eval().to(device)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for chunk in _chunks(list(hidden), batch_size):
            tensor = torch.from_numpy(np.stack(chunk)).to(device=device, dtype=torch.float32)
            outputs.append(projector(tensor).detach().float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _predict_and_value(
    current: np.ndarray,
    actual_next: np.ndarray,
    actions: np.ndarray,
    *,
    predictor: Any,
    value_head: Any,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    current_tensor = torch.from_numpy(current).to(device=device, dtype=torch.float32)
    actual_next_tensor = torch.from_numpy(actual_next).to(device=device, dtype=torch.float32)
    action_tensor = torch.from_numpy(actions).to(device=device, dtype=torch.long)
    previous = torch.empty((len(actions), 0), device=device, dtype=torch.long)
    with torch.inference_mode():
        predicted = predictor.rollout_from_history(
            current_tensor.unsqueeze(1),
            previous,
            action_tensor.unsqueeze(1),
        )[:, 0]
        current_q = value_head(current_tensor.mean(dim=-2))
        actual_next_q = value_head(actual_next_tensor.mean(dim=-2))
        predicted_next_q = value_head(predicted.mean(dim=-2))
    return tuple(
        tensor.detach().float().cpu().numpy()
        for tensor in (predicted, current_q, actual_next_q, predicted_next_q)
    )  # type: ignore[return-value]


def _subset_metrics(
    indices: np.ndarray,
    *,
    current: np.ndarray,
    actual_next: np.ndarray,
    predicted: np.ndarray,
    current_dino: np.ndarray,
    next_dino: np.ndarray,
    current_q: np.ndarray,
    actual_next_q: np.ndarray,
    predicted_q: np.ndarray,
    current_actions: np.ndarray,
    next_actions: np.ndarray,
    current_returns: np.ndarray,
    next_returns: np.ndarray,
) -> dict[str, float]:
    return combination_metrics(
        current_state=current[indices],
        actual_next_state=actual_next[indices],
        predicted_next_state=predicted[indices],
        current_dino=current_dino[indices],
        next_dino=next_dino[indices],
        current_q=current_q[indices],
        actual_next_q=actual_next_q[indices],
        predicted_next_q=predicted_q[indices],
        current_actions=current_actions[indices],
        next_actions=next_actions[indices],
        current_returns=current_returns[indices],
        next_returns=next_returns[indices],
    )


def _render_html(result: dict[str, Any]) -> str:
    combinations = result["combinations"]
    columns = (
        "current_dino_rmse",
        "actual_next_dino_rmse",
        "behavior_copy_rmse",
        "behavior_predicted_rmse",
        "behavior_predicted_vs_copy_skill",
        "predicted_next_dino_rmse",
        "dino_predicted_vs_copy_skill",
        "current_state_std",
        "actual_next_state_std",
        "predicted_next_state_std",
        "value_actual_next_executed_rmse",
        "value_predicted_next_executed_rmse",
    )
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    rows = "".join(
        "<tr><td>" + html.escape(name) + "</td>" + "".join(
            f"<td>{metrics['overall'][column]:.6f}</td>" for column in columns
        ) + "</tr>"
        for name, metrics in combinations.items()
    )
    drift_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.6f}</td></tr>"
        for name, value in result["state_drift_rmse"].items()
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID58 SFT checkpoint state matrix</title>
<style>body{{font-family:system-ui;margin:20px;background:#f5f7fa;color:#17202a}}table{{border-collapse:collapse;background:white;font-size:12px}}th,td{{padding:6px;border:1px solid #ccd;text-align:right}}th:first-child,td:first-child{{text-align:left;white-space:nowrap}}.warn{{background:#fff4d6;padding:12px}}</style></head><body>
<h1>ID58 SFT1 / ID74 checkpoint state matrix</h1>
<p class=\"warn\">Read-only pre-RL validation diagnostic. ID74 WM and ValueHead on non-ID74 cells measure cross-component compatibility only. No validated goal labels were available, so no goal probe or counterfactual claim is included.</p>
<p>Samples: {result['sample_count']}; sources {html.escape(str(result['source_counts']))}; actions {html.escape(str(result['action_counts']))}; steps {html.escape(str(result['step_counts']))}; git <code>{html.escape(result['git_commit'])}</code>.</p>
<h2>Overall matrix</h2><table><tr><th>combination</th>{head}</tr>{rows}</table>
<h2>Selected state drift</h2><table><tr><th>comparison</th><th>RMSE</th></tr>{drift_rows}</table>
<p>Full per-source/per-action metrics, sample identities, checkpoint hashes, and raw float32 tensors are in <code>result.json</code> and <code>matrix_states.npz</code>.</p>
</body></html>"""


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--dino-grid-cache-root", type=Path, required=True)
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--id74-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=32)
    parser.add_argument("--max-step-index", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", default="nimloth-recon")
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from nimloth.backbone import build_input_builder, load_backbone
    from nimloth.backbone.dino_grid import CachedDINOGridTargets, DINOV2_LARGE_IDENTITY
    from nimloth.backbone.qwen25vl.factory import build_vision_ema
    from nimloth.rollout.transitions import expand_record_transitions
    from nimloth.wm.grid import (
        SharedSlotProjector,
        TemporalSpatialGridPredictor,
        load_sft1_slot_projector,
    )
    from nimloth.wm.value_head import ValueHead

    if not torch.cuda.is_available():
        raise RuntimeError("ID58 requires one CUDA GPU")
    device = torch.device("cuda:0")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create output_dir before evaluator starts")
    if any(output_dir.iterdir()):
        allowed = {"README.md", "wandb"}
        unexpected = {path.name for path in output_dir.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(f"fresh output directory contains unexpected files: {sorted(unexpected)}")

    records = _read_jsonl(args.val_jsonl)
    selected_records = select_early_transition_records(
        records,
        per_source=args.samples_per_source,
        max_step_index=args.max_step_index,
        expected_sources=DEFAULT_SOURCES,
    )
    samples = []
    metadata = []
    for record in selected_records:
        transitions = expand_record_transitions(record)
        step_index = int(record["_selected_step_index"])
        samples.append(transitions[step_index])
        metadata.append(
            {
                "record_id": str(record["id"]),
                "data_source": str(record["data_source"]),
                "seed": int(record["env_seed"]),
                "success": bool(record["success"]),
                "step_index": step_index,
                "current_action": int(transitions[step_index].action_index),
                "next_action": int(transitions[step_index + 1].action_index),
                "current_return": float(transitions[step_index].action_value_target),
                "next_return": float(transitions[step_index + 1].action_value_target),
                "current_image_path": str(transitions[step_index].current_image_path),
                "next_image_path": str(transitions[step_index].next_image_path),
            }
        )

    print(json.dumps({"selected_samples": len(samples), "sources": DEFAULT_SOURCES}), flush=True)
    sft1_args = _backbone_args(args.sft1_checkpoint, resume=False)
    sft1_args.max_pixels = args.max_pixels
    loaded_sft1 = load_backbone(
        sft1_args,
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
    )
    builder = build_input_builder(
        loaded_sft1,
        max_length=args.max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    current_batches = _build_batches(builder, samples, next_state=False, batch_size=args.batch_size)
    next_batches = _build_batches(builder, samples, next_state=True, batch_size=args.batch_size)
    sft1_current_hidden = _encode(loaded_sft1.backbone, current_batches, label="sft1_current")
    sft1_next_hidden = _encode(loaded_sft1.backbone, next_batches, label="sft1_next")
    del loaded_sft1
    torch.cuda.empty_cache()

    id74_args = _backbone_args(args.sft1_checkpoint, resume=True)
    id74_args.max_pixels = args.max_pixels
    loaded_id74 = load_backbone(
        id74_args,
        device=device,
        latent_token_count=16,
        model_parallel_size=1,
        resume_dir=args.id74_checkpoint,
        resume_state_path=args.id74_checkpoint / "training_state.pt",
    )
    id74_current_hidden = _encode(loaded_id74.backbone, current_batches, label="id74_online_current")
    id74_next_hidden = _encode(loaded_id74.backbone, next_batches, label="id74_online_next")
    vision_ema = build_vision_ema(
        enabled=True,
        decay=0.999,
        llm=loaded_id74.backbone.model,
        resume_path=args.id74_checkpoint / "vision_ema.pt",
        device=device,
    )
    if vision_ema is None:
        raise RuntimeError("ID74 vision EMA failed to load")
    with vision_ema.use_ema_weights(loaded_id74.backbone.model):
        id74_ema_current_hidden = _encode(loaded_id74.backbone, current_batches, label="id74_ema_current")
        id74_ema_next_hidden = _encode(loaded_id74.backbone, next_batches, label="id74_ema_next")
    del loaded_id74, vision_ema, current_batches, next_batches
    torch.cuda.empty_cache()

    sft1_projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        grid_tokens=16,
        qwen_hidden_dim=2048,
        state_dim=1024,
        map_location=device,
    )
    id74_projector = SharedSlotProjector(
        input_dim=2048,
        output_dim=1024,
        hidden_dim=2048,
        grid_tokens=16,
    ).to(device)
    id74_projector.load_state_dict(
        torch.load(args.id74_checkpoint / "state_proj.pt", map_location=device, weights_only=True)
    )

    hidden_modes = {
        "sft1_backbone": (sft1_current_hidden, sft1_next_hidden),
        "id74_online_backbone": (id74_current_hidden, id74_next_hidden),
        "id74_ema_backbone": (id74_ema_current_hidden, id74_ema_next_hidden),
    }
    projectors = {"sft1_projector": sft1_projector, "id74_projector": id74_projector}
    states: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for backbone_name, (current_hidden, next_hidden) in hidden_modes.items():
        for projector_name, projector in projectors.items():
            name = f"{backbone_name}__{projector_name}"
            states[name] = (
                _project(current_hidden, projector, device=device),
                _project(next_hidden, projector, device=device),
            )

    dino = CachedDINOGridTargets.from_cache_root(
        args.dino_grid_cache_root,
        identity=DINOV2_LARGE_IDENTITY,
        grid_size=4,
    )
    current_dino = dino.load([sample.current_image_path for sample in samples], device=torch.device("cpu")).numpy()
    next_dino = dino.load([sample.next_image_path for sample in samples], device=torch.device("cpu")).numpy()
    predictor = TemporalSpatialGridPredictor.load_checkpoint(
        args.id74_checkpoint / "wm_predictor", map_location=device
    ).to(device).eval()
    value_head = ValueHead.load_checkpoint(
        args.id74_checkpoint / "value_head", emb_dim=1024, map_location=device
    ).to(device).eval()

    current_actions = np.asarray([row["current_action"] for row in metadata], dtype=np.int64)
    next_actions = np.asarray([row["next_action"] for row in metadata], dtype=np.int64)
    current_returns = np.asarray([row["current_return"] for row in metadata], dtype=np.float64)
    next_returns = np.asarray([row["next_return"] for row in metadata], dtype=np.float64)
    sources = np.asarray([row["data_source"] for row in metadata])

    combinations: dict[str, Any] = {}
    predicted_states: dict[str, np.ndarray] = {}
    q_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (current, actual_next) in states.items():
        predicted, current_q, actual_next_q, predicted_q = _predict_and_value(
            current,
            actual_next,
            current_actions,
            predictor=predictor,
            value_head=value_head,
            device=device,
        )
        predicted_states[name] = predicted
        q_arrays[name] = (current_q, actual_next_q, predicted_q)
        all_indices = np.arange(len(samples))
        overall = _subset_metrics(
            all_indices,
            current=current,
            actual_next=actual_next,
            predicted=predicted,
            current_dino=current_dino,
            next_dino=next_dino,
            current_q=current_q,
            actual_next_q=actual_next_q,
            predicted_q=predicted_q,
            current_actions=current_actions,
            next_actions=next_actions,
            current_returns=current_returns,
            next_returns=next_returns,
        )
        by_source = {}
        for source in DEFAULT_SOURCES:
            indices = np.flatnonzero(sources == source)
            by_source[source] = _subset_metrics(
                indices,
                current=current,
                actual_next=actual_next,
                predicted=predicted,
                current_dino=current_dino,
                next_dino=next_dino,
                current_q=current_q,
                actual_next_q=actual_next_q,
                predicted_q=predicted_q,
                current_actions=current_actions,
                next_actions=next_actions,
                current_returns=current_returns,
                next_returns=next_returns,
            )
        by_action = {}
        for action in sorted(set(current_actions.tolist())):
            indices = np.flatnonzero(current_actions == action)
            by_action[str(action)] = _subset_metrics(
                indices,
                current=current,
                actual_next=actual_next,
                predicted=predicted,
                current_dino=current_dino,
                next_dino=next_dino,
                current_q=current_q,
                actual_next_q=actual_next_q,
                predicted_q=predicted_q,
                current_actions=current_actions,
                next_actions=next_actions,
                current_returns=current_returns,
                next_returns=next_returns,
            )
        combinations[name] = {"overall": overall, "by_source": by_source, "by_action": by_action}

    def state_rmse(left: str, right: str) -> float:
        left_both = np.concatenate(states[left], axis=0)
        right_both = np.concatenate(states[right], axis=0)
        return _rmse(left_both, right_both)

    state_drift = {
        "backbone_drift_with_sft1_projector_sft1_to_id74_online": state_rmse(
            "sft1_backbone__sft1_projector", "id74_online_backbone__sft1_projector"
        ),
        "projector_drift_on_sft1_backbone_sft1_to_id74": state_rmse(
            "sft1_backbone__sft1_projector", "sft1_backbone__id74_projector"
        ),
        "projector_drift_on_id74_online_sft1_to_id74": state_rmse(
            "id74_online_backbone__sft1_projector", "id74_online_backbone__id74_projector"
        ),
        "vision_ema_drift_with_id74_projector_online_to_ema": state_rmse(
            "id74_online_backbone__id74_projector", "id74_ema_backbone__id74_projector"
        ),
    }

    source_counts = {source: int(np.sum(sources == source)) for source in DEFAULT_SOURCES}
    action_counts = {
        str(action): int(np.sum(current_actions == action))
        for action in sorted(set(current_actions.tolist()))
    }
    step_counts = {
        str(step): sum(int(row["step_index"]) == step for row in metadata)
        for step in sorted({int(row["step_index"]) for row in metadata})
    }
    result: dict[str, Any] = {
        "schema": "nimloth_sft_checkpoint_state_matrix_v1",
        "read_only": True,
        "sample_count": len(samples),
        "source_counts": source_counts,
        "action_counts": action_counts,
        "step_counts": step_counts,
        "selection": metadata,
        "combinations": combinations,
        "state_drift_rmse": state_drift,
        "interpretation_limits": {
            "cross_component": "ID74 WM and ValueHead on non-ID74 cells measure compatibility only",
            "goal_probe": "not run: no validated goal labels or matched real goal counterfactual pairs",
            "cot": "all states use the actual recorded observation-conditioned CoT",
            "terminal": "not used; every selected step0 has an exact recorded next decision state",
        },
        "data": {
            "val_jsonl": str(args.val_jsonl.resolve()),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
            "dino_grid_cache_root": str(args.dino_grid_cache_root.resolve()),
        },
        "checkpoints": {
            "sft1": _checkpoint_files(args.sft1_checkpoint),
            "id74": _checkpoint_files(args.id74_checkpoint),
        },
        "git_commit": args.git_commit,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }

    payload: dict[str, np.ndarray] = {
        "current_dino": current_dino.astype(np.float32),
        "next_dino": next_dino.astype(np.float32),
        "current_actions": current_actions.astype(np.int64),
        "next_actions": next_actions.astype(np.int64),
        "current_returns": current_returns.astype(np.float32),
        "next_returns": next_returns.astype(np.float32),
    }
    for name, (current, actual_next) in states.items():
        payload[f"{name}__current"] = current.astype(np.float32)
        payload[f"{name}__actual_next"] = actual_next.astype(np.float32)
        payload[f"{name}__predicted_next"] = predicted_states[name].astype(np.float32)
        current_q, actual_next_q, predicted_q = q_arrays[name]
        payload[f"{name}__current_q"] = current_q.astype(np.float32)
        payload[f"{name}__actual_next_q"] = actual_next_q.astype(np.float32)
        payload[f"{name}__predicted_next_q"] = predicted_q.astype(np.float32)
    payload_path = output_dir / "matrix_states.npz"
    _atomic_npz(payload_path, payload)
    result["matrix_states"] = {
        "path": payload_path.name,
        "bytes": payload_path.stat().st_size,
        "sha256": _sha256(payload_path),
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.html").write_text(_render_html(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    run_handle = None
    try:
        import wandb

        run_handle = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="never",
            config={
                "read_only": True,
                "samples_per_source": args.samples_per_source,
                "max_step_index": args.max_step_index,
                "batch_size": args.batch_size,
                "sft1_checkpoint": str(args.sft1_checkpoint),
                "id74_checkpoint": str(args.id74_checkpoint),
                "git_commit": args.git_commit,
            },
        )
        result = run(args)
        flattened = {}
        for name, section in result["combinations"].items():
            for key, value in section["overall"].items():
                flattened[f"matrix/{name}/{key}"] = value
        flattened.update({f"drift/{key}": value for key, value in result["state_drift_rmse"].items()})
        run_handle.log(flattened)
        run_handle.summary.update(
            {
                "status": "passed",
                "sample_count": result["sample_count"],
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                "matrix_states_sha256": result["matrix_states"]["sha256"],
            }
        )
        run_handle.finish(exit_code=0)
        return 0
    except BaseException:
        if run_handle is not None:
            run_handle.summary["status"] = "failed"
            run_handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
