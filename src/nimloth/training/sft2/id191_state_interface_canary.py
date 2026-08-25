"""Run the ID191 state-interface direction canary from frozen pre-RL evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from nimloth.eval.action_outcome_predictability_probe import (
    _fit_epoch_selection,
    _train_final_probe,
    average_precision,
    binary_probe_metrics,
)
from nimloth.eval.frozen_state_goal_probe import (
    _build_cache_index,
    _classification_metrics,
    _fit_linear_probe,
    _inner_split,
    _train_final_probe as _train_final_goal_probe,
    aggregate_task_probe_features,
    goal_probe_gate,
)
from nimloth.eval.id75_action_outcome_audit import binary_auc, parse_step_action_success
from nimloth.eval.sft_checkpoint_state_matrix import _backbone_args, _checkpoint_files, _sha256
from nimloth.training.sft2.state_interface_canary import (
    ResidualStateInterfaceCanary,
    StateInterfaceCanaryConfig,
    canary_gate,
    grouped_record_selection,
    normalized_multitask_loss,
    save_canary_checkpoint,
    visual_state_metrics,
)

_MOVEMENT_ACTIONS = (0, 2, 3)
_ACTION_NAMES = {0: "move_forward", 2: "move_right", 3: "move_left"}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_probabilities(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    counts = Counter(values.tolist())
    weights = np.asarray([1.0 / counts[value] for value in values.tolist()], dtype=np.float64)
    return weights / weights.sum()


def _sample_indices(
    indices: np.ndarray,
    probabilities: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.choice(indices, size=size, replace=True, p=probabilities)


def _encode_hidden(
    *,
    checkpoint: Path,
    samples: Sequence[Any],
    projector: Any,
    expected_state: np.ndarray,
    device: Any,
    batch_size: int,
    max_length: int,
    max_pixels: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Recompute same-generation hidden and prove its projection matches ID60."""

    import torch
    from nimloth.backbone import build_input_builder, load_backbone

    args = _backbone_args(checkpoint, resume=False)
    args.max_pixels = max_pixels
    loaded = load_backbone(args, device=device, latent_token_count=16, model_parallel_size=1)
    loaded.backbone.eval()
    projector.eval().to(device=device, dtype=torch.float32)
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    builder = build_input_builder(
        loaded,
        max_length=max_length,
        latent_token_count=16,
        mask_latent_query_labels=True,
    )
    hidden_chunks: list[np.ndarray] = []
    squared_error = 0.0
    maximum_error = 0.0
    element_count = 0
    started = time.monotonic()
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            batch = builder.build(
                [sample.prefix_messages for sample in chunk],
                [sample.prefix_image_paths for sample in chunk],
                include_labels=False,
            )
            hidden = loaded.backbone(batch, include_lm_loss=False).hidden.detach().float()
            projected = projector(hidden).detach().float().cpu().numpy()
            reference = expected_state[start : start + len(chunk)].astype(np.float32)
            difference = projected.astype(np.float64) - reference.astype(np.float64)
            squared_error += float(np.square(difference).sum())
            maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
            element_count += difference.size
            hidden_chunks.append(hidden.cpu().numpy().astype(np.float32))
            batch_index = start // batch_size + 1
            if batch_index % 25 == 0 or start + len(chunk) == len(samples):
                print(
                    json.dumps(
                        {
                            "hidden_encode_batch": batch_index,
                            "hidden_encode_batches": math.ceil(len(samples) / batch_size),
                        }
                    ),
                    flush=True,
                )
    hidden_array = np.concatenate(hidden_chunks, axis=0)
    del loaded
    torch.cuda.empty_cache()
    if hidden_array.shape != (len(samples), 16, 2048):
        raise ValueError(f"hidden cache shape mismatch: {hidden_array.shape}")
    if hidden_array.dtype != np.float32 or not np.isfinite(hidden_array).all():
        raise ValueError("hidden cache is not finite float32")
    identity = {
        "projected_state_rmse": float(math.sqrt(squared_error / element_count)),
        "projected_state_max_abs": maximum_error,
        "seconds": float(time.monotonic() - started),
    }
    if identity["projected_state_rmse"] > 1e-6 or maximum_error > 1e-4:
        raise ValueError(f"recomputed hidden does not reproduce ID60 state: {identity}")
    return hidden_array, identity


def _candidate_batches(
    model: ResidualStateInterfaceCanary,
    hidden: np.ndarray,
    baseline: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(hidden), batch_size):
            stop = min(len(hidden), start + batch_size)
            state = model.calibrated_state(
                torch.from_numpy(hidden[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(baseline[start:stop]).to(device=device, dtype=torch.float32),
            )
            chunks.append(state.detach().float().cpu().numpy())
    result = np.concatenate(chunks, axis=0).astype(np.float32)
    if result.shape != baseline.shape or not np.isfinite(result).all():
        raise ValueError("candidate state inference is invalid")
    return result


def _macro_goal_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    prediction = np.asarray(logits).argmax(axis=1)
    target = np.asarray(labels, dtype=np.int64)
    return float(
        np.mean(
            [float((prediction[target == label] == label).mean()) for label in sorted(set(target))]
        )
    )


def _selection_metrics(
    *,
    model: ResidualStateInterfaceCanary,
    hidden: np.ndarray,
    baseline: np.ndarray,
    dino: np.ndarray,
    state_record_index: np.ndarray,
    visual_record_indices: np.ndarray,
    goal_state_indices: np.ndarray,
    goal_labels: np.ndarray,
    transition_indices: np.ndarray,
    transition_current_index: np.ndarray,
    transition_actions: np.ndarray,
    transition_outcomes: np.ndarray,
    device: Any,
    batch_size: int,
) -> dict[str, float]:
    import torch

    model.eval()
    selected_state_indices = np.flatnonzero(
        np.isin(state_record_index, visual_record_indices)
    )
    candidate = _candidate_batches(
        model,
        hidden[selected_state_indices],
        baseline[selected_state_indices],
        device=device,
        batch_size=batch_size,
    )
    visual = visual_state_metrics(
        candidate,
        dino[selected_state_indices],
        baseline[selected_state_indices],
    )
    with torch.inference_mode():
        goal_state = model.calibrated_state(
            torch.from_numpy(hidden[goal_state_indices]).to(device),
            torch.from_numpy(baseline[goal_state_indices]).to(device),
        )
        goal_logits = model.goal_logits(goal_state).float().cpu().numpy()
    action_auc: list[float] = []
    for action in _MOVEMENT_ACTIONS:
        indices = transition_indices[transition_actions[transition_indices] == action]
        if len(indices) < 2:
            raise ValueError(f"selection action {action} is empty")
        labels = transition_outcomes[indices]
        if not labels.any() or labels.all():
            raise ValueError(f"selection action {action} lacks both outcomes")
        state_indices = transition_current_index[indices]
        with torch.inference_mode():
            state = model.calibrated_state(
                torch.from_numpy(hidden[state_indices]).to(device),
                torch.from_numpy(baseline[state_indices]).to(device),
            )
            logits = model.outcome_logits(
                state,
                torch.from_numpy(transition_actions[indices]).to(device),
            ).float().cpu().numpy()
        action_auc.append(binary_auc(labels, logits))
    return {
        "goal_macro_top1": _macro_goal_accuracy(goal_logits, goal_labels),
        "outcome_macro_auc": float(np.mean(action_auc)),
        "visual_cosine": visual["candidate_dino_cosine"],
        "baseline_visual_cosine": visual["baseline_dino_cosine"],
        "residual_fraction_max": visual["residual_fraction_max"],
    }


def _train_canary(
    *,
    config: StateInterfaceCanaryConfig,
    hidden: np.ndarray,
    baseline: np.ndarray,
    dino: np.ndarray,
    state_record_index: np.ndarray,
    visual_record_indices: np.ndarray,
    goal_state_indices: np.ndarray,
    goal_labels: np.ndarray,
    transition_indices: np.ndarray,
    transition_current_index: np.ndarray,
    transition_actions: np.ndarray,
    transition_outcomes: np.ndarray,
    device: Any,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    anchor_weight: float,
    seed: int,
    selection: dict[str, np.ndarray] | None,
    patience: int | None,
) -> tuple[ResidualStateInterfaceCanary, int, list[dict[str, float]]]:
    import torch

    torch.manual_seed(seed)
    model = ResidualStateInterfaceCanary(config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    visual_indices = np.flatnonzero(np.isin(state_record_index, visual_record_indices))
    goal_probabilities = _balanced_probabilities(goal_labels)
    strata = np.asarray(
        [f"{int(action)}:{int(outcome)}" for action, outcome in zip(
            transition_actions[transition_indices],
            transition_outcomes[transition_indices],
            strict=True,
        )]
    )
    outcome_probabilities = _balanced_probabilities(strata)
    visual_probabilities = np.full(len(visual_indices), 1.0 / len(visual_indices))
    visual_reference = float(
        1.0
        - np.mean(
            np.sum(baseline[visual_indices].astype(np.float64) * dino[visual_indices], axis=-1)
            / np.maximum(
                np.linalg.norm(baseline[visual_indices].astype(np.float64), axis=-1)
                * np.linalg.norm(dino[visual_indices].astype(np.float64), axis=-1),
                1e-12,
            )
        )
    )
    goal_reference = math.log(config.goal_classes)
    outcome_reference = math.log(2.0)
    steps_per_epoch = math.ceil(
        max(len(visual_indices), len(goal_state_indices), len(transition_indices)) / batch_size
    )
    rng = np.random.default_rng(seed)
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_score = -float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        accumulated = {"loss": [], "visual": [], "goal": [], "outcome": [], "anchor": []}
        for _ in range(steps_per_epoch):
            visual_batch = _sample_indices(
                visual_indices,
                visual_probabilities,
                size=batch_size,
                rng=rng,
            )
            goal_local = _sample_indices(
                np.arange(len(goal_state_indices)),
                goal_probabilities,
                size=batch_size,
                rng=rng,
            )
            goal_batch = goal_state_indices[goal_local]
            outcome_local = _sample_indices(
                np.arange(len(transition_indices)),
                outcome_probabilities,
                size=batch_size,
                rng=rng,
            )
            outcome_batch = transition_indices[outcome_local]
            outcome_state_batch = transition_current_index[outcome_batch]
            optimizer.zero_grad(set_to_none=True)
            total, components = normalized_multitask_loss(
                model=model,
                visual_hidden=torch.from_numpy(hidden[visual_batch]).to(device),
                visual_baseline=torch.from_numpy(baseline[visual_batch]).to(device),
                visual_dino=torch.from_numpy(dino[visual_batch]).to(device),
                goal_hidden=torch.from_numpy(hidden[goal_batch]).to(device),
                goal_baseline=torch.from_numpy(baseline[goal_batch]).to(device),
                goal_labels=torch.from_numpy(goal_labels[goal_local]).to(device),
                outcome_hidden=torch.from_numpy(hidden[outcome_state_batch]).to(device),
                outcome_baseline=torch.from_numpy(baseline[outcome_state_batch]).to(device),
                outcome_actions=torch.from_numpy(transition_actions[outcome_batch]).to(device),
                outcome_labels=torch.from_numpy(transition_outcomes[outcome_batch]).to(device),
                visual_reference_loss=visual_reference,
                goal_reference_loss=goal_reference,
                outcome_reference_loss=outcome_reference,
                anchor_weight=anchor_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError("state-interface canary loss is non-finite")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            accumulated["loss"].append(float(total.detach().item()))
            for key, value in components.items():
                accumulated[key].append(float(value.detach().item()))
        row = {"epoch": float(epoch)}
        row.update({f"train_{key}": float(np.mean(value)) for key, value in accumulated.items()})
        if selection is not None:
            metrics = _selection_metrics(
                model=model,
                hidden=hidden,
                baseline=baseline,
                dino=dino,
                state_record_index=state_record_index,
                visual_record_indices=selection["record_indices"],
                goal_state_indices=selection["goal_state_indices"],
                goal_labels=selection["goal_labels"],
                transition_indices=selection["transition_indices"],
                transition_current_index=transition_current_index,
                transition_actions=transition_actions,
                transition_outcomes=transition_outcomes,
                device=device,
                batch_size=batch_size,
            )
            row.update({f"selection_{key}": value for key, value in metrics.items()})
            visual_ok = (
                metrics["visual_cosine"] >= metrics["baseline_visual_cosine"] - 0.005
                and metrics["residual_fraction_max"] <= config.max_residual_fraction + 1e-5
            )
            score = (
                metrics["goal_macro_top1"]
                + metrics["outcome_macro_auc"]
                + metrics["visual_cosine"]
                if visual_ok
                else -float("inf")
            )
            row["selection_score"] = score
            if score > best_score + 1e-6:
                best_score = score
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if selection is not None and patience is not None and stale >= patience:
            break
    return model, best_epoch, history


def _goal_probe(
    *,
    features: np.ndarray,
    record_metadata: list[dict[str, Any]],
    seed: int,
    device: Any,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    labels = np.asarray([row["goal"] for row in record_metadata])
    task_keys = np.asarray([row["observed_task_key"] for row in record_metadata])
    groups = np.asarray([row["inner_group_key"] for row in record_metadata])
    split = np.asarray([row["split"] for row in record_metadata])
    train_mask = split == "train"
    val_mask = (split == "val") & ~np.isin(groups, sorted(set(groups[train_mask])))
    train = aggregate_task_probe_features(
        features=features[train_mask],
        task_keys=task_keys[train_mask],
        labels=labels[train_mask],
        leakage_group_keys=groups[train_mask],
    )
    val = aggregate_task_probe_features(
        features=features[val_mask],
        task_keys=task_keys[val_mask],
        labels=labels[val_mask],
        leakage_group_keys=groups[val_mask],
    )
    classes = sorted(set(train.labels.tolist()))
    class_index = {name: index for index, name in enumerate(classes)}
    seen = np.isin(val.labels, classes)
    train_labels = np.asarray([class_index[label] for label in train.labels], dtype=np.int64)
    val_labels = np.asarray([class_index[label] for label in val.labels[seen]], dtype=np.int64)
    inner = _inner_split(train.leakage_group_keys, train.labels)
    selected_epochs = _fit_linear_probe(
        train.features[~inner],
        train_labels[~inner],
        train.features[inner],
        train_labels[inner],
        class_count=len(classes),
        device=device,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
    )
    logits, weights = _train_final_goal_probe(
        train.features,
        train_labels,
        val.features[seen],
        class_count=len(classes),
        device=device,
        seed=seed,
        epochs=selected_epochs,
    )
    metrics, correct = _classification_metrics(logits, val_labels, classes)
    metrics.update(
        {
            "selected_epochs": selected_epochs,
            "train_observed_tasks": len(train.features),
            "external_observed_tasks": len(val_labels),
            "unseen_external_labels": sorted(set(val.labels[~seen].tolist())),
        }
    )
    return metrics, correct, {key: value.astype(np.float32) for key, value in weights.items()}


def _paired_accuracy_bootstrap(
    candidate_correct: np.ndarray,
    reference_correct: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    difference = np.asarray(candidate_correct, dtype=np.float64) - np.asarray(
        reference_correct, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    output = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        width = min(500, draws - start)
        index = rng.integers(0, len(difference), size=(width, len(difference)))
        output[start : start + width] = difference[index].mean(axis=1)
    return {
        "draws": draws,
        "mean_difference": float(difference.mean()),
        "lower_95": float(np.quantile(output, 0.025)),
        "upper_95": float(np.quantile(output, 0.975)),
    }


def _paired_auc_difference(
    labels: np.ndarray,
    candidate_logits: np.ndarray,
    reference_logits: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    target = np.asarray(labels, dtype=np.bool_)
    candidate = np.asarray(candidate_logits, dtype=np.float64)
    reference = np.asarray(reference_logits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    candidate_values: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(target), size=len(target))
        sampled = target[index]
        if not sampled.any() or sampled.all():
            continue
        candidate_auc = binary_auc(sampled, candidate[index])
        reference_auc = binary_auc(sampled, reference[index])
        candidate_values.append(candidate_auc)
        values.append(candidate_auc - reference_auc)
    if len(values) < draws // 2:
        raise RuntimeError("too few valid paired AUC draws")
    difference = np.asarray(values)
    candidate_array = np.asarray(candidate_values)
    return {
        "requested_draws": draws,
        "valid_draws": len(values),
        "candidate_lower_95": float(np.quantile(candidate_array, 0.025)),
        "candidate_upper_95": float(np.quantile(candidate_array, 0.975)),
        "mean_difference": float(difference.mean()),
        "lower_95": float(np.quantile(difference, 0.025)),
        "upper_95": float(np.quantile(difference, 0.975)),
    }


def _outcome_probes(
    *,
    feature_sets: dict[str, np.ndarray],
    transition_current_index: np.ndarray,
    transition_actions: np.ndarray,
    transition_split: np.ndarray,
    transition_record_index: np.ndarray,
    transition_external_eligible: np.ndarray,
    outcomes: np.ndarray,
    record_metadata: list[dict[str, Any]],
    seed: int,
    device: Any,
    max_epochs: int,
    patience: int,
    bootstrap_draws: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    group = np.asarray(
        [record_metadata[index]["inner_group_key"] for index in transition_record_index]
    )
    train_mask = transition_split == 0
    external_mask = (transition_split == 1) & transition_external_eligible
    output: dict[str, Any] = {}
    saved_weights: dict[str, np.ndarray] = {}
    for action in _MOVEMENT_ACTIONS:
        action_train = train_mask & (transition_actions == action)
        action_external = external_mask & (transition_actions == action)
        train_indices = np.flatnonzero(action_train)
        external_indices = np.flatnonzero(action_external)
        inner = np.asarray(
            [
                int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 10 == 0
                for value in group[train_indices]
            ],
            dtype=np.bool_,
        )
        if not inner.any() or inner.all():
            raise ValueError(f"outcome action {action} inner split is empty")
        feature_result: dict[str, Any] = {}
        external_logits: dict[str, np.ndarray] = {}
        for feature_offset, (name, feature) in enumerate(feature_sets.items()):
            flat = feature[transition_current_index].reshape(len(transition_current_index), -1)
            selected_epoch, _ = _fit_epoch_selection(
                flat[train_indices[~inner]],
                outcomes[train_indices[~inner]],
                flat[train_indices[inner]],
                outcomes[train_indices[inner]],
                learning_rate=3e-3,
                weight_decay=1e-2,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed + action * 20 + feature_offset,
                device=device,
            )
            logits, weights = _train_final_probe(
                flat[train_indices],
                outcomes[train_indices],
                flat[external_indices],
                learning_rate=3e-3,
                weight_decay=1e-2,
                epochs=selected_epoch,
                seed=seed + action * 20 + feature_offset,
                device=device,
            )
            external_logits[name] = logits
            feature_result[name] = {
                **binary_probe_metrics(
                    outcomes[external_indices],
                    logits,
                    train_success_rate=float(outcomes[train_indices].mean()),
                ),
                "selected_epoch": selected_epoch,
                "parameter_count": int(flat.shape[1] + 1),
            }
            for key, value in weights.items():
                saved_weights[f"action_{action}__{name}__{key}"] = value.astype(np.float32)
            del flat
        paired: dict[str, Any] = {}
        for reference in ("state", "dino"):
            paired[f"candidate_minus_{reference}"] = _paired_auc_difference(
                outcomes[external_indices],
                external_logits["candidate"],
                external_logits[reference],
                seed=seed + action * 100 + (0 if reference == "state" else 1),
                draws=bootstrap_draws,
            )
        output[str(action)] = {
            "action_name": _ACTION_NAMES[action],
            "train_count": len(train_indices),
            "external_count": len(external_indices),
            "features": feature_result,
            "paired_bootstrap": paired,
        }
    return output, saved_weights


def _validate_saved_npz(path: Path, expected: dict[str, tuple[np.dtype, tuple[int, ...]]]) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError(f"saved NPZ keys differ: {saved.files}")
        for key, (dtype, shape) in expected.items():
            value = saved[key]
            if value.dtype != dtype or value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"saved NPZ array {key} is invalid")


def _render_html(result: dict[str, Any]) -> str:
    outcome_rows = "".join(
        f"<tr><td>{section['action_name']}</td>"
        f"<td>{section['features']['state']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['hidden']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['candidate']['roc_auc']:.4f}</td>"
        f"<td>{section['features']['dino']['roc_auc']:.4f}</td></tr>"
        for section in result["outcome_probe"].values()
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID191 state interface canary</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID191 state-interface direction canary</h1><p>Overall gate: <strong>{html.escape(str(result['gate']['passed']))}</strong></p>
<p>Goal micro top1: baseline {result['goal_probe']['state']['micro_top1']:.4f}, hidden {result['goal_probe']['hidden']['micro_top1']:.4f}, candidate {result['goal_probe']['candidate']['micro_top1']:.4f}, DINO {result['goal_probe']['dino']['micro_top1']:.4f}.</p>
<table><tr><th>action</th><th>state AUC</th><th>hidden AUC</th><th>candidate AUC</th><th>DINO AUC</th></tr>{outcome_rows}</table>
<p>The adapter is a diagnostic direction test only. No old WM, ValueHead, MCTS, or RL component was updated or reused.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--state-cache-metadata", type=Path, required=True)
    parser.add_argument("--id60-result", type=Path, required=True)
    parser.add_argument("--id71-result", type=Path, required=True)
    parser.add_argument("--sft1-checkpoint", type=Path, required=True)
    parser.add_argument("--actor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=12000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--adapter-rank", type=int, default=64)
    parser.add_argument("--max-residual-fraction", type=float, default=0.1)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--anchor-weight", type=float, default=0.25)
    parser.add_argument("--probe-max-epochs", type=int, default=300)
    parser.add_argument("--probe-patience", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42191)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def _validate_wandb_identity(args: argparse.Namespace) -> None:
    if args.wandb_project != "nimloth-sft2":
        raise ValueError("ID191 W&B project must be nimloth-sft2")
    if not args.wandb_run_id.startswith("nimloth-sft2-id191-state-interface-canary"):
        raise ValueError("ID191 W&B run ID is outside the locked namespace")
    if os.environ.get("WANDB_PROJECT") != args.wandb_project:
        raise ValueError("effective ID191 W&B project differs")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_run_id:
        raise ValueError("effective ID191 W&B run ID differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nimloth.wm.grid import load_sft1_slot_projector

    if not torch.cuda.is_available():
        raise RuntimeError("ID191 requires one CUDA GPU")
    if args.max_residual_fraction != 0.1:
        raise ValueError("ID191 residual bound is fixed at 0.1")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create ID191 output")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID191 output is not fresh")

    id60 = json.loads(args.id60_result.read_text(encoding="utf-8"))
    id71 = json.loads(args.id71_result.read_text(encoding="utf-8"))
    if id60.get("schema") != "nimloth_frozen_state_goal_probe_v1":
        raise ValueError("ID60 schema mismatch")
    if id71.get("schema") != "nimloth_frozen_state_action_outcome_probe_v1":
        raise ValueError("ID71 schema mismatch")
    if _sha256(args.state_cache) != id60["state_cache"]["sha256"]:
        raise ValueError("ID60 cache hash mismatch")
    if _sha256(args.state_cache_metadata) != id60["state_cache"]["metadata_sha256"]:
        raise ValueError("ID60 metadata hash mismatch")
    if _checkpoint_files(args.sft1_checkpoint) != id60["checkpoints"]["sft1"]:
        raise ValueError("SFT1 checkpoint identity differs from ID60")
    if _checkpoint_files(args.actor_checkpoint) != id60["checkpoints"]["actor"]:
        raise ValueError("actor checkpoint identity differs from ID60")

    train_records = _load_jsonl(args.train_jsonl)
    val_records = _load_jsonl(args.val_jsonl)
    records = train_records + val_records
    metadata = json.loads(args.state_cache_metadata.read_text(encoding="utf-8"))
    record_metadata = metadata["records"]
    if len(record_metadata) != len(records):
        raise ValueError("record metadata count mismatch")
    for row, record in zip(record_metadata, records, strict=True):
        if row["record_id"] != str(record["id"]):
            raise ValueError("record order differs from ID60")

    with np.load(args.state_cache, allow_pickle=False) as cache:
        baseline = cache["state"].astype(np.float32)
        dino = cache["dino"].astype(np.float32)
        transition_current = cache["transition_current_index"].astype(np.int64)
        transition_actions = cache["transition_action"].astype(np.int64)
        transition_split = cache["transition_split"].astype(np.int64)
        transition_record = cache["transition_record_index"].astype(np.int64)
        transition_eligible = cache["transition_external_eligible"].astype(np.bool_)
        state_record = cache["state_record_index"].astype(np.int64)
        state_step = cache["state_step_index"].astype(np.int64)
        initial_state = cache["record_initial_state_index"].astype(np.int64)

    train_meta_copy = [dict(row) for row in record_metadata[: len(train_records)]]
    val_meta_copy = [dict(row) for row in record_metadata[len(train_records) :]]
    train_prompts, train_states, _ = _build_cache_index(
        train_records,
        train_meta_copy,
        max_step_index=3,
        split_code=0,
        record_offset=0,
        state_offset=0,
    )
    val_prompts, val_states, _ = _build_cache_index(
        val_records,
        val_meta_copy,
        max_step_index=3,
        split_code=1,
        record_offset=len(train_records),
        state_offset=len(train_prompts),
    )
    prompts = train_prompts + val_prompts
    rebuilt_states = train_states + val_states
    if len(prompts) != len(baseline) or len(rebuilt_states) != len(metadata["states"]):
        raise ValueError("rebuilt prompt count differs from ID60")
    for rebuilt, source in zip(rebuilt_states, metadata["states"], strict=True):
        if (rebuilt["record_index"], rebuilt["step_index"]) != (
            source["record_index"],
            source["step_index"],
        ):
            raise ValueError("rebuilt prompt ordering differs from ID60")

    device = torch.device("cuda:0")
    projector = load_sft1_slot_projector(
        args.sft1_checkpoint,
        qwen_hidden_dim=2048,
        state_dim=1024,
        grid_tokens=16,
        map_location=device,
        dtype=torch.float32,
    )
    hidden, hidden_identity = _encode_hidden(
        checkpoint=args.actor_checkpoint,
        samples=prompts,
        projector=projector,
        expected_state=baseline,
        device=device,
        batch_size=args.encode_batch_size,
        max_length=args.max_length,
        max_pixels=args.max_pixels,
    )
    hidden_path = output_dir / "frozen_same_generation_hidden.npz"
    _atomic_npz(hidden_path, {"hidden": hidden})
    _validate_saved_npz(hidden_path, {"hidden": (np.dtype("float32"), hidden.shape)})

    goals = np.asarray([row["goal"] for row in record_metadata])
    classes = sorted(set(goals[: len(train_records)].tolist()))
    class_index = {name: index for index, name in enumerate(classes)}
    goal_labels = np.asarray([class_index[value] for value in goals[: len(train_records)]], dtype=np.int64)
    train_groups = np.asarray(
        [row["inner_group_key"] for row in record_metadata[: len(train_records)]]
    )
    selection_record_local = grouped_record_selection(train_groups, goals[: len(train_records)])
    fit_record_indices = np.flatnonzero(~selection_record_local)
    selection_record_indices = np.flatnonzero(selection_record_local)
    train_record_indices = np.arange(len(train_records), dtype=np.int64)

    outcomes = np.asarray(
        [
            parse_step_action_success(records[record], int(state_step[current]))
            for record, current in zip(transition_record, transition_current, strict=True)
        ],
        dtype=np.bool_,
    )
    movement = np.isin(transition_actions, _MOVEMENT_ACTIONS)
    train_transition = (transition_split == 0) & movement
    fit_transition = train_transition & np.isin(transition_record, fit_record_indices)
    selection_transition = train_transition & np.isin(transition_record, selection_record_indices)
    fit_transition_indices = np.flatnonzero(fit_transition)
    selection_transition_indices = np.flatnonzero(selection_transition)
    train_transition_indices = np.flatnonzero(train_transition)

    config = StateInterfaceCanaryConfig(
        adapter_rank=args.adapter_rank,
        goal_classes=len(classes),
        max_residual_fraction=args.max_residual_fraction,
    )
    _, selected_epoch, selection_history = _train_canary(
        config=config,
        hidden=hidden,
        baseline=baseline,
        dino=dino,
        state_record_index=state_record,
        visual_record_indices=fit_record_indices,
        goal_state_indices=initial_state[fit_record_indices],
        goal_labels=goal_labels[fit_record_indices],
        transition_indices=fit_transition_indices,
        transition_current_index=transition_current,
        transition_actions=transition_actions,
        transition_outcomes=outcomes,
        device=device,
        batch_size=args.batch_size,
        epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        anchor_weight=args.anchor_weight,
        seed=args.seed,
        selection={
            "record_indices": selection_record_indices,
            "goal_state_indices": initial_state[selection_record_indices],
            "goal_labels": goal_labels[selection_record_indices],
            "transition_indices": selection_transition_indices,
        },
        patience=args.patience,
    )
    final_model, _, final_history = _train_canary(
        config=config,
        hidden=hidden,
        baseline=baseline,
        dino=dino,
        state_record_index=state_record,
        visual_record_indices=train_record_indices,
        goal_state_indices=initial_state[train_record_indices],
        goal_labels=goal_labels,
        transition_indices=train_transition_indices,
        transition_current_index=transition_current,
        transition_actions=transition_actions,
        transition_outcomes=outcomes,
        device=device,
        batch_size=args.batch_size,
        epochs=selected_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        anchor_weight=args.anchor_weight,
        seed=args.seed,
        selection=None,
        patience=None,
    )
    candidate = _candidate_batches(
        final_model,
        hidden,
        baseline,
        device=device,
        batch_size=args.batch_size,
    )

    train_state_hashes = {
        row["image_sha256"]
        for row in metadata["states"]
        if int(row["record_index"]) < len(train_records)
    }
    external_state_mask = np.asarray(
        [
            int(row["record_index"]) >= len(train_records)
            and row["image_sha256"] not in train_state_hashes
            for row in metadata["states"]
        ],
        dtype=np.bool_,
    )
    external_state_indices = np.flatnonzero(external_state_mask)
    visual = visual_state_metrics(
        candidate[external_state_indices],
        dino[external_state_indices],
        baseline[external_state_indices],
    )

    initial_features = {
        "state": baseline[initial_state].mean(axis=1),
        "hidden": hidden[initial_state].mean(axis=1),
        "candidate": candidate[initial_state].mean(axis=1),
        "dino": dino[initial_state].mean(axis=1),
    }
    goal_results: dict[str, Any] = {}
    goal_correct: dict[str, np.ndarray] = {}
    probe_arrays: dict[str, np.ndarray] = {}
    for offset, (name, feature) in enumerate(initial_features.items()):
        metrics, correct, weights = _goal_probe(
            features=feature,
            record_metadata=record_metadata,
            seed=args.seed + offset,
            device=device,
            max_epochs=args.probe_max_epochs,
            patience=args.probe_patience,
        )
        goal_results[name] = metrics
        goal_correct[name] = correct
        for key, value in weights.items():
            probe_arrays[f"goal__{name}__{key}"] = value
    majority_label, _ = Counter(goals[: len(train_records)].tolist()).most_common(1)[0]
    val_metadata = record_metadata[len(train_records) :]
    train_leakage_groups = {
        row["inner_group_key"] for row in record_metadata[: len(train_records)]
    }
    val_external = np.asarray(
        [row["inner_group_key"] not in train_leakage_groups for row in val_metadata],
        dtype=np.bool_,
    )
    val_goal_rows = aggregate_task_probe_features(
        features=initial_features["candidate"][len(train_records) :][val_external],
        task_keys=np.asarray([row["observed_task_key"] for row in val_metadata])[val_external],
        labels=goals[len(train_records) :][val_external],
        leakage_group_keys=np.asarray([row["inner_group_key"] for row in val_metadata])[val_external],
    )
    represented = np.isin(val_goal_rows.labels, classes)
    majority_top1 = float((val_goal_rows.labels[represented] == majority_label).mean())
    goal_bootstrap = _paired_accuracy_bootstrap(
        goal_correct["candidate"],
        goal_correct["dino"],
        seed=args.seed,
        draws=args.bootstrap_draws,
    )
    candidate_goal_gate = goal_probe_gate(
        state_micro_top1=goal_results["candidate"]["micro_top1"],
        state_macro_top1=goal_results["candidate"]["macro_top1"],
        dino_micro_top1=goal_results["dino"]["micro_top1"],
        dino_macro_top1=goal_results["dino"]["macro_top1"],
        majority_top1=majority_top1,
        paired_bootstrap_lower=goal_bootstrap["lower_95"],
    )

    outcome_results, outcome_weights = _outcome_probes(
        feature_sets={
            "state": baseline,
            "hidden": hidden,
            "candidate": candidate,
            "dino": dino,
        },
        transition_current_index=transition_current,
        transition_actions=transition_actions,
        transition_split=transition_split,
        transition_record_index=transition_record,
        transition_external_eligible=transition_eligible,
        outcomes=outcomes,
        record_metadata=record_metadata,
        seed=args.seed,
        device=device,
        max_epochs=args.probe_max_epochs,
        patience=args.probe_patience,
        bootstrap_draws=args.bootstrap_draws,
    )
    probe_arrays.update(outcome_weights)

    hidden_goal_support = (
        goal_results["hidden"]["micro_top1"] >= goal_results["state"]["micro_top1"] + 0.02
        and goal_results["hidden"]["macro_top1"] >= goal_results["state"]["macro_top1"] + 0.02
    )
    hidden_outcome_support = all(
        outcome_results[str(action)]["features"]["hidden"]["roc_auc"]
        >= outcome_results[str(action)]["features"]["state"]["roc_auc"]
        for action in (2, 3)
    )
    hidden_support = bool(hidden_goal_support and hidden_outcome_support)
    outcome_checks: dict[str, bool] = {}
    outcome_gate_detail: dict[str, Any] = {}
    for action in _MOVEMENT_ACTIONS:
        section = outcome_results[str(action)]
        candidate_metrics = section["features"]["candidate"]
        versus_state = section["paired_bootstrap"]["candidate_minus_state"]
        versus_dino = section["paired_bootstrap"]["candidate_minus_dino"]
        checks = {
            "above_chance": versus_state["candidate_lower_95"] > 0.5,
            "not_below_dino": versus_dino["lower_95"] >= -0.02,
            "improves_state": (
                versus_state["lower_95"] > 0.0
                if action in (2, 3)
                else versus_state["lower_95"] >= -0.02
            ),
            "brier_beats_constant": candidate_metrics["brier"]
            <= candidate_metrics["constant_train_rate_baseline"]["brier"],
        }
        outcome_gate_detail[str(action)] = checks
        outcome_checks[_ACTION_NAMES[action]] = bool(all(checks.values()))
    gate = canary_gate(
        visual_metrics=visual,
        goal_gate_passed=bool(candidate_goal_gate["passed"]),
        outcome_checks=outcome_checks,
        hidden_probe_supports_calibration=hidden_support,
    )

    checkpoint_path = output_dir / "diagnostic_state_adapter"
    save_canary_checkpoint(final_model, checkpoint_path)
    external_path = output_dir / "external_candidate_state.npz"
    external_arrays = {
        "state": candidate[external_state_indices].astype(np.float32),
        "source_state_index": external_state_indices.astype(np.int64),
    }
    _atomic_npz(external_path, external_arrays)
    _validate_saved_npz(
        external_path,
        {
            "state": (np.dtype("float32"), external_arrays["state"].shape),
            "source_state_index": (np.dtype("int64"), external_arrays["source_state_index"].shape),
        },
    )
    weights_path = output_dir / "diagnostic_readouts.npz"
    _atomic_npz(weights_path, probe_arrays)
    with np.load(weights_path, allow_pickle=False) as saved:
        for key in saved.files:
            if saved[key].dtype != np.float32 or not np.isfinite(saved[key]).all():
                raise ValueError(f"diagnostic readout {key} is invalid")
    history_path = output_dir / "training_history.json"
    _atomic_json(
        history_path,
        {
            "selected_epoch": selected_epoch,
            "selection_phase": selection_history,
            "final_phase": final_history,
        },
    )

    result: dict[str, Any] = {
        "schema": "nimloth_state_interface_direction_canary_v1",
        "authorization": "diagnostic direction canary only; no downstream use",
        "state_semantics": "one unified K16 visual-goal state; no factorized branches",
        "cot_semantics": "actual archived observation-conditioned assistant response",
        "trainable_modules": [
            "bounded zero-initialized rank-64 residual adapter from same-generation hidden",
            "training-only goal head",
            "training-only action-specific success head",
            "fresh diagnostic linear readouts",
        ],
        "frozen_modules": [
            "ID176 actor/Qwen/vision",
            "SFT1 SharedSlotProjector",
            "DINO",
            "all old WM/ValueHead/policy/planner/RL components",
        ],
        "old_checkpoint_reuse": {"ID74_WM": False, "ID75_WM": False, "ValueHead": False},
        "row_task_identity_available": False,
        "selected_epoch": selected_epoch,
        "hidden_identity": hidden_identity,
        "visual": visual,
        "goal_probe": goal_results,
        "goal_candidate_minus_dino_bootstrap": goal_bootstrap,
        "goal_gate": candidate_goal_gate,
        "outcome_probe": outcome_results,
        "hidden_direction_support": {
            "goal": hidden_goal_support,
            "lateral_outcome": hidden_outcome_support,
            "passed": hidden_support,
        },
        "outcome_gate_detail": outcome_gate_detail,
        "gate": gate,
        "artifacts": {
            "hidden_cache": {
                "path": hidden_path.name,
                "sha256": _sha256(hidden_path),
                "dtype": "float32",
                "shape": list(hidden.shape),
            },
            "adapter": {
                "path": checkpoint_path.name,
                "weights_sha256": _sha256(checkpoint_path / "diagnostic_adapter.pt"),
                "config_sha256": _sha256(checkpoint_path / "canary_config.json"),
                "downstream_use_authorized": False,
            },
            "external_candidate_state": {
                "path": external_path.name,
                "sha256": _sha256(external_path),
            },
            "diagnostic_readouts": {
                "path": weights_path.name,
                "sha256": _sha256(weights_path),
                "downstream_use_authorized": False,
            },
            "training_history": {
                "path": history_path.name,
                "sha256": _sha256(history_path),
            },
        },
        "sources": {
            "id60_result_sha256": _sha256(args.id60_result),
            "id71_result_sha256": _sha256(args.id71_result),
            "state_cache_sha256": _sha256(args.state_cache),
            "state_metadata_sha256": _sha256(args.state_cache_metadata),
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
        },
        "hyperparameters": {
            "adapter_rank": args.adapter_rank,
            "max_residual_fraction": args.max_residual_fraction,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "anchor_weight": args.anchor_weight,
            "seed": args.seed,
        },
        "git_commit": args.git_commit,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.html").write_text(_render_html(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_wandb_identity(args)
    import wandb

    handle = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={
            "diagnostic_direction_canary": True,
            "git_commit": args.git_commit,
            "adapter_rank": args.adapter_rank,
            "max_residual_fraction": args.max_residual_fraction,
        },
    )
    try:
        if handle.project != args.wandb_project or handle.id != args.wandb_run_id:
            raise RuntimeError("initialized W&B identity differs from locked ID191 identity")
        result = run(args)
        payload = {
            "gate/passed": int(result["gate"]["passed"]),
            "goal/state_micro": result["goal_probe"]["state"]["micro_top1"],
            "goal/hidden_micro": result["goal_probe"]["hidden"]["micro_top1"],
            "goal/candidate_micro": result["goal_probe"]["candidate"]["micro_top1"],
            "goal/dino_micro": result["goal_probe"]["dino"]["micro_top1"],
        }
        for action, section in result["outcome_probe"].items():
            for feature in ("state", "hidden", "candidate", "dino"):
                payload[f"outcome/action_{action}_{feature}_auc"] = section["features"][feature][
                    "roc_auc"
                ]
        handle.log(payload)
        handle.summary.update(
            {
                "status": "passed",
                "canary_gate_passed": result["gate"]["passed"],
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                **payload,
            }
        )
        handle.finish(exit_code=0)
        return 0
    except BaseException:
        handle.summary["status"] = "failed"
        handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
