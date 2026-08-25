"""Probe whether frozen K16 state predicts movement execution success."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nimloth.eval.id75_action_outcome_audit import binary_auc, parse_step_action_success

_ACTION_NAMES = {
    0: "move_forward",
    2: "move_right",
    3: "move_left",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Tie-aware average precision for success as the positive outcome."""

    target = np.asarray(labels, dtype=np.bool_)
    value = np.asarray(scores, dtype=np.float64)
    if target.shape != value.shape or target.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("average-precision labels/scores must be aligned finite vectors")
    positives = int(target.sum())
    if positives == 0 or positives == len(target):
        raise ValueError("average precision requires both classes")
    order = np.argsort(-value, kind="mergesort")
    ordered_scores = value[order]
    ordered_target = target[order]
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    area = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and ordered_scores[stop] == ordered_scores[start]:
            stop += 1
        true_positives += int(ordered_target[start:stop].sum())
        false_positives += int((~ordered_target[start:stop]).sum())
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
        start = stop
    return float(area)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponent = np.exp(value[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    if bins < 1:
        raise ValueError("ECE bins must be positive")
    total = len(labels)
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index + 1 == bins:
            mask = (probabilities >= edges[index]) & (probabilities <= edges[index + 1])
        else:
            mask = (probabilities >= edges[index]) & (probabilities < edges[index + 1])
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return value


def binary_probe_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    train_success_rate: float,
    ece_bins: int = 10,
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.bool_)
    score = np.asarray(logits, dtype=np.float64)
    if target.shape != score.shape or target.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("binary probe labels/logits must be aligned finite vectors")
    if not target.any() or target.all():
        raise ValueError("binary probe metrics require both outcomes")
    if not 0.0 < train_success_rate < 1.0:
        raise ValueError("training success rate must be strictly between zero and one")
    probabilities = _sigmoid(score)
    predictions = probabilities >= 0.5
    positive_accuracy = float(predictions[target].mean())
    negative_accuracy = float((~predictions[~target]).mean())
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    nll = float(-np.mean(target * np.log(clipped) + (~target) * np.log(1.0 - clipped)))
    brier = float(np.mean(np.square(probabilities - target.astype(np.float64))))
    baseline_probability = float(train_success_rate)
    baseline_nll = float(
        -np.mean(
            target * np.log(baseline_probability)
            + (~target) * np.log(1.0 - baseline_probability)
        )
    )
    baseline_brier = float(
        np.mean(np.square(baseline_probability - target.astype(np.float64)))
    )
    return {
        "count": len(target),
        "success_count": int(target.sum()),
        "failed_count": int((~target).sum()),
        "success_rate": float(target.mean()),
        "roc_auc": binary_auc(target, score),
        "pr_auc": average_precision(target, score),
        "pr_auc_prevalence_baseline": float(target.mean()),
        "accuracy": float((predictions == target).mean()),
        "balanced_accuracy": 0.5 * (positive_accuracy + negative_accuracy),
        "successful_accuracy": positive_accuracy,
        "failed_accuracy": negative_accuracy,
        "nll": nll,
        "brier": brier,
        "ece": _ece(target, probabilities, ece_bins),
        "constant_train_rate_baseline": {
            "probability": baseline_probability,
            "nll": baseline_nll,
            "brier": baseline_brier,
        },
    }


def grouped_selection_mask(group_keys: np.ndarray, *, modulo: int = 10) -> np.ndarray:
    groups = np.asarray(group_keys).astype(str)
    if groups.ndim != 1 or modulo < 2:
        raise ValueError("grouped selection requires a vector and modulo >=2")
    choices = {
        group: int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % modulo == 0
        for group in sorted(set(groups.tolist()))
    }
    selected = np.asarray([choices[group] for group in groups], dtype=np.bool_)
    if not selected.any() or selected.all():
        raise ValueError("grouped probe selection split is empty")
    return selected


def _standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float32)
    mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-3))


def _transform(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(features, dtype=np.float32) - mean) / std).astype(np.float32)


def _fit_epoch_selection(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    selection_features: np.ndarray,
    selection_labels: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    seed: int,
    device: Any = "cuda",
) -> tuple[int, list[dict[str, float]]]:
    import torch

    mean, std = _standardizer(fit_features)
    fit_x = torch.from_numpy(_transform(fit_features, mean, std)).to(device)
    fit_y = torch.from_numpy(np.asarray(fit_labels, dtype=np.float32)).to(device)
    selection_x = torch.from_numpy(_transform(selection_features, mean, std)).to(device)
    torch.manual_seed(seed)
    model = torch.nn.Linear(fit_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_epoch = 1
    best_auc = -float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(fit_x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, fit_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("outcome probe loss is non-finite")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            selection_logits = model(selection_x).squeeze(1).float().cpu().numpy()
        auc = binary_auc(selection_labels, selection_logits)
        history.append({"epoch": float(epoch), "fit_bce": float(loss.item()), "selection_auc": auc})
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best_epoch, history


def _train_final_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    query_features: np.ndarray,
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    seed: int,
    device: Any = "cuda",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    import torch

    mean, std = _standardizer(train_features)
    train_x = torch.from_numpy(_transform(train_features, mean, std)).to(device)
    train_y = torch.from_numpy(np.asarray(train_labels, dtype=np.float32)).to(device)
    query_x = torch.from_numpy(_transform(query_features, mean, std)).to(device)
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError("final outcome probe loss is non-finite")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        query_logits = model(query_x).squeeze(1).float().cpu().numpy()
    weights = {
        "weight": model.weight.detach().float().cpu().numpy(),
        "bias": model.bias.detach().float().cpu().numpy(),
        "feature_mean": mean,
        "feature_std": std,
    }
    return query_logits, weights


def _paired_auc_bootstrap(
    labels: np.ndarray,
    state_logits: np.ndarray,
    dino_logits: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    target = np.asarray(labels, dtype=np.bool_)
    state = np.asarray(state_logits, dtype=np.float64)
    dino = np.asarray(dino_logits, dtype=np.float64)
    rng = np.random.default_rng(seed)
    state_auc: list[float] = []
    dino_auc: list[float] = []
    for _ in range(draws):
        index = rng.integers(0, len(target), size=len(target))
        sampled = target[index]
        if not sampled.any() or sampled.all():
            continue
        state_auc.append(binary_auc(sampled, state[index]))
        dino_auc.append(binary_auc(sampled, dino[index]))
    if len(state_auc) < max(100, draws // 2):
        raise RuntimeError("too few valid paired outcome bootstrap draws")
    state_array = np.asarray(state_auc)
    dino_array = np.asarray(dino_auc)
    difference = state_array - dino_array
    return {
        "requested_draws": draws,
        "valid_draws": len(state_array),
        "state_lower_95": float(np.quantile(state_array, 0.025)),
        "state_upper_95": float(np.quantile(state_array, 0.975)),
        "dino_lower_95": float(np.quantile(dino_array, 0.025)),
        "dino_upper_95": float(np.quantile(dino_array, 0.975)),
        "state_minus_dino_mean": float(difference.mean()),
        "state_minus_dino_lower_95": float(np.quantile(difference, 0.025)),
        "state_minus_dino_upper_95": float(np.quantile(difference, 0.975)),
        "state_above_chance": bool(float(np.quantile(state_array, 0.025)) > 0.5),
        "state_above_dino": bool(float(np.quantile(difference, 0.025)) > 0.0),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _render_html(result: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{section['action_name']}</td><td>{section['state']['roc_auc']:.4f}</td>"
        f"<td>{section['dino']['roc_auc']:.4f}</td><td>{section['paired_bootstrap']['state_minus_dino_mean']:.4f}</td></tr>"
        for section in result["per_action"].values()
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID71 outcome probe</title>
<style>body{{font-family:system-ui;max-width:900px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID71 frozen-state action-outcome probe</h1><table><tr><th>action</th><th>state AUC</th><th>DINO AUC</th><th>state-DINO</th></tr>{rows}</table>
<p>Only matched diagnostic linear readouts were trained. Actor, projector, DINO and WM stayed frozen.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--state-cache-metadata", type=Path, required=True)
    parser.add_argument("--id60-result", type=Path, required=True)
    parser.add_argument("--id61-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42071)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def _validate_wandb_identity(args: argparse.Namespace) -> None:
    if args.wandb_project != "nimloth-recon":
        raise ValueError("ID71 W&B project must be nimloth-recon")
    if not args.wandb_run_id.startswith("nimloth-recon-id71-action-outcome-probe"):
        raise ValueError("ID71 W&B ID is outside the locked namespace")
    if os.environ.get("WANDB_PROJECT") != args.wandb_project:
        raise ValueError("effective WANDB_PROJECT differs from locked ID71 project")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_run_id:
        raise ValueError("effective WANDB_RUN_ID differs from locked ID71 identity")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("ID71 requires one CUDA GPU")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create ID71 output")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID71 output is not fresh")
    id60 = json.loads(args.id60_result.read_text(encoding="utf-8"))
    id61 = json.loads(args.id61_result.read_text(encoding="utf-8"))
    if id60.get("schema") != "nimloth_frozen_state_goal_probe_v1":
        raise ValueError("ID60 schema mismatch")
    if id61.get("schema") != "nimloth_id75_action_outcome_audit_v1":
        raise ValueError("ID61 schema mismatch")
    if _sha256(args.state_cache) != id60["state_cache"]["sha256"]:
        raise ValueError("ID60 state cache hash mismatch")
    if _sha256(args.state_cache_metadata) != id60["state_cache"]["metadata_sha256"]:
        raise ValueError("ID60 metadata hash mismatch")

    train_records = _load_jsonl(args.train_jsonl)
    val_records = _load_jsonl(args.val_jsonl)
    records = train_records + val_records
    metadata = json.loads(args.state_cache_metadata.read_text(encoding="utf-8"))
    if len(metadata["records"]) != len(records):
        raise ValueError("record metadata count mismatch")
    for index, (row, record) in enumerate(zip(metadata["records"], records, strict=True)):
        if row["record_id"] != str(record["id"]):
            raise ValueError(f"record alignment differs at {index}")

    with np.load(args.state_cache, allow_pickle=False) as cache:
        state = cache["state"].astype(np.float32)
        dino = cache["dino"].astype(np.float32)
        current_index = cache["transition_current_index"].astype(np.int64)
        actions = cache["transition_action"].astype(np.int64)
        split = cache["transition_split"].astype(np.int64)
        record_index = cache["transition_record_index"].astype(np.int64)
        eligible = cache["transition_external_eligible"].astype(np.bool_)
        state_step = cache["state_step_index"].astype(np.int64)
    outcomes = np.asarray(
        [
            parse_step_action_success(records[record], int(state_step[current]))
            for record, current in zip(record_index, current_index, strict=True)
        ],
        dtype=np.bool_,
    )
    record_groups = np.asarray(
        [metadata["records"][record]["inner_group_key"] for record in record_index]
    )
    train_mask = split == 0
    external_mask = (split == 1) & eligible
    if set(record_groups[train_mask]) & set(record_groups[external_mask]):
        raise ValueError("train/external exact initial-image groups overlap")
    current_state = state[current_index].reshape(len(current_index), -1)
    current_dino = dino[current_index].reshape(len(current_index), -1)

    result_actions: dict[str, Any] = {}
    weight_arrays: dict[str, np.ndarray] = {}
    history: dict[str, Any] = {}
    for action, action_name in _ACTION_NAMES.items():
        action_train = train_mask & (actions == action)
        action_external = external_mask & (actions == action)
        selected_local = grouped_selection_mask(record_groups[action_train])
        train_indices = np.flatnonzero(action_train)
        fit_indices = train_indices[~selected_local]
        selection_indices = train_indices[selected_local]
        external_indices = np.flatnonzero(action_external)
        for name, indices in (
            ("fit", fit_indices),
            ("selection", selection_indices),
            ("external", external_indices),
        ):
            labels = outcomes[indices]
            if not labels.any() or labels.all():
                raise ValueError(f"action {action} {name} lacks both outcomes")
        feature_logits: dict[str, np.ndarray] = {}
        feature_results: dict[str, Any] = {}
        for feature_offset, (feature_name, features) in enumerate(
            (("state", current_state), ("dino", current_dino))
        ):
            selected_epoch, selected_history = _fit_epoch_selection(
                features[fit_indices],
                outcomes[fit_indices],
                features[selection_indices],
                outcomes[selection_indices],
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                patience=args.patience,
                seed=args.seed + action * 10 + feature_offset,
            )
            logits, weights = _train_final_probe(
                features[train_indices],
                outcomes[train_indices],
                features[external_indices],
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=selected_epoch,
                seed=args.seed + action * 10 + feature_offset,
            )
            feature_logits[feature_name] = logits
            feature_results[feature_name] = {
                **binary_probe_metrics(
                    outcomes[external_indices],
                    logits,
                    train_success_rate=float(outcomes[train_indices].mean()),
                ),
                "selected_epoch": selected_epoch,
                "parameter_count": int(features.shape[1] + 1),
            }
            history[f"action_{action}_{feature_name}"] = selected_history
            for key, value in weights.items():
                weight_arrays[f"action_{action}__{feature_name}__{key}"] = value.astype(np.float32)
        paired = _paired_auc_bootstrap(
            outcomes[external_indices],
            feature_logits["state"],
            feature_logits["dino"],
            seed=args.seed + action,
            draws=args.bootstrap_draws,
        )
        result_actions[str(action)] = {
            "action_name": action_name,
            "fit_count": len(fit_indices),
            "selection_count": len(selection_indices),
            "external_count": len(external_indices),
            "state": feature_results["state"],
            "dino": feature_results["dino"],
            "paired_bootstrap": paired,
            "id75_predicted_change_success_auc": id61["external_validation"]["per_action"][str(action)][
                "predicted_change_success_auc"
            ],
        }

    weights_path = output_dir / "diagnostic_action_outcome_readouts.npz"
    _atomic_npz(weights_path, weight_arrays)
    with np.load(weights_path, allow_pickle=False) as saved:
        if set(saved.files) != set(weight_arrays):
            raise ValueError("saved outcome readout keys differ")
        for key in saved.files:
            if saved[key].dtype != np.float32 or not np.isfinite(saved[key]).all():
                raise ValueError(f"saved outcome readout {key} is invalid")
    history_path = output_dir / "selection_history.json"
    _atomic_json(history_path, history)
    result: dict[str, Any] = {
        "schema": "nimloth_frozen_state_action_outcome_probe_v1",
        "trainable_modules": ["matched_action_specific_linear_readouts"],
        "frozen_modules": ["ID176 actor/Qwen", "vision", "SFT1 SharedSlotProjector", "DINO", "ID75 WM"],
        "optimizer_updates_to_project_modules": 0,
        "row_task_identity_available": False,
        "split_semantics": "archive train/external boundary; exact initial/current/next images decontaminated by ID60, inner selection groups exact initial-image hashes",
        "label_semantics": "exact archived environment feedback after each action",
        "feature_semantics": "matched flattened K16 float32 state and DINO with fit-only per-dimension standardization",
        "per_action": result_actions,
        "probe_weights": {
            "path": weights_path.name,
            "sha256": _sha256(weights_path),
            "downstream_use_authorized": False,
        },
        "selection_history": {
            "path": history_path.name,
            "sha256": _sha256(history_path),
        },
        "sources": {
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
            "state_cache_sha256": _sha256(args.state_cache),
            "id60_result_sha256": _sha256(args.id60_result),
            "id61_result_sha256": _sha256(args.id61_result),
        },
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "bootstrap_draws": args.bootstrap_draws,
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
        config={"git_commit": args.git_commit, "diagnostic_readout_only": True},
    )
    try:
        if handle.project != args.wandb_project or handle.id != args.wandb_run_id:
            raise RuntimeError("initialized W&B identity differs from locked ID71 identity")
        result = run(args)
        payload: dict[str, Any] = {}
        for action, section in result["per_action"].items():
            payload[f"action_{action}/state_auc"] = section["state"]["roc_auc"]
            payload[f"action_{action}/dino_auc"] = section["dino"]["roc_auc"]
            payload[f"action_{action}/state_minus_dino_auc"] = section["paired_bootstrap"][
                "state_minus_dino_mean"
            ]
        handle.log(payload)
        handle.summary.update(
            {
                "status": "passed",
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
