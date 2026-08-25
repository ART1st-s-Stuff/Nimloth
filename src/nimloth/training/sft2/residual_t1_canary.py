"""Train a zero-copy-initialized T1 residual WM on the frozen ID60 state cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nimloth.eval.sft_checkpoint_state_matrix import _sha256
from nimloth.wm.grid import (
    GridPredictorConfig,
    ResidualTemporalSpatialGridPredictor,
)


def _finite(name: str, value: np.ndarray, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite rank-{ndim} array")
    return array


def _transition_section(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    dino: np.ndarray,
) -> dict[str, float]:
    current64 = current.astype(np.float64)
    target64 = target.astype(np.float64)
    prediction64 = prediction.astype(np.float64)
    dino64 = dino.astype(np.float64)
    copy_mse = float(np.mean(np.square(current64 - target64)))
    prediction_mse = float(np.mean(np.square(prediction64 - target64)))
    copy_dino_rmse = float(np.sqrt(np.mean(np.square(current64 - dino64))))
    prediction_dino_rmse = float(np.sqrt(np.mean(np.square(prediction64 - dino64))))
    return {
        "count": int(len(current)),
        "copy_rmse": float(np.sqrt(copy_mse)),
        "prediction_rmse": float(np.sqrt(prediction_mse)),
        "copy_relative_skill": float(1.0 - prediction_mse / max(copy_mse, 1e-12)),
        "copy_next_dino_rmse": copy_dino_rmse,
        "prediction_next_dino_rmse": prediction_dino_rmse,
    }


def residual_t1_metrics(
    *,
    current_state: np.ndarray,
    actual_next_state: np.ndarray,
    predicted_next_state: np.ndarray,
    next_dino: np.ndarray,
    actions: np.ndarray,
    minimum_primary_count: int = 20,
) -> dict[str, Any]:
    """Report natural-distribution copy-relative metrics and strict action gates."""

    current = _finite("current_state", current_state, 3).astype(np.float32)
    target = _finite("actual_next_state", actual_next_state, 3).astype(np.float32)
    prediction = _finite("predicted_next_state", predicted_next_state, 3).astype(np.float32)
    dino = _finite("next_dino", next_dino, 3).astype(np.float32)
    action = np.asarray(actions, dtype=np.int64)
    if not (
        current.shape == target.shape == prediction.shape == dino.shape
        and action.shape == (len(current),)
    ):
        raise ValueError("T1 metric arrays do not align")
    if minimum_primary_count < 1:
        raise ValueError("minimum_primary_count must be positive")

    overall = _transition_section(current, target, prediction, dino)
    per_action: dict[str, dict[str, float]] = {}
    primary: list[int] = []
    for action_id in sorted(set(action.tolist())):
        mask = action == action_id
        per_action[str(action_id)] = _transition_section(
            current[mask], target[mask], prediction[mask], dino[mask]
        )
        if int(mask.sum()) >= minimum_primary_count:
            primary.append(action_id)
    if not primary:
        raise ValueError("no action has enough validation support for a primary gate")
    primary_skills = [per_action[str(index)]["copy_relative_skill"] for index in primary]
    target_std = float(target.astype(np.float64).std())
    prediction_std = float(prediction.astype(np.float64).std())
    std_ratio = prediction_std / max(target_std, 1e-12)
    checks = {
        "all_primary_action_skill_positive": bool(all(skill > 0.0 for skill in primary_skills)),
        "macro_primary_skill_positive": bool(float(np.mean(primary_skills)) > 0.0),
        "overall_skill_positive": bool(overall["copy_relative_skill"] > 0.0),
        "std_ratio_in_range": bool(0.9 <= std_ratio <= 1.1),
        "next_dino_no_worse_than_copy": bool(
            overall["prediction_next_dino_rmse"] <= overall["copy_next_dino_rmse"]
        ),
    }
    return {
        "overall": overall,
        "per_action": per_action,
        "primary_actions": primary,
        "macro_primary_skill": float(np.mean(primary_skills)),
        "predicted_std": prediction_std,
        "actual_next_std": target_std,
        "predicted_actual_std_ratio": std_ratio,
        "gate": {
            **checks,
            "passed": bool(all(checks.values())),
            "strong_overall_skill_above_0p2": bool(overall["copy_relative_skill"] > 0.2),
        },
    }


def _group_selection_mask(group_keys: np.ndarray) -> np.ndarray:
    selected = np.asarray(
        [int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % 10 == 0 for key in group_keys],
        dtype=bool,
    )
    if not selected.any() or selected.all():
        raise ValueError("deterministic T1 inner task split is empty")
    return selected


def _predict_batches(
    model: Any,
    current: np.ndarray,
    actions: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> np.ndarray:
    import torch

    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(current), batch_size):
            stop = min(len(current), start + batch_size)
            prediction = model(
                torch.from_numpy(current[start:stop]).to(device=device, dtype=torch.float32),
                torch.from_numpy(actions[start:stop]).to(device=device, dtype=torch.long),
            )
            output.append(prediction.float().cpu().numpy())
    return np.concatenate(output, axis=0)


def _selection_score(metrics: dict[str, Any]) -> float:
    skills = [
        section["copy_relative_skill"]
        for section in metrics["per_action"].values()
        if int(section["count"]) >= 10
    ]
    if not skills:
        return float(metrics["overall"]["copy_relative_skill"])
    return float(np.mean(skills))


def _sampling_weights(actions: np.ndarray) -> np.ndarray:
    counts = Counter(np.asarray(actions, dtype=np.int64).tolist())
    largest = max(counts.values())
    per_action = {
        action: min(8.0, float(np.sqrt(largest / count)))
        for action, count in counts.items()
    }
    weights = np.asarray([per_action[int(action)] for action in actions], dtype=np.float64)
    return weights / weights.sum()


def _train_model(
    *,
    model: Any,
    current: np.ndarray,
    target: np.ndarray,
    actions: np.ndarray,
    device: Any,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    phase: str,
    step_log: list[dict[str, Any]],
    selection: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    patience: int | None = None,
) -> tuple[Any, int, list[dict[str, float]]]:
    import torch

    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    probabilities = torch.from_numpy(_sampling_weights(actions))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_score = -float("inf")
    stale = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        sampled = torch.multinomial(
            probabilities,
            num_samples=len(actions),
            replacement=True,
            generator=generator,
        ).numpy()
        model.train()
        losses: list[float] = []
        for start in range(0, len(sampled), batch_size):
            indices = sampled[start : start + batch_size]
            x = torch.from_numpy(current[indices]).to(device=device, dtype=torch.float32)
            y = torch.from_numpy(target[indices]).to(device=device, dtype=torch.float32)
            a = torch.from_numpy(actions[indices]).to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, a)
            loss = torch.nn.functional.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("T1 training loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            loss_value = float(loss.detach().item())
            losses.append(loss_value)
            step_log.append(
                {
                    "phase": phase,
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss_value,
                }
            )
        row: dict[str, float] = {"epoch": float(epoch), "train_mse": float(np.mean(losses))}
        if selection is not None:
            select_current, select_target, select_actions, select_dino = selection
            prediction = _predict_batches(
                model,
                select_current,
                select_actions,
                device=device,
                batch_size=batch_size,
            )
            metrics = residual_t1_metrics(
                current_state=select_current,
                actual_next_state=select_target,
                predicted_next_state=prediction,
                next_dino=select_dino,
                actions=select_actions,
                minimum_primary_count=10,
            )
            score = _selection_score(metrics)
            row["selection_macro_action_skill"] = score
            row["selection_overall_skill"] = float(metrics["overall"]["copy_relative_skill"])
            if score > best_score + 1e-6:
                best_score = score
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            history.append(row)
            if patience is not None and stale >= patience:
                break
        else:
            history.append(row)
    return model, best_epoch, history


def _write_step_log(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "epoch", "global_step", "loss"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _save_checkpoint_atomic(model: ResidualTemporalSpatialGridPredictor, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    model.save_checkpoint(temporary)
    os.replace(temporary, path)


def _render_html(result: dict[str, Any]) -> str:
    metrics = result["external_validation"]
    rows = "".join(
        f"<tr><td>{action}</td><td>{section['count']}</td><td>{section['copy_rmse']:.5f}</td>"
        f"<td>{section['prediction_rmse']:.5f}</td><td>{section['copy_relative_skill']:.5f}</td></tr>"
        for action, section in metrics["per_action"].items()
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID75 residual T1 canary</title>
<style>body{{font-family:system-ui;max-width:900px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID75 frozen-SFT1 Residual-T1 canary</h1><p>Canary gate: <strong>{html.escape(str(metrics['gate']['passed']))}</strong>; overall skill {metrics['overall']['copy_relative_skill']:.5f}.</p>
<table><tr><th>action</th><th>N</th><th>copy RMSE</th><th>prediction RMSE</th><th>skill</th></tr>{rows}</table>
<p>Only the residual predictor was trained. Raw DINO loss weight was exactly zero.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--state-cache-metadata", type=Path, required=True)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--minimum-primary-count", type=int, default=20)
    parser.add_argument("--raw-dino-loss-weight", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42075)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.raw_dino_loss_weight != 0.0:
        raise ValueError("ID75 raw DINO training loss must be exactly zero")
    if not torch.cuda.is_available():
        raise RuntimeError("ID75 requires one CUDA GPU")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create output_dir")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID75 output is not fresh")

    probe = json.loads(args.probe_result.read_text(encoding="utf-8"))
    if probe.get("schema") != "nimloth_frozen_state_goal_probe_v1":
        raise ValueError("ID75 probe result schema mismatch")
    expected_cache_hash = probe["state_cache"]["sha256"]
    if _sha256(args.state_cache) != expected_cache_hash:
        raise ValueError("ID75 state cache hash differs from ID60 result")
    metadata_hash = probe["state_cache"]["metadata_sha256"]
    if _sha256(args.state_cache_metadata) != metadata_hash:
        raise ValueError("ID75 state cache metadata hash differs from ID60 result")

    with np.load(args.state_cache, allow_pickle=False) as cache:
        required = {
            "state",
            "dino",
            "transition_current_index",
            "transition_next_index",
            "transition_action",
            "transition_split",
            "transition_record_index",
            "transition_external_eligible",
        }
        if not required.issubset(cache.files):
            raise ValueError(f"ID60 cache is missing arrays: {sorted(required - set(cache.files))}")
        state = cache["state"].astype(np.float32)
        dino = cache["dino"].astype(np.float32)
        current_index = cache["transition_current_index"].astype(np.int64)
        next_index = cache["transition_next_index"].astype(np.int64)
        actions = cache["transition_action"].astype(np.int64)
        split = cache["transition_split"].astype(np.int64)
        record_index = cache["transition_record_index"].astype(np.int64)
        external_eligible = cache["transition_external_eligible"].astype(np.bool_)
    if state.shape != dino.shape or state.ndim != 3 or state.shape[1:] != (16, 1024):
        raise ValueError("ID75 frozen state/DINO arrays have invalid shape")
    if not np.isfinite(state).all() or not np.isfinite(dino).all():
        raise ValueError("ID75 frozen state/DINO arrays are non-finite")

    metadata = json.loads(args.state_cache_metadata.read_text(encoding="utf-8"))
    records = metadata["records"]
    group_keys = np.asarray([records[index]["inner_group_key"] for index in record_index])
    current = state[current_index]
    target = state[next_index]
    next_dino = dino[next_index]
    train_mask = split == 0
    raw_val_mask = split == 1
    val_mask = raw_val_mask & external_eligible
    train_task_selection = _group_selection_mask(group_keys[train_mask])
    fit_mask = train_mask.copy()
    selection_mask = train_mask.copy()
    fit_mask[train_mask] = ~train_task_selection
    selection_mask[train_mask] = train_task_selection
    if set(group_keys[fit_mask]) & set(group_keys[selection_mask]):
        raise ValueError("T1 fit/selection exact-image groups overlap")
    if set(group_keys[train_mask]) & set(group_keys[val_mask]):
        raise ValueError("T1 train/external-validation exact-image groups overlap")

    config = GridPredictorConfig(
        grid_tokens=16,
        emb_dim=1024,
        history_size=1,
        action_dim=8,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
    )
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    selection_model = ResidualTemporalSpatialGridPredictor(config)
    if not selection_model.is_zero_initialized():
        raise RuntimeError("T1 selection model is not zero initialized")
    zero_probe = torch.from_numpy(current[fit_mask][: min(8, int(fit_mask.sum()))])
    zero_action = torch.from_numpy(actions[fit_mask][: len(zero_probe)])
    with torch.inference_mode():
        zero_prediction = selection_model(zero_probe, zero_action)
    if not torch.equal(zero_prediction, zero_probe):
        raise RuntimeError("T1 zero initialization is not exact copy")

    step_log: list[dict[str, Any]] = []
    selection_model, selected_epoch, selection_history = _train_model(
        model=selection_model,
        current=current[fit_mask],
        target=target[fit_mask],
        actions=actions[fit_mask],
        device=device,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        phase="inner_selection",
        step_log=step_log,
        selection=(
            current[selection_mask],
            target[selection_mask],
            actions[selection_mask],
            next_dino[selection_mask],
        ),
        patience=args.patience,
    )
    del selection_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    final_model = ResidualTemporalSpatialGridPredictor(config)
    if not final_model.is_zero_initialized():
        raise RuntimeError("T1 final model is not zero initialized")
    final_model, _, final_history = _train_model(
        model=final_model,
        current=current[train_mask],
        target=target[train_mask],
        actions=actions[train_mask],
        device=device,
        epochs=selected_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        phase="final_train",
        step_log=step_log,
    )
    prediction = _predict_batches(
        final_model,
        current[val_mask],
        actions[val_mask],
        device=device,
        batch_size=args.batch_size,
    )
    metrics = residual_t1_metrics(
        current_state=current[val_mask],
        actual_next_state=target[val_mask],
        predicted_next_state=prediction,
        next_dino=next_dino[val_mask],
        actions=actions[val_mask],
        minimum_primary_count=args.minimum_primary_count,
    )

    checkpoint_dir = output_dir / "residual_t1_checkpoint"
    _save_checkpoint_atomic(final_model.cpu(), checkpoint_dir)
    _write_step_log(output_dir / "train_step_log.csv", step_log)
    history = {"inner_selection": selection_history, "final_train": final_history}
    _atomic_json(output_dir / "train_history.json", history)
    result: dict[str, Any] = {
        "schema": "nimloth_frozen_sft1_residual_t1_canary_v1",
        "trainable_modules": ["ResidualTemporalSpatialGridPredictor"],
        "frozen_modules": [
            "ID176 actor/Qwen",
            "vision",
            "SFT1 SharedSlotProjector",
            "DINO",
        ],
        "absent_modules": ["ValueHead", "planner", "policy", "RL optimizer"],
        "raw_dino_training_loss_weight": 0.0,
        "zero_initial_prediction_exact_copy": True,
        "selected_epoch": selected_epoch,
        "counts": {
            "fit_transitions": int(fit_mask.sum()),
            "inner_selection_transitions": int(selection_mask.sum()),
            "all_train_transitions": int(train_mask.sum()),
            "external_validation_transitions": int(val_mask.sum()),
            "external_validation_excluded_cross_split_image": int(
                raw_val_mask.sum() - val_mask.sum()
            ),
            "train_action_counts": {
                str(key): value for key, value in sorted(Counter(actions[train_mask].tolist()).items())
            },
            "validation_action_counts": {
                str(key): value for key, value in sorted(Counter(actions[val_mask].tolist()).items())
            },
        },
        "external_validation": metrics,
        "checkpoint": {
            "path": checkpoint_dir.name,
            "predictor_sha256": _sha256(checkpoint_dir / "predictor.pt"),
            "config_sha256": _sha256(checkpoint_dir / "config.json"),
            "downstream_use_authorized": False,
        },
        "step_log": {
            "path": "train_step_log.csv",
            "sha256": _sha256(output_dir / "train_step_log.csv"),
            "rows": len(step_log),
        },
        "source": {
            "id60_result": str(args.probe_result.resolve()),
            "id60_result_sha256": _sha256(args.probe_result),
            "state_cache": str(args.state_cache.resolve()),
            "state_cache_sha256": expected_cache_hash,
            "goal_gate_passed": bool(probe["goal_gate"]["passed"]),
        },
        "hyperparameters": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "model": {
                "depth": 6,
                "heads": 16,
                "dim_head": 64,
                "mlp_dim": 2048,
                "dropout": 0.1,
            },
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
    import wandb

    run_handle = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={
            "state_cache": str(args.state_cache),
            "raw_dino_loss_weight": args.raw_dino_loss_weight,
            "seed": args.seed,
            "git_commit": args.git_commit,
        },
    )
    try:
        result = run(args)
        metrics = result["external_validation"]
        payload = {
            "t1/overall_skill": metrics["overall"]["copy_relative_skill"],
            "t1/macro_primary_skill": metrics["macro_primary_skill"],
            "t1/std_ratio": metrics["predicted_actual_std_ratio"],
            "t1/predicted_next_dino_rmse": metrics["overall"]["prediction_next_dino_rmse"],
            "t1/copy_next_dino_rmse": metrics["overall"]["copy_next_dino_rmse"],
        }
        for action, section in metrics["per_action"].items():
            payload[f"t1/action_{action}_skill"] = section["copy_relative_skill"]
        run_handle.log(payload)
        run_handle.summary.update(
            {
                "status": "passed",
                "canary_gate_passed": metrics["gate"]["passed"],
                "goal_gate_passed": result["source"]["goal_gate_passed"],
                "selected_epoch": result["selected_epoch"],
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                "predictor_sha256": result["checkpoint"]["predictor_sha256"],
            }
        )
        run_handle.finish(exit_code=0)
        return 0
    except BaseException:
        run_handle.summary["status"] = "failed"
        run_handle.finish(exit_code=1)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
