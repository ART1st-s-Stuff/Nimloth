"""Audit whether action-execution outcomes explain ID75 per-action WM failure."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_SUCCESS_FEEDBACK = "Last action is executed successfully."
_FAILURE_FEEDBACK = "Last action is not executed successfully."
_ACTION_NAMES = (
    "move_forward",
    "move_backward",
    "move_right",
    "move_left",
    "turn_right",
    "turn_left",
    "look_up",
    "look_down",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_step_action_success(record: dict[str, Any], step_index: int) -> bool:
    """Read the exact environment feedback emitted after one recorded action."""

    actions = record.get("action_indices")
    observations = record.get("observation_texts")
    if not isinstance(actions, list) or not isinstance(observations, list):
        raise ValueError("trajectory lacks action_indices/observation_texts")
    if not 0 <= step_index < len(actions) or len(observations) != len(actions) + 1:
        raise ValueError("trajectory action/observation alignment is invalid")
    feedback = str(observations[step_index + 1])
    successful = _SUCCESS_FEEDBACK in feedback
    failed = _FAILURE_FEEDBACK in feedback
    if int(successful) + int(failed) != 1:
        raise ValueError("step feedback must contain exactly one authoritative outcome")
    return successful


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute tie-aware ROC AUC without a sklearn dependency."""

    target = np.asarray(labels, dtype=np.bool_)
    value = np.asarray(scores, dtype=np.float64)
    if target.shape != value.shape or target.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("binary AUC labels/scores must be aligned finite vectors")
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("binary AUC requires both classes")
    rank_sum = float(_average_ranks(value)[target].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 1 or not np.isfinite(data).all() or draws < 1:
        raise ValueError("bootstrap input must be a nonempty finite vector")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        width = min(500, draws - start)
        indices = rng.integers(0, len(data), size=(width, len(data)))
        means[start : start + width] = data[indices].mean(axis=1)
    return {
        "draws": draws,
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _outcome_section(
    current: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    current64 = np.asarray(current, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    prediction64 = np.asarray(prediction, dtype=np.float64)
    if not (
        current64.shape == target64.shape == prediction64.shape
        and current64.ndim == 3
        and len(current64) > 0
    ):
        raise ValueError("outcome metric state tensors must align and be nonempty")
    axes = tuple(range(1, current64.ndim))
    copy_row_mse = np.square(current64 - target64).mean(axis=axes)
    prediction_row_mse = np.square(prediction64 - target64).mean(axis=axes)
    actual_change = np.sqrt(copy_row_mse)
    predicted_change = np.sqrt(np.square(prediction64 - current64).mean(axis=axes))
    copy_mse = float(copy_row_mse.mean())
    prediction_mse = float(prediction_row_mse.mean())
    return {
        "count": len(current64),
        "copy_rmse": float(np.sqrt(copy_mse)),
        "prediction_rmse": float(np.sqrt(prediction_mse)),
        "copy_relative_skill": float(1.0 - prediction_mse / max(copy_mse, 1e-12)),
        "prediction_minus_copy_mse": float(prediction_mse - copy_mse),
        "copy_minus_prediction_mse_bootstrap": _bootstrap_mean_interval(
            copy_row_mse - prediction_row_mse,
            seed=seed,
            draws=draws,
        ),
        "actual_change_rms_mean": float(actual_change.mean()),
        "actual_change_rms_median": float(np.median(actual_change)),
        "predicted_change_rms_mean": float(predicted_change.mean()),
        "predicted_change_rms_median": float(np.median(predicted_change)),
    }


def stratified_outcome_metrics(
    *,
    current_state: np.ndarray,
    actual_next_state: np.ndarray,
    predicted_next_state: np.ndarray,
    action_success: np.ndarray,
    bootstrap_seed: int = 42061,
    bootstrap_draws: int = 10000,
) -> dict[str, Any]:
    """Compare WM vs copy separately on successful and failed executions."""

    current = np.asarray(current_state, dtype=np.float32)
    target = np.asarray(actual_next_state, dtype=np.float32)
    prediction = np.asarray(predicted_next_state, dtype=np.float32)
    success = np.asarray(action_success, dtype=np.bool_)
    if not (
        current.shape == target.shape == prediction.shape
        and current.ndim == 3
        and success.shape == (len(current),)
        and np.isfinite(current).all()
        and np.isfinite(target).all()
        and np.isfinite(prediction).all()
    ):
        raise ValueError("stratified outcome arrays do not align")
    if not success.any() or success.all():
        raise ValueError("stratified outcome metrics require both outcomes")
    output: dict[str, Any] = {}
    for offset, (name, mask) in enumerate((
        ("failed", ~success),
        ("successful", success),
    )):
        output[name] = _outcome_section(
            current[mask],
            target[mask],
            prediction[mask],
            seed=bootstrap_seed + offset,
            draws=bootstrap_draws,
        )
    actual_change = np.sqrt(np.square(target.astype(np.float64) - current).mean(axis=(1, 2)))
    predicted_change = np.sqrt(
        np.square(prediction.astype(np.float64) - current).mean(axis=(1, 2))
    )
    output["failure_rate"] = float((~success).mean())
    output["actual_change_success_auc"] = binary_auc(success, actual_change)
    output["predicted_change_success_auc"] = binary_auc(success, predicted_change)
    return output


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _archive_outcome_counts(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {index: Counter() for index in range(len(_ACTION_NAMES))}
    for record in records:
        for step_index, action in enumerate(record["action_indices"]):
            outcome = "successful" if parse_step_action_success(record, step_index) else "failed"
            counts[int(action)][outcome] += 1
    return {
        str(action): {
            "action_name": _ACTION_NAMES[action],
            "count": int(sum(counts[action].values())),
            "successful": int(counts[action]["successful"]),
            "failed": int(counts[action]["failed"]),
            "failure_rate": (
                float(counts[action]["failed"] / sum(counts[action].values()))
                if counts[action]
                else None
            ),
        }
        for action in range(len(_ACTION_NAMES))
    }


def _count_outcomes(actions: np.ndarray, success: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for action in sorted(set(actions[mask].tolist())):
        action_mask = mask & (actions == action)
        count = int(action_mask.sum())
        failed = int((~success[action_mask]).sum())
        output[str(action)] = {
            "action_name": _ACTION_NAMES[action],
            "count": count,
            "successful": count - failed,
            "failed": failed,
            "failure_rate": float(failed / count),
        }
    return output


def _predict(model: Any, state: np.ndarray, actions: np.ndarray, *, batch_size: int) -> np.ndarray:
    import torch

    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(state), batch_size):
            stop = min(len(state), start + batch_size)
            rows.append(
                model(
                    torch.from_numpy(state[start:stop]).cuda(non_blocking=True),
                    torch.from_numpy(actions[start:stop]).cuda(non_blocking=True),
                ).float().cpu().numpy()
            )
    return np.concatenate(rows)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _render_html(result: dict[str, Any]) -> str:
    move_left = result["external_validation"]["per_action"]["3"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{section['count']}</td><td>{section['copy_rmse']:.5f}</td>"
        f"<td>{section['prediction_rmse']:.5f}</td><td>{section['copy_relative_skill']:.5f}</td></tr>"
        for name, section in (("failed", move_left["failed"]), ("successful", move_left["successful"]))
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><title>ID61 action-outcome audit</title>
<style>body{{font-family:system-ui;max-width:900px;margin:auto;padding:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd;padding:7px;text-align:right}}</style></head><body>
<h1>ID61 ID75 action-outcome audit</h1><p>Move-left train/external failure rates: {result['hypothesis']['move_left_train_failure_rate']:.3f} / {result['hypothesis']['move_left_external_failure_rate']:.3f}.</p>
<table><tr><th>move_left outcome</th><th>N</th><th>copy RMSE</th><th>prediction RMSE</th><th>skill</th></tr>{rows}</table>
<p>Predicted-change success AUC: {move_left['predicted_change_success_auc']:.4f}. Labels are exact archived environment feedback.</p></body></html>"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--state-cache-metadata", type=Path, required=True)
    parser.add_argument("--id60-result", type=Path, required=True)
    parser.add_argument("--id75-result", type=Path, required=True)
    parser.add_argument("--predictor-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42061)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--wandb-run-id", required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nimloth.wm.grid import ResidualTemporalSpatialGridPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("ID61 requires one CUDA GPU for frozen ID75 inference")
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError("runner must create output_dir")
    if {path.name for path in output_dir.iterdir()} - {"README.md", "wandb"}:
        raise FileExistsError("ID61 output is not fresh")

    id60 = json.loads(args.id60_result.read_text(encoding="utf-8"))
    id75 = json.loads(args.id75_result.read_text(encoding="utf-8"))
    if id60.get("schema") != "nimloth_frozen_state_goal_probe_v1":
        raise ValueError("ID60 result schema mismatch")
    if id75.get("schema") != "nimloth_frozen_sft1_residual_t1_canary_v1":
        raise ValueError("ID75 result schema mismatch")
    if _sha256(args.state_cache) != id60["state_cache"]["sha256"]:
        raise ValueError("ID60 state cache hash mismatch")
    if _sha256(args.state_cache_metadata) != id60["state_cache"]["metadata_sha256"]:
        raise ValueError("ID60 state metadata hash mismatch")
    if _sha256(args.predictor_checkpoint / "predictor.pt") != id75["checkpoint"]["predictor_sha256"]:
        raise ValueError("ID75 predictor hash mismatch")

    train_records = _load_jsonl(args.train_jsonl)
    val_records = _load_jsonl(args.val_jsonl)
    all_records = train_records + val_records
    metadata = json.loads(args.state_cache_metadata.read_text(encoding="utf-8"))
    if metadata.get("schema") != "nimloth_frozen_state_cache_v1":
        raise ValueError("ID60 cache metadata schema mismatch")
    if len(metadata["records"]) != len(all_records):
        raise ValueError("ID60 metadata record count differs from archives")
    for index, (row, record) in enumerate(zip(metadata["records"], all_records, strict=True)):
        if row["record_id"] != str(record["id"]):
            raise ValueError(f"ID60 record order differs at index {index}")

    with np.load(args.state_cache, allow_pickle=False) as cache:
        state = cache["state"].astype(np.float32)
        current_index = cache["transition_current_index"].astype(np.int64)
        next_index = cache["transition_next_index"].astype(np.int64)
        actions = cache["transition_action"].astype(np.int64)
        split = cache["transition_split"].astype(np.int64)
        record_index = cache["transition_record_index"].astype(np.int64)
        eligible = cache["transition_external_eligible"].astype(np.bool_)
        state_step = cache["state_step_index"].astype(np.int64)
    outcomes = np.asarray(
        [
            parse_step_action_success(all_records[record], int(state_step[current]))
            for record, current in zip(record_index, current_index, strict=True)
        ],
        dtype=np.bool_,
    )
    train_mask = split == 0
    external_mask = (split == 1) & eligible
    current = state[current_index]
    target = state[next_index]
    model = ResidualTemporalSpatialGridPredictor.load_checkpoint(
        args.predictor_checkpoint,
        map_location="cpu",
    ).to(device="cuda:0", dtype=torch.float32)
    prediction = _predict(
        model,
        current[external_mask],
        actions[external_mask],
        batch_size=args.batch_size,
    )
    external_current = current[external_mask]
    external_target = target[external_mask]
    external_actions = actions[external_mask]
    external_outcomes = outcomes[external_mask]

    per_action: dict[str, Any] = {}
    for action in sorted(set(external_actions.tolist())):
        mask = external_actions == action
        if external_outcomes[mask].any() and not external_outcomes[mask].all():
            per_action[str(action)] = {
                "action_name": _ACTION_NAMES[action],
                **stratified_outcome_metrics(
                    current_state=external_current[mask],
                    actual_next_state=external_target[mask],
                    predicted_next_state=prediction[mask],
                    action_success=external_outcomes[mask],
                    bootstrap_seed=args.seed + action * 10,
                    bootstrap_draws=args.bootstrap_draws,
                ),
            }
        else:
            per_action[str(action)] = {
                "action_name": _ACTION_NAMES[action],
                "count": int(mask.sum()),
                "failure_rate": float((~external_outcomes[mask]).mean()),
                "stratified_metrics_available": False,
            }

    state_rows = metadata["states"]
    image_unchanged = np.asarray(
        [
            state_rows[current]["image_sha256"] == state_rows[next_row]["image_sha256"]
            for current, next_row in zip(current_index, next_index, strict=True)
        ],
        dtype=np.bool_,
    )
    move_left_train = train_mask & (actions == 3)
    move_left_external = external_mask & (actions == 3)
    train_failure_rate = float((~outcomes[move_left_train]).mean())
    external_failure_rate = float((~outcomes[move_left_external]).mean())
    move_left_metrics = per_action["3"]
    result: dict[str, Any] = {
        "schema": "nimloth_id75_action_outcome_audit_v1",
        "read_only": True,
        "optimizer_updates": 0,
        "new_checkpoints": 0,
        "label_semantics": (
            "authoritative exact archived environment feedback in observation_texts[t+1]; "
            "trajectory-level success is not used"
        ),
        "archive_all_steps": {
            "train": _archive_outcome_counts(train_records),
            "validation": _archive_outcome_counts(val_records),
        },
        "early_cache": {
            "train": _count_outcomes(actions, outcomes, train_mask),
            "external_validation": _count_outcomes(actions, outcomes, external_mask),
        },
        "external_validation": {
            "count": int(external_mask.sum()),
            "per_action": per_action,
        },
        "move_left_image_proxy": {
            "successful_exact_image_unchanged_rate": float(
                image_unchanged[move_left_external & outcomes].mean()
            ),
            "failed_exact_image_unchanged_rate": float(
                image_unchanged[move_left_external & ~outcomes].mean()
            ),
        },
        "hypothesis": {
            "move_left_train_failure_rate": train_failure_rate,
            "move_left_external_failure_rate": external_failure_rate,
            "failure_rate_shift_external_minus_train": external_failure_rate - train_failure_rate,
            "move_left_failed_is_majority_train": train_failure_rate > 0.5,
            "move_left_failed_is_majority_external": external_failure_rate > 0.5,
            "wm_failed_subset_skill": move_left_metrics["failed"]["copy_relative_skill"],
            "wm_successful_subset_skill": move_left_metrics["successful"]["copy_relative_skill"],
            "wm_predicted_change_success_auc": move_left_metrics[
                "predicted_change_success_auc"
            ],
            "actual_state_change_success_auc": move_left_metrics[
                "actual_change_success_auc"
            ],
        },
        "sources": {
            "train_jsonl": str(args.train_jsonl.resolve()),
            "train_jsonl_sha256": _sha256(args.train_jsonl),
            "val_jsonl": str(args.val_jsonl.resolve()),
            "val_jsonl_sha256": _sha256(args.val_jsonl),
            "state_cache_sha256": _sha256(args.state_cache),
            "id60_result_sha256": _sha256(args.id60_result),
            "id75_result_sha256": _sha256(args.id75_result),
            "predictor_sha256": _sha256(args.predictor_checkpoint / "predictor.pt"),
        },
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "git_commit": args.git_commit,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
    }
    _atomic_json(output_dir / "result.json", result)
    (output_dir / "summary.html").write_text(_render_html(result), encoding="utf-8")
    return result


def _validate_wandb_identity(args: argparse.Namespace) -> None:
    if args.wandb_project != "nimloth-recon":
        raise ValueError(f"ID61 W&B project must be nimloth-recon, got {args.wandb_project!r}")
    if not args.wandb_run_id.startswith("nimloth-recon-id61-id75-action-outcome-audit"):
        raise ValueError("ID61 W&B run ID is outside the locked experiment namespace")
    if os.environ.get("WANDB_PROJECT") != args.wandb_project:
        raise ValueError("effective WANDB_PROJECT differs from the locked ID61 project")
    if os.environ.get("WANDB_RUN_ID") != args.wandb_run_id:
        raise ValueError("effective WANDB_RUN_ID differs from the locked ID61 identity")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_wandb_identity(args)
    import wandb

    handle = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=args.wandb_run_id,
        resume="never",
        config={"git_commit": args.git_commit, "read_only": True},
    )
    try:
        if handle.project != args.wandb_project or handle.id != args.wandb_run_id:
            raise RuntimeError("initialized W&B identity differs from the locked ID61 identity")
        result = run(args)
        hypothesis = result["hypothesis"]
        handle.log({f"move_left/{key}": value for key, value in hypothesis.items()})
        handle.summary.update(
            {
                "status": "passed",
                "result_json_sha256": _sha256(args.output_dir / "result.json"),
                **hypothesis,
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
