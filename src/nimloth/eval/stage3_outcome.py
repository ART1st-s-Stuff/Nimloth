"""Offline paired Stage3 outcome diagnostics from production-forward exports.

Rows are window/action positions, not independent trajectories. Bootstrap units
are complete trajectories; repeated sliding-window positions remain paired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from nimloth.eval.id75_action_outcome_audit import binary_auc
from nimloth.eval.action_outcome_predictability_probe import average_precision


class OutcomeRowsWriter:
    """Callback for stage3.evaluate; one exclusive file per distributed rank."""

    def __init__(self, path: Path, *, outcome_available: bool) -> None:
        self.stream = Path(path).open("x", encoding="utf-8")
        self.outcome_available = outcome_available
        self.row_count = 0

    def close(self) -> None:
        self.stream.close()

    def __call__(self, batch, output) -> None:
        diagnostic = output.diagnostics
        if not diagnostic or "dino_targets" not in diagnostic:
            raise ValueError("paired outcome export requires DINO diagnostics")
        prediction = diagnostic["predicted_states"].float().cpu()
        target = diagnostic["dino_targets"].float().cpu()
        horizon = getattr(batch, "prediction_horizon", 1)
        prediction = prediction.reshape(batch.batch_size, horizon, *prediction.shape[-2:])
        target = target.reshape_as(prediction)
        current = output.current_state.detach().float().cpu()
        encoded_copy = current[:, None].expand_as(prediction)
        if "current_dino_targets" not in diagnostic:
            raise ValueError("fixed-teacher copy baseline requires current DINO targets")
        teacher_current = diagnostic["current_dino_targets"].float().cpu()
        copy = teacher_current[:, None].expand_as(prediction)
        actions = batch.action_sequences.cpu()
        labels = batch.outcome_targets.reshape(batch.batch_size, horizon).cpu()
        masks = batch.outcome_mask.reshape(batch.batch_size, horizon).cpu()
        logits = diagnostic.get("outcome_logits")
        if self.outcome_available and logits is None:
            raise ValueError("outcome-capable export is missing logits")
        logits = logits.reshape(batch.batch_size, horizon).cpu() if logits is not None else None
        keys = batch.current_keys
        for row, (trajectory, start) in enumerate(keys):
            if not batch.sample_weights[row]:
                continue
            for step in range(horizon):
                entry = {
                    "trajectory": trajectory, "window_start": int(start), "horizon_step": step + 1,
                    "pooled_wm_features": prediction[row, step].mean(dim=0).tolist(),
                    "action": int(actions[row, step]),
                    "outcome": bool(labels[row, step]) if masks[row, step] else None,
                    "target_sha256": hashlib.sha256(target[row, step].contiguous().numpy().tobytes()).hexdigest(),
                    "dino_mse": float((prediction[row, step] - target[row, step]).square().mean()),
                    "copy_mse": float((copy[row, step] - target[row, step]).square().mean()),
                    "copy_definition": "frozen_teacher_initial_observation_persisted_to_each_horizon",
                    "current_target_sha256": hashlib.sha256(teacher_current[row].contiguous().numpy().tobytes()).hexdigest(),
                    "encoded_copy_mse": float((encoded_copy[row, step] - target[row, step]).square().mean()),
                }
                if self.outcome_available:
                    entry["outcome_logit"] = float(logits[row, step])
                if not all(np.isfinite(entry[key]) for key in ("dino_mse", "copy_mse")):
                    raise ValueError("non-finite prediction error")
                self.stream.write(json.dumps(entry, allow_nan=False) + "\n")
                self.row_count += 1
        self.stream.flush()


def outcome_metrics(labels, logits, *, baseline_rate: float) -> dict:
    labels, logits = np.asarray(labels, dtype=float), np.asarray(logits, dtype=float)
    if labels.shape != logits.shape or not labels.size or not np.isin(labels, [0, 1]).all():
        raise ValueError("nonempty aligned binary labels and logits required")
    if not np.isfinite(logits).all() or not 0 <= baseline_rate <= 1:
        raise ValueError("finite logits and valid training baseline required")
    probability = np.exp(-np.logaddexp(0, -logits))
    bce = np.logaddexp(0, logits) - labels * logits
    both = np.unique(labels).size == 2
    ece = 0.0
    for index in range(10):
        mask = (probability >= index / 10) & (probability < (index + 1) / 10 if index < 9 else probability <= 1)
        if mask.any():
            ece += mask.mean() * abs(probability[mask].mean() - labels[mask].mean())
    baseline = np.clip(baseline_rate, 1e-12, 1 - 1e-12)
    return {
        "count": len(labels), "bce": float(bce.mean()),
        "roc_auc": float(binary_auc(labels, logits)) if both else None,
        "pr_auc_average_precision": float(average_precision(labels, logits)) if both else None,
        "balanced_accuracy": float(((probability[labels == 1] >= .5).mean() + (probability[labels == 0] < .5).mean()) / 2) if both else None,
        "brier": float(np.mean((probability - labels) ** 2)), "ece_10_bins": float(ece),
        "train_rate": baseline_rate,
        "baseline_bce": float(np.mean(-labels * np.log(baseline) - (1-labels) * np.log1p(-baseline))),
        "baseline_brier": float(np.mean((baseline_rate-labels)**2)),
    }


def summarize_rows(rows: list[dict], *, train_action_rates: dict[int, float]) -> dict:
    if not rows:
        raise ValueError("evaluation rows must not be empty")
    groups = {"overall": rows}
    for row in rows:
        for name in (f"step/{row['horizon_step']}", f"action/{row['action']}/outcome/{row['outcome']}"):
            groups.setdefault(name, []).append(row)
    result = {}
    for key, group in groups.items():
        mse = float(np.mean([r["dino_mse"] for r in group]))
        copy = float(np.mean([r["copy_mse"] for r in group]))
        result[key] = {"count": len(group), "trajectories": len({r["trajectory"] for r in group}),
                       "dino_mse": mse, "rmse": mse ** .5, "copy_rmse": copy ** .5,
                       "copy_relative_skill": 1 - mse / copy if copy > 0 else None}
    result["outcome_by_action"] = {}
    for action in sorted({r["action"] for r in rows}):
        group = [r for r in rows if r["action"] == action and r["outcome"] is not None and "outcome_logit" in r]
        if group:
            if action not in train_action_rates:
                raise ValueError(f"missing train-only success rate for action {action}")
            result["outcome_by_action"][str(action)] = outcome_metrics(
                [r["outcome"] for r in group], [r["outcome_logit"] for r in group], baseline_rate=train_action_rates[action])
    valid = [r for r in rows if r["outcome"] is not None and "outcome_logit" in r]
    if valid:
        rates = np.array([train_action_rates[r["action"]] for r in valid])
        aggregate = outcome_metrics([r["outcome"] for r in valid], [r["outcome_logit"] for r in valid], baseline_rate=float(rates.mean()))
        labels = np.array([r["outcome"] for r in valid], dtype=float)
        clipped = np.clip(rates, 1e-12, 1-1e-12)
        aggregate["baseline_bce"] = float(np.mean(-labels*np.log(clipped)-(1-labels)*np.log1p(-clipped)))
        aggregate["baseline_brier"] = float(np.mean((rates-labels)**2))
        aggregate["baseline_definition"] = "per_action_train_rate"
        result["outcome_overall"] = aggregate
    return result


def paired_trajectory_bootstrap(control: list[dict], treatment: list[dict], *, repetitions=2000, seed=42) -> dict:
    def keyed(rows):
        result = {}
        for row in rows:
            key = (row["trajectory"], row["window_start"], row["horizon_step"])
            if key in result:
                raise ValueError("duplicate trajectory/window/horizon key")
            result[key] = row
        return result
    left, right = keyed(control), keyed(treatment)
    if not left or left.keys() != right.keys():
        raise ValueError("paired evaluation requires exactly matching row identities")
    deltas = {}
    for key, row in left.items():
        other = right[key]
        if any(row[k] != other[k] for k in ("action", "outcome", "target_sha256", "current_target_sha256")):
            raise ValueError("paired action/label/fixed DINO target mismatch")
        deltas.setdefault(key[0], []).append(other["dino_mse"] - row["dino_mse"])
    means = np.array([np.mean(values) for values in deltas.values()])
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rng = np.random.default_rng(seed)
    draws = np.array([means[rng.integers(0, len(means), len(means))].mean() for _ in range(repetitions)])
    return {"unit": "trajectory", "aggregation": "trajectory_macro_mean", "trajectories": len(means),
            "treatment_minus_control_dino_mse": float(means.mean()),
            "ci95": [float(x) for x in np.quantile(draws, [.025, .975])] if len(means) > 1 else None}


def compare_rows(control: list[dict], treatment: list[dict], *, train_action_rates: dict[int, float], repetitions=2000, seed=42) -> dict:
    """Complete paired summary; fail before comparison on any lineage mismatch."""
    paired = {"overall": paired_trajectory_bootstrap(control, treatment, repetitions=repetitions, seed=seed)}
    predicates = {f"step/{step}": lambda row, step=step: row["horizon_step"] == step
                  for step in sorted({r["horizon_step"] for r in control})}
    predicates["failed"] = lambda row: row["outcome"] is False
    for action in sorted({r["action"] for r in control}):
        for success in (False, True):
            predicates[f"action/{action}/outcome/{success}"] = lambda row, action=action, success=success: row["action"] == action and row["outcome"] is success
    for name, predicate in predicates.items():
        left, right = [r for r in control if predicate(r)], [r for r in treatment if predicate(r)]
        paired[name] = paired_trajectory_bootstrap(left, right, repetitions=repetitions, seed=seed) if left else None
    return {"control": summarize_rows(control, train_action_rates=train_action_rates),
            "treatment": summarize_rows(treatment, train_action_rates=train_action_rates),
            "paired": paired}


def matched_probes(control_train, treatment_train, control_eval, treatment_eval, *, epochs=300, learning_rate=3e-3, weight_decay=1e-2, seed=42071, device="cpu"):
    """Fit identical linear readouts using only each arm's training features.

    Uses the existing frozen-probe AdamW/default LR/weight-decay/seed. Both arms
    receive exactly the same fixed 300-epoch budget; eval never selects epochs.
    """
    from nimloth.eval.action_outcome_predictability_probe import _train_final_probe
    paired_trajectory_bootstrap(control_train, treatment_train, repetitions=1)
    paired_trajectory_bootstrap(control_eval, treatment_eval, repetitions=1)
    if {r["trajectory"] for r in control_train} & {r["trajectory"] for r in control_eval}:
        raise ValueError("probe fit/eval trajectory identities overlap")
    order = lambda row: (row["trajectory"], row["window_start"], row["horizon_step"])
    rates = train_action_rates(control_train)
    output, weights = {}, {}
    for name, train, evaluation in (("control", control_train, control_eval), ("treatment", treatment_train, treatment_eval)):
        train = sorted((r for r in train if r["outcome"] is not None), key=order)
        evaluation = sorted((r for r in evaluation if r["outcome"] is not None), key=order)
        if not train or not evaluation:
            raise ValueError("probe requires labeled train and eval features")
        x = np.asarray([r["pooled_wm_features"] for r in train], dtype=np.float32)
        q = np.asarray([r["pooled_wm_features"] for r in evaluation], dtype=np.float32)
        if x.ndim != 2 or q.ndim != 2 or x.shape[1] != q.shape[1] or not np.isfinite(x).all() or not np.isfinite(q).all():
            raise ValueError("invalid pooled WM feature matrices")
        logits, parameters = _train_final_probe(x, np.array([r["outcome"] for r in train], dtype=np.float32), q,
            epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, seed=seed, device=device)
        rows = [dict(row, outcome_logit=float(logit)) for row, logit in zip(evaluation, logits, strict=True)]
        output[name] = summarize_rows(rows, train_action_rates=rates)
        weights[name] = parameters
    output["config"] = dict(epochs=epochs, learning_rate=learning_rate, weight_decay=weight_decay, seed=seed,
                            feature="mean_pool_predicted_wm_grid", standardization="train_only", selection="fixed_budget_no_eval_selection")
    return output, weights


def train_action_rates(rows):
    rates = {}
    for action in sorted({r["action"] for r in rows}):
        # Sliding windows repeat transitions at different rollout depths. The
        # environment action rate counts each executed transition exactly once.
        labels = {}
        for row in rows:
            if row["action"] == action and row["outcome"] is not None:
                key = (row["trajectory"], row["window_start"] + row["horizon_step"] - 1)
                if key in labels and labels[key] != row["outcome"]:
                    raise ValueError("inconsistent repeated transition label")
                labels[key] = row["outcome"]
        if labels:
            rates[action] = float(np.mean(list(labels.values())))
    return rates


def main(argv=None):
    """Compare rank exports and optionally fit matched offline probes."""
    import argparse
    import glob
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-train", nargs="+", required=True)
    parser.add_argument("--treatment-train", nargs="+", required=True)
    parser.add_argument("--control-eval", nargs="+", required=True)
    parser.add_argument("--treatment-eval", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-probes", action="store_true")
    parser.add_argument("--probe-device", default="cpu")
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    identities = []
    def read(patterns):
        paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
        if not paths:
            raise FileNotFoundError(f"no export files match {patterns}")
        path = Path(paths[0])
        manifest_path = path.parent / (path.name.split("_rank_")[0] + ".complete.json")
        identities.append(json.loads(manifest_path.read_text()))
        return load_completed_exports([Path(path) for path in paths])
    ct, tt, ce, te = (read(paths) for paths in (args.control_train, args.treatment_train, args.control_eval, args.treatment_eval))
    for key in ("train_sha256", "eval_sha256", "initial_config_sha256", "initial_training_state_sha256", "seed", "source_commit"):
        if any(key not in item["identity"] for item in identities) or len({item["identity"][key] for item in identities}) != 1:
            raise ValueError(f"export identity mismatch or missing {key}")
    paired_trajectory_bootstrap(ct, tt, repetitions=1)
    result = compare_rows(ce, te, train_action_rates=train_action_rates(ct), repetitions=args.bootstrap_repetitions)
    result["export_manifests"] = identities
    weights = None
    if args.fit_probes:
        result["matched_probes"], weights = matched_probes(ct, tt, ce, te, epochs=args.probe_epochs, device=args.probe_device)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    if weights is not None:
        weights_path = args.output.with_suffix(".probe_weights.npz")
        with weights_path.open("xb") as stream:
            np.savez(stream, **{f"{arm}_{key}": value for arm, parameters in weights.items() for key, value in parameters.items()})
    return 0





def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_completed_exports(paths):
    manifests = {path.parent / (path.name.split("_rank_")[0] + ".complete.json") for path in paths}
    if len(manifests) != 1:
        raise ValueError("one complete export phase is required per CLI input")
    manifest_path = next(iter(manifests))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or not manifest.get("identity") or not manifest.get("checkpoint_identity"):
        raise ValueError("export phase is not committed with complete identity")
    files = manifest["files"]
    if {p.name for p in paths} != {entry["name"] for entry in files}:
        raise ValueError("all rank exports in the manifest must be supplied")
    rows = []
    for entry in files:
        path = manifest_path.parent / entry["name"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("export file hash mismatch")
        rank_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rank_rows) != entry["rows"]:
            raise ValueError("export rank row count mismatch")
        rows.extend(rank_rows)
    if len(rows) != manifest["expected_rows"]:
        raise ValueError("export window count mismatch")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
