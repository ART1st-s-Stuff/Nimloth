"""Calibrate conservative Qwen override gate thresholds from fork samples.

The gate is intentionally small and task-agnostic: it only decides whether a
learned candidate may override Qwen's action using predicted value margin plus
WM/value uncertainty.  Labels come from same-state fork groups, not from
hand-written navigation rules.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork-jsonl", nargs="+", required=True, help="fork_samples.jsonl files, globs, or directories.")
    p.add_argument("--output-json", required=True)
    p.add_argument("--target-field", default="continuation_reward")
    p.add_argument("--fallback-target-field", default="continuation_reward")
    p.add_argument("--min-outcome-delta", type=float, default=0.02)
    p.add_argument("--min-effective-lr-scale", type=float, default=0.0)
    p.add_argument("--min-precision", type=float, default=0.6)
    p.add_argument("--min-coverage", type=float, default=0.01)
    p.add_argument("--margin-grid", default="0.0,0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2")
    p.add_argument("--max-value-std-grid", default="0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.3")
    p.add_argument("--max-wm-unc-grid", default="0.01,0.03,0.05,0.08,0.1,0.15,0.2,0.3,0.5")
    return p.parse_args()


def _expand(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("**/fork_samples.jsonl")))
        else:
            matches = sorted(glob.glob(item))
            out.extend(Path(m) for m in matches) if matches else out.append(p)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def _f(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def _i(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _grid(spec: str) -> list[float]:
    vals = []
    for item in str(spec).split(","):
        item = item.strip()
        if item:
            vals.append(float(item))
    return sorted(set(vals))


def _group_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("rollout_id", "")),
            str(row.get("episode_id", "")),
            str(row.get("eval_set", "")),
            str(row.get("task_key", "")),
            str(row.get("step", "")),
        ]
    )


def _target(row: dict[str, Any], target_field: str, fallback_field: str) -> float:
    if target_field in row:
        return _f(row.get(target_field), 0.0)
    return _f(row.get(fallback_field), 0.0)


def _score_by_action(row: dict[str, Any], action_id: int) -> float | None:
    scores = row.get("value_policy_score_by_action")
    if not isinstance(scores, list):
        scores = row.get("policy_scores")
    if isinstance(scores, list) and 0 <= int(action_id) < len(scores):
        return _f(scores[int(action_id)], 0.0)
    return None


def _make_examples(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped") or row.get("skip_for_training"):
            continue
        if _f(row.get("effective_lr_scale", 1.0), 1.0) < float(args.min_effective_lr_scale):
            continue
        grouped[_group_key(row)].append(row)

    examples: list[dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) < 2:
            continue
        qwen_action = _i(group[0].get("qwen_proposed_action_id", group[0].get("selected_action_id", -1)))
        if qwen_action < 0:
            continue
        by_action = {_i(r.get("candidate_action_id", r.get("action_id", -1))): r for r in group}
        qwen_row = by_action.get(qwen_action)
        if qwen_row is None:
            continue
        best_row = max(group, key=lambda r: _target(r, args.target_field, args.fallback_target_field))
        best_action = _i(best_row.get("candidate_action_id", best_row.get("action_id", -1)))
        if best_action < 0 or best_action == qwen_action:
            continue
        qwen_target = _target(qwen_row, args.target_field, args.fallback_target_field)
        best_target = _target(best_row, args.target_field, args.fallback_target_field)
        outcome_delta = best_target - qwen_target
        if abs(outcome_delta) < float(args.min_outcome_delta):
            continue
        best_score = _score_by_action(best_row, best_action)
        qwen_score = _score_by_action(best_row, qwen_action)
        if best_score is None or qwen_score is None:
            continue
        examples.append(
            {
                "group_key": key,
                "qwen_action_id": qwen_action,
                "candidate_action_id": best_action,
                "outcome_delta": outcome_delta,
                "label_better_than_qwen": outcome_delta > 0.0,
                "predicted_margin_over_qwen": float(best_score - qwen_score),
                "candidate_value_std": _f(best_row.get("candidate_predicted_value_std", 0.0), 0.0),
                "candidate_wm_uncertainty": _f(best_row.get("candidate_pred_uncertainty", 0.0), 0.0),
                "effective_lr_scale": _f(best_row.get("effective_lr_scale", 1.0), 1.0),
            }
        )
    return examples


def _evaluate(examples: list[dict[str, Any]], *, margin: float, max_value_std: float, max_wm_unc: float) -> dict[str, Any]:
    selected = [
        ex for ex in examples
        if _f(ex["predicted_margin_over_qwen"]) >= margin
        and _f(ex["candidate_value_std"]) <= max_value_std
        and _f(ex["candidate_wm_uncertainty"]) <= max_wm_unc
    ]
    tp = sum(1 for ex in selected if bool(ex["label_better_than_qwen"]))
    fp = len(selected) - tp
    possible = sum(1 for ex in examples if bool(ex["label_better_than_qwen"]))
    precision = tp / max(1, len(selected))
    recall = tp / max(1, possible)
    coverage = len(selected) / max(1, len(examples))
    mean_delta = sum(_f(ex["outcome_delta"]) for ex in selected) / max(1, len(selected))
    return {
        "margin": margin,
        "max_value_std": max_value_std,
        "max_pred_uncertainty": max_wm_unc,
        "selected": len(selected),
        "true_positive": tp,
        "false_positive": fp,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "mean_selected_outcome_delta": mean_delta,
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    files = _expand(args.fork_jsonl)
    rows: list[dict[str, Any]] = []
    per_file: dict[str, int] = {}
    for path in files:
        n = 0
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                        n += 1
        per_file[str(path)] = n
    examples = _make_examples(rows, args)
    candidates = [
        _evaluate(examples, margin=m, max_value_std=vs, max_wm_unc=wu)
        for m in _grid(args.margin_grid)
        for vs in _grid(args.max_value_std_grid)
        for wu in _grid(args.max_wm_unc_grid)
    ]
    feasible = [
        c for c in candidates
        if float(c["precision"]) >= float(args.min_precision)
        and float(c["coverage"]) >= float(args.min_coverage)
        and int(c["selected"]) > 0
    ]
    ranked = feasible or [c for c in candidates if int(c["selected"]) > 0] or candidates
    best = max(
        ranked,
        key=lambda c: (
            float(c["precision"]),
            float(c["mean_selected_outcome_delta"]),
            float(c["recall"]),
            -float(c["coverage"]),
        ),
    )
    return {
        "gate_config_version": "eb_nav_qwen_override_gate_v1",
        "override_margin": float(best["margin"]),
        "override_max_value_std": float(best["max_value_std"]),
        "override_max_pred_uncertainty": float(best["max_pred_uncertainty"]),
        "selection_metrics": best,
        "calibration": {
            "num_input_records": len(rows),
            "num_examples": len(examples),
            "num_feasible_grid_points": len(feasible),
            "target_field": str(args.target_field),
            "fallback_target_field": str(args.fallback_target_field),
            "min_outcome_delta": float(args.min_outcome_delta),
            "min_effective_lr_scale": float(args.min_effective_lr_scale),
            "input_files": per_file,
        },
        "top_grid_points": sorted(
            candidates,
            key=lambda c: (
                float(c["precision"]),
                float(c["mean_selected_outcome_delta"]),
                float(c["recall"]),
                -float(c["coverage"]),
            ),
            reverse=True,
        )[:20],
    }


def main() -> None:
    args = parse_args()
    result = calibrate(args)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
