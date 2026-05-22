"""Compare EB-Nav gated override evaluation against direct Qwen baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-summary", required=True, help="summary.json from direct Qwen eval.")
    p.add_argument("--gate-summary", required=True, help="summary.json from Qwen+gate override eval.")
    p.add_argument("--output-json", default="", help="Optional comparison output path.")
    p.add_argument("--min-success-delta", type=float, default=0.0, help="Gate success_rate - baseline success_rate must be at least this value.")
    p.add_argument("--max-collision-delta", type=float, default=0.0, help="Gate collision_rate - baseline collision_rate must be at most this value.")
    p.add_argument("--max-planner-failure-delta", type=float, default=0.0, help="Gate planner_failure_rate - baseline planner_failure_rate must be at most this value.")
    return p.parse_args()


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _f(obj: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(obj.get(key, default))
    except Exception:
        return default


def compare(args: argparse.Namespace) -> dict[str, Any]:
    baseline = _load(args.baseline_summary)
    gate = _load(args.gate_summary)
    success_delta = _f(gate, "task_success_rate") - _f(baseline, "task_success_rate")
    collision_delta = _f(gate, "collision_rate") - _f(baseline, "collision_rate")
    planner_failure_delta = _f(gate, "planner_failure_rate") - _f(baseline, "planner_failure_rate")
    checks = {
        "success_nonregression": success_delta >= float(args.min_success_delta),
        "collision_nonregression": collision_delta <= float(args.max_collision_delta),
        "planner_failure_nonregression": planner_failure_delta <= float(args.max_planner_failure_delta),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_success_delta": float(args.min_success_delta),
            "max_collision_delta": float(args.max_collision_delta),
            "max_planner_failure_delta": float(args.max_planner_failure_delta),
        },
        "metrics": {
            "baseline_task_success_rate": _f(baseline, "task_success_rate"),
            "gate_task_success_rate": _f(gate, "task_success_rate"),
            "success_delta": success_delta,
            "baseline_collision_rate": _f(baseline, "collision_rate"),
            "gate_collision_rate": _f(gate, "collision_rate"),
            "collision_delta": collision_delta,
            "baseline_planner_failure_rate": _f(baseline, "planner_failure_rate"),
            "gate_planner_failure_rate": _f(gate, "planner_failure_rate"),
            "planner_failure_delta": planner_failure_delta,
            "gate_override_rate": _f(gate, "override_rate"),
            "gate_same_as_qwen_rate": _f(gate, "same_as_qwen_rate"),
        },
        "inputs": {
            "baseline_summary": str(args.baseline_summary),
            "gate_summary": str(args.gate_summary),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    result = compare(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if str(args.output_json).strip():
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
