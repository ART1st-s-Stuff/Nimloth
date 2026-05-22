"""Build Qwen planner correction SFT data from EB-Nav fork samples.

This converts same-state counterfactual fork groups into planner SFT records:
the target action is the candidate with the best reliable fork outcome.  Sample
weight is inherited from fork reliability/novelty/learnability metadata, so
repeated or unreliable samples can be skipped without action-specific weighting.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = {
    0: "Move forward by 0.25",
    1: "Move backward by 0.25",
    2: "Move rightward by 0.25",
    3: "Move leftward by 0.25",
    4: "Rotate to the right by 90 degrees.",
    5: "Rotate to the left by 90 degrees.",
    6: "Tilt the camera upward by 30 degrees.",
    7: "Tilt the camera downward by 30 degrees.",
}


def build_planner_special_response(*, cot: str, action_id: int) -> str:
    return f"<think>{cot}</think><|latent_token|><|action_start|><|action_{int(action_id)}|><|action_end|>"


def validate_planner_special_output(response: str) -> tuple[bool, str, int | None]:
    for action_id in range(8):
        if f"<|action_{action_id}|>" in response:
            return True, "", action_id
    return False, "missing action id token", None


ACTION_LIST = "\n".join(f"{idx}: {name}" for idx, name in sorted(ACTION_NAMES.items()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fork-jsonl", nargs="+", required=True, help="fork_samples.jsonl files, globs, or directories.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--target-field", default="continuation_reward")
    p.add_argument("--fallback-target-field", default="continuation_reward")
    p.add_argument("--min-outcome-gap", type=float, default=0.02)
    p.add_argument("--min-effective-lr-scale", type=float, default=0.0)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260523)
    p.add_argument("--max-history-actions", type=int, default=0)
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


def _repo_rel(path: str, repo: Path) -> str:
    raw = Path(str(path))
    if raw.is_absolute():
        try:
            return str(raw.resolve().relative_to(repo.resolve()))
        except Exception:
            return str(raw)
    return str(path)


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


def _format_history(action_ids: list[Any], *, max_actions: int) -> str:
    ids = [_i(x, -1) for x in action_ids]
    ids = [x for x in ids if x >= 0]
    if max_actions > 0:
        ids = ids[-int(max_actions):]
    if not ids:
        return "No previous actions in this episode."
    lines = []
    offset = 0 if max_actions <= 0 else max(0, len(action_ids) - len(ids))
    for idx, action_id in enumerate(ids):
        lines.append(f"{idx + offset:02d}. action {action_id} ({ACTION_NAMES.get(action_id, f'action_{action_id}')})")
    return "\n".join(lines)


def _build_prompt(row: dict[str, Any], *, max_history_actions: int) -> str:
    return f"""You are a robot navigating a home from a first-person image.

Available actions:
{ACTION_LIST}

Goal instruction:
{str(row.get("instruction", ""))}

Complete action history for this episode before the current image:
{_format_history(list(row.get("history_actions", []) or []), max_actions=max_history_actions)}

Choose the single next expert navigation action for the current image.
Use the image, goal instruction, and complete action history. Do not use
distance-to-goal values.

Respond exactly in this format:
<think>brief expert navigation reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>
where N is one action id from 0 to 7."""


def _build_cot(row: dict[str, Any], action_id: int) -> str:
    action_name = ACTION_NAMES.get(int(action_id), f"action_{action_id}")
    reasons = list(row.get("fork_trigger_reasons", []) or [])
    reason_text = ", ".join(str(x) for x in reasons[:3]) if reasons else "the current state is informative"
    return (
        f"The current image and full history define a high-information decision point ({reason_text}). "
        f"Among the reliable counterfactual candidates, the expert next action is {action_name}."
    )


def _load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    per_file: dict[str, int] = {}
    for path in paths:
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
    return rows, per_file


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repo = Path(args.repo_root).resolve()
    rows, per_file = _load_rows(_expand(args.fork_jsonl))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("skipped") or row.get("skip_for_training"):
            continue
        if _f(row.get("effective_lr_scale", 1.0), 1.0) < float(args.min_effective_lr_scale):
            continue
        grouped[_group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    skipped = Counter()
    action_counts: Counter[int] = Counter()
    for key, group in grouped.items():
        if len(group) < 2:
            skipped["too_few_candidates"] += 1
            continue
        ranked = sorted(group, key=lambda r: _target(r, args.target_field, args.fallback_target_field), reverse=True)
        best = ranked[0]
        second = ranked[1]
        outcome_gap = _target(best, args.target_field, args.fallback_target_field) - _target(second, args.target_field, args.fallback_target_field)
        if outcome_gap < float(args.min_outcome_gap):
            skipped["small_outcome_gap"] += 1
            continue
        action_id = _i(best.get("candidate_action_id", best.get("action_id", -1)))
        if action_id < 0 or action_id > 7:
            skipped["bad_action"] += 1
            continue
        image = _repo_rel(str(best.get("image_t", "")), repo)
        if not image:
            skipped["missing_image"] += 1
            continue
        prompt = _build_prompt(best, max_history_actions=int(args.max_history_actions))
        cot = _build_cot(best, action_id)
        response = build_planner_special_response(cot=cot, action_id=action_id)
        valid, reason, parsed = validate_planner_special_output(response)
        if not valid or parsed != action_id:
            raise RuntimeError(f"invalid generated response for {key}: {reason}")
        effective_lr_scale = _f(best.get("effective_lr_scale", 1.0), 1.0)
        record = {
            "id": f"fork_qwen_correction_{len(out):08d}",
            "image": image,
            "prompt": prompt,
            "response": response,
            "instruction": str(best.get("instruction", "")),
            "cot": cot,
            "action_id": action_id,
            "action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
            "sample_weight": effective_lr_scale,
            "effective_lr_scale": effective_lr_scale,
            "sample_reliability": _f(best.get("sample_reliability", 1.0), 1.0),
            "sample_novelty": _f(best.get("sample_novelty", 1.0), 1.0),
            "sample_learnability": _f(best.get("sample_learnability", 1.0), 1.0),
            "metadata": {
                "source": "eb_nav_fork_counterfactual_qwen_correction",
                "group_key": key,
                "target_field": str(args.target_field),
                "outcome_gap": outcome_gap,
                "best_target": _target(best, args.target_field, args.fallback_target_field),
                "second_target": _target(second, args.target_field, args.fallback_target_field),
                "qwen_proposed_action_id": _i(best.get("qwen_proposed_action_id", best.get("selected_action_id", -1))),
                "selected_action_id": _i(best.get("selected_action_id", -1)),
                "candidate_actions": list(best.get("candidate_actions", []) or []),
                "fork_trigger_reasons": list(best.get("fork_trigger_reasons", []) or []),
            },
        }
        out.append(record)
        action_counts[action_id] += 1

    summary = {
        "input_files": per_file,
        "input_records": len(rows),
        "fork_groups": len(grouped),
        "records": len(out),
        "skipped": dict(skipped),
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
        "target_field": str(args.target_field),
        "fallback_target_field": str(args.fallback_target_field),
        "min_outcome_gap": float(args.min_outcome_gap),
        "min_effective_lr_scale": float(args.min_effective_lr_scale),
    }
    return out, summary


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records, summary = build_records(args)
    if not records:
        raise RuntimeError(f"no fork Qwen correction records built: {summary}")
    rng = random.Random(int(args.seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    test_n = max(1, int(round(len(records) * float(args.test_fraction)))) if len(records) > 1 else 0
    test_ids = set(indices[:test_n])
    train = [item for idx, item in enumerate(records) if idx not in test_ids]
    test = [item for idx, item in enumerate(records) if idx in test_ids]
    out = Path(args.output_dir)
    _write_jsonl(out / "qwen_planner_fork_correction_train.jsonl", train)
    _write_jsonl(out / "qwen_planner_fork_correction_test.jsonl", test)
    summary.update(
        {
            "output_dir": str(out),
            "seed": int(args.seed),
            "test_fraction": float(args.test_fraction),
            "train_records": len(train),
            "test_records": len(test),
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
