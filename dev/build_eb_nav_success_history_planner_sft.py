"""Build planner SFT records from successful EB-Nav expert rollouts.

Each training record uses the current pre-action image and the full action
history from the beginning of the episode up to, but not including, the target
expert action. Failed expert actions are kept in history so the planner can
learn recovery behavior from subsequent successful-rollout states.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import random
from pathlib import Path
from typing import Any

from src.data.eb_nav_dataset import ACTION_NAMES
from src.vlm.qwen_planner import build_planner_special_response, validate_planner_special_output


ACTION_LIST = "\n".join(
    f"{idx}: {name}" for idx, name in sorted(ACTION_NAMES.items())
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _repo_rel(path: str, repo: Path) -> str:
    path_str = str(path)
    if path_str.startswith("images/"):
        path_str = f"datasets/EB-Nav/{path_str}"
    raw = Path(path_str)
    if raw.is_absolute():
        try:
            return str(raw.resolve().relative_to(repo.resolve()))
        except Exception:
            pass
    return path_str


def _action_from_step(step: dict[str, Any]) -> dict[str, Any] | None:
    plans = step.get("executable_plan")
    if not isinstance(plans, list) or not plans:
        return None
    plan = plans[0] if isinstance(plans[0], dict) else {}
    raw_action = plan.get("action", [])
    if not isinstance(raw_action, list) or not raw_action:
        return None
    action_id = _safe_int(raw_action[0], -1)
    if action_id < 0 or action_id > 7:
        return None
    return {
        "action_id": action_id,
        "action_name": str(raw_action[1] if len(raw_action) > 1 else ACTION_NAMES.get(action_id, f"action_{action_id}")),
        "action_success": bool(plan.get("action_success", True)),
        "env_feedback": str(plan.get("env_feedback", "")),
        "img_path": str(plan.get("img_path", "")),
    }


def _format_full_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No previous actions in this episode."
    lines = []
    for item in history:
        status = "success" if bool(item.get("action_success")) else "failed"
        feedback = str(item.get("env_feedback", "")).strip()
        if feedback:
            feedback = f"; env feedback: {feedback}"
        lines.append(
            f"{int(item['step_index']):02d}. action {int(item['action_id'])} "
            f"({item['action_name']}) -> {status}{feedback}"
        )
    return "\n".join(lines)


def _build_prompt(*, instruction: str, full_history: list[dict[str, Any]]) -> str:
    return f"""You are a robot navigating a home from a first-person image.

Available actions:
{ACTION_LIST}

Goal instruction:
{instruction}

Complete action history for this episode before the current image:
{_format_full_history(full_history)}

Choose the single next expert navigation action for the current image.
Use the image, goal instruction, and complete action history. If a previous
action failed, avoid repeating the same blocked move unless the view has changed.
Do not use distance-to-goal values.

Respond exactly in this format:
<think>brief expert navigation reasoning</think><|latent_token|><|action_start|><|action_N|><|action_end|>
where N is one action id from 0 to 7."""


def _build_cot(step: dict[str, Any], history: list[dict[str, Any]], action_id: int) -> str:
    visual = str(step.get("visual_description", "")).strip()
    reasoning = str(step.get("reasoning_and_reflection", "")).strip()
    action_name = ACTION_NAMES.get(int(action_id), f"action_{action_id}")
    if visual or reasoning:
        parts = []
        if visual:
            parts.append(visual)
        if reasoning:
            parts.append(reasoning)
        parts.append(f"Considering the full action history, the expert next action is {action_name}.")
        return " ".join(parts)
    if history and not bool(history[-1].get("action_success", True)):
        return (
            f"The previous action failed, so I use the current image and full history to recover. "
            f"The expert next action is {action_name}."
        )
    return f"I inspect the current image and full history, then choose the expert next action: {action_name}."


def _episode_split_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("model_name", "")),
        str(record.get("eval_set", "")),
        str(record.get("episode_id", "")),
    )


def _build_records(records: list[dict[str, Any]], repo: Path, *, max_history_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "successful_episodes": 0,
        "skipped_unsuccessful_episodes": 0,
        "trajectory_steps": 0,
        "records": 0,
        "missing_action": 0,
        "invalid_response": 0,
        "action_distribution": {},
        "previous_action_failed_records": 0,
    }
    action_counts: Counter[int] = Counter()
    history: list[dict[str, Any]]
    for ep_idx, record in enumerate(records):
        if _safe_float(record.get("success"), 0.0) != 1.0:
            stats["skipped_unsuccessful_episodes"] += 1
            continue
        trajectory = record.get("trajectory")
        if not isinstance(trajectory, list):
            continue
        stats["successful_episodes"] += 1
        history = []
        for step_idx, step in enumerate(trajectory):
            if not isinstance(step, dict):
                continue
            stats["trajectory_steps"] += 1
            action = _action_from_step(step)
            if action is None:
                stats["missing_action"] += 1
                continue
            action_id = int(action["action_id"])
            prompt = _build_prompt(instruction=str(record.get("instruction", "")), full_history=history)
            if max_history_chars > 0 and len(prompt) > max_history_chars:
                raise ValueError(
                    f"prompt exceeded --max-history-chars at episode={record.get('episode_id')} "
                    f"step={step_idx}: len={len(prompt)}"
                )
            cot = _build_cot(step, history, action_id)
            response = build_planner_special_response(cot=cot, action_id=action_id)
            valid, reason, parsed_action = validate_planner_special_output(response)
            if not valid or parsed_action != action_id:
                stats["invalid_response"] += 1
                raise RuntimeError(f"invalid generated response: {reason}")
            input_image = _repo_rel(str(step.get("input_image_path", "")), repo)
            split_key = _episode_split_key(record)
            out.append(
                {
                    "id": f"success_history_ep{ep_idx:06d}_step{step_idx:03d}",
                    "image": input_image,
                    "prompt": prompt,
                    "response": response,
                    "instruction": str(record.get("instruction", "")),
                    "cot": cot,
                    "action_id": action_id,
                    "action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                    "episode_success": 1.0,
                    "history_len": len(history),
                    "full_history": list(history),
                    "previous_action_failed": bool(history and not bool(history[-1].get("action_success", True))),
                    "source_input_image": input_image,
                    "source_next_image": _repo_rel(str(action.get("img_path", "")), repo),
                    "split_key": list(split_key),
                    "metadata": {
                        "source": "eb_nav_success_expert_rollout",
                        "model_name": split_key[0],
                        "eval_set": split_key[1],
                        "episode_id": split_key[2],
                        "step_index": step_idx,
                    },
                }
            )
            action_counts[action_id] += 1
            if history and not bool(history[-1].get("action_success", True)):
                stats["previous_action_failed_records"] += 1
            history.append(
                {
                    "step_index": step_idx,
                    "action_id": action_id,
                    "action_name": ACTION_NAMES.get(action_id, f"action_{action_id}"),
                    "action_success": bool(action["action_success"]),
                    "env_feedback": str(action.get("env_feedback", "")),
                }
            )
    stats["records"] = len(out)
    stats["action_distribution"] = {str(k): int(v) for k, v in sorted(action_counts.items())}
    return out, stats


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--max-history-chars", type=int, default=0)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    raw_records = json.load(open(args.dataset, encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise RuntimeError(f"dataset is not a list: {args.dataset}")
    all_records, stats = _build_records(raw_records, repo, max_history_chars=max(0, int(args.max_history_chars)))
    if not all_records:
        raise RuntimeError("no planner SFT records built")

    by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in all_records:
        by_episode[tuple(str(x) for x in item["split_key"])].append(item)
    keys = sorted(by_episode)
    rng = random.Random(int(args.seed))
    rng.shuffle(keys)
    test_count = max(1, int(round(len(keys) * float(args.test_fraction))))
    test_keys = set(keys[:test_count])
    train_records = [item for key in keys if key not in test_keys for item in by_episode[key]]
    test_records = [item for key in keys if key in test_keys for item in by_episode[key]]

    out = Path(args.output_dir)
    _write_jsonl(out / "qwen_planner_success_history_train.jsonl", train_records)
    _write_jsonl(out / "qwen_planner_success_history_test.jsonl", test_records)
    summary = {
        "dataset": args.dataset,
        "output_dir": str(out),
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
        "split_key": ["model_name", "eval_set", "episode_id"],
        "num_episode_keys": len(keys),
        "num_train_episode_keys": len(keys) - len(test_keys),
        "num_test_episode_keys": len(test_keys),
        "train_records": len(train_records),
        "test_records": len(test_records),
        **stats,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
