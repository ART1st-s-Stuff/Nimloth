"""Convert EB-Nav rollout transitions JSONL into CustomJointSequenceDataset JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TransitionRow:
    source_file: str
    source_line: int
    rollout_id: str
    step: int
    image_t: str
    image_next: str
    sampled_action_id: int
    reward: float
    instruction: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transitions-jsonl",
        nargs="+",
        required=True,
        help="One or more input transitions.jsonl files.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output CustomJointSequenceDataset JSONL path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path (default: next to output).")
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--future-len", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument(
        "--history-mode",
        choices=("previous", "repeat"),
        default="previous",
        help="How to build history_images/history_actions.",
    )
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Skip image file existence checks.",
    )
    return parser.parse_args()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _one_hot(action_id: int, action_dim: int) -> list[float]:
    vec = [0.0] * action_dim
    if 0 <= action_id < action_dim:
        vec[action_id] = 1.0
    return vec


def _load_transitions(paths: list[str]) -> tuple[list[TransitionRow], dict[str, int]]:
    rows: list[TransitionRow] = []
    stats = {
        "input_files": 0,
        "input_lines": 0,
        "input_rows": 0,
        "json_decode_errors": 0,
    }
    for path_str in paths:
        path = Path(path_str)
        if not path.is_file():
            continue
        stats["input_files"] += 1
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stats["input_lines"] += 1
                raw = line.strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    stats["json_decode_errors"] += 1
                    continue
                row = TransitionRow(
                    source_file=str(path),
                    source_line=line_no,
                    rollout_id=str(item.get("rollout_id", "unknown")),
                    step=_safe_int(item.get("step"), default=0),
                    image_t=str(item.get("image_t", "")),
                    image_next=str(item.get("image_next", "")),
                    sampled_action_id=_safe_int(item.get("sampled_action_id"), default=-1),
                    reward=_safe_float(item.get("reward"), default=0.0),
                    instruction=str(item.get("instruction", "")),
                )
                rows.append(row)
                stats["input_rows"] += 1
    return rows, stats


def _image_ok(path_str: str, allow_missing_images: bool) -> bool:
    if not path_str:
        return False
    if allow_missing_images:
        return True
    return Path(path_str).is_file()


def _build_history(
    rows: list[TransitionRow],
    current_index: int,
    history_len: int,
    action_dim: int,
    history_mode: str,
) -> tuple[list[str], list[list[float]]]:
    current = rows[current_index]
    if history_mode == "repeat":
        return [current.image_t] * history_len, [[0.0] * action_dim for _ in range(history_len)]

    left = max(0, current_index - history_len + 1)
    indices = list(range(left, current_index + 1))
    pad_count = history_len - len(indices)

    earliest_image = rows[indices[0]].image_t if indices else current.image_t
    history_images = [earliest_image] * pad_count + [rows[idx].image_t for idx in indices]
    history_actions = [[0.0] * action_dim for _ in range(pad_count)] + [
        _one_hot(rows[idx].sampled_action_id, action_dim) for idx in indices
    ]
    return history_images, history_actions


def _group_rollouts(rows: list[TransitionRow]) -> dict[str, list[TransitionRow]]:
    grouped: dict[str, list[TransitionRow]] = {}
    for row in rows:
        rollout_key = f"{row.source_file}::{row.rollout_id}"
        grouped.setdefault(rollout_key, []).append(row)
    for rollout_rows in grouped.values():
        rollout_rows.sort(key=lambda x: x.step)
    return grouped


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1")
    if args.future_len < 1:
        raise ValueError("--future-len must be >= 1")
    if args.action_dim < 1:
        raise ValueError("--action-dim must be >= 1")

    rows, stats = _load_transitions(args.transitions_jsonl)
    grouped = _group_rollouts(rows)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped_missing_images = 0
    output_rows = 0
    action_counts: Counter[int] = Counter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for rollout_id, rollout_rows in grouped.items():
            for idx, row in enumerate(rollout_rows):
                if not _image_ok(row.image_t, args.allow_missing_images) or not _image_ok(
                    row.image_next, args.allow_missing_images
                ):
                    skipped_missing_images += 1
                    continue

                history_images, history_actions = _build_history(
                    rollout_rows,
                    current_index=idx,
                    history_len=int(args.history_len),
                    action_dim=int(args.action_dim),
                    history_mode=str(args.history_mode),
                )

                action_vec = _one_hot(row.sampled_action_id, int(args.action_dim))
                future_len = int(args.future_len)
                record: dict[str, Any] = {
                    "history_images": history_images,
                    "history_actions": history_actions,
                    "future_images": [row.image_next] * future_len,
                    "future_actions": [action_vec] * future_len,
                    "future_action_ids": [int(row.sampled_action_id)] * future_len,
                    "future_rewards": [float(row.reward)] * future_len,
                    "instruction": row.instruction,
                    "prompt": row.instruction,
                    "rollout_id": rollout_id,
                    "metadata": {
                        "source_file": row.source_file,
                        "source_line": row.source_line,
                        "step": row.step,
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_rows += 1
                action_counts[row.sampled_action_id] += 1

    summary_path = Path(args.summary_json) if str(args.summary_json).strip() else output_path.with_name("summary.json")
    summary = {
        **stats,
        "output_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "history_len": int(args.history_len),
        "future_len": int(args.future_len),
        "action_dim": int(args.action_dim),
        "history_mode": str(args.history_mode),
        "allow_missing_images": bool(args.allow_missing_images),
        "num_rollouts": len(grouped),
        "output_rows": output_rows,
        "skipped_missing_images": skipped_missing_images,
        "action_distribution": {str(k): int(v) for k, v in sorted(action_counts.items())},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    summary = convert(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
