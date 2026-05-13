"""Convert original EB-Nav dataset into CustomJointSequenceDataset JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.eb_nav_dataset import EBNavSequenceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True, help="EB-Nav dataset JSON path.")
    parser.add_argument(
        "--images-base-dir",
        required=True,
        help="Base directory for EB-Nav images used by EBNavSequenceDataset.",
    )
    parser.add_argument(
        "--reward-cache-path",
        default="",
        help="Optional reward cache JSON/JSONL path.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Output CustomJointSequenceDataset JSONL path.")
    parser.add_argument("--summary-json", default="", help="Optional summary JSON path.")
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    return parser.parse_args()


def _normalize_image_path(path_str: str, images_base_dir: Path, repo_cwd: Path) -> str:
    raw = str(path_str or "").strip()
    if not raw:
        return raw
    path = Path(raw)
    if path.is_absolute():
        return str(path.resolve())

    candidate_repo = (repo_cwd / path).resolve()
    if candidate_repo.is_file():
        return str(candidate_repo)

    candidate_img = (images_base_dir / path).resolve()
    if candidate_img.is_file():
        return str(candidate_img)

    return str(candidate_img)


def _normalize_sequence(seq: dict[str, Any], images_base_dir: Path, repo_cwd: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "history_images": [
            _normalize_image_path(p, images_base_dir=images_base_dir, repo_cwd=repo_cwd)
            for p in seq.get("history_images", [])
        ],
        "history_actions": [[float(x) for x in action] for action in seq.get("history_actions", [])],
        "future_images": [
            _normalize_image_path(p, images_base_dir=images_base_dir, repo_cwd=repo_cwd)
            for p in seq.get("future_images", [])
        ],
        "future_actions": [[float(x) for x in action] for action in seq.get("future_actions", [])],
    }

    if "future_action_ids" in seq:
        record["future_action_ids"] = [int(x) for x in seq.get("future_action_ids", [])]
    if "future_rewards" in seq:
        record["future_rewards"] = [float(x) for x in seq.get("future_rewards", [])]

    for key in ("instruction", "prompt"):
        if key in seq and seq[key] is not None:
            record[key] = str(seq[key])

    metadata: dict[str, Any] = {}
    for key in ("episode_idx", "episode_id", "model_name", "success"):
        if key in seq:
            metadata[key] = seq[key]
    if metadata:
        record["metadata"] = metadata
    return record


def convert(args: argparse.Namespace) -> dict[str, Any]:
    if args.history_len < 1:
        raise ValueError("--history-len must be >= 1")
    if args.temporal_stride < 1:
        raise ValueError("--temporal-stride must be >= 1")
    if args.action_dim < 1:
        raise ValueError("--action-dim must be >= 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")

    dataset_path = Path(args.dataset_path)
    images_base_dir = Path(args.images_base_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json) if str(args.summary_json).strip() else output_path.with_name("summary.json")

    dataset = EBNavSequenceDataset(
        json_path=str(dataset_path),
        images_base_dir=str(images_base_dir),
        action_dim=int(args.action_dim),
        history_len=int(args.history_len),
        temporal_stride=int(args.temporal_stride),
        reward_cache_path=str(args.reward_cache_path).strip() or None,
    )

    total = len(dataset)
    start = min(int(args.start_index), total)
    end = total
    if int(args.max_samples) > 0:
        end = min(total, start + int(args.max_samples))

    repo_cwd = Path.cwd()
    exported = 0
    action_counts: Counter[int] = Counter()

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx in range(start, end):
            seq = dataset[idx]
            record = _normalize_sequence(seq, images_base_dir=images_base_dir, repo_cwd=repo_cwd)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1
            future_action_ids = record.get("future_action_ids")
            if isinstance(future_action_ids, list) and future_action_ids:
                action_counts[int(future_action_ids[0])] += 1

    summary = {
        "dataset_path": str(dataset_path),
        "images_base_dir": str(images_base_dir),
        "reward_cache_path": str(args.reward_cache_path).strip(),
        "output_jsonl": str(output_path),
        "summary_json": str(summary_path),
        "history_len": int(args.history_len),
        "temporal_stride": int(args.temporal_stride),
        "action_dim": int(args.action_dim),
        "total_dataset_sequences": total,
        "start_index": start,
        "end_index_exclusive": end,
        "exported_count": exported,
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
