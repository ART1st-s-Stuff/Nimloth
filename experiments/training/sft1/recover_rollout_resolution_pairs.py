#!/usr/bin/env python3
"""Recover paired rollout identities when async recorder order corrupted metadata.

The legacy validation dumper zipped asynchronously ordered rollout records with
input-order metadata.  The wrong metadata is still a permutation within each
validation batch.  This tool recovers the batch from that permutation, obtains
the actual source from ``config_id``, groups by instruction, and resolves
repeated instructions by minimum-RMSE matching of the initial frames.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from math import comb
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


_INSTRUCTION_RE = re.compile(
    r"Human Instruction: (.*?)\nDecide your next action", re.DOTALL
)
_EVAL_SET_RE = re.compile(r"eval_set=([^,)]+)")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _control_info(path: Path, batch_size: int):
    rows = pd.read_parquet(path).to_dict("records")
    key_to_batch: dict[tuple[str, int], int] = {}
    source_by_batch_eval: dict[tuple[int, str], str] = {}
    for index, row in enumerate(rows):
        extra = row["extra_info"]
        source = str(row["data_source"])
        seed = int(extra["seed"])
        eval_set = str(extra["env_config"]["eval_set"])
        batch = index // batch_size
        key = (source, seed)
        if key in key_to_batch:
            raise ValueError(f"duplicate control key {key}")
        key_to_batch[key] = batch
        source_key = (batch, eval_set)
        previous = source_by_batch_eval.setdefault(source_key, source)
        if previous != source:
            raise ValueError(
                f"batch {batch} eval_set {eval_set!r} maps to multiple sources"
            )
    return key_to_batch, source_by_batch_eval


def _instruction(row: dict[str, Any]) -> str:
    match = _INSTRUCTION_RE.search(str(row.get("output_str", "")))
    if match is None:
        raise ValueError("rollout row lacks an initial Human Instruction")
    return match.group(1)


def _actual_eval_set(row: dict[str, Any]) -> str:
    match = _EVAL_SET_RE.search(str(row.get("config_id", "")))
    if match is None:
        raise ValueError(f"cannot parse eval_set from config_id={row.get('config_id')!r}")
    return match.group(1)


def _group_rows(
    rows: list[dict[str, Any]],
    key_to_batch: dict[tuple[str, int], int],
    source_by_batch_eval: dict[tuple[int, str], str],
):
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        reported_key = (str(row.get("data_source")), int(row.get("env_seed")))
        if reported_key not in key_to_batch:
            raise ValueError(f"reported metadata key absent from control parquet: {reported_key}")
        batch = key_to_batch[reported_key]
        eval_set = _actual_eval_set(row)
        source_key = (batch, eval_set)
        if source_key not in source_by_batch_eval:
            raise ValueError(f"runtime source absent from control batch: {source_key}")
        source = source_by_batch_eval[source_key]
        grouped[(batch, source, _instruction(row))].append(row)
    return grouped


def _initial_frame(row: dict[str, Any]) -> np.ndarray:
    paths = row.get("image_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("rollout row has no image_paths")
    with Image.open(paths[0]) as image:
        resized = image.convert("RGB").resize((255, 255), Image.Resampling.BICUBIC)
        return np.asarray(resized, dtype=np.float32)


def _minimum_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError(f"assignment cost must be square, got {cost.shape}")
    size = cost.shape[0]

    @lru_cache(maxsize=None)
    def solve(old_index: int, used_mask: int):
        if old_index == size:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for new_index in range(size):
            if used_mask & (1 << new_index):
                continue
            tail_cost, tail = solve(old_index + 1, used_mask | (1 << new_index))
            candidate = (float(cost[old_index, new_index]) + tail_cost, (new_index,) + tail)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best

    _, columns = solve(0, 0)
    return list(enumerate(columns))


def _is_success(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict) or "success" not in metrics:
        raise ValueError("raw rollout row lacks metrics.success")
    value = metrics["success"]
    return value if isinstance(value, bool) else float(value or 0.0) >= 1.0


def _mcnemar_exact(old_only: int, new_only: int) -> float:
    discordant = old_only + new_only
    if discordant == 0:
        return 1.0
    lower = min(old_only, new_only)
    one_sided = sum(comb(discordant, index) for index in range(lower + 1)) / (
        2**discordant
    )
    return min(1.0, 2.0 * one_sided)


def compare_runtime_pairs(
    old_jsonl: Path,
    new_jsonl: Path,
    old_control_parquet: Path,
    new_control_parquet: Path,
    *,
    batch_size: int,
) -> dict[str, Any]:
    old_rows = _load_jsonl(old_jsonl)
    new_rows = _load_jsonl(new_jsonl)
    old_control = _control_info(old_control_parquet, batch_size)
    new_control = _control_info(new_control_parquet, batch_size)
    old_groups = _group_rows(old_rows, *old_control)
    new_groups = _group_rows(new_rows, *new_control)
    if set(old_groups) != set(new_groups):
        raise ValueError("runtime identity groups differ between runs")

    pairs: list[dict[str, Any]] = []
    assigned_costs: list[float] = []
    unassigned_costs: list[float] = []
    multi_record_groups = 0
    for group_key in sorted(old_groups):
        old_group = old_groups[group_key]
        new_group = new_groups[group_key]
        if len(old_group) != len(new_group):
            raise ValueError(
                f"runtime identity group count differs for {group_key}: "
                f"{len(old_group)} vs {len(new_group)}"
            )
        if len(old_group) > 1:
            multi_record_groups += 1
        old_frames = [_initial_frame(row) for row in old_group]
        new_frames = [_initial_frame(row) for row in new_group]
        cost = np.asarray(
            [
                [float(np.sqrt(np.mean((old - new) ** 2))) for new in new_frames]
                for old in old_frames
            ]
        )
        batch, source, instruction = group_key
        assignment = _minimum_assignment(cost)
        assigned_indexes = set(assignment)
        if len(old_group) > 1:
            unassigned_costs.extend(
                float(cost[old_index, new_index])
                for old_index in range(len(old_group))
                for new_index in range(len(new_group))
                if (old_index, new_index) not in assigned_indexes
            )
        for pair_index, (old_index, new_index) in enumerate(assignment):
            old_row = old_group[old_index]
            new_row = new_group[new_index]
            rmse = float(cost[old_index, new_index])
            assigned_costs.append(rmse)
            pairs.append(
                {
                    "runtime_pair_id": f"batch{batch}:{source}:{instruction}:{pair_index}",
                    "batch": batch,
                    "data_source": source,
                    "instruction": instruction,
                    "initial_frame_rmse": rmse,
                    "old_env_id": old_row.get("env_id"),
                    "new_env_id": new_row.get("env_id"),
                    "old_reported_seed": old_row.get("env_seed"),
                    "new_reported_seed": new_row.get("env_seed"),
                    "old_success": _is_success(old_row),
                    "new_success": _is_success(new_row),
                }
            )

    paired = {
        "both_success": 0,
        "old_only_success": 0,
        "new_only_success": 0,
        "both_failure": 0,
    }
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        old_success = pair["old_success"]
        new_success = pair["new_success"]
        if old_success and new_success:
            outcome = "both_success"
        elif old_success:
            outcome = "old_only_success"
        elif new_success:
            outcome = "new_only_success"
        else:
            outcome = "both_failure"
        paired[outcome] += 1
        counts = by_source[pair["data_source"]]
        counts["records"] += 1
        counts["old_success"] += int(old_success)
        counts["new_success"] += int(new_success)

    source_summary = {}
    for source, counts in sorted(by_source.items()):
        records = counts["records"]
        source_summary[source] = {
            **dict(counts),
            "old_success_rate": counts["old_success"] / records,
            "new_success_rate": counts["new_success"] / records,
            "success_rate_delta": (
                counts["new_success"] - counts["old_success"]
            )
            / records,
        }
    old_success = sum(pair["old_success"] for pair in pairs)
    new_success = sum(pair["new_success"] for pair in pairs)
    return {
        "records": len(pairs),
        "old_success": old_success,
        "new_success": new_success,
        "old_success_rate": old_success / len(pairs),
        "new_success_rate": new_success / len(pairs),
        "success_rate_delta": (new_success - old_success) / len(pairs),
        "paired": paired,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(
            paired["old_only_success"], paired["new_only_success"]
        ),
        "by_data_source": source_summary,
        "matching": {
            "method": "control-batch + runtime-config + instruction + minimum initial-frame RMSE",
            "groups": len(old_groups),
            "multi_record_groups": multi_record_groups,
            "min_assigned_initial_frame_rmse": min(assigned_costs),
            "median_assigned_initial_frame_rmse": float(np.median(assigned_costs)),
            "max_assigned_initial_frame_rmse": max(assigned_costs),
            "min_unassigned_initial_frame_rmse_in_multi_groups": min(
                unassigned_costs
            ) if unassigned_costs else None,
            "seed_labels_recovered": False,
        },
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-jsonl", type=Path, required=True)
    parser.add_argument("--new-jsonl", type=Path, required=True)
    parser.add_argument("--old-control-parquet", type=Path, required=True)
    parser.add_argument("--new-control-parquet", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare_runtime_pairs(
        args.old_jsonl,
        args.new_jsonl,
        args.old_control_parquet,
        args.new_control_parquet,
        batch_size=args.batch_size,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
