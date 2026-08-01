#!/usr/bin/env python3
"""Validate three variable-local-world agents on an 8+4+4 allocation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def path_host(path: Path) -> str:
    header = next(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("job=")
    )
    return dict(item.split("=", 1) for item in header.split())["host"]


def validate_allocation(run_root: Path, job_id: str) -> dict[str, object]:
    paths = sorted(run_root.glob(f"allocation_{job_id}_agent*.log"))
    assert len(paths) == 3, f"expected 3 logical agents, got {len(paths)}"
    ranks: set[int] = set()
    host_uuids: dict[str, set[str]] = {}
    all_uuids: set[str] = set()
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        header = next(line for line in lines if line.startswith("job="))
        fields = dict(item.split("=", 1) for item in header.split())
        rank = int(fields["agent_rank"])
        assert rank not in ranks
        ranks.add(rank)
        host = fields["host"]
        gpu_rows = [
            line.removeprefix("gpu=")
            for line in lines
            if line.startswith("gpu=")
        ]
        expected_gpus = 8 if rank == 0 else 4
        assert len(gpu_rows) == expected_gpus
        assert all("H800" in row for row in gpu_rows)
        uuids = {row.split(",", 1)[0].strip() for row in gpu_rows}
        assert len(uuids) == expected_gpus
        assert not (host_uuids.setdefault(host, set()) & uuids)
        host_uuids[host].update(uuids)
        assert not (all_uuids & uuids)
        all_uuids.update(uuids)

    host_task_counts = Counter(path_host(path) for path in paths)
    assert ranks == set(range(3))
    assert sorted(host_task_counts.values()) == [1, 1, 1]
    assert sorted(len(uuids) for uuids in host_uuids.values()) == [4, 4, 8]
    assert len(all_uuids) == 16
    return {
        "logical_agents": 3,
        "physical_nodes": 3,
        "gpu_uuids": 16,
        "tasks_per_host": sorted(host_task_counts.values()),
    }


def main() -> None:
    args = parse_args()
    print(validate_allocation(args.run_root, args.job_id))


if __name__ == "__main__":
    main()
