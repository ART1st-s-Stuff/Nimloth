#!/usr/bin/env python3
"""Validate sixteen 1-GPU agents on a 4+4+4+2+1+1 allocation."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    return parser.parse_args()


def validate_allocation(run_root: Path, job_id: str) -> dict[str, object]:
    paths = sorted(run_root.glob(f"allocation_{job_id}_agent*.log"))
    assert len(paths) == 16, f"expected 16 logical agents, got {len(paths)}"
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
        gpu_rows = [line.removeprefix("gpu=") for line in lines if line.startswith("gpu=")]
        assert len(gpu_rows) == 1
        assert "H800" in gpu_rows[0]
        uuid = gpu_rows[0].split(",", 1)[0].strip()
        assert uuid not in all_uuids
        assert uuid not in host_uuids.setdefault(host, set())
        host_uuids[host].add(uuid)
        all_uuids.add(uuid)

    host_task_counts = Counter(
        next(
            dict(item.split("=", 1) for item in line.split())["host"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("job=")
        )
        for path in paths
    )
    assert ranks == set(range(16))
    assert sorted(host_task_counts.values()) == [1, 1, 2, 4, 4, 4]
    assert sorted(len(uuids) for uuids in host_uuids.values()) == [1, 1, 2, 4, 4, 4]
    assert len(all_uuids) == 16
    return {
        "logical_agents": 16,
        "physical_nodes": 6,
        "gpu_uuids": 16,
        "tasks_per_host": sorted(host_task_counts.values()),
    }


def main() -> None:
    args = parse_args()
    print(validate_allocation(args.run_root, args.job_id))


if __name__ == "__main__":
    main()
