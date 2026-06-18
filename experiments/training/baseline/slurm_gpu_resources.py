#!/usr/bin/env python3
"""Print per-node Slurm GPU/CPU/memory availability.

This script is intentionally dependency-free and parses `scontrol show nodes`.
It is useful when deciding whether a job can fit on a single node or whether
available GPUs are fragmented across nodes.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class NodeResource:
    partition: str
    node: str
    state: str
    total_gpu: int
    alloc_gpu: int
    free_gpu: int
    total_cpu: int
    alloc_cpu: int
    free_cpu: int
    real_mem_mb: int | None
    alloc_mem_mb: int | None
    free_mem_mb: int | None


def parse_int(value: str | None) -> int | None:
    if value is None or value in {"", "N/A"}:
        return None
    m = re.match(r"(\d+)", value)
    return int(m.group(1)) if m else None


def parse_nodes() -> list[NodeResource]:
    output = subprocess.check_output(["scontrol", "show", "nodes"], text=True, errors="replace")
    rows: list[NodeResource] = []
    for block in output.strip().split("\n\n"):
        if not block.strip():
            continue
        fields: dict[str, str] = {}
        for token in re.split(r"\s+", block.strip()):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value

        gres = fields.get("Gres", "")
        if "gpu" not in gres:
            continue

        total_gpu = 0
        m = re.search(r"gpu:(\d+)", gres)
        if m:
            total_gpu = int(m.group(1))

        alloc_gpu = 0
        m = re.search(r"gres/gpu=(\d+)", fields.get("AllocTRES", ""))
        if m:
            alloc_gpu = int(m.group(1))

        total_cpu = parse_int(fields.get("CPUTot")) or 0
        alloc_cpu = parse_int(fields.get("CPUAlloc")) or 0
        real_mem = parse_int(fields.get("RealMemory"))
        alloc_mem = parse_int(fields.get("AllocMem"))
        free_mem = parse_int(fields.get("FreeMem"))

        rows.append(
            NodeResource(
                partition=fields.get("Partitions", "unknown"),
                node=fields.get("NodeName", "unknown"),
                state=fields.get("State", "unknown"),
                total_gpu=total_gpu,
                alloc_gpu=alloc_gpu,
                free_gpu=max(total_gpu - alloc_gpu, 0),
                total_cpu=total_cpu,
                alloc_cpu=alloc_cpu,
                free_cpu=max(total_cpu - alloc_cpu, 0),
                real_mem_mb=real_mem,
                alloc_mem_mb=alloc_mem,
                free_mem_mb=free_mem,
            )
        )
    return rows


def mem_gb(value_mb: int | None) -> str:
    if value_mb is None:
        return "?"
    return f"{value_mb / 1024:.1f}G"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", action="append", help="Only show this partition. Can be repeated.")
    parser.add_argument("--only-free-gpu", action="store_true", help="Only show nodes with at least one free GPU.")
    parser.add_argument("--min-free-gpu", type=int, default=0, help="Only show nodes with at least this many free GPUs.")
    args = parser.parse_args()

    rows = parse_nodes()
    if args.partition:
        wanted = set(args.partition)
        rows = [row for row in rows if row.partition in wanted]
    if args.only_free_gpu:
        rows = [row for row in rows if row.free_gpu > 0]
    if args.min_free_gpu:
        rows = [row for row in rows if row.free_gpu >= args.min_free_gpu]

    rows.sort(key=lambda r: (r.partition, r.node))

    print("partition node state gpu_free/alloc/total cpu_free/alloc/total free_mem real_mem")
    for row in rows:
        print(
            f"{row.partition:<8} {row.node:<7} {row.state:<12} "
            f"{row.free_gpu:>2}/{row.alloc_gpu:<2}/{row.total_gpu:<2} "
            f"{row.free_cpu:>3}/{row.alloc_cpu:<3}/{row.total_cpu:<3} "
            f"{mem_gb(row.free_mem_mb):>8} {mem_gb(row.real_mem_mb):>8}"
        )

    print("\nsummary_by_partition")
    summary: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    for row in rows:
        values = summary[row.partition]
        values[0] += 1
        values[1] += row.total_gpu
        values[2] += row.alloc_gpu
        values[3] += row.free_gpu
        values[4] += 1 if row.free_gpu > 0 else 0
    for partition in sorted(summary):
        nodes, total_gpu, alloc_gpu, free_gpu, nodes_with_free = summary[partition]
        print(
            f"{partition}: nodes={nodes} nodes_with_free_gpu={nodes_with_free} "
            f"gpu_free/alloc/total={free_gpu}/{alloc_gpu}/{total_gpu}"
        )


if __name__ == "__main__":
    main()
