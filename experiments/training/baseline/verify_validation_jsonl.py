#!/usr/bin/env python3
"""Verify VAGEN validation JSONL has stable env metadata and expected composition."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_rows(rows: list[dict], *, expect_base: int, expect_common: int) -> dict:
    if not rows:
        raise SystemExit("empty jsonl")

    required = ("env_seed", "data_source", "eval_set")
    missing = [f for f in required if f not in rows[0]]
    if missing:
        raise SystemExit(f"missing fields in jsonl: {missing}")

    counts = Counter(str(r["data_source"]) for r in rows)
    pairs = [(r["data_source"], r["env_seed"]) for r in rows]
    unique_pairs = len(set(pairs))

    if counts.get("navigation_base", 0) != expect_base:
        raise SystemExit(f"navigation_base count {counts.get('navigation_base', 0)} != {expect_base}")
    if counts.get("navigation_common", 0) != expect_common:
        raise SystemExit(f"navigation_common count {counts.get('navigation_common', 0)} != {expect_common}")
    if unique_pairs != len(rows):
        raise SystemExit(
            f"duplicate (data_source, env_seed): {len(rows) - unique_pairs} duplicates in {len(rows)} rows"
        )

    seeds = sorted(int(r["env_seed"]) for r in rows if r["data_source"] == "navigation_base")
    if len(seeds) == expect_base and (seeds[0], seeds[-1]) != (1, expect_base):
        raise SystemExit(f"unexpected base env_seed range: {seeds[0]}..{seeds[-1]}")

    return {
        "n": len(rows),
        "counts": dict(counts),
        "unique_env_instances": unique_pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path, help="validation/{step}.jsonl")
    ap.add_argument("--expect-base", type=int, default=60)
    ap.add_argument("--expect-common", type=int, default=60)
    args = ap.parse_args()

    stats = verify_rows(load_rows(args.jsonl), expect_base=args.expect_base, expect_common=args.expect_common)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
