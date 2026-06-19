#!/usr/bin/env python3
"""Verify VAGEN validation JSONL has stable env metadata and expected composition."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_SPEC: dict[str, dict[str, object]] = {
    "navigation_base": {"count": 60, "eval_set": "base"},
    "navigation_common": {"count": 60, "eval_set": "common_sense"},
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_rows(
    rows: list[dict],
    *,
    spec: dict[str, dict[str, object]],
    val_repeat: int = 1,
) -> dict:
    if not rows:
        raise SystemExit("empty jsonl")

    required = ("env_seed", "data_source", "eval_set")
    missing = [f for f in required if f not in rows[0]]
    if missing:
        raise SystemExit(f"missing fields in jsonl: {missing}")

    counts: Counter[str] = Counter()
    eval_set_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: Counter[tuple[str, object]] = Counter()

    for row in rows:
        src = str(row["data_source"])
        counts[src] += 1
        eval_set_by_source[src][str(row["eval_set"])] += 1
        pair_counts[(src, row["env_seed"])] += 1

    for src, rule in spec.items():
        exp_count = int(rule["count"])
        exp_eval_set = rule.get("eval_set")
        got = counts.get(src, 0)
        if got != exp_count:
            raise SystemExit(f"{src} count {got} != {exp_count}")
        if exp_eval_set is not None:
            allowed = {str(exp_eval_set)}
            observed = set(eval_set_by_source[src])
            if observed != allowed:
                raise SystemExit(
                    f"{src} eval_set mismatch: expected {exp_eval_set!r}, observed {dict(eval_set_by_source[src])}"
                )

    overflow = {pair: count for pair, count in pair_counts.items() if count > val_repeat}
    if overflow:
        raise SystemExit(
            f"duplicate (data_source, env_seed) above val_repeat={val_repeat}: {overflow}"
        )

    base_seeds = sorted(int(r["env_seed"]) for r in rows if r["data_source"] == "navigation_base")
    exp_base = int(spec["navigation_base"]["count"])
    if base_seeds and len(base_seeds) == exp_base and (base_seeds[0], base_seeds[-1]) != (1, exp_base):
        raise SystemExit(f"unexpected navigation_base env_seed range: {base_seeds[0]}..{base_seeds[-1]}")

    return {
        "n": len(rows),
        "counts": dict(counts),
        "eval_sets": {src: dict(counter) for src, counter in eval_set_by_source.items()},
        "unique_env_instances": len(pair_counts),
        "val_repeat": val_repeat,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path, help="validation/{step}.jsonl")
    ap.add_argument("--expect-base", type=int, default=60)
    ap.add_argument("--expect-common", type=int, default=60)
    ap.add_argument("--base-eval-set", default="base")
    ap.add_argument("--common-eval-set", default="common_sense")
    ap.add_argument("--val-repeat", type=int, default=1)
    args = ap.parse_args()

    spec = {
        "navigation_base": {"count": args.expect_base, "eval_set": args.base_eval_set},
        "navigation_common": {"count": args.expect_common, "eval_set": args.common_eval_set},
    }
    stats = verify_rows(load_rows(args.jsonl), spec=spec, val_repeat=args.val_repeat)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
