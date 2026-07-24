#!/usr/bin/env python3
"""Report static success-label prevalence from a rollout JSONL dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.wm.statistics import dataset_rollout_success_rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=-1)
    args = ap.parse_args()

    rate = dataset_rollout_success_rate(args.val_jsonl, max_records=args.max_records)
    print(json.dumps({"jsonl": str(args.val_jsonl), "dataset_rollout_success_rate": rate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
