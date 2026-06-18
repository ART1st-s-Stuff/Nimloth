#!/usr/bin/env python3
"""Report trajectory-level val success rate from Nimloth jsonl (offline metric)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.eval.rollout import val_rollout_success_rate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-jsonl", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=-1)
    args = ap.parse_args()

    rate = val_rollout_success_rate(args.val_jsonl, max_records=args.max_records)
    print(json.dumps({"val_jsonl": str(args.val_jsonl), "val_rollout_success_rate": rate}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
