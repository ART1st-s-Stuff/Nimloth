#!/usr/bin/env python3
"""Independently recompute a reconstruction runtime-contract payload SHA256."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training.sft1.vagen_step60_collect import (
    source_runtime_contract_payload_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    digest = source_runtime_contract_payload_sha256(contract)
    if contract.get("contract_payload_sha256") != digest:
        raise ValueError("runtime contract claimed payload SHA256 mismatch")
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
