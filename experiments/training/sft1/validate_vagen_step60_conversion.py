#!/usr/bin/env python3
"""Independently validate a published VAGEN step60 conversion directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.training.sft1.vagen_step60_convert import (
    validate_conversion_output,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = validate_conversion_output(args.output_dir)
    print(json.dumps(manifest["result"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
