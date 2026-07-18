#!/usr/bin/env python3
"""Validate a Nimloth checkpoint view with the pinned VERL model classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.training.rl.verl_checkpoint import (
    validate_transformers449_checkpoint_view,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = validate_transformers449_checkpoint_view(args.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
