#!/usr/bin/env python3
"""Create a non-mutating Transformers-4.49 view of a 4.55 Nimloth HF export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nimloth.training.rl.verl_checkpoint import (
    prepare_transformers449_checkpoint_view,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weight-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="hardlink avoids duplicating model shards and is the default",
    )
    args = parser.parse_args()
    manifest = prepare_transformers449_checkpoint_view(
        args.source,
        args.output,
        weight_mode=args.weight_mode,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
