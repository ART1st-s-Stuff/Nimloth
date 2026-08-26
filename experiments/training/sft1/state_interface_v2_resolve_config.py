#!/usr/bin/env python3
"""Publish one explicit immutable launch-locked SFT1-v2 config."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nimloth.training.sft1.launch_config import (
    SFT1V2LaunchResolution,
    resolve_launch_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in (
        "repo", "expected_commit", "interpreter", "cache_output_dir", "run_dir",
        "wandb_run_name", "wandb_run_id", "processor_sha256", "tokenizer_sha256",
        "prompt_template_sha256", "token_table_sha256",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in (
        "world_size", "max_sequence_length", "max_padded_tokens",
        "max_rows_per_micro_batch", "rows_per_rank_update",
        "teacher_batch_size", "checkpoint_cadence_steps", "minimum_free_bytes",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, required=True)
    args = parser.parse_args()
    values = {
        name: getattr(args, name)
        for name in SFT1V2LaunchResolution.__dataclass_fields__
    }
    config = resolve_launch_config(
        args.template,
        args.output,
        SFT1V2LaunchResolution(**values),
    )
    print(json.dumps({
        "config_identity": config.identity,
        "resolved": asdict(config),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
