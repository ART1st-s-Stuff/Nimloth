#!/usr/bin/env python3
"""Build a deterministic one-row VAGEN navigation dataset for the online gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30002)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_config = {
        "render_mode": "vision",
        "prompt_format": "source_eval_mode",
        "use_state_reward": False,
        "eval_set": "base_train",
    }
    row = {
        "data_source": "navigation",
        "prompt": [{"role": "user", "content": ""}],
        "extra_info": {
            "split": "train",
            "env_name": "navigation",
            "env_config": env_config,
            "seed": int(args.seed),
        },
    }
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    Dataset.from_list([row]).to_parquet(train_path)
    Dataset.from_list([row]).to_parquet(val_path)
    manifest = {
        "schema_version": 1,
        "purpose": "mechanics-only online VAGEN rollout gate",
        "dataset_split": "base_train",
        "seed": int(args.seed),
        "rows": 1,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "env_config": env_config,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
