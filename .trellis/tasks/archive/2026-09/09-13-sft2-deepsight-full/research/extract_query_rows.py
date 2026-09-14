#!/usr/bin/env python3
"""Extract the 64 latent query rows from one checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_tensor(root: Path, name: str) -> torch.Tensor:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    with safe_open(
        root / index["weight_map"][name], framework="pt", device="cpu"
    ) as handle:
        return handle.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    added = json.loads((args.checkpoint / "added_tokens.json").read_text())
    names = ["<|latent_state|>"] + [
        f"<|latent_state_{index}|>" for index in range(1, 64)
    ]
    ids = [int(added[name]) for name in names]
    row_ids = torch.tensor(ids, dtype=torch.long)
    bundle = {
        "checkpoint": str(args.checkpoint),
        "names": names,
        "ids": ids,
        "model.embed_tokens.weight": load_tensor(
            args.checkpoint, "model.embed_tokens.weight"
        ).index_select(0, row_ids),
        "lm_head.weight": load_tensor(
            args.checkpoint, "lm_head.weight"
        ).index_select(0, row_ids),
    }
    torch.save(bundle, args.output)
    print(json.dumps({"output": str(args.output), "ids": [min(ids), max(ids)]}))


if __name__ == "__main__":
    main()
