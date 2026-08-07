#!/usr/bin/env python3
"""Create a reproducible, fresh PlannerPolicyHead warm-start artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nimloth.wm import PlannerPolicyHead


def _dimensions(value_head_checkpoint: Path) -> tuple[int, int, int]:
    state_path = value_head_checkpoint / "value_head.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"missing ValueHead checkpoint: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    first = state.get("net.0.weight")
    last = state.get("net.2.weight")
    if (
        first is None
        or last is None
        or first.ndim != 2
        or last.ndim != 2
        or last.shape[1] != first.shape[0]
    ):
        raise ValueError("ValueHead checkpoint has an incompatible architecture")
    return int(first.shape[1]), int(first.shape[0]), int(last.shape[0])


def initialize(
    *,
    value_head_checkpoint: Path,
    output_dir: Path,
    seed: int,
) -> PlannerPolicyHead:
    """Initialize the same MLP shape without copying critic parameters."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty PolicyHead directory: {output_dir}"
        )
    emb_dim, hidden_dim, num_actions = _dimensions(value_head_checkpoint)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        policy_head = PlannerPolicyHead(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
        )
    policy_head.save_checkpoint(output_dir)
    metadata = {
        "artifact_type": "nimloth_planner_policy_head_v1",
        "initialization": "fresh_torch_default",
        "seed": int(seed),
        "architecture_reference": str(value_head_checkpoint.resolve()),
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "num_actions": num_actions,
        "copied_value_head_parameters": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return policy_head


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    initialize(
        value_head_checkpoint=args.value_head_checkpoint,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps({"planner_policy_head": str(args.output_dir.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
