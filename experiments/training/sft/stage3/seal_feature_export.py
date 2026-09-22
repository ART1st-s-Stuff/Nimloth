"""Seal completed Stage3 eval-only feature shards without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nimloth.eval.stage3_outcome import file_sha256
from nimloth.rollout.fresh import auxiliary_artifact_fingerprint, policy_artifact_fingerprint
from nimloth.training.sft.stage3.diagnostics import seal_existing_dino_feature_export


def checkpoint_identity(
    checkpoint: Path,
    eval_jsonl: Path,
    *,
    generation_source_commit: str,
    attention_implementation: str,
    launch_script: Path,
) -> tuple[int, dict]:
    checkpoint = checkpoint.resolve()
    state = torch.load(checkpoint / "rl_state.pt", map_location="cpu", weights_only=False)
    step = int(state["global_step"])
    auxiliaries = {
        name: checkpoint / path
        for name, path in {
            "projector": "state_proj.pt",
            "world_model": "wm_predictor",
            "value_head": "value_head",
            "outcome_head": "outcome_head.pt",
            "rl_state": "rl_state.pt",
        }.items()
    }
    identity = {
        "contract": "stage3_rl_fixed_eval_v1",
        "checkpoint": str(checkpoint),
        "global_step": step,
        "policy_fingerprint": policy_artifact_fingerprint(checkpoint),
        "auxiliary_fingerprints": {
            name: auxiliary_artifact_fingerprint(path)
            for name, path in auxiliaries.items()
        },
        "eval_jsonl": str(eval_jsonl.resolve()),
        "eval_sha256": file_sha256(eval_jsonl),
        "generation_source_commit": generation_source_commit,
        "attention_implementation": attention_implementation,
        "launch_script_sha256": file_sha256(launch_script),
    }
    return step, identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--generation-source-commit", required=True)
    parser.add_argument("--attention-implementation", required=True)
    parser.add_argument("--launch-script", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    step, identity = checkpoint_identity(
        args.checkpoint,
        args.eval_jsonl,
        generation_source_commit=args.generation_source_commit,
        attention_implementation=args.attention_implementation,
        launch_script=args.launch_script,
    )
    for rank in range(args.world_size):
        seal_existing_dino_feature_export(
            args.directory, rank=rank, step=step, identity=identity
        )
    (args.directory / "EXPORT_IDENTITY.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
