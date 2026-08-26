"""Thin code-canary preflight; it deliberately does not launch training or services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from nimloth.training.sft1.config import load_sft1_v2_config
from nimloth.training.sft1.manifest import load_sft1_v2_manifest
from nimloth.training.verl.source import verify_pinned_vagen_verl_source


def validate_sft1_v2_canary_inputs(
    *,
    config_path: Path,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Validate identities and return a non-launching resolved summary."""

    config = load_sft1_v2_config(config_path)
    manifest = load_sft1_v2_manifest(manifest_path)
    verify_pinned_vagen_verl_source(repo_root)
    if config.state.objective_version != manifest.objective_version:
        raise ValueError("config/manifest state objective mismatch")
    if config.state.grid_tokens != manifest.query_count:
        raise ValueError("config/manifest query count mismatch")
    if config.state.action_dim != manifest.action_count:
        raise ValueError("config/manifest action count mismatch")
    return {
        "status": "code_canary_preflight_only",
        "config_identity": config.identity,
        "manifest_identity": manifest.identity,
        "objective_version": manifest.objective_version,
        "query_count": manifest.query_count,
        "state_dim": config.state.state_dim,
        "action_count": manifest.action_count,
        "train_split": manifest.train_split,
        "external_validation_split": manifest.external_validation_split,
        "launch_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_sft1_v2_canary_inputs(
        config_path=args.config,
        manifest_path=args.manifest,
        repo_root=args.repo_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = ["build_parser", "main", "validate_sft1_v2_canary_inputs"]
