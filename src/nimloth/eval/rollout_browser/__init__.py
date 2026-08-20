"""Offline, auditable rollout browsers for environment-based evaluations."""

from __future__ import annotations

from typing import Any

from .schema import (
    EVALUATION_MANIFEST_SCHEMA,
    ROLLOUT_AUDIT_SCHEMA,
    validate_rollout_audit,
)


def __getattr__(name: str) -> Any:
    """Keep schema-only users independent from torch/Pillow rollout dependencies."""

    if name in {"RolloutBrowserArtifact", "rollout_trajectory_artifact"}:
        from .sft_adapter import RolloutBrowserArtifact, rollout_trajectory_artifact

        return {
            "RolloutBrowserArtifact": RolloutBrowserArtifact,
            "rollout_trajectory_artifact": rollout_trajectory_artifact,
        }[name]
    if name in {
        "finalize_evaluation_browser",
        "merge_evaluation_browsers",
        "write_evaluation_browser",
        "write_evaluation_browser_batch",
    }:
        from .storage import (
            finalize_evaluation_browser,
            merge_evaluation_browsers,
            write_evaluation_browser,
            write_evaluation_browser_batch,
        )

        return {
            "finalize_evaluation_browser": finalize_evaluation_browser,
            "merge_evaluation_browsers": merge_evaluation_browsers,
            "write_evaluation_browser": write_evaluation_browser,
            "write_evaluation_browser_batch": write_evaluation_browser_batch,
        }[name]
    raise AttributeError(name)


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "ROLLOUT_AUDIT_SCHEMA",
    "RolloutBrowserArtifact",
    "rollout_trajectory_artifact",
    "validate_rollout_audit",
    "finalize_evaluation_browser",
    "merge_evaluation_browsers",
    "write_evaluation_browser",
    "write_evaluation_browser_batch",
]
