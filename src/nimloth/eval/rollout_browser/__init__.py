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
    if name == "write_evaluation_browser":
        from .storage import write_evaluation_browser

        return write_evaluation_browser
    raise AttributeError(name)


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA",
    "ROLLOUT_AUDIT_SCHEMA",
    "RolloutBrowserArtifact",
    "rollout_trajectory_artifact",
    "validate_rollout_audit",
    "write_evaluation_browser",
]
