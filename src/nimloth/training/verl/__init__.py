"""Reusable VERL execution infrastructure; stage objectives remain in their owners."""

from nimloth.training.verl.source import (
    PINNED_VAGEN_COMMIT,
    PINNED_VERL_COMMIT,
    require_pinned_verl_import,
    verify_pinned_vagen_verl_source,
)

__all__ = [
    "PINNED_VAGEN_COMMIT",
    "PINNED_VERL_COMMIT",
    "require_pinned_verl_import",
    "verify_pinned_vagen_verl_source",
]
