"""Offline evaluation utilities.

Keep this package import lightweight. Some eval submodules depend on optional
world-model submodules (for example external/le-wm), so importing
``nimloth.eval.representation_ablation`` must not eagerly import rollout code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nimloth.eval.rollout import val_rollout_success_rate

__all__ = ["val_rollout_success_rate"]


def __getattr__(name: str):
    if name == "val_rollout_success_rate":
        from nimloth.eval.rollout import val_rollout_success_rate

        return val_rollout_success_rate
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
