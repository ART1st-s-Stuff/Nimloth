"""Exact parent VAGEN and nested VERL source verification."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import subprocess


PINNED_VAGEN_COMMIT = "9f1e89eb8c9839a406b6e62aa75703494a79e5b5"
PINNED_VERL_COMMIT = "494f264494b2525f2c13595f63ac4912963e6d2f"


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"cannot verify git source with arguments {args!r}") from error


@lru_cache(maxsize=4)
def verify_pinned_vagen_verl_source(repo_root: Path) -> None:
    """Verify parent/nested gitlinks and checked-out commits without importing VERL."""

    root = Path(repo_root).resolve()
    vagen = root / "external" / "VAGEN"
    verl = vagen / "verl"
    checks = {
        "parent VAGEN gitlink": _git_output("-C", str(root), "rev-parse", "HEAD:external/VAGEN"),
        "checked-out VAGEN": _git_output("-C", str(vagen), "rev-parse", "HEAD"),
        "nested VERL gitlink": _git_output("-C", str(vagen), "rev-parse", "HEAD:verl"),
        "checked-out VERL": _git_output("-C", str(verl), "rev-parse", "HEAD"),
    }
    expected = {
        "parent VAGEN gitlink": PINNED_VAGEN_COMMIT,
        "checked-out VAGEN": PINNED_VAGEN_COMMIT,
        "nested VERL gitlink": PINNED_VERL_COMMIT,
        "checked-out VERL": PINNED_VERL_COMMIT,
    }
    mismatches = {
        name: (checks[name], expected[name])
        for name in checks
        if checks[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"VAGEN/VERL source identity mismatch: {mismatches}")


@lru_cache(maxsize=4)
def require_pinned_verl_import(repo_root: Path) -> type:
    """Return DataProto only when its import resolves to the nested gitlink."""

    import verl

    expected = (Path(repo_root).resolve() / "external" / "VAGEN" / "verl").resolve()
    module_path = Path(verl.__file__).resolve()
    actual = module_path.parents[1]
    if actual != expected:
        raise RuntimeError(
            "VERL import does not resolve to the pinned nested gitlink: "
            f"expected={expected}, actual={actual}"
        )
    return verl.DataProto


__all__ = [
    "PINNED_VAGEN_COMMIT",
    "PINNED_VERL_COMMIT",
    "require_pinned_verl_import",
    "verify_pinned_vagen_verl_source",
]
