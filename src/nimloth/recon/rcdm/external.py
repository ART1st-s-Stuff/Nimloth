"""Helpers for importing the vendored RCDM code from ``external/RCDM``.

The RCDM repository is kept as a git submodule and is not assumed to be
installed into the active Python environment.  These helpers add the submodule
root to ``sys.path`` at runtime, keeping Nimloth code separate from upstream
RCDM files.
"""

from __future__ import annotations

import sys
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_RCDM_ROOT = _REPO_ROOT / "external" / "RCDM"


def rcdm_root(path: str | Path | None = None) -> Path:
    """Return the RCDM checkout path and validate its package directory exists."""

    root = Path(path).expanduser().resolve() if path is not None else _DEFAULT_RCDM_ROOT
    package_dir = root / "guided_diffusion_rcdm"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            f"RCDM package not found at {package_dir}. "
            "Run `git submodule update --init external/RCDM` first."
        )
    return root


def ensure_rcdm_importable(path: str | Path | None = None) -> Path:
    """Add ``external/RCDM`` to ``sys.path`` and return its resolved path."""

    root = rcdm_root(path)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
