#!/usr/bin/env python3
"""Thin fail-closed controller for the approved SFT1-v2 phase graph."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nimloth.training.sft1.controller import main


if __name__ == "__main__":
    raise SystemExit(main())
