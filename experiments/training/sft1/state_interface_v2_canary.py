#!/usr/bin/env python3
"""Validate the SFT1-v2 code-canary contract without launching an experiment."""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from nimloth.training.sft1.entrypoint import main


if __name__ == "__main__":
    raise SystemExit(main())
