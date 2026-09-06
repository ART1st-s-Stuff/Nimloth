#!/usr/bin/env python3
"""Compatibility entry for the shared, verified early-stage checkpoint export."""
from pathlib import Path
import sys

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
from nimloth.training.sft.stage1 import checkpoint_export as _implementation

def __getattr__(name):
    return getattr(_implementation, name)

main = _implementation.main
if __name__ == "__main__":
    raise SystemExit(main())
