#!/usr/bin/env python3
"""Compatibility entry for SFT1; implementation lives in training.sft.stage1."""
from pathlib import Path
import sys

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
from nimloth.training.sft.stage1.trainer import main

if __name__ == "__main__":
    raise SystemExit(main())
