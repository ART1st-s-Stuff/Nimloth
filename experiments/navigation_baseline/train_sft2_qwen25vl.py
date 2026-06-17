#!/usr/bin/env python3
"""Deprecated wrapper — use experiments/training/sft2/train.py."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "training" / "sft2" / "train.py"
    runpy.run_path(str(target), run_name="__main__")
