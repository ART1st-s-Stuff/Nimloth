#!/usr/bin/env python3
"""Compatibility CLI for the shared SFT/RL real-environment rollout producer."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from nimloth.training.sft.evaluation import rollout as _implementation

# Preserve private helpers used by historical evaluation launchers.
globals().update({name: value for name, value in vars(_implementation).items()
                  if not name.startswith("__")})

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
