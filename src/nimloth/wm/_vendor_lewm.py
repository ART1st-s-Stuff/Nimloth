"""Load LeWM modules from the ``external/le-wm`` git submodule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_LEWM_ROOT = Path(__file__).resolve().parents[3] / "external" / "le-wm"
_LOADED: dict[str, ModuleType] = {}


def lewm_root() -> Path:
    if not _LEWM_ROOT.is_dir():
        raise ImportError(
            f"LeWM submodule not found at {_LEWM_ROOT}. "
            "Run: git submodule update --init external/le-wm"
        )
    return _LEWM_ROOT


def _load_lewm_file(module_name: str, filename: str) -> ModuleType:
    if module_name in _LOADED:
        return _LOADED[module_name]

    path = lewm_root() / filename
    if not path.is_file():
        raise ImportError(f"LeWM file missing: {path}")

    spec = importlib.util.spec_from_file_location(f"nimloth_lewm.{module_name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load LeWM module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _LOADED[module_name] = module
    return module


_module = _load_lewm_file("module", "module.py")

ARPredictor = _module.ARPredictor
Embedder = _module.Embedder
MLP = _module.MLP

__all__ = ["ARPredictor", "Embedder", "MLP", "lewm_root"]
