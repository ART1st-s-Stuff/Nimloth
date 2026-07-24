"""与具体训练阶段无关的配置文件加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load YAML configuration") from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data
