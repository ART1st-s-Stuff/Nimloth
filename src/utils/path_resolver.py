"""路径解析工具。"""

from __future__ import annotations

import json
from pathlib import Path


def resolve_latest_path(path_text: str) -> Path:
    """解析 latest 软链接风格路径。"""
    candidate = Path(path_text)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if len(parts) >= 3 and parts[-2] == "latest":
        group_dir = Path(*parts[:-2])
        meta_path = group_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            latest = metadata.get("latest")
            if isinstance(latest, str):
                latest_path = group_dir / latest / parts[-1]
                if latest_path.exists():
                    return latest_path
    return candidate
