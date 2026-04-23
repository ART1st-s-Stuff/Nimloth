"""统一的模型路径解析工具。"""

from __future__ import annotations

import json
from pathlib import Path


def _safe_read_latest_name(meta_path: Path) -> str | None:
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    latest = payload.get("latest")
    return latest if isinstance(latest, str) and latest else None


def resolve_latest_model_file(base_dir: str | Path, candidate_files: list[str]) -> Path | None:
    """解析最近可用模型文件；按 candidate_files 顺序优先级匹配。"""
    root = Path(base_dir)
    if not root.exists() or not root.is_dir():
        return None
    candidates = [name for name in candidate_files if isinstance(name, str) and name.strip()]
    if not candidates:
        return None

    # 先尝试 metadata.latest 指向的目录。
    latest_name = _safe_read_latest_name(root / "metadata.json")
    if latest_name:
        latest_dir = root / latest_name
        if latest_dir.exists() and latest_dir.is_dir():
            for file_name in candidates:
                file_path = latest_dir / file_name
                if file_path.exists():
                    return file_path

    # 再按目录修改时间倒序，寻找首个满足候选文件的 run。
    run_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        for file_name in candidates:
            file_path = run_dir / file_name
            if file_path.exists():
                return file_path
    return None

