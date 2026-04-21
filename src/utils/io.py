"""文件读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

