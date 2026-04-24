"""文件读写工具。"""

from __future__ import annotations

import atexit
from pathlib import Path
import time
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    output.write_text(__import__("json").dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 每路径一个 buffer：批量累积，达到阈值或超时后一次性写入。
# 消除了 96 workers 下的文件句柄争用 + 每次 open 抖动开销。
# ---------------------------------------------------------------------------

class _PathBuffer:
    __slots__ = ("path", "lines", "last_flush", "flush_interval", "flush_size")

    def __init__(self, path: str, flush_size: int = 200, flush_seconds: float = 2.0) -> None:
        self.path = path
        self.lines: list[str] = []
        self.last_flush = time.time()
        self.flush_interval = flush_seconds
        self.flush_size = flush_size

    def append(self, line: str) -> None:
        self.lines.append(line)
        now = time.time()
        if len(self.lines) >= self.flush_size or now - self.last_flush >= self.flush_interval:
            self._flush()

    def _flush(self) -> None:
        if not self.lines:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.writelines(self.lines)
        self.lines.clear()
        self.last_flush = time.time()

    def flush(self) -> None:
        self._flush()


_buf_map: dict[str, _PathBuffer] = {}
_MAX_BUFFERS = 200  # 最多缓存 200 个路径


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    key = str(Path(path).resolve())
    if key in _buf_map:
        buf = _buf_map[key]
        buf.append(__import__("json").dumps(data, ensure_ascii=False) + "\n")
        return

    ensure_dir(Path(key).parent)
    buf = _PathBuffer(key)
    _buf_map[key] = buf
    buf.append(__import__("json").dumps(data, ensure_ascii=False) + "\n")

    # 超过上限时清理最老的 buffer（防止内存泄漏）
    if len(_buf_map) > _MAX_BUFFERS:
        oldest_key = next(iter(_buf_map))
        _buf_map.pop(oldest_key, None)


def flush_all_buffers() -> None:
    for buf in _buf_map.values():
        buf.flush()


atexit.register(flush_all_buffers)