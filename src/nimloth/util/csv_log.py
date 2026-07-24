"""训练阶段共用的固定列 CSV 记录器。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CSVRecordWriter:
    """按声明顺序写 header 与记录，缺失字段写为空字符串。"""

    path: Path
    columns: tuple[str, ...]

    def ensure_header(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(self.columns)

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [record.get(column, "") for column in self.columns]
            )
