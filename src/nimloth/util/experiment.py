"""实验标识和输出目录创建。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


def _validate_path_component(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"{field} must be a non-empty path component")
    if Path(normalized).name != normalized:
        raise ValueError(f"{field} must not contain path separators: {value!r}")
    return normalized


def create_experiment_dir(
    *,
    root: Path,
    stage: str,
    name: str,
    run_date: date | None = None,
    exist_ok: bool = False,
) -> Path:
    """创建 ``root/stage/YYYY-MM-DD/name`` 并返回其绝对路径。"""

    stage_component = _validate_path_component(stage, field="stage")
    name_component = _validate_path_component(name, field="name")
    day = (run_date or date.today()).isoformat()
    output_dir = Path(root) / stage_component / day / name_component
    output_dir.mkdir(parents=True, exist_ok=exist_ok)
    return output_dir.resolve()


@dataclass(frozen=True)
class Experiment:
    stage: str
    git_commit: str
    name: str = ""

    def create_directory(
        self,
        root: Path,
        *,
        run_date: date | None = None,
        exist_ok: bool = False,
    ) -> Path:
        if not self.name:
            raise ValueError("Experiment.name is required to create an output directory")
        return create_experiment_dir(
            root=root,
            stage=self.stage,
            name=self.name,
            run_date=run_date,
            exist_ok=exist_ok,
        )
