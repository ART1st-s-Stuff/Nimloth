"""统一 rollout trajectory 的 JSONL 持久化。"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from nimloth.rollout.schema import RolloutTrajectory
from nimloth.rollout.validation import validate_rollout_trajectory


def save_trajectories(
    trajectories: list[RolloutTrajectory],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "trajectories.jsonl"
    lines: list[str] = []
    for trajectory in trajectories:
        validate_rollout_trajectory(trajectory)
        lines.append(
            json.dumps(
                trajectory.to_record(),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_dir,
            prefix=".trajectories.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.writelines(lines)
        temporary_path.replace(jsonl_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return jsonl_path


def load_trajectories(jsonl_path: Path) -> list[RolloutTrajectory]:
    trajectories: list[RolloutTrajectory] = []
    opener = gzip.open if jsonl_path.suffix == ".gz" else Path.open
    with opener(jsonl_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                trajectories.append(RolloutTrajectory.from_record(json.loads(line)))
    return trajectories
