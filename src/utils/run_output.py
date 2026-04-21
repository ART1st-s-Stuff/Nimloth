"""统一管理任务输出目录与 metadata 索引。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from src.utils.io import ensure_dir


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_metadata(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {"latest": None, "runs": [], "updated_at": _now_iso()}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"latest": None, "runs": [], "updated_at": _now_iso()}
    if not isinstance(data, dict):
        return {"latest": None, "runs": [], "updated_at": _now_iso()}
    runs = data.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    return {
        "latest": data.get("latest"),
        "runs": runs,
        "updated_at": data.get("updated_at", _now_iso()),
    }


def _write_metadata(meta_path: Path, payload: dict) -> None:
    payload["updated_at"] = _now_iso()
    tmp_path = meta_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(meta_path)


def refresh_run_metadata(parent_dir: str | Path) -> dict:
    root = ensure_dir(parent_dir)
    meta_path = root / "metadata.json"
    runs = [p for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"]
    runs_sorted = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    meta_runs = [{"name": p.name, "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")} for p in runs_sorted]
    payload = _read_metadata(meta_path)
    payload["runs"] = meta_runs
    payload["latest"] = meta_runs[0]["name"] if meta_runs else None
    _write_metadata(meta_path, payload)
    return payload


def _find_latest_run_dir(task_root: Path) -> Path | None:
    meta_path = task_root / "metadata.json"
    meta = _read_metadata(meta_path)
    latest_name = meta.get("latest")
    if isinstance(latest_name, str):
        latest_path = task_root / latest_name
        if latest_path.exists() and latest_path.is_dir():
            return latest_path
    candidates = [p for p in task_root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_run_output_dir(path_segments: list[str], resume: bool = False) -> Path:
    """按 path_segments/<datetime> 构建并创建输出目录。"""
    task_root = ensure_dir(Path(*path_segments))
    if resume:
        latest = _find_latest_run_dir(task_root)
        if latest is not None:
            refresh_run_metadata(task_root)
            return latest
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = ensure_dir(task_root / timestamp)
    refresh_run_metadata(task_root)
    return run_dir

