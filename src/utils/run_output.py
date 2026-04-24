"""统一管理任务输出目录、metadata 索引与运行状态。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from src.utils.io import ensure_dir

RUN_STATUS_FILE = "run_status.json"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"


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
    # 确保父目录存在（并发场景下可能已被删除）
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 使用 replace 原子性替换
    try:
        tmp_path.replace(meta_path)
    except OSError:
        # 极少数情况下 replace 可能失败，尝试直接写入
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    run_dir = task_root / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = task_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    run_dir = ensure_dir(run_dir)
    refresh_run_metadata(task_root)
    return run_dir


def read_run_status(run_dir: str | Path) -> dict:
    """读取 run 状态，缺省返回 unknown。"""
    path = Path(run_dir) / RUN_STATUS_FILE
    if not path.exists():
        return {"status": "unknown"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "unknown"}
    if not isinstance(payload, dict):
        return {"status": "unknown"}
    status = payload.get("status")
    if not isinstance(status, str):
        payload["status"] = "unknown"
    return payload


def write_run_status(run_dir: str | Path, status: str, **extra: object) -> Path:
    """写入 run 状态文件。"""
    run_path = ensure_dir(run_dir)
    payload = {"status": status, "updated_at": _now_iso(), **extra}
    status_path = run_path / RUN_STATUS_FILE
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        tmp_path.replace(status_path)
    except OSError:
        status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return status_path


def is_run_completed(run_dir: str | Path) -> bool:
    return str(read_run_status(run_dir).get("status", "")).lower() == RUN_STATUS_COMPLETED


def resolve_training_run_dir(path_segments: list[str], force_new: bool = False) -> tuple[Path, bool]:
    """
    解析训练输出目录。

    返回值: (run_dir, resumed)
    resumed=True 表示续用未完成 latest run；False 表示新建 run。
    """
    task_root = ensure_dir(Path(*path_segments))
    if not force_new:
        latest = _find_latest_run_dir(task_root)
        if latest is not None and not is_run_completed(latest):
            refresh_run_metadata(task_root)
            return latest, True
    return build_run_output_dir(path_segments=path_segments, resume=False), False

