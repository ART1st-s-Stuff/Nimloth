"""Progress 服务：单服务查看数据集与训练进度。"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import threading
from typing import Any

import gradio as gr
import plotly.graph_objects as go

# 侧边栏固定为纵向按钮栈，避免 Radio 在部分主题下呈「胶囊/浮动」观感。
_PROGRESS_APP_CSS = """
.app-shell { align-items: stretch !important; flex-wrap: nowrap !important; }
.app-sidebar {
  flex: 0 0 14.5rem !important;
  max-width: 15.5rem !important;
  min-width: 13rem !important;
  border-right: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary);
  padding: 0.75rem 0.6rem 1rem 0.6rem;
  border-radius: 0 !important;
  gap: 0.5rem !important;
}
.app-sidebar .markdown, .app-sidebar .prose {
  font-size: 0.85rem;
  margin: 0 0 0.4rem 0.1rem !important;
  color: var(--body-text-color-subdued);
}
.nav-stack { display: flex; flex-direction: column; gap: 0.45rem; width: 100%; }
.nav-stack button { width: 100% !important; justify-content: flex-start !important; text-align: left !important; }
.main-content {
  flex: 1 1 0 !important;
  min-width: 0 !important;
  padding: 0.35rem 0.75rem 1.25rem 1rem;
}
.main-content > .gr-group { gap: 0.75rem; }
"""
_CACHE_ROOT = Path(".cache") / "visualize"
_FIGURE_CACHE_DIR = _CACHE_ROOT / "figures"
_GRADIO_TEMP_DIR = Path(".cache") / "gradio"
_OFFLINE_WANDB_HISTORY_CACHE: dict[str, tuple[float, dict[str, list[tuple[float, float]]]]] = {}
_ZT_ST_COT_TOOLS: dict[str, Any] | None = None
_WM_TRAJ_TOOLS: dict[str, Any] | None = None
_DEFAULT_TASK_TEXTS = [
    "在当前房间里找到出口并接近门口。",
    "从房间移动到走廊并继续前进到目标点。",
    "沿走廊移动并最终接近电梯区域。",
]

# WM training subprocess management
_WM_TRAIN_PROCESS: dict[str, Any] = {}  # wm_name -> process info
_WM_TRAIN_LOGS: dict[str, list[str]] = {}  # wm_name -> log lines


def _list_wm_configs() -> list[str]:
    """列出所有 WM 配置。"""
    wm_dir = Path("configs/wm")
    if not wm_dir.exists():
        return []
    return [p.stem for p in wm_dir.glob("*.yaml")]


def _start_wm_training(
    wm_name: str,
    split: str,
    epochs: float,
    batch_size: float,
    overrides_json: str,
) -> tuple[str, str, list[list[str]]]:
    """启动 WM 训练子进程。"""
    global _WM_TRAIN_PROCESS, _WM_TRAIN_LOGS

    if not wm_name:
        return "未选择 WM 配置", "", _get_active_trainings_table()

    if wm_name in _WM_TRAIN_PROCESS:
        proc_info = _WM_TRAIN_PROCESS[wm_name]
        if proc_info["process"].poll() is None:
            return f"{wm_name} 已在训练中 (PID={proc_info['pid']})", "", _get_active_trainings_table()

    # Build command
    import json
    import subprocess
    import threading

    cmd = ["uv", "run", "python", "-m", "src.train.train_wm", f"wm={wm_name}"]
    cmd += [f"dataset.manifests.train=datasets/ai2thor/{split}"]
    cmd += [f"train.epochs={int(epochs)}"]
    cmd += [f"train.batch_size={int(batch_size)}"]

    # Parse overrides
    if overrides_json.strip():
        try:
            overrides = json.loads(overrides_json)
            for k, v in overrides.items():
                cmd.append(f"{k}={v}")
        except json.JSONDecodeError as e:
            return f"JSON 解析错误: {e}", "", _get_active_trainings_table()

    # Start subprocess
    log_file = Path(f".cache/visualize/train_{wm_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
        )

    _WM_TRAIN_PROCESS[wm_name] = {
        "process": proc,
        "pid": proc.pid,
        "start_time": datetime.now().isoformat(),
        "log_file": str(log_file),
        "cmd": " ".join(cmd),
    }
    _WM_TRAIN_LOGS[wm_name] = []

    # Start log reader thread
    def _read_log():
        with open(log_file, "r") as f:
            f.seek(0, 2)  # seek to end
            while proc.poll() is None:
                line = f.readline()
                if line:
                    _WM_TRAIN_LOGS[wm_name].append(line.rstrip())
                    if len(_WM_TRAIN_LOGS[wm_name]) > 500:
                        _WM_TRAIN_LOGS[wm_name].pop(0)
                import time
                time.sleep(0.5)

    thread = threading.Thread(target=_read_log, daemon=True)
    thread.start()

    return f"已启动 {wm_name} (PID={proc.pid})\n命令: {' '.join(cmd)}", "", _get_active_trainings_table()


def _stop_wm_training(wm_name: str) -> tuple[str, str, list[list[str]]]:
    """停止 WM 训练子进程。"""
    global _WM_TRAIN_PROCESS, _WM_TRAIN_LOGS

    if wm_name not in _WM_TRAIN_PROCESS:
        return f"{wm_name} 未在训练", "", _get_active_trainings_table()

    proc_info = _WM_TRAIN_PROCESS[wm_name]
    proc = proc_info["process"]

    if proc.is_alive():
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    del _WM_TRAIN_PROCESS[wm_name]
    if wm_name in _WM_TRAIN_LOGS:
        del _WM_TRAIN_LOGS[wm_name]

    return f"已停止 {wm_name}", "", _get_active_trainings_table()


def _get_active_trainings_table() -> list[list[str]]:
    """获取当前训练进程表格。"""
    rows = []
    for wm_name, info in _WM_TRAIN_PROCESS.items():
        proc = info["process"]
        status = "running" if proc.poll() is None else "finished"
        rows.append([
            wm_name,
            info.get("cmd", "").split("wm=")[-1].split()[0] if "wm=" in info.get("cmd", "") else wm_name,
            info["start_time"],
            status,
            str(info["pid"]),
        ])
    return rows


def _configure_gradio_temp_dir() -> None:
    """将 Gradio 临时目录固定到项目内可写路径。"""
    gradio_temp = _GRADIO_TEMP_DIR.resolve()
    gradio_temp.mkdir(parents=True, exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = str(gradio_temp)


def _get_zt_st_cot_tools() -> dict[str, Any]:
    """懒加载 dev.webui.test_zt_st_cot，缩短服务冷启动时间。"""
    global _ZT_ST_COT_TOOLS
    if _ZT_ST_COT_TOOLS is not None:
        return _ZT_ST_COT_TOOLS
    module = importlib.import_module("dev.webui.test_zt_st_cot")
    _ZT_ST_COT_TOOLS = {
        "build_visual_payload_from_result_json_text": module.build_visual_payload_from_result_json_text,
        "list_dev_history": module.list_dev_history,
        "list_rollout_runs": module.list_rollout_runs,
        "list_task_texts": module.list_task_texts,
        "load_dev_history": module.load_dev_history,
        "run_zt_st_cot_test": module.run_zt_st_cot_test,
    }
    return _ZT_ST_COT_TOOLS


def _get_wm_traj_tools() -> dict[str, Any]:
    """懒加载 dev.webui.test_wm_traj_compare，缩短服务冷启动时间。"""
    global _WM_TRAJ_TOOLS
    if _WM_TRAJ_TOOLS is not None:
        return _WM_TRAJ_TOOLS
    module = importlib.import_module("dev.webui.test_wm_traj_compare")
    _WM_TRAJ_TOOLS = {
        "list_wm_test_runs": module.list_wm_test_runs,
        "list_wm_traj_history": module.list_wm_traj_history,
        "load_wm_traj_history": module.load_wm_traj_history,
        "run_wm_traj_compare_test": module.run_wm_traj_compare_test,
    }
    return _WM_TRAJ_TOOLS


def _build_env_context(metadata: dict[str, Any]) -> str:
    scene = metadata.get("scene")
    distance = metadata.get("target_distance")
    collided = metadata.get("collided")
    grasped = metadata.get("grasped")
    parts: list[str] = []
    if scene is not None:
        parts.append(f"scene={scene}")
    if isinstance(distance, (int, float)):
        parts.append(f"target_distance={float(distance):.3f}m")
    if isinstance(collided, bool):
        parts.append(f"collided={'yes' if collided else 'no'}")
    if isinstance(grasped, bool):
        parts.append(f"grasped={'yes' if grasped else 'no'}")
    if not parts:
        return "env=unknown"
    return " | ".join(parts)


def _list_data_runs(base: Path) -> list[Path]:
    """列出 base 下的 run 目录（包含 manifest_worker_*.jsonl 文件的目录）。"""
    import re
    if not base.exists():
        return []
    runs = []
    for path in base.iterdir():
        if path.is_dir():
            # 检查是否包含 manifest_worker_*.jsonl 文件
            has_worker_manifest = any(
                re.match(r"^manifest_worker_\d+_.+\.jsonl$", p.name)
                for p in path.iterdir() if p.is_file()
            )
            if has_worker_manifest:
                runs.append(path)
    return sorted(runs, reverse=True)


def _list_runs_with_file(base: Path, required_file: str) -> list[Path]:
    """列出 base 下包含指定文件的 run 目录。"""
    if not base.exists():
        return []
    runs = [
        path
        for path in base.iterdir()
        if path.is_dir() and (path / required_file).exists()
    ]
    return sorted(runs, reverse=True)


def _load_manifest_samples(run_dir: Path, limit: int = 5000) -> list[dict[str, Any]]:
    """从 run 目录读取 manifest_worker_*.jsonl 文件。"""
    import re
    samples: list[dict[str, Any]] = []
    worker_files = [
        p for p in run_dir.iterdir()
        if p.is_file() and p.suffix == ".jsonl" and re.match(r"^manifest_worker_\d+_.+\.jsonl$", p.name)
    ]
    if not worker_files:
        return []
    for wf in sorted(worker_files):
        with wf.open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                if len(samples) >= limit:
                    return samples
                text = line.strip()
                if not text:
                    continue
                samples.append(json.loads(text))
    return samples


def _build_stats(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "未读取到样本。"
    scene_counter: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    episode_ids: list[int] = []
    for sample in samples:
        metadata = sample.get("metadata", {})
        scene_counter.update([str(metadata.get("scene", "unknown"))])
        action_counter.update([str(sample.get("action", "unknown"))])
        episode_ids.append(int(sample.get("episode_id", -1)))
    top_scene = "\n".join(f"- {scene}: {count}" for scene, count in scene_counter.most_common(10))
    top_action = "\n".join(f"- {action}: {count}" for action, count in action_counter.most_common())
    return (
        f"样本数（读取上限内）：{len(samples)}\n"
        f"episode 范围：{min(episode_ids)} ~ {max(episode_ids)}\n\n"
        f"scene 分布（Top10）:\n{top_scene}\n\n"
        f"动作分布:\n{top_action}"
    )


def _build_rows(samples: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for sample in samples[:200]:
        metadata = sample.get("metadata", {})
        rows.append(
            [
                str(metadata.get("scene", "unknown")),
                str(sample.get("episode_id", "")),
                str(sample.get("step_id", "")),
                str(sample.get("action", "")),
                str(sample.get("image_path", "")),
                _build_env_context(metadata),
            ]
        )
    return rows


def _build_gallery_items_from_rows(rows: list[list[str]]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for row in rows[:80]:
        if len(row) < 5:
            continue
        image_path = row[4]
        if not image_path or not Path(image_path).exists():
            continue
        caption = f"scene={row[0]}, ep={row[1]}, step={row[2]}"
        items.append((image_path, caption))
    return items


def _select_gallery_image(gallery_items: list[tuple[str, str]], evt: gr.SelectData) -> str | None:
    if not gallery_items:
        return None
    if evt.index is None or evt.index < 0 or evt.index >= len(gallery_items):
        return None
    image_path, _ = gallery_items[evt.index]
    return image_path if Path(image_path).exists() else None


def _ensure_cache_dirs() -> None:
    _FIGURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _figure_cache_path(payload: dict[str, Any], title: str) -> Path:
    _ensure_cache_dirs()
    serialized = json.dumps({"title": title, "payload": payload}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]
    return _FIGURE_CACHE_DIR / f"{digest}.json"


def _load_cached_figure(payload: dict[str, Any], title: str) -> go.Figure | None:
    cache_path = _figure_cache_path(payload=payload, title=title)
    if not cache_path.exists():
        return None
    fig_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    figure_json = fig_payload.get("figure")
    if not isinstance(figure_json, dict):
        return None
    return go.Figure(figure_json)


def _save_cached_figure(payload: dict[str, Any], title: str, fig: go.Figure) -> None:
    cache_path = _figure_cache_path(payload=payload, title=title)
    data = {
        "title": title,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "figure": json.loads(fig.to_json()),
    }
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _build_umap_figure(payload: dict[str, Any], title: str) -> go.Figure:
    cached = _load_cached_figure(payload=payload, title=title)
    if cached is not None:
        return cached
    points = payload.get("points", []) if isinstance(payload, dict) else []
    if len(points) > 300:
        stride = max(1, len(points) // 300)
        points = points[::stride][:300]
    fig = go.Figure()
    if not points:
        fig.update_layout(
            title=f"{title}（暂无数据）",
            margin={"l": 0, "r": 0, "b": 0, "t": 40},
        )
        _save_cached_figure(payload=payload, title=title, fig=fig)
        return fig
    xs = [float(point["x"]) for point in points]
    ys = [float(point["y"]) for point in points]
    zs = [float(point["z"]) for point in points]
    labels = [
        f"ep={point.get('episode_id', 'N/A')}, step={point.get('step_id', 'N/A')}"
        for point in points
    ]
    colors = [float(point.get("step_id", idx)) for idx, point in enumerate(points)]
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers",
            marker={"size": 6, "color": colors, "colorscale": "Viridis", "opacity": 0.9},
            text=labels,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        scene={"xaxis_title": "UMAP-1", "yaxis_title": "UMAP-2", "zaxis_title": "UMAP-3"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    _save_cached_figure(payload=payload, title=title, fig=fig)
    return fig


def _build_empty_figure(title: str, note: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=f"{title}（{note}）",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _extract_rollout_ids(payload: dict[str, Any]) -> list[str]:
    points = payload.get("points", []) if isinstance(payload, dict) else []
    rollout_ids = sorted({str(point.get("rollout_id", "")) for point in points if point.get("rollout_id")})
    return rollout_ids


def _color_for_rollout(idx: int) -> str:
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    return palette[idx % len(palette)]


def _build_wm_traj_compare_figure(
    payload: dict[str, Any],
    selected_rollouts: list[str] | None,
    connect_rollout_adjacent: bool,
    connect_gt_pred: bool,
) -> go.Figure:
    points = payload.get("points", []) if isinstance(payload, dict) else []
    if not points:
        return _build_empty_figure("WM 轨迹对比", "暂无数据")
    all_rollouts = _extract_rollout_ids(payload)
    active_rollouts = set(selected_rollouts) if selected_rollouts else set(all_rollouts)
    if not active_rollouts:
        return _build_empty_figure("WM 轨迹对比", "未选择 rollout")

    filtered = [point for point in points if str(point.get("rollout_id", "")) in active_rollouts]
    if not filtered:
        return _build_empty_figure("WM 轨迹对比", "当前筛选无数据")

    fig = go.Figure()
    color_map = {rollout_id: _color_for_rollout(idx) for idx, rollout_id in enumerate(sorted(active_rollouts))}
    marker_symbol = {"real": "circle", "pred": "diamond"}
    source_label = {"real": "Encoder(gt)", "pred": "WM(pred)"}

    grouped_by_rollout_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    grouped_by_rollout_state: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for point in filtered:
        rollout_id = str(point.get("rollout_id", "unknown"))
        source = str(point.get("source", "real"))
        state_index = int(point.get("state_index", -1))
        grouped_by_rollout_source.setdefault((rollout_id, source), []).append(point)
        grouped_by_rollout_state.setdefault((rollout_id, state_index), []).append(point)

    for (rollout_id, source), group_points in sorted(grouped_by_rollout_source.items()):
        ordered = sorted(group_points, key=lambda x: int(x.get("state_index", -1)))
        xs = [float(p["x"]) for p in ordered]
        ys = [float(p["y"]) for p in ordered]
        zs = [float(p["z"]) for p in ordered]
        labels = [
            f"rollout={rollout_id}<br>source={source_label.get(source, source)}<br>step={p.get('step_id', 'N/A')}"
            for p in ordered
        ]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker={
                    "size": 6,
                    "symbol": marker_symbol.get(source, "circle"),
                    "color": color_map.get(rollout_id, "#1f77b4"),
                    "opacity": 0.9,
                },
                name=f"{rollout_id} / {source_label.get(source, source)}",
                text=labels,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        if connect_rollout_adjacent and len(ordered) >= 2:
            fig.add_trace(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line={"color": color_map.get(rollout_id, "#1f77b4"), "width": 3},
                    name=f"{rollout_id} / 相邻状态连线 / {source_label.get(source, source)}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if connect_gt_pred:
        for (rollout_id, state_index), group_points in sorted(grouped_by_rollout_state.items()):
            by_source = {str(point.get("source", "")): point for point in group_points}
            real_point = by_source.get("real")
            pred_point = by_source.get("pred")
            # 第一个状态通常只有 ground truth，没有预测点，按需求跳过。
            if real_point is None or pred_point is None:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=[float(real_point["x"]), float(pred_point["x"])],
                    y=[float(real_point["y"]), float(pred_point["y"])],
                    z=[float(real_point["z"]), float(pred_point["z"])],
                    mode="lines",
                    line={"color": color_map.get(rollout_id, "#1f77b4"), "width": 2, "dash": "dash"},
                    name=f"{rollout_id} / gt-pred 连线",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title="WM 轨迹对比（同一映射空间）",
        scene={"xaxis_title": "UMAP-1", "yaxis_title": "UMAP-2", "zaxis_title": "UMAP-3"},
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
        legend={"orientation": "h"},
    )
    return fig


def _collect_phase2_runs(models_root: str = "models") -> list[Path]:
    root = Path(models_root) / "wm"
    if not root.exists():
        return []
    run_dirs: list[Path] = []
    for wm_dir in root.iterdir():
        if not wm_dir.is_dir():
            continue
        for run_dir in wm_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "phase2_eval_metrics.json").exists():
                run_dirs.append(run_dir)
    return sorted(run_dirs, reverse=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_divergence_figure(normal_scores: list[float], abnormal_scores: list[float]) -> go.Figure:
    if not normal_scores and not abnormal_scores:
        return _build_empty_figure("散度分布", "暂无数据")
    fig = go.Figure()
    if normal_scores:
        fig.add_trace(
            go.Histogram(
                x=normal_scores,
                name="normal",
                opacity=0.65,
                nbinsx=35,
            )
        )
    if abnormal_scores:
        fig.add_trace(
            go.Histogram(
                x=abnormal_scores,
                name="abnormal",
                opacity=0.65,
                nbinsx=35,
            )
        )
    fig.update_layout(
        barmode="overlay",
        title="散度分布对比",
        xaxis_title="divergence",
        yaxis_title="count",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _build_drift_figure(steps: list[int], fd: list[float], cd: list[float]) -> go.Figure:
    if not steps:
        return _build_empty_figure("长程 Drift 曲线", "暂无数据")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=fd, mode="lines+markers", name="FD"))
    fig.add_trace(go.Scatter(x=steps, y=cd, mode="lines+markers", name="CD"))
    fig.update_layout(
        title="长程 Drift 曲线",
        xaxis_title="rollout_step",
        yaxis_title="distance",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _phase2_summary(payload: dict[str, Any], run_dir: Path) -> str:
    kpis = payload.get("kpis", {})
    threshold = payload.get("threshold_coverage", {})
    meta = payload.get("meta", {})
    return (
        f"run: {run_dir}\n"
        f"wm: {meta.get('wm_name', 'N/A')}\n"
        f"samples: {meta.get('num_samples', 'N/A')}\n"
        f"wm_mse: {kpis.get('wm_mse', 'N/A')}\n"
        f"latent_fd_mean: {kpis.get('latent_fd_mean', 'N/A')}\n"
        f"latent_cd_mean: {kpis.get('latent_cd_mean', 'N/A')}\n"
        f"divergence_auroc: {kpis.get('divergence_auroc', 'N/A')}\n"
        f"idm_action_mse: {kpis.get('idm_action_mse', 'N/A')}\n"
        f"theta_div: {threshold.get('theta_div', 'N/A')}\n"
        f"normal_trigger_rate: {threshold.get('normal_trigger_rate', 'N/A')}\n"
        f"abnormal_trigger_rate: {threshold.get('abnormal_trigger_rate', 'N/A')}"
    )


def _phase2_compare_rows(run_dirs: list[Path]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run_dir in run_dirs[:300]:
        payload = _read_json(run_dir / "phase2_eval_metrics.json")
        kpis = payload.get("kpis", {})
        threshold = payload.get("threshold_coverage", {})
        rows.append(
            [
                str(run_dir),
                str(kpis.get("wm_mse", "N/A")),
                str(kpis.get("divergence_auroc", "N/A")),
                str(kpis.get("latent_fd_mean", "N/A")),
                str(kpis.get("latent_cd_mean", "N/A")),
                str(kpis.get("idm_action_mse", "N/A")),
                str(threshold.get("theta_div", "N/A")),
                str(threshold.get("abnormal_trigger_rate", "N/A")),
            ]
        )
    return rows


def _load_phase2_dashboard(selected_run: str, models_root: str = "models") -> tuple[str, list[list[str]], go.Figure, go.Figure, str]:
    run_dirs = _collect_phase2_runs(models_root=models_root)
    compare_rows = _phase2_compare_rows(run_dirs)
    if not run_dirs:
        empty = _build_empty_figure("Phase2 图表", "未找到 phase2_eval_metrics.json")
        return "未找到评估结果。请先执行 scripts/phase2/wm_evaluate.sh。", [], empty, empty, ""
    target = Path(selected_run) if selected_run else run_dirs[0]
    payload = _read_json(target / "phase2_eval_metrics.json")
    if not payload:
        empty = _build_empty_figure("Phase2 图表", "评估文件为空")
        return f"评估文件不存在或为空: {target}", compare_rows, empty, empty, ""
    dist = payload.get("divergence_distribution", {})
    drift = payload.get("drift_curve", {})
    report_path = target / "phase2_eval_report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    return (
        _phase2_summary(payload=payload, run_dir=target),
        compare_rows,
        _build_divergence_figure(
            normal_scores=[float(x) for x in dist.get("normal_scores", [])],
            abnormal_scores=[float(x) for x in dist.get("abnormal_scores", [])],
        ),
        _build_drift_figure(
            steps=[int(x) for x in drift.get("steps", [])],
            fd=[float(x) for x in drift.get("fd", [])],
            cd=[float(x) for x in drift.get("cd", [])],
        ),
        report_text,
    )


def _select_dataset_run(run_name: str) -> tuple[str, list[list[str]], list[tuple[str, str]], str]:
    if not run_name:
        return "未选择运行目录。", [], [], ""
    run_dir = Path(run_name)
    samples = _load_manifest_samples(run_dir=run_dir)
    stats = _build_stats(samples)
    rows = _build_rows(samples)
    gallery_items = _build_gallery_items_from_rows(rows)
    gallery_notice = "未找到可展示的图片路径。" if not gallery_items else f"已加载 {len(gallery_items)} 张缩略图。"
    return stats, rows, gallery_items, gallery_notice


def _load_dev_history_light(run_dir: str) -> tuple[str, str, list[tuple[str, str]], str]:
    zt_tools = _get_zt_st_cot_tools()
    summary, result_json_text = zt_tools["load_dev_history"](run_dir)
    if not run_dir or not result_json_text:
        return summary, result_json_text, [], ""
    gallery_items, _, _, _, notice = zt_tools["build_visual_payload_from_result_json_text"](result_json_text)
    history_json_preview = result_json_text
    if len(history_json_preview) > 25000:
        history_json_preview = history_json_preview[:25000] + "\n...（已截断，完整内容请查看结果目录中的 result.json）"
    return summary, history_json_preview, gallery_items, notice


def _load_dev_history_umap_plots(result_json_text: str) -> tuple[go.Figure, go.Figure, go.Figure, str]:
    if not result_json_text:
        empty_fig = _build_umap_figure({}, "UMAP 3D")
        return empty_fig, empty_fig, empty_fig, "未选择历史目录。"
    zt_tools = _get_zt_st_cot_tools()
    _, dino_payload, qwen_payload, st_payload, notice = zt_tools["build_visual_payload_from_result_json_text"](
        result_json_text
    )
    return (
        _build_umap_figure(dino_payload, "z_t_dino UMAP 3D"),
        _build_umap_figure(qwen_payload, "z_t_qwen UMAP 3D"),
        _build_umap_figure(st_payload, "s_t UMAP 3D"),
        notice,
    )


def _switch_section(selected: str) -> tuple[gr.update, gr.update, gr.update, gr.update, gr.update, gr.update]:
    return (
        gr.update(visible=selected == "数据集进度"),
        gr.update(visible=selected == "训练进度"),
        gr.update(visible=selected == "Phase2评估"),
        gr.update(visible=selected == "Dev/Test z_t-s_t-CoT"),
        gr.update(visible=selected == "WM轨迹对比"),
        gr.update(visible=selected == "WM训练"),
    )


def _nav_labels() -> tuple[str, str, str, str, str, str]:
    return ("数据集进度", "训练进度", "Phase2评估", "Dev/Test z_t-s_t-CoT", "WM轨迹对比", "WM训练")


def _apply_nav(
    selected: str,
) -> tuple[
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
    gr.update,
]:
    """侧边栏：切换主面板可见性，并同步导航按钮的主次样式。"""
    d, t, m, dev, wm_traj, wm_train = _switch_section(selected)
    labels = _nav_labels()
    btn_updates = tuple(
        gr.update(variant="primary" if selected == lab else "secondary") for lab in labels
    )
    return (d, t, m, dev, wm_traj, wm_train) + btn_updates


def _list_training_runs(models_root: str = "models") -> list[Path]:
    wm_root = Path(models_root) / "wm"
    if not wm_root.exists():
        return []
    run_dirs: list[Path] = []
    for wm_dir in wm_root.iterdir():
        if not wm_dir.is_dir():
            continue
        run_dirs.extend(_list_runs_with_file(wm_dir, required_file="train_metrics.json"))
    return sorted(run_dirs, reverse=True)


def _read_train_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _build_training_rows(run_dirs: list[Path]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run_dir in run_dirs[:300]:
        metrics = _read_train_metrics(run_dir / "train_metrics.json")
        last_loss = metrics.get("last_loss", "N/A")
        has_ckpt = (run_dir / "wm.pt").exists()
        status = "completed" if has_ckpt else "incomplete"
        rows.append(
            [
                str(run_dir),
                str(last_loss),
                "yes" if has_ckpt else "no",
                status,
            ]
        )
    return rows


def _training_summary(rows: list[list[str]]) -> str:
    if not rows:
        return "未找到训练运行目录（models/wm/<wm-config-name>/<datetime>）。"
    completed = sum(1 for row in rows if row[3] == "completed")
    incomplete = len(rows) - completed
    return (
        f"训练运行总数: {len(rows)}\n"
        f"已完成（存在 wm.pt）: {completed}\n"
        f"未完成/异常: {incomplete}"
    )


def _load_training_progress(models_root: str = "models") -> tuple[str, list[list[str]]]:
    run_dirs = _list_training_runs(models_root=models_root)
    rows = _build_training_rows(run_dirs)
    return _training_summary(rows), rows


def _load_training_progress_page(page: float, page_size: float = 100, models_root: str = "models") -> tuple[str, list[list[str]]]:
    run_dirs = _list_training_runs(models_root=models_root)
    rows = _build_training_rows(run_dirs)
    page_idx = max(1, int(page))
    per_page = max(20, int(page_size))
    start = (page_idx - 1) * per_page
    end = start + per_page
    page_rows = rows[start:end]
    total_pages = max(1, (len(rows) + per_page - 1) // per_page)
    summary = _training_summary(rows) + f"\n当前页: {page_idx}/{total_pages}（每页 {per_page} 条）"
    return summary, page_rows


def _list_offline_wandb_runs(wandb_root: str = "wandb") -> list[Path]:
    """扫描本地离线 W&B 运行目录。"""
    offline_root = Path(wandb_root) / "wandb"
    if not offline_root.exists():
        return []
    run_dirs = [
        path
        for path in offline_root.iterdir()
        if path.is_dir() and path.name.startswith("offline-run-") and any(path.glob("run-*.wandb"))
    ]
    return sorted(run_dirs, reverse=True)


def _extract_wandb_offline_history(run_dir: Path) -> dict[str, list[tuple[float, float]]]:
    """从 run-*.wandb 里提取 history 曲线点，返回 {metric: [(step, value), ...]}。"""
    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return {}
    run_file = wandb_files[0]
    cache_key = str(run_file.resolve())
    mtime = float(run_file.stat().st_mtime)
    cached = _OFFLINE_WANDB_HISTORY_CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        from wandb.proto import wandb_internal_pb2  # type: ignore
        from wandb.sdk.internal import datastore  # type: ignore
    except Exception:
        return {}

    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    data_store = datastore.DataStore()
    data_store.open_for_scan(str(wandb_files[0]))
    while True:
        data = data_store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if not record.history.item:
            continue
        # W&B 离线记录中 step 可能位于 history.step 或 _step。
        record_step = float(record.history.step.num) if record.history.step else None
        for item in record.history.item:
            metric_key = item.key if item.key else ".".join(item.nested_key)
            if not metric_key:
                continue
            try:
                value = float(item.value_json)
            except Exception:
                continue
            if metric_key == "_step":
                continue
            if metric_key.startswith("_"):
                continue
            step = record_step
            if step is None:
                # 回退：若本条记录没有显式 step，则用当前序列长度近似。
                step = float(len(series[metric_key]))
            series[metric_key].append((step, value))
    for metric, points in series.items():
        points.sort(key=lambda x: x[0])
    parsed = dict(series)
    _OFFLINE_WANDB_HISTORY_CACHE[cache_key] = (mtime, parsed)
    return parsed


def _build_offline_wandb_curve_figure(
    run_dir: str,
    selected_metrics: list[str] | None,
) -> go.Figure:
    if not run_dir:
        return _build_empty_figure("离线 W&B Loss 曲线", "未选择 run")
    series = _extract_wandb_offline_history(Path(run_dir))
    if not series:
        return _build_empty_figure("离线 W&B Loss 曲线", "未读取到离线 history")
    metric_names = selected_metrics or sorted(series.keys())
    fig = go.Figure()
    for metric in metric_names:
        points = series.get(metric, [])
        if not points:
            continue
        sampled_points = _downsample_points(points)
        xs = [p[0] for p in sampled_points]
        ys = [p[1] for p in sampled_points]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=metric))
    if not fig.data:
        return _build_empty_figure("离线 W&B Loss 曲线", "当前指标无有效点")
    fig.update_layout(
        title="离线 W&B 指标曲线",
        xaxis_title="step/epoch",
        yaxis_title="value",
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def _offline_wandb_metric_choices(run_dir: str) -> gr.update:
    if not run_dir:
        return gr.update(choices=[], value=[])
    series = _extract_wandb_offline_history(Path(run_dir))
    metric_candidates = [name for name in sorted(series.keys()) if "loss" in name or name.startswith("train/")]
    default_values = [name for name in metric_candidates if "loss" in name][:3]
    return gr.update(choices=metric_candidates, value=default_values or metric_candidates[:3])


def _refresh_offline_run_choices() -> gr.update:
    runs = [str(path) for path in _list_offline_wandb_runs()]
    return gr.update(choices=runs, value=(runs[0] if runs else None))


def _downsample_points(points: list[tuple[float, float]], max_points: int = 1500) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    stride = max(1, len(points) // max_points)
    return points[::stride][:max_points]


def _run_zt_test_for_ui(
    run_dir: str,
    task_text: str,
    max_steps: float,
) -> tuple[str, list[list[Any]], str, go.Figure, go.Figure, go.Figure, list[tuple[str, str]], str]:
    zt_tools = _get_zt_st_cot_tools()
    (
        summary,
        table,
        output_dir,
        dino_payload,
        qwen_payload,
        st_payload,
        gallery_items,
        notice,
    ) = zt_tools["run_zt_st_cot_test"](run_dir=run_dir, task_text=task_text, max_steps=int(max_steps))
    return (
        summary,
        table,
        output_dir,
        _build_umap_figure(dino_payload, "z_t_dino UMAP 3D"),
        _build_umap_figure(qwen_payload, "z_t_qwen UMAP 3D"),
        _build_umap_figure(st_payload, "s_t UMAP 3D"),
        gallery_items,
        notice,
    )


def _run_wm_traj_test_for_ui(
    wm_run_dir: str,
    episodes_per_scene: float,
    max_steps_per_episode: float,
    outputs_root: str,
) -> tuple[str, str, go.Figure, str, float, str, dict[str, Any], gr.update]:
    wm_tools = _get_wm_traj_tools()
    (
        summary,
        output_dir,
        real_payload,
        pred_payload,
        overlay_payload,
        notice,
    ) = wm_tools["run_wm_traj_compare_test"](
        wm_run_dir=wm_run_dir,
        episodes_per_scene=int(episodes_per_scene),
        max_steps_per_episode=int(max_steps_per_episode),
        outputs_root=outputs_root,
    )
    return (
        summary,
        output_dir,
        _build_wm_traj_compare_figure(
            payload=overlay_payload,
            selected_rollouts=_extract_rollout_ids(overlay_payload),
            connect_rollout_adjacent=True,
            connect_gt_pred=False,
        ),
        notice,
        100.0,
        "执行完成",
        overlay_payload,
        gr.update(choices=_extract_rollout_ids(overlay_payload), value=_extract_rollout_ids(overlay_payload)),
    )


def _run_wm_traj_test_stream_for_ui(
    wm_run_dir: str,
    split: str,
    scene_filter: str,
    max_rows: float,
    outputs_root: str,
):
    progress_queue: queue.Queue[tuple[float, str]] = queue.Queue()
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, str] = {}

    def _progress_cb(pct: float, status: str) -> None:
        progress_queue.put((float(pct), str(status)))

    # Parse scene filter
    scene_filter_list = [s.strip() for s in scene_filter.split(",") if s.strip()] if scene_filter else None

    # 用 run_wm_traj_compare_test 的进度回调驱动页面进度条。
    def _worker_with_progress() -> None:
        wm_tools = _get_wm_traj_tools()
        try:
            summary, output_dir, real_payload, pred_payload, overlay_payload, notice = wm_tools[
                "run_wm_traj_compare_test"
            ](
                wm_run_dir=wm_run_dir,
                split=split,
                scene_filter=scene_filter_list,
                max_rows=int(max_rows) if max_rows > 0 else 0,
                outputs_root=outputs_root,
                progress_callback=_progress_cb,
            )
            rollout_ids = _extract_rollout_ids(overlay_payload)
            result_holder["value"] = (
                summary,
                output_dir,
                _build_wm_traj_compare_figure(
                    payload=overlay_payload,
                    selected_rollouts=rollout_ids,
                    connect_rollout_adjacent=True,
                    connect_gt_pred=False,
                ),
                notice,
                100.0,
                "执行完成",
                overlay_payload,
                gr.update(choices=rollout_ids, value=rollout_ids),
            )
        except Exception as exc:  # pragma: no cover
            error_holder["value"] = str(exc)

    worker = threading.Thread(target=_worker_with_progress, daemon=True)
    worker.start()

    while worker.is_alive() or not progress_queue.empty():
        try:
            pct, status = progress_queue.get(timeout=0.2)
            yield (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                pct,
                status,
                gr.update(),
                gr.update(),
            )
        except queue.Empty:
            continue

    if "value" in error_holder:
        yield (
            f"执行失败: {error_holder['value']}",
            "",
            _build_empty_figure("WM 轨迹对比", "执行失败"),
            "",
            100.0,
            "执行失败",
            {"feature": "wm_traj_shared", "points": [], "warning": ""},
            gr.update(choices=[], value=[]),
        )
        return

    if "value" in result_holder:
        yield result_holder["value"]


def _build_wm_traj_stream_runner(outputs_root: str):
    def _runner(run_dir: str, split: str, scene_filter: str, max_rows: float):
        yield from _run_wm_traj_test_stream_for_ui(
            wm_run_dir=run_dir,
            split=split,
            scene_filter=scene_filter,
            max_rows=max_rows,
            outputs_root=outputs_root,
        )

    return _runner


def _load_wm_traj_history_for_ui(run_dir: str) -> tuple[str, str, go.Figure, str, dict[str, Any], gr.update]:
    wm_tools = _get_wm_traj_tools()
    summary, preview_json, real_payload, pred_payload, overlay_payload, notice = wm_tools["load_wm_traj_history"](
        run_dir
    )
    rollout_ids = _extract_rollout_ids(overlay_payload)
    return (
        summary,
        preview_json,
        _build_wm_traj_compare_figure(
            payload=overlay_payload,
            selected_rollouts=rollout_ids,
            connect_rollout_adjacent=True,
            connect_gt_pred=False,
        ),
        notice,
        overlay_payload,
        gr.update(choices=rollout_ids, value=rollout_ids),
    )


def _refresh_wm_traj_plot_for_ui(
    payload: dict[str, Any],
    selected_rollouts: list[str],
    connect_rollout_adjacent: bool,
    connect_gt_pred: bool,
) -> go.Figure:
    return _build_wm_traj_compare_figure(
        payload=payload,
        selected_rollouts=selected_rollouts,
        connect_rollout_adjacent=bool(connect_rollout_adjacent),
        connect_gt_pred=bool(connect_gt_pred),
    )


def build_app(dataset_root: str = "datasets", models_root: str = "models", outputs_root: str = "outputs") -> gr.Blocks:
    runs = _list_data_runs(base=Path(dataset_root) / "ai2thor")
    run_choices = [str(path) for path in runs]
    default_run = run_choices[0] if run_choices else None
    offline_wandb_runs = _list_offline_wandb_runs()
    offline_wandb_choices = [str(path) for path in offline_wandb_runs]
    default_offline_wandb_run = offline_wandb_choices[0] if offline_wandb_choices else None
    lab_ds, lab_tr, lab_mi, lab_dev, lab_wm_traj, lab_wm_train = _nav_labels()
    wm_traj_stream_runner = _build_wm_traj_stream_runner(outputs_root=outputs_root)
    initial_task_texts = list(_DEFAULT_TASK_TEXTS)
    with gr.Blocks(title="Flower Progress Server") as app:
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(elem_classes=["app-sidebar"]):
                gr.Markdown("**导航**")
                with gr.Column(elem_classes=["nav-stack"]):
                    nav_btn_dataset = gr.Button(lab_ds, variant="primary", size="lg")
                    nav_btn_train = gr.Button(lab_tr, variant="secondary", size="lg")
                    nav_btn_misc = gr.Button(lab_mi, variant="secondary", size="lg")
                    nav_btn_dev = gr.Button(lab_dev, variant="secondary", size="lg")
                    nav_btn_wm_traj = gr.Button(lab_wm_traj, variant="secondary", size="lg")
                    nav_btn_wm_train = gr.Button(lab_wm_train, variant="secondary", size="lg")

            with gr.Column(elem_classes=["main-content"]):
                gr.Markdown("# Flower Progress Server")
                gr.Markdown("数据集、训练进度、Phase2 评估与 Dev 测试集中在同一页面；左侧为分区入口。")

                with gr.Group(visible=True) as dataset_panel:
                    gr.Markdown("### 数据集进度")
                    with gr.Row():
                        run_selector = gr.Dropdown(
                            choices=run_choices,
                            value=default_run,
                            label="数据收集运行目录",
                            scale=5,
                            allow_custom_value=True,
                        )
                        dataset_refresh_btn = gr.Button("刷新", scale=0, min_width=88)
                    with gr.Accordion("统计信息", open=True):
                        stats_box = gr.Textbox(
                            label="",
                            value="点击右侧“刷新”后加载。",
                            placeholder="选择目录后显示统计…",
                            lines=16,
                            max_lines=24,
                            show_label=False,
                        )
                    with gr.Accordion("样本表（最多 200 条）", open=True):
                        sample_table = gr.Dataframe(
                            headers=["scene", "episode_id", "step_id", "action", "image_path", "env_context"],
                            label="",
                            wrap=True,
                        )
                    with gr.Accordion("图片：缩略图与大图", open=True):
                        dataset_gallery_notice = gr.Textbox(label="", value="尚未加载，点击“刷新”后显示。", lines=1, show_label=False)
                        with gr.Row():
                            with gr.Column(scale=3, min_width=320):
                                dataset_gallery = gr.Gallery(
                                    label="缩略图",
                                    columns=5,
                                    height=300,
                                    object_fit="contain",
                                    preview=True,
                                )
                            with gr.Column(scale=2, min_width=260):
                                dataset_large_image = gr.Image(
                                    label="大图（点击左侧缩略图）",
                                    type="filepath",
                                    height=420,
                                )
                    dataset_refresh_btn.click(
                        fn=_select_dataset_run,
                        inputs=[run_selector],
                        outputs=[stats_box, sample_table, dataset_gallery, dataset_gallery_notice],
                    )
                    dataset_gallery.select(
                        fn=_select_gallery_image,
                        inputs=[dataset_gallery],
                        outputs=[dataset_large_image],
                    )

                with gr.Group(visible=False) as train_panel:
                    gr.Markdown("### 训练进度")
                    with gr.Row():
                        train_page = gr.Number(value=1, precision=0, minimum=1, label="页码", scale=1)
                        train_refresh_btn = gr.Button("刷新训练进度", variant="secondary", scale=0, min_width=120)
                    with gr.Accordion("汇总", open=True):
                        train_summary = gr.Textbox(label="", lines=6, show_label=False)
                    with gr.Accordion("运行列表", open=True):
                        train_table = gr.Dataframe(
                            headers=["run_dir", "last_loss", "has_wm_pt", "status"],
                            label="",
                            wrap=True,
                        )
                    train_refresh_btn.click(
                        fn=lambda page: _load_training_progress_page(page=page, page_size=100, models_root=models_root),
                        inputs=[train_page],
                        outputs=[train_summary, train_table],
                    )
                    with gr.Accordion("离线 W&B 曲线", open=False):
                        offline_run_selector = gr.Dropdown(
                            choices=offline_wandb_choices,
                            value=default_offline_wandb_run,
                            label="离线 W&B run 目录",
                            allow_custom_value=True,
                        )
                        offline_metric_selector = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="指标（建议选择 loss 相关）",
                        )
                        with gr.Row():
                            offline_refresh_runs_btn = gr.Button("刷新 run 列表", variant="secondary")
                            offline_refresh_metrics_btn = gr.Button("加载指标", variant="secondary")
                            offline_refresh_curve_btn = gr.Button("刷新曲线", variant="primary")
                        offline_curve_plot = gr.Plot(label="离线 W&B 曲线")
                    offline_refresh_runs_btn.click(
                        fn=_refresh_offline_run_choices,
                        inputs=[],
                        outputs=[offline_run_selector],
                    )
                    offline_refresh_metrics_btn.click(
                        fn=_offline_wandb_metric_choices,
                        inputs=[offline_run_selector],
                        outputs=[offline_metric_selector],
                    )
                    offline_refresh_curve_btn.click(
                        fn=_build_offline_wandb_curve_figure,
                        inputs=[offline_run_selector, offline_metric_selector],
                        outputs=[offline_curve_plot],
                    )

                with gr.Group(visible=False) as wm_train_panel:
                    gr.Markdown("### WM 训练控制")
                    with gr.Accordion("快速训练", open=True):
                        with gr.Row():
                            wm_config_selector = gr.Dropdown(
                                choices=_list_wm_configs(),
                                value="cfm_dinov2m",
                                label="WM 配置",
                                scale=2,
                            )
                            wm_train_refresh_btn = gr.Button("刷新", scale=0, min_width=80)
                        with gr.Row():
                            train_split_selector = gr.Dropdown(
                                choices=["train", "val"],
                                value="train",
                                label="数据集 split",
                                scale=1,
                            )
                            train_epochs = gr.Number(value=4, label="epochs", precision=0, minimum=1, scale=1)
                            train_batch_size = gr.Number(value=16, label="batch_size", precision=0, minimum=1, scale=1)
                        train_overrides = gr.Textbox(
                            label="参数覆盖（JSON格式，如 {\"train.epochs\": 8}）",
                            value="",
                            placeholder='{"train.epochs": 8, "train.batch_size": 32}',
                            lines=2,
                        )
                        with gr.Row():
                            start_train_btn = gr.Button("开始训练", variant="primary", scale=0, min_width=120)
                            stop_train_btn = gr.Button("停止训练", variant="stop", scale=0, min_width=100)

                    with gr.Accordion("训练状态", open=True):
                        train_status = gr.Textbox(label="", value="未开始训练", lines=8, show_label=False)
                        train_log = gr.Textbox(label="实时日志", value="", lines=12, max_lines=500, show_label=False)

                    with gr.Accordion("已启动的训练", open=False):
                        active_trainings = gr.Dataframe(
                            headers=["wm_name", "config", "start_time", "status", "pid"],
                            label="",
                        )

                    # Training button handlers
                    wm_train_refresh_btn.click(
                        fn=lambda: gr.update(choices=_list_wm_configs()),
                        inputs=[],
                        outputs=[wm_config_selector],
                    )
                    start_train_btn.click(
                        fn=_start_wm_training,
                        inputs=[wm_config_selector, train_split_selector, train_epochs, train_batch_size, train_overrides],
                        outputs=[train_status, train_log, active_trainings],
                    )
                    stop_train_btn.click(
                        fn=_stop_wm_training,
                        inputs=[wm_config_selector],
                        outputs=[train_status, train_log, active_trainings],
                    )

                with gr.Group(visible=False) as misc_panel:
                    gr.Markdown("### Phase2 评估 Dashboard")
                    with gr.Row():
                        phase2_run_selector = gr.Dropdown(
                            choices=[str(path) for path in _collect_phase2_runs(models_root=models_root)],
                            value=None,
                            label="评估运行目录",
                            scale=5,
                            allow_custom_value=True,
                        )
                        phase2_refresh_btn = gr.Button("刷新评估", variant="secondary", scale=0, min_width=120)
                    with gr.Accordion("评估摘要", open=True):
                        phase2_summary_box = gr.Textbox(label="", value="点击“刷新评估”后加载。", lines=12, show_label=False)
                    with gr.Accordion("多运行对比", open=True):
                        phase2_compare_table = gr.Dataframe(
                            headers=[
                                "run_dir",
                                "wm_mse",
                                "divergence_auroc",
                                "latent_fd_mean",
                                "latent_cd_mean",
                                "idm_action_mse",
                                "theta_div",
                                "abnormal_trigger_rate",
                            ],
                            label="",
                            wrap=True,
                        )
                    with gr.Accordion("散度分布对比", open=True):
                        phase2_div_plot = gr.Plot(label="normal/abnormal divergence")
                    with gr.Accordion("长程 Drift 曲线", open=True):
                        phase2_drift_plot = gr.Plot(label="drift curve")
                    with gr.Accordion("Markdown 报告", open=False):
                        phase2_report_box = gr.Textbox(label="", lines=16, max_lines=40, show_label=False)
                    phase2_refresh_btn.click(
                        fn=lambda run: _load_phase2_dashboard(selected_run=run, models_root=models_root),
                        inputs=[phase2_run_selector],
                        outputs=[phase2_summary_box, phase2_compare_table, phase2_div_plot, phase2_drift_plot, phase2_report_box],
                    )
                    phase2_refresh_btn.click(
                        fn=lambda: gr.update(choices=[str(path) for path in _collect_phase2_runs(models_root=models_root)]),
                        inputs=[],
                        outputs=[phase2_run_selector],
                    )

                with gr.Group(visible=False) as dev_panel:
                    gr.Markdown("### Dev / Test：z_t、s_t、CoT")
                    with gr.Accordion("运行参数", open=True):
                        with gr.Row():
                            rollout_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="rollout 目录",
                                scale=4,
                                allow_custom_value=True,
                            )
                            refresh_rollout_btn = gr.Button("刷新列表", scale=0, min_width=100)
                        task_selector = gr.Dropdown(choices=initial_task_texts, value=initial_task_texts[0], label="任务文本")
                        with gr.Row():
                            max_steps = gr.Number(value=10, label="step 上限", precision=0, minimum=1, scale=1)
                            run_btn = gr.Button("执行测试", variant="primary", scale=0, min_width=120)

                    with gr.Accordion("本次运行输出", open=True):
                        test_summary = gr.Textbox(label="摘要", lines=7)
                        test_output_dir = gr.Textbox(label="输出目录", lines=1)
                        test_table = gr.Dataframe(
                            headers=["episode_id", "step_id", "image_path", "z_t_dino_mean", "z_t_qwen_mean", "s_t_mean"],
                            label="数值预览",
                            wrap=True,
                        )

                    with gr.Accordion("图片：本次运行", open=True):
                        test_gallery_notice = gr.Textbox(label="", lines=1, show_label=False)
                        with gr.Row():
                            with gr.Column(scale=3, min_width=300):
                                test_gallery = gr.Gallery(
                                    label="缩略图",
                                    columns=5,
                                    height=280,
                                    object_fit="contain",
                                    preview=True,
                                )
                            with gr.Column(scale=2, min_width=260):
                                test_large_image = gr.Image(
                                    label="大图",
                                    type="filepath",
                                    height=400,
                                )

                    with gr.Accordion("UMAP 三维（按特征分别）", open=True):
                        gr.Markdown("下图可旋转；颜色大致对应 step。")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=200):
                                dino_plot = gr.Plot(label="z_t_dino", show_label=True)
                            with gr.Column(scale=1, min_width=200):
                                qwen_plot = gr.Plot(label="z_t_qwen", show_label=True)
                            with gr.Column(scale=1, min_width=200):
                                st_plot = gr.Plot(label="s_t", show_label=True)

                    gr.Markdown("#### 历史 Dev 输出")
                    with gr.Accordion("历史目录与 JSON", open=True):
                        with gr.Row():
                            history_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="历史运行目录",
                                scale=4,
                                allow_custom_value=True,
                            )
                            refresh_history_btn = gr.Button("刷新", scale=0, min_width=88)
                        history_summary = gr.Textbox(label="摘要", lines=5)
                        history_json = gr.Textbox(label="result.json", lines=14, max_lines=40)

                    with gr.Accordion("图片：历史结果", open=True):
                        history_gallery_notice = gr.Textbox(label="", lines=1, show_label=False)
                        with gr.Row():
                            with gr.Column(scale=3, min_width=300):
                                history_gallery = gr.Gallery(
                                    label="缩略图",
                                    columns=5,
                                    height=280,
                                    object_fit="contain",
                                    preview=True,
                                )
                            with gr.Column(scale=2, min_width=260):
                                history_large_image = gr.Image(
                                    label="大图",
                                    type="filepath",
                                    height=400,
                                )

                    with gr.Accordion("UMAP：历史结果", open=True):
                        load_history_umap_btn = gr.Button("加载历史 UMAP 图", variant="secondary")
                        with gr.Row(equal_height=True):
                            with gr.Column(scale=1, min_width=200):
                                history_dino_plot = gr.Plot(label="z_t_dino", show_label=True)
                            with gr.Column(scale=1, min_width=200):
                                history_qwen_plot = gr.Plot(label="z_t_qwen", show_label=True)
                            with gr.Column(scale=1, min_width=200):
                                history_st_plot = gr.Plot(label="s_t", show_label=True)

                    run_btn.click(
                        fn=_run_zt_test_for_ui,
                        inputs=[rollout_selector, task_selector, max_steps],
                        outputs=[test_summary, test_table, test_output_dir, dino_plot, qwen_plot, st_plot, test_gallery, test_gallery_notice],
                    )
                    refresh_rollout_btn.click(
                        fn=lambda: gr.update(
                            choices=_get_zt_st_cot_tools()["list_rollout_runs"](dataset_root=f"{dataset_root}/ai2thor")
                        ),
                        inputs=[],
                        outputs=[rollout_selector],
                    )
                    test_gallery.select(
                        fn=_select_gallery_image,
                        inputs=[test_gallery],
                        outputs=[test_large_image],
                    )

                    refresh_history_btn.click(
                        fn=lambda: gr.update(choices=_get_zt_st_cot_tools()["list_dev_history"](outputs_root=outputs_root)),
                        inputs=[],
                        outputs=[history_selector],
                    )
                    history_selector.change(
                        fn=_load_dev_history_light,
                        inputs=[history_selector],
                        outputs=[history_summary, history_json, history_gallery, history_gallery_notice],
                    )
                    history_gallery.select(
                        fn=_select_gallery_image,
                        inputs=[history_gallery],
                        outputs=[history_large_image],
                    )
                    load_history_umap_btn.click(
                        fn=_load_dev_history_umap_plots,
                        inputs=[history_json],
                        outputs=[history_dino_plot, history_qwen_plot, history_st_plot, history_gallery_notice],
                    )

                with gr.Group(visible=False) as wm_traj_panel:
                    gr.Markdown("### WM 轨迹对比测试")
                    wm_traj_payload_state = gr.State({"feature": "wm_traj_shared", "points": [], "warning": ""})
                    with gr.Accordion("运行参数", open=True):
                        with gr.Row():
                            wm_run_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="WM 运行目录（包含 wm.pt）",
                                scale=3,
                                allow_custom_value=True,
                            )
                            refresh_wm_run_btn = gr.Button("刷新", scale=0, min_width=80)
                        with gr.Row():
                            wm_traj_split = gr.Dropdown(
                                choices=["test", "val", "train"],
                                value="test",
                                label="数据集 split",
                                scale=1,
                            )
                            wm_traj_scene_filter = gr.Textbox(
                                value="",
                                label="Scene 过滤（如 FloorPlan1,FloorPlan2）",
                                scale=2,
                                placeholder="留空表示全部场景",
                            )
                        with gr.Row():
                            wm_traj_max_rows = gr.Number(value=10000, label="最大样本数", precision=0, minimum=1, scale=1)
                            run_wm_traj_btn = gr.Button("执行 WM 轨迹测试", variant="primary", scale=0, min_width=140)

                    with gr.Accordion("本次 WM 轨迹测试输出", open=True):
                        wm_traj_progress = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=0,
                            step=1,
                            label="执行进度（%）",
                            interactive=False,
                        )
                        wm_traj_progress_text = gr.Textbox(label="进度状态", value="未开始", lines=1)
                        wm_traj_summary = gr.Textbox(label="摘要", lines=7)
                        wm_traj_output_dir = gr.Textbox(label="输出目录", lines=1)
                        wm_traj_notice = gr.Textbox(label="提示", lines=2)
                        wm_rollout_filter = gr.CheckboxGroup(
                            choices=[],
                            value=[],
                            label="显示 rollout（不同 rollout 不同颜色）",
                        )
                        with gr.Row():
                            wm_connect_rollout_adjacent = gr.Checkbox(
                                value=True,
                                label="连接同一 rollout 的相邻状态",
                            )
                            wm_connect_gt_pred = gr.Checkbox(
                                value=False,
                                label="连接同一状态的 ground truth 与 predicted（首状态无pred）",
                            )
                        wm_traj_plot = gr.Plot(label="统一映射可视化（Encoder/WM 同空间）", show_label=True)

                    with gr.Accordion("WM 轨迹测试历史", open=True):
                        with gr.Row():
                            wm_traj_history_selector = gr.Dropdown(
                                choices=[],
                                value=None,
                                label="历史运行目录",
                                scale=4,
                                allow_custom_value=True,
                            )
                            refresh_wm_traj_history_btn = gr.Button("刷新历史", scale=0, min_width=100)
                        wm_traj_history_summary = gr.Textbox(label="历史摘要", lines=6)
                        wm_traj_history_preview = gr.Textbox(label="历史预览", lines=10, max_lines=20)
                        wm_traj_history_notice = gr.Textbox(label="提示", lines=2)

                    refresh_wm_run_btn.click(
                        fn=lambda: gr.update(choices=_get_wm_traj_tools()["list_wm_test_runs"](models_root=models_root)),
                        inputs=[],
                        outputs=[wm_run_selector],
                    )
                    run_wm_traj_btn.click(
                        fn=wm_traj_stream_runner,
                        inputs=[wm_run_selector, wm_traj_split, wm_traj_scene_filter, wm_traj_max_rows],
                        outputs=[
                            wm_traj_summary,
                            wm_traj_output_dir,
                            wm_traj_plot,
                            wm_traj_notice,
                            wm_traj_progress,
                            wm_traj_progress_text,
                            wm_traj_payload_state,
                            wm_rollout_filter,
                        ],
                    )
                    run_wm_traj_btn.click(
                        fn=lambda: gr.update(
                            choices=_get_wm_traj_tools()["list_wm_traj_history"](outputs_root=outputs_root)
                        ),
                        inputs=[],
                        outputs=[wm_traj_history_selector],
                    )
                    refresh_wm_traj_history_btn.click(
                        fn=lambda: gr.update(
                            choices=_get_wm_traj_tools()["list_wm_traj_history"](outputs_root=outputs_root)
                        ),
                        inputs=[],
                        outputs=[wm_traj_history_selector],
                    )
                    wm_traj_history_selector.change(
                        fn=_load_wm_traj_history_for_ui,
                        inputs=[wm_traj_history_selector],
                        outputs=[
                            wm_traj_history_summary,
                            wm_traj_history_preview,
                            wm_traj_plot,
                            wm_traj_history_notice,
                            wm_traj_payload_state,
                            wm_rollout_filter,
                        ],
                    )
                    wm_rollout_filter.change(
                        fn=_refresh_wm_traj_plot_for_ui,
                        inputs=[wm_traj_payload_state, wm_rollout_filter, wm_connect_rollout_adjacent, wm_connect_gt_pred],
                        outputs=[wm_traj_plot],
                    )
                    wm_connect_rollout_adjacent.change(
                        fn=_refresh_wm_traj_plot_for_ui,
                        inputs=[wm_traj_payload_state, wm_rollout_filter, wm_connect_rollout_adjacent, wm_connect_gt_pred],
                        outputs=[wm_traj_plot],
                    )
                    wm_connect_gt_pred.change(
                        fn=_refresh_wm_traj_plot_for_ui,
                        inputs=[wm_traj_payload_state, wm_rollout_filter, wm_connect_rollout_adjacent, wm_connect_gt_pred],
                        outputs=[wm_traj_plot],
                    )

        nav_outputs = [
            dataset_panel,
            train_panel,
            misc_panel,
            dev_panel,
            wm_traj_panel,
            wm_train_panel,
            nav_btn_dataset,
            nav_btn_train,
            nav_btn_misc,
            nav_btn_dev,
            nav_btn_wm_traj,
            nav_btn_wm_train,
        ]
        nav_btn_dataset.click(fn=lambda: _apply_nav(lab_ds), inputs=[], outputs=nav_outputs)
        nav_btn_train.click(fn=lambda: _apply_nav(lab_tr), inputs=[], outputs=nav_outputs)
        nav_btn_misc.click(fn=lambda: _apply_nav(lab_mi), inputs=[], outputs=nav_outputs)
        nav_btn_dev.click(fn=lambda: _apply_nav(lab_dev), inputs=[], outputs=nav_outputs)
        nav_btn_wm_traj.click(fn=lambda: _apply_nav(lab_wm_traj), inputs=[], outputs=nav_outputs)
        nav_btn_wm_train.click(fn=lambda: _apply_nav(lab_wm_train), inputs=[], outputs=nav_outputs)

    return app


def main() -> None:
    _configure_gradio_temp_dir()
    app = build_app(dataset_root="datasets", models_root="models", outputs_root="outputs")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue=gr.themes.colors.slate),
        css=_PROGRESS_APP_CSS,
    )


if __name__ == "__main__":
    main()

