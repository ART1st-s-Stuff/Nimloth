"""Progress 服务：单服务查看数据集与训练进度。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import gradio as gr
import plotly.graph_objects as go
from dev.webui.test_zt_st_cot import (
    build_visual_payload_from_result_json_text,
    list_dev_history,
    list_rollout_runs,
    list_task_texts,
    load_dev_history,
    run_zt_st_cot_test,
)

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


def _list_runs(base: Path, required_file: str) -> list[Path]:
    if not base.exists():
        return []
    runs = [
        path
        for path in base.iterdir()
        if path.is_dir() and (path / required_file).exists()
    ]
    return sorted(runs, reverse=True)


def _load_manifest_samples(run_dir: Path, limit: int = 5000) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest.jsonl"
    if not manifest_path.exists():
        return []
    samples: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx >= limit:
                break
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
    summary, result_json_text = load_dev_history(run_dir)
    if not run_dir or not result_json_text:
        return summary, result_json_text, [], ""
    gallery_items, _, _, _, notice = build_visual_payload_from_result_json_text(result_json_text)
    history_json_preview = result_json_text
    if len(history_json_preview) > 25000:
        history_json_preview = history_json_preview[:25000] + "\n...（已截断，完整内容请查看结果目录中的 result.json）"
    return summary, history_json_preview, gallery_items, notice


def _load_dev_history_umap_plots(result_json_text: str) -> tuple[go.Figure, go.Figure, go.Figure, str]:
    if not result_json_text:
        empty_fig = _build_umap_figure({}, "UMAP 3D")
        return empty_fig, empty_fig, empty_fig, "未选择历史目录。"
    _, dino_payload, qwen_payload, st_payload, notice = build_visual_payload_from_result_json_text(result_json_text)
    return (
        _build_umap_figure(dino_payload, "z_t_dino UMAP 3D"),
        _build_umap_figure(qwen_payload, "z_t_qwen UMAP 3D"),
        _build_umap_figure(st_payload, "s_t UMAP 3D"),
        notice,
    )


def _switch_section(selected: str) -> tuple[gr.update, gr.update, gr.update, gr.update]:
    return (
        gr.update(visible=selected == "数据集进度"),
        gr.update(visible=selected == "训练进度"),
        gr.update(visible=selected == "校准与Rollout"),
        gr.update(visible=selected == "Dev/Test z_t-s_t-CoT"),
    )


def _nav_labels() -> tuple[str, str, str, str]:
    return ("数据集进度", "训练进度", "校准与Rollout", "Dev/Test z_t-s_t-CoT")


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
]:
    """侧边栏：切换主面板可见性，并同步四个导航按钮的主次样式。"""
    d, t, m, dev = _switch_section(selected)
    labels = _nav_labels()
    btn_updates = tuple(
        gr.update(variant="primary" if selected == lab else "secondary") for lab in labels
    )
    return (d, t, m, dev) + btn_updates


def _list_training_runs(models_root: str = "models") -> list[Path]:
    return _list_runs(Path(models_root) / "wm" / "cfm", required_file="train_metrics.json")


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


def _load_calib_and_rollout_placeholder(models_root: str = "models", outputs_root: str = "outputs") -> str:
    calib_base = Path(models_root) / "wm" / "cfm"
    calib_runs = [p for p in calib_base.iterdir() if p.is_dir() and (p / "theta_div.json").exists()] if calib_base.exists() else []
    rollout_base = Path(outputs_root) / "phase3" / "rollout"
    rollout_exists = rollout_base.exists()
    return (
        f"校准运行数: {len(calib_runs)}\n"
        f"rollout 目录存在: {'yes' if rollout_exists else 'no'}\n\n"
        "后续将在本面板接入 rollout 轨迹、关键帧与不确定度变化曲线。"
    )


def _run_zt_test_for_ui(
    run_dir: str,
    task_text: str,
    max_steps: float,
) -> tuple[str, list[list[Any]], str, go.Figure, go.Figure, go.Figure, list[tuple[str, str]], str]:
    (
        summary,
        table,
        output_dir,
        dino_payload,
        qwen_payload,
        st_payload,
        gallery_items,
        notice,
    ) = run_zt_st_cot_test(run_dir=run_dir, task_text=task_text, max_steps=int(max_steps))
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


def build_app(dataset_root: str = "datasets", models_root: str = "models", outputs_root: str = "outputs") -> gr.Blocks:
    runs = _list_runs(base=Path(dataset_root) / "ai2thor", required_file="manifest.jsonl")
    run_choices = [str(path) for path in runs]
    default_run = run_choices[0] if run_choices else None
    lab_ds, lab_tr, lab_mi, lab_dev = _nav_labels()
    with gr.Blocks(title="Flower Progress Server") as app:
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(elem_classes=["app-sidebar"]):
                gr.Markdown("**导航**")
                with gr.Column(elem_classes=["nav-stack"]):
                    nav_btn_dataset = gr.Button(lab_ds, variant="primary", size="lg")
                    nav_btn_train = gr.Button(lab_tr, variant="secondary", size="lg")
                    nav_btn_misc = gr.Button(lab_mi, variant="secondary", size="lg")
                    nav_btn_dev = gr.Button(lab_dev, variant="secondary", size="lg")

            with gr.Column(elem_classes=["main-content"]):
                gr.Markdown("# Flower Progress Server")
                gr.Markdown("数据集、训练进度、校准/Rollout 占位与 Dev 测试集中在同一页面；左侧为分区入口。")

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
                        dataset_gallery_notice = gr.Textbox(label="", lines=1, show_label=False)
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
                    app.load(
                        fn=_select_dataset_run,
                        inputs=[run_selector],
                        outputs=[stats_box, sample_table, dataset_gallery, dataset_gallery_notice],
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

                with gr.Group(visible=False) as misc_panel:
                    gr.Markdown("### 校准与 Rollout")
                    misc_refresh_btn = gr.Button("刷新状态", variant="secondary")
                    with gr.Accordion("状态说明", open=True):
                        misc_box = gr.Textbox(label="", lines=10, show_label=False)
                    misc_refresh_btn.click(
                        fn=lambda: _load_calib_and_rollout_placeholder(models_root=models_root, outputs_root=outputs_root),
                        inputs=[],
                        outputs=[misc_box],
                    )

                with gr.Group(visible=False) as dev_panel:
                    gr.Markdown("### Dev / Test：z_t、s_t、CoT")
                    with gr.Accordion("运行参数", open=True):
                        with gr.Row():
                            rollout_selector = gr.Dropdown(
                                choices=list_rollout_runs(dataset_root=f"{dataset_root}/ai2thor"),
                                value=None,
                                label="rollout 目录",
                                scale=4,
                                allow_custom_value=True,
                            )
                            refresh_rollout_btn = gr.Button("刷新列表", scale=0, min_width=100)
                        task_selector = gr.Dropdown(
                            choices=list_task_texts(),
                            value=list_task_texts()[0] if list_task_texts() else None,
                            label="任务文本",
                        )
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
                                choices=list_dev_history(outputs_root=outputs_root),
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
                        fn=lambda: gr.update(choices=list_rollout_runs(dataset_root=f"{dataset_root}/ai2thor")),
                        inputs=[],
                        outputs=[rollout_selector],
                    )
                    test_gallery.select(
                        fn=_select_gallery_image,
                        inputs=[test_gallery],
                        outputs=[test_large_image],
                    )

                    refresh_history_btn.click(
                        fn=lambda: gr.update(choices=list_dev_history(outputs_root=outputs_root)),
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

        nav_outputs = [dataset_panel, train_panel, misc_panel, dev_panel, nav_btn_dataset, nav_btn_train, nav_btn_misc, nav_btn_dev]
        nav_btn_dataset.click(fn=lambda: _apply_nav(lab_ds), inputs=[], outputs=nav_outputs)
        nav_btn_train.click(fn=lambda: _apply_nav(lab_tr), inputs=[], outputs=nav_outputs)
        nav_btn_misc.click(fn=lambda: _apply_nav(lab_mi), inputs=[], outputs=nav_outputs)
        nav_btn_dev.click(fn=lambda: _apply_nav(lab_dev), inputs=[], outputs=nav_outputs)

    return app


def main() -> None:
    app = build_app(dataset_root="datasets", models_root="models", outputs_root="outputs")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue=gr.themes.colors.slate),
        css=_PROGRESS_APP_CSS,
    )


if __name__ == "__main__":
    main()

