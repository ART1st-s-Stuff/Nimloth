"""Progress 服务：单服务查看数据集与训练进度。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import gradio as gr


def _list_runs(outputs_root: str, phase: str, task: str, required_file: str) -> list[Path]:
    base = Path(outputs_root) / phase / task
    if not base.exists():
        return []
    runs = [
        path
        for path in base.glob("*/*")
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
                str(sample.get("label_text", "")),
            ]
        )
    return rows


def _select_dataset_run(run_name: str) -> tuple[str, list[list[str]]]:
    if not run_name:
        return "未选择运行目录。", []
    run_dir = Path(run_name)
    samples = _load_manifest_samples(run_dir=run_dir)
    stats = _build_stats(samples)
    rows = _build_rows(samples)
    return stats, rows


def _list_training_runs(models_root: str = "models") -> list[Path]:
    return _list_runs(models_root, phase="phase2", task="wm_training", required_file="train_metrics.json")


def _read_train_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _build_training_rows(run_dirs: list[Path]) -> list[list[str]]:
    rows: list[list[str]] = []
    for run_dir in run_dirs:
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
        return "未找到训练运行目录（models/phase2/wm_training/<date>/<time>）。"
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


def _load_calib_and_rollout_placeholder(models_root: str = "models", outputs_root: str = "outputs") -> str:
    calib_base = Path(models_root) / "phase2" / "wm_calibration"
    calib_runs = [p for p in calib_base.glob("*/*") if (p / "theta_div.json").exists()] if calib_base.exists() else []
    rollout_base = Path(outputs_root) / "phase3" / "rollout"
    rollout_exists = rollout_base.exists()
    return (
        f"校准运行数: {len(calib_runs)}\n"
        f"rollout 目录存在: {'yes' if rollout_exists else 'no'}\n\n"
        "后续将在本面板接入 rollout 轨迹、关键帧与不确定度变化曲线。"
    )


def build_app(dataset_root: str = "datasets", models_root: str = "models", outputs_root: str = "outputs") -> gr.Blocks:
    runs = _list_runs(
        outputs_root=dataset_root,
        phase="phase1",
        task="wm_data_collection",
        required_file="manifest.jsonl",
    )
    run_choices = [str(path) for path in runs]
    default_run = run_choices[0] if run_choices else None
    with gr.Blocks(title="Flower Progress Server") as app:
        gr.Markdown("## Flower Progress Server")
        gr.Markdown("同一个服务内查看数据集 progress、训练 progress，并预留校准/rollout 面板。")
        with gr.Tabs():
            with gr.Tab("数据集进度"):
                run_selector = gr.Dropdown(
                    choices=run_choices,
                    value=default_run,
                    label="选择数据收集运行目录（datasets/phase1/wm_data_collection/...）",
                )
                dataset_refresh_btn = gr.Button("刷新数据集")
                stats_box = gr.Textbox(label="统计信息", lines=18)
                sample_table = gr.Dataframe(
                    headers=["scene", "episode_id", "step_id", "action", "image_path", "label_text"],
                    label="样本预览（最多200条）",
                )
                dataset_refresh_btn.click(
                    fn=_select_dataset_run,
                    inputs=[run_selector],
                    outputs=[stats_box, sample_table],
                )
                app.load(fn=_select_dataset_run, inputs=[run_selector], outputs=[stats_box, sample_table])

            with gr.Tab("训练进度"):
                train_refresh_btn = gr.Button("刷新训练进度")
                train_summary = gr.Textbox(label="训练统计", lines=5)
                train_table = gr.Dataframe(
                    headers=["run_dir", "last_loss", "has_wm_pt", "status"],
                    label="训练运行列表",
                )
                train_refresh_btn.click(
                    fn=lambda: _load_training_progress(models_root=models_root),
                    inputs=[],
                    outputs=[train_summary, train_table],
                )
                app.load(
                    fn=lambda: _load_training_progress(models_root=models_root),
                    inputs=[],
                    outputs=[train_summary, train_table],
                )

            with gr.Tab("校准与Rollout"):
                misc_refresh_btn = gr.Button("刷新校准/Rollout状态")
                misc_box = gr.Textbox(label="扩展面板状态", lines=8)
                misc_refresh_btn.click(
                    fn=lambda: _load_calib_and_rollout_placeholder(models_root=models_root, outputs_root=outputs_root),
                    inputs=[],
                    outputs=[misc_box],
                )
                app.load(
                    fn=lambda: _load_calib_and_rollout_placeholder(models_root=models_root, outputs_root=outputs_root),
                    inputs=[],
                    outputs=[misc_box],
                )
    return app


def main() -> None:
    app = build_app(dataset_root="datasets", models_root="models", outputs_root="outputs")
    app.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()

