"""WM 轨迹对比测试：在线采样 + 真实/预测轨迹降维。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from omegaconf import OmegaConf
import torch
import umap

from src.data.collector import CollectConfig, run_collection
from src.utils.io import ensure_dir
from src.utils.model_provider import resolve_latest_model_file
from src.wm.encoders import build_wm_image_encoder
from src.wm.factory import build_world_model

TEST_SCENES = ["FloorPlan1", "FloorPlan101", "FloorPlan201"]
_CACHE_ROOT = Path(".cache") / "visualize"
_UMAP_CACHE_DIR = _CACHE_ROOT / "umap"


@dataclass
class TrajPoint:
    scene: str
    episode_id: int
    step_id: int
    rollout_id: str
    state_index: int
    source: str
    latent: list[float]


def _ensure_cache_dirs() -> None:
    _UMAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _umap_cache_key(feature_name: str, points: list[list[float]]) -> str:
    arr = np.asarray(points, dtype=np.float32)
    hasher = hashlib.sha256()
    hasher.update(feature_name.encode("utf-8"))
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(arr.tobytes())
    return hasher.hexdigest()[:24]


def _load_umap_cache(feature_name: str, cache_key: str) -> list[list[float]] | None:
    _ensure_cache_dirs()
    cache_path = _UMAP_CACHE_DIR / f"{feature_name}_{cache_key}.json"
    if not cache_path.exists():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    points = payload.get("points", [])
    if not isinstance(points, list):
        return None
    return points


def _save_umap_cache(feature_name: str, cache_key: str, embedded: list[list[float]]) -> None:
    _ensure_cache_dirs()
    cache_path = _UMAP_CACHE_DIR / f"{feature_name}_{cache_key}.json"
    payload = {
        "feature": feature_name,
        "cache_key": cache_key,
        "points": embedded,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_umap_3d(points: list[list[float]], feature_name: str) -> tuple[list[list[float]], str | None]:
    if len(points) < 3:
        return [], "样本数少于3，跳过 UMAP 三维降维。"
    cache_key = _umap_cache_key(feature_name=feature_name, points=points)
    cached = _load_umap_cache(feature_name=feature_name, cache_key=cache_key)
    if cached is not None:
        return cached, None
    arr = np.asarray(points, dtype=np.float32)
    n_neighbors = min(15, max(2, len(points) - 1))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
    try:
        embedded = reducer.fit_transform(arr)
    except Exception as exc:
        return [], f"UMAP 计算失败: {exc}"
    embedded_list = embedded.astype(np.float32).tolist()
    _save_umap_cache(feature_name=feature_name, cache_key=cache_key, embedded=embedded_list)
    return embedded_list, None


def _load_rows(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not manifest_path.exists():
        return rows
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def _build_collect_config(
    dataset_cfg: Any,
    output_dir: Path,
    scenes: list[str],
    episodes_per_scene: int,
    max_steps: int,
) -> CollectConfig:
    collect_cfg = dataset_cfg.collect
    return CollectConfig(
        backend=str(dataset_cfg.backend),
        scenes=list(scenes),
        seed=114514,
        output_dir=str(output_dir),
        num_episodes_per_scene=int(episodes_per_scene),
        max_steps_per_episode=int(max_steps),
        num_workers=max(1, min(int(collect_cfg.num_workers), len(TEST_SCENES))),
        image_height=int(collect_cfg.image_height),
        image_width=int(collect_cfg.image_width),
        action_weights=[float(w) for w in collect_cfg.action_weights],
        move_ahead_range=[float(v) for v in collect_cfg.continuous_action.move_ahead_range],
        yaw_delta_range=[float(v) for v in collect_cfg.continuous_action.yaw_delta_range],
        pitch_delta_range=[float(v) for v in collect_cfg.continuous_action.pitch_delta_range],
        rw_move_target_range=[float(v) for v in collect_cfg.random_walk.move_target_range],
        rw_yaw_target_range=[float(v) for v in collect_cfg.random_walk.yaw_target_range],
        rw_pitch_target_range=[float(v) for v in collect_cfg.random_walk.pitch_target_range],
        rw_move_speed_range=[float(v) for v in collect_cfg.random_walk.move_speed_range],
        rw_yaw_speed_range=[float(v) for v in collect_cfg.random_walk.yaw_speed_range],
        rw_pitch_speed_range=[float(v) for v in collect_cfg.random_walk.pitch_speed_range],
        rw_target_jitter_ratio=float(collect_cfg.random_walk.target_jitter_ratio),
        rw_speed_jitter_ratio=float(collect_cfg.random_walk.speed_jitter_ratio),
        rw_single_axis_prob=float(collect_cfg.random_walk.single_axis_prob),
        depth_threshold_m=float(collect_cfg.anti_collision.depth_threshold_m),
        avoid_probability=float(collect_cfg.anti_collision.avoid_probability),
        wall_hit_probability=float(collect_cfg.anti_collision.wall_hit_probability),
        max_failed_actions=int(collect_cfg.anti_collision.max_failed_actions),
        min_move_scale_when_near_wall=float(collect_cfg.anti_collision.min_move_scale_when_near_wall),
        rotate_boost_when_near_wall=float(collect_cfg.anti_collision.rotate_boost_when_near_wall),
        recovery_enable=bool(collect_cfg.recovery.enable),
        recovery_enter_failed_actions=int(collect_cfg.recovery.enter_failed_actions),
        recovery_enter_low_depth_steps=int(collect_cfg.recovery.enter_low_depth_steps),
        recovery_steps=int(collect_cfg.recovery.recover_steps),
        recovery_backoff_distance_range=[float(v) for v in collect_cfg.recovery.backoff_distance_range],
        recovery_yaw_range=[float(v) for v in collect_cfg.recovery.recover_yaw_range],
        recovery_small_turn_range=[float(v) for v in collect_cfg.recovery.small_turn_range],
        recovery_small_turn_steps_range=[int(v) for v in collect_cfg.recovery.small_turn_steps_range],
        recovery_safe_depth_threshold_m=float(collect_cfg.recovery.safe_depth_threshold_m),
        recovery_success_exit_steps=int(collect_cfg.recovery.success_exit_steps),
        recovery_navmesh_assist_steps=int(collect_cfg.recovery.navmesh_assist_steps),
        pitch_control_enable=bool(collect_cfg.pitch_control.enable),
        pitch_control_threshold_deg=float(collect_cfg.pitch_control.threshold_deg),
        pitch_control_over_limit_steps=int(collect_cfg.pitch_control.over_limit_steps),
        pitch_control_recenter_step_deg=[float(v) for v in collect_cfg.pitch_control.recenter_step_deg],
        pitch_control_safe_band_deg=float(collect_cfg.pitch_control.safe_band_deg),
        navmesh_rollout_ratio=float(collect_cfg.navmesh.rollout_ratio),
        navmesh_target_refresh_steps=int(collect_cfg.navmesh.target_refresh_steps),
        navmesh_neighbor_radius=float(collect_cfg.navmesh.neighbor_radius),
        visibility_distance=float(dataset_cfg.visibility_distance),
        grid_size=float(dataset_cfg.grid_size),
        render_depth_image=bool(dataset_cfg.render_depth_image),
        render_instance_segmentation=bool(dataset_cfg.render_instance_segmentation),
        ai2thor_platform=str(dataset_cfg.ai2thor_platform),
        ai2thor_cache_dir=str(dataset_cfg.ai2thor_cache_dir),
        resume=False,
    )


def _resolve_wm_config(wm_name: str) -> Any:
    target = Path("configs/wm") / f"{wm_name}.yaml"
    if not target.exists():
        target = Path("configs/wm/cfm_dinov2m.yaml")
    return OmegaConf.load(target)


def _resolve_wm_ckpt_path(wm_run_path: Path) -> Path | None:
    """优先使用 EMA 权重，不存在时回退普通权重。"""
    resolved = resolve_latest_model_file(wm_run_path, ["wm_ema.pt", "wm.pt"])
    if resolved is not None:
        return resolved
    # 兼容仅传入单个 run 目录的场景。
    for name in ["wm_ema.pt", "wm.pt"]:
        candidate = wm_run_path / name
        if candidate.exists():
            return candidate
    return None


def _wm_ckpt_type(ckpt_path: Path) -> str:
    if ckpt_path.name == "wm_ema.pt":
        return "ema"
    return "base"


def _build_model(wm_cfg: Any, action_dim: int, wm_ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_world_model(
        wm_cfg=wm_cfg,
        train_cfg=None,
        action_dim=int(action_dim),
        device=device,
    )
    model.load_state_dict(torch.load(wm_ckpt_path, map_location=device))
    model.eval()
    return model


def _action_vec(row: dict[str, Any], action_dim: int) -> torch.Tensor:
    base = torch.tensor(
        [
            float(row.get("move_ahead_distance", 0.0)),
            float(row.get("delta_yaw", 0.0)),
            float(row.get("delta_pitch", 0.0)),
        ],
        dtype=torch.float32,
    )
    if base.numel() >= action_dim:
        return base[:action_dim]
    return torch.cat([base, torch.zeros(action_dim - base.numel(), dtype=torch.float32)], dim=0)


def _group_episodes(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        scene = str(metadata.get("scene", "unknown"))
        episode_id = int(row.get("episode_id", -1))
        grouped[(scene, episode_id)].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item.get("step_id", -1)))
    return grouped


def _build_umap_payload(feature_name: str, points: list[TrajPoint]) -> tuple[dict[str, Any], str | None]:
    embedded, warning = _compute_umap_3d([item.latent for item in points], feature_name=feature_name)
    if not embedded and points:
        # UMAP 失败时退化为前三维投影，保证前端仍可查看轨迹结构。
        fallback_coords = []
        for item in points:
            vec = item.latent
            x = float(vec[0]) if len(vec) > 0 else 0.0
            y = float(vec[1]) if len(vec) > 1 else 0.0
            z = float(vec[2]) if len(vec) > 2 else 0.0
            fallback_coords.append([x, y, z])
        embedded = fallback_coords
        warning = warning or "UMAP 失败，已退化为前三维投影。"
    if not embedded:
        return {"feature": feature_name, "points": [], "warning": warning or ""}, warning
    payload_points: list[dict[str, Any]] = []
    for idx, coords in enumerate(embedded):
        item = points[idx]
        payload_points.append(
            {
                "x": float(coords[0]),
                "y": float(coords[1]),
                "z": float(coords[2]),
                "episode_id": item.episode_id,
                "step_id": item.step_id,
                "scene": item.scene,
                "rollout_id": item.rollout_id,
                "state_index": item.state_index,
                "source": item.source,
            }
        )
    return {"feature": feature_name, "points": payload_points, "warning": ""}, None


def _build_shared_umap_payloads(
    real_points: list[TrajPoint],
    pred_points: list[TrajPoint],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    all_points = real_points + pred_points
    overlay_payload, overlay_warning = _build_umap_payload("wm_traj_shared", all_points)
    real_payload = {
        "feature": "wm_traj_real",
        "points": [point for point in overlay_payload.get("points", []) if point.get("source") == "real"],
        "warning": overlay_payload.get("warning", ""),
    }
    pred_payload = {
        "feature": "wm_traj_pred",
        "points": [point for point in overlay_payload.get("points", []) if point.get("source") == "pred"],
        "warning": overlay_payload.get("warning", ""),
    }
    warnings = [msg for msg in [overlay_warning, overlay_payload.get("warning", "")] if msg]
    return real_payload, pred_payload, overlay_payload, sorted(set(warnings))


def list_wm_test_runs(models_root: str = "models") -> list[str]:
    root = Path(models_root) / "wm"
    if not root.exists():
        return []
    runs: list[Path] = []
    for wm_dir in root.iterdir():
        if not wm_dir.is_dir():
            continue
        for run_dir in wm_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if (run_dir / "wm_ema.pt").exists() or (run_dir / "wm.pt").exists():
                runs.append(run_dir)
    return [str(path) for path in sorted(runs, reverse=True)]


def list_wm_traj_history(outputs_root: str = "outputs") -> list[str]:
    base = Path(outputs_root) / "dev"
    if not base.exists():
        return []
    runs: list[Path] = []
    for day_dir in base.iterdir():
        if not day_dir.is_dir():
            continue
        for task_dir in day_dir.iterdir():
            if not task_dir.is_dir() or not task_dir.name.startswith("test_wm_traj_compare"):
                continue
            for run_dir in task_dir.iterdir():
                if run_dir.is_dir() and (run_dir / "result.json").exists():
                    runs.append(run_dir)
    return [str(path) for path in sorted(runs, reverse=True)]


def _build_output_dir(outputs_root: str = "outputs") -> Path:
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(outputs_root) / "dev" / day / "test_wm_traj_compare" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_wm_traj_compare_test(
    wm_run_dir: str,
    episodes_per_scene: int = 1,
    max_steps_per_episode: int = 16,
    outputs_root: str = "outputs",
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    def _report(progress: float, status: str) -> None:
        if progress_callback is None:
            return
        progress_callback(float(max(0.0, min(100.0, progress))), status)

    _report(1.0, "初始化参数...")
    if not wm_run_dir:
        empty = {"feature": "", "points": [], "warning": ""}
        return "未选择 WM 运行目录。", "", empty, empty, empty, ""
    wm_run_path = Path(wm_run_dir)
    wm_ckpt_path = _resolve_wm_ckpt_path(wm_run_path)
    if wm_ckpt_path is None:
        empty = {"feature": "", "points": [], "warning": ""}
        return f"缺少 WM checkpoint（wm_ema.pt/wm.pt）: {wm_run_path}", "", empty, empty, empty, ""

    dataset_cfg = OmegaConf.load(Path("configs/dataset/ai2thor.yaml"))
    wm_name = wm_run_path.parent.name
    wm_cfg = _resolve_wm_config(wm_name=wm_name)
    out_dir = _build_output_dir(outputs_root=outputs_root)
    collect_root = ensure_dir(out_dir / "collection")
    _report(5.0, "开始在线采样（FloorPlan1/101/201）...")
    rows: list[dict[str, Any]] = []
    scene_errors: dict[str, str] = {}
    success_manifests: list[str] = []
    total_scenes = max(1, len(TEST_SCENES))
    for scene_idx, scene in enumerate(TEST_SCENES, start=1):
        _report(5.0 + 30.0 * (scene_idx - 1) / total_scenes, f"采样场景 {scene_idx}/{total_scenes}：{scene}")
        scene_collect_dir = ensure_dir(collect_root / scene)
        try:
            collect_cfg = _build_collect_config(
                dataset_cfg=dataset_cfg,
                output_dir=scene_collect_dir,
                scenes=[scene],
                episodes_per_scene=max(1, int(episodes_per_scene)),
                max_steps=max(1, int(max_steps_per_episode)),
            )
            manifest_path = run_collection(collect_cfg)
            rows.extend(_load_rows(manifest_path))
            success_manifests.append(str(manifest_path))
        except Exception as exc:
            scene_errors[scene] = str(exc)
    _report(38.0, "采样完成，准备加载编码器与WM...")

    if not rows:
        empty = {"feature": "", "points": [], "warning": ""}
        error_text = "；".join([f"{scene}: {err}" for scene, err in sorted(scene_errors.items())])
        return f"采样结果为空。{error_text}", str(out_dir), empty, empty, empty, ""

    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if encoder is None:
        empty = {"feature": "", "points": [], "warning": ""}
        return "当前 WM 未启用 encoder，无法构建真实轨迹。", str(out_dir), empty, empty, empty, ""

    _report(45.0, "加载 WM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(
        wm_cfg=wm_cfg,
        action_dim=int(dataset_cfg.action_dim),
        wm_ckpt_path=wm_ckpt_path,
        device=device,
    )
    history_len = int(wm_cfg.history_len)
    action_dim = int(dataset_cfg.action_dim)

    episode_rows = _group_episodes(rows)
    real_points: list[TrajPoint] = []
    pred_points: list[TrajPoint] = []
    scene_counter: dict[str, int] = defaultdict(int)

    total_episodes = max(1, len(episode_rows))
    _report(55.0, "开始编码真实轨迹与滚动预测...")
    with torch.no_grad():
        for epi_idx, ((scene, episode_id), seq_rows) in enumerate(episode_rows.items(), start=1):
            _report(
                55.0 + 30.0 * (epi_idx - 1) / total_episodes,
                f"处理 episode {epi_idx}/{total_episodes}：scene={scene}, ep={episode_id}",
            )
            rollout_id = f"{scene}#ep{episode_id}"
            latents: list[torch.Tensor] = []
            for state_index, row in enumerate(seq_rows):
                image_path = str(row.get("image_path", ""))
                z = encoder.encode_image_path(image_path).z.float().cpu()
                latents.append(z)
                real_points.append(
                    TrajPoint(
                        scene=scene,
                        episode_id=episode_id,
                        step_id=int(row.get("step_id", -1)),
                        rollout_id=rollout_id,
                        state_index=state_index,
                        source="real",
                        latent=z.reshape(-1).tolist(),
                    )
                )
            if len(seq_rows) < history_len + 1:
                continue
            scene_counter[scene] += 1
            # 每步监督后使用 GT latent 校准 history window，避免纯 predicted latent rollout。
            teacher_z = torch.stack(latents[:history_len], dim=0).unsqueeze(0).to(device)
            teacher_a = torch.stack([_action_vec(row, action_dim=action_dim) for row in seq_rows[:history_len]], dim=0).unsqueeze(0).to(device)
            num_steps = len(seq_rows) - history_len
            for step_idx in range(num_steps):
                teacher_a[:, -1, :] = _action_vec(seq_rows[history_len + step_idx - 1], action_dim=action_dim).to(device)
                pred = model.predict_next(teacher_z, teacher_a).squeeze(0).float().cpu()
                state_index = history_len + step_idx
                pred_points.append(
                    TrajPoint(
                        scene=scene,
                        episode_id=episode_id,
                        step_id=int(seq_rows[state_index].get("step_id", -1)),
                        rollout_id=rollout_id,
                        state_index=state_index,
                        source="pred",
                        latent=pred.reshape(-1).tolist(),
                    )
                )
                # history window 向前滑动：使用对应 GT latent 校准
                gt_next = latents[state_index].to(device)
                teacher_z = torch.cat([teacher_z[:, 1:, ...], gt_next.unsqueeze(0).unsqueeze(1)], dim=1)
                if step_idx < num_steps - 1:
                    next_action = _action_vec(seq_rows[state_index], action_dim=action_dim).unsqueeze(0).unsqueeze(1).to(device)
                    teacher_a = torch.cat([teacher_a[:, 1:, :], next_action], dim=1)

    _report(88.0, "降维可视化（UMAP）处理中...")
    real_payload, pred_payload, overlay_payload, warnings = _build_shared_umap_payloads(real_points, pred_points)
    warning_text = "；".join(sorted(set(warnings))) if warnings else ""

    result_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "wm_run_dir": str(wm_run_path),
        "wm_ckpt_path": str(wm_ckpt_path),
        "wm_ckpt_type": _wm_ckpt_type(wm_ckpt_path),
        "test_scenes": list(TEST_SCENES),
        "collect_manifest_paths": success_manifests,
        "episodes_per_scene": int(episodes_per_scene),
        "max_steps_per_episode": int(max_steps_per_episode),
        "rows_total": len(rows),
        "scene_counts": dict(sorted(scene_counter.items())),
        "scene_errors": scene_errors,
        "real_points_total": len(real_points),
        "pred_points_total": len(pred_points),
        "umap": {
            "real": real_payload,
            "pred": pred_payload,
            "overlay": overlay_payload,
        },
    }
    _report(96.0, "写入结果文件...")
    (out_dir / "result.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = (
        f"执行完成：WM轨迹对比测试\n"
        f"wm_run={wm_run_path}\n"
        f"manifest_count={len(success_manifests)}\n"
        f"rows_total={len(rows)}\n"
        f"real_points={len(real_points)} pred_points={len(pred_points)}\n"
        f"scene_episodes={dict(sorted(scene_counter.items()))}"
    )
    if scene_errors:
        summary += "\nscene_errors=" + str(scene_errors)
    _report(100.0, "执行完成")
    return summary, str(out_dir), real_payload, pred_payload, overlay_payload, warning_text


def load_wm_traj_history(run_dir: str) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any], str]:
    empty = {"feature": "", "points": [], "warning": ""}
    if not run_dir:
        return "未选择历史运行目录。", "", empty, empty, empty, ""
    result_path = Path(run_dir) / "result.json"
    if not result_path.exists():
        return f"缺少结果文件: {result_path}", "", empty, empty, empty, ""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    umap_payload = payload.get("umap", {})
    real_payload = dict(umap_payload.get("real", empty))
    pred_payload = dict(umap_payload.get("pred", empty))
    overlay_payload = dict(umap_payload.get("overlay", empty))
    summary = (
        f"wm_run={payload.get('wm_run_dir', '')}\n"
        f"manifest_count={len(payload.get('collect_manifest_paths', []))}\n"
        f"rows_total={payload.get('rows_total', 0)}\n"
        f"real_points={payload.get('real_points_total', 0)} pred_points={payload.get('pred_points_total', 0)}"
    )
    preview = {
        "created_at": payload.get("created_at", ""),
        "wm_run_dir": payload.get("wm_run_dir", ""),
        "test_scenes": payload.get("test_scenes", []),
        "rows_total": payload.get("rows_total", 0),
        "real_points_total": payload.get("real_points_total", 0),
        "pred_points_total": payload.get("pred_points_total", 0),
        "scene_counts": payload.get("scene_counts", {}),
        "scene_errors": payload.get("scene_errors", {}),
    }
    return summary, json.dumps(preview, ensure_ascii=False, indent=2), real_payload, pred_payload, overlay_payload, ""
