"""Phase 1 数据采集主逻辑。"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import dataclasses as dc
import json
from multiprocessing import Queue
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.ai2thor_env import Ai2ThorEnvAdapter, Ai2ThorEnvConfig
from src.data.env_adapter import EnvAdapter
from src.data.mock_env import MockEnvAdapter, MockEnvConfig
from src.data.schema import FrameState
from src.utils.io import append_jsonl, ensure_dir, flush_all_buffers


@dataclass
class CollectConfig:
    backend: str
    scenes: list[str]
    seed: int
    output_dir: str
    num_episodes_per_scene: int
    max_steps_per_episode: int
    num_workers: int
    image_height: int
    image_width: int
    action_weights: list[float]
    move_ahead_range: list[float]
    yaw_delta_range: list[float]
    pitch_delta_range: list[float]
    rw_move_target_range: list[float]
    rw_yaw_target_range: list[float]
    rw_pitch_target_range: list[float]
    rw_move_speed_range: list[float]
    rw_yaw_speed_range: list[float]
    rw_pitch_speed_range: list[float]
    rw_target_jitter_ratio: float
    rw_speed_jitter_ratio: float
    rw_single_axis_prob: float
    depth_threshold_m: float
    avoid_probability: float
    wall_hit_probability: float
    max_failed_actions: int
    min_move_scale_when_near_wall: float
    rotate_boost_when_near_wall: float
    recovery_enable: bool
    recovery_enter_failed_actions: int
    recovery_enter_low_depth_steps: int
    recovery_steps: int
    recovery_backoff_distance_range: list[float]
    recovery_yaw_range: list[float]
    recovery_small_turn_range: list[float]
    recovery_small_turn_steps_range: list[int]
    recovery_safe_depth_threshold_m: float
    recovery_success_exit_steps: int
    recovery_navmesh_assist_steps: int
    pitch_control_enable: bool
    pitch_control_threshold_deg: float
    pitch_control_over_limit_steps: int
    pitch_control_recenter_step_deg: list[float]
    pitch_control_safe_band_deg: float
    navmesh_rollout_ratio: float
    navmesh_target_refresh_steps: int
    navmesh_neighbor_radius: float
    visibility_distance: float
    grid_size: float
    render_depth_image: bool
    render_instance_segmentation: bool
    ai2thor_platform: str
    ai2thor_cache_dir: str
    resume: bool
    merge_interval_episodes: int = 0
    ai2thor_gpu_device: int | None = None


ACTION_SPACE = ["RandomWalk", "NavMesh", "ForcedRotate", "Done"]


def _validate_action_weights(action_weights: list[float]) -> list[float]:
    if len(action_weights) != len(ACTION_SPACE):
        raise ValueError(
            f"action_weights 长度应为 {len(ACTION_SPACE)}，当前为 {len(action_weights)}。"
        )
    if any(weight < 0 for weight in action_weights):
        raise ValueError("action_weights 不能包含负数。")
    weight_sum = float(sum(action_weights))
    if weight_sum <= 0:
        raise ValueError("action_weights 之和必须大于 0。")
    return [float(weight) / weight_sum for weight in action_weights]


def _clip_range(v: float, v_min: float, v_max: float) -> float:
    return float(min(max(v, v_min), v_max))


def _distance_sq_2d(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = float(a.get("x", 0.0)) - float(b.get("x", 0.0))
    dz = float(a.get("z", 0.0)) - float(b.get("z", 0.0))
    return dx * dx + dz * dz


def _uniform_in_range(rng: np.random.Generator, value_range: list[float]) -> float:
    if len(value_range) != 2:
        raise ValueError(f"range 配置必须包含2个元素，当前为 {value_range}")
    low = float(min(value_range[0], value_range[1]))
    high = float(max(value_range[0], value_range[1]))
    return float(rng.uniform(low, high))


def _wrap_angle_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _estimate_incident_angle_deg(position_history: list[dict[str, float]], current_yaw: float) -> float:
    """根据历史轨迹估计入射夹角（0~180度）。"""
    if len(position_history) < 2:
        return 90.0
    start = position_history[0]
    end = position_history[-1]
    dx = float(end.get("x", 0.0)) - float(start.get("x", 0.0))
    dz = float(end.get("z", 0.0)) - float(start.get("z", 0.0))
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 90.0
    traj_yaw = float(np.degrees(np.arctan2(dx, dz)))
    diff = abs(_wrap_angle_deg(current_yaw - traj_yaw))
    return float(min(180.0, max(0.0, diff)))


def _apply_jitter(base: float, jitter_ratio: float, rng: np.random.Generator) -> float:
    ratio = max(0.0, float(jitter_ratio))
    factor = 1.0 + float(rng.uniform(-ratio, ratio))
    return float(base * factor)


def _consume_axis(remaining: float, speed: float) -> tuple[float, float]:
    if abs(remaining) <= 1e-6 or speed <= 1e-6:
        return 0.0, 0.0
    step = float(np.sign(remaining) * min(abs(remaining), speed))
    return step, float(remaining - step)


def _build_env(cfg: CollectConfig) -> EnvAdapter:
    if cfg.backend == "mock":
        return MockEnvAdapter(
            MockEnvConfig(
                image_height=cfg.image_height,
                image_width=cfg.image_width,
                seed=cfg.seed,
            )
        )
    if cfg.backend == "ai2thor":
        try:
            return Ai2ThorEnvAdapter(
                Ai2ThorEnvConfig(
                    scene=cfg.scenes[0],
                    seed=cfg.seed,
                    image_height=cfg.image_height,
                    image_width=cfg.image_width,
                    visibility_distance=cfg.visibility_distance,
                    grid_size=cfg.grid_size,
                    render_depth_image=cfg.render_depth_image,
                    render_instance_segmentation=cfg.render_instance_segmentation,
                    platform=cfg.ai2thor_platform,
                    cache_dir=cfg.ai2thor_cache_dir,
                    gpu_device=cfg.ai2thor_gpu_device,
                )
            )
        except Exception as exc:
            raise RuntimeError(f"ai2thor 初始化失败: {exc}") from exc
    raise ValueError(f"不支持的采集后端: {cfg.backend}")


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _collect_scene(
    cfg: CollectConfig,
    scene: str,
    worker_id: int,
    progress_queue: Queue | None = None,
) -> tuple[Path, int]:
    out_dir = ensure_dir(cfg.output_dir)
    img_dir = ensure_dir(out_dir / "images")
    manifest = out_dir / f"manifest_worker_{worker_id}_{scene}.jsonl"
    existing_count = _count_jsonl_lines(manifest) if cfg.resume else 0
    if not cfg.resume and manifest.exists():
        manifest.unlink()

    scene_cfg = CollectConfig(
        backend=cfg.backend,
        scenes=[scene],
        seed=cfg.seed + worker_id,
        output_dir=cfg.output_dir,
        num_episodes_per_scene=cfg.num_episodes_per_scene,
        max_steps_per_episode=cfg.max_steps_per_episode,
        num_workers=1,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        action_weights=list(cfg.action_weights),
        move_ahead_range=list(cfg.move_ahead_range),
        yaw_delta_range=list(cfg.yaw_delta_range),
        pitch_delta_range=list(cfg.pitch_delta_range),
        rw_move_target_range=list(cfg.rw_move_target_range),
        rw_yaw_target_range=list(cfg.rw_yaw_target_range),
        rw_pitch_target_range=list(cfg.rw_pitch_target_range),
        rw_move_speed_range=list(cfg.rw_move_speed_range),
        rw_yaw_speed_range=list(cfg.rw_yaw_speed_range),
        rw_pitch_speed_range=list(cfg.rw_pitch_speed_range),
        rw_target_jitter_ratio=cfg.rw_target_jitter_ratio,
        rw_speed_jitter_ratio=cfg.rw_speed_jitter_ratio,
        rw_single_axis_prob=cfg.rw_single_axis_prob,
        depth_threshold_m=cfg.depth_threshold_m,
        avoid_probability=cfg.avoid_probability,
        wall_hit_probability=cfg.wall_hit_probability,
        max_failed_actions=cfg.max_failed_actions,
        min_move_scale_when_near_wall=cfg.min_move_scale_when_near_wall,
        rotate_boost_when_near_wall=cfg.rotate_boost_when_near_wall,
        recovery_enable=cfg.recovery_enable,
        recovery_enter_failed_actions=cfg.recovery_enter_failed_actions,
        recovery_enter_low_depth_steps=cfg.recovery_enter_low_depth_steps,
        recovery_steps=cfg.recovery_steps,
        recovery_backoff_distance_range=list(cfg.recovery_backoff_distance_range),
        recovery_yaw_range=list(cfg.recovery_yaw_range),
        recovery_small_turn_range=list(cfg.recovery_small_turn_range),
        recovery_small_turn_steps_range=list(cfg.recovery_small_turn_steps_range),
        recovery_safe_depth_threshold_m=cfg.recovery_safe_depth_threshold_m,
        recovery_success_exit_steps=cfg.recovery_success_exit_steps,
        recovery_navmesh_assist_steps=cfg.recovery_navmesh_assist_steps,
        pitch_control_enable=cfg.pitch_control_enable,
        pitch_control_threshold_deg=cfg.pitch_control_threshold_deg,
        pitch_control_over_limit_steps=cfg.pitch_control_over_limit_steps,
        pitch_control_recenter_step_deg=list(cfg.pitch_control_recenter_step_deg),
        pitch_control_safe_band_deg=cfg.pitch_control_safe_band_deg,
        navmesh_rollout_ratio=cfg.navmesh_rollout_ratio,
        navmesh_target_refresh_steps=cfg.navmesh_target_refresh_steps,
        navmesh_neighbor_radius=cfg.navmesh_neighbor_radius,
        visibility_distance=cfg.visibility_distance,
        grid_size=cfg.grid_size,
        render_depth_image=cfg.render_depth_image,
        render_instance_segmentation=cfg.render_instance_segmentation,
        ai2thor_platform=cfg.ai2thor_platform,
        ai2thor_cache_dir=cfg.ai2thor_cache_dir,
        resume=cfg.resume,
        ai2thor_gpu_device=cfg.ai2thor_gpu_device,
    )
    env = _build_env(scene_cfg)
    sample_count = 0
    pending_progress = 0
    report_interval = 32
    total_samples = cfg.num_episodes_per_scene * cfg.max_steps_per_episode
    start_episode = min(cfg.num_episodes_per_scene, existing_count // cfg.max_steps_per_episode)
    action_probs = _validate_action_weights(cfg.action_weights)

    def _report_progress(delta: int) -> None:
        nonlocal pending_progress
        if progress_queue is None or delta <= 0:
            return
        pending_progress += int(delta)
        if pending_progress >= report_interval:
            progress_queue.put(pending_progress)
            pending_progress = 0
    try:
        scene_tag = scene.lower()
        for episode_id in range(start_episode, cfg.num_episodes_per_scene):
            step_result = env.reset(episode_id=episode_id)
            episode_rng = np.random.default_rng(scene_cfg.seed * 1_000_003 + episode_id)
            rw_remaining_move = 0.0
            rw_remaining_yaw = 0.0
            rw_remaining_pitch = 0.0
            rw_move_speed = 0.0
            rw_yaw_speed = 0.0
            rw_pitch_speed = 0.0
            failed_actions = 0
            near_wall_streak = 0
            recovery_state = "Normal"
            recovery_step = 0
            recovery_turn_steps_total = 0
            recovery_turn_sign = 1.0
            recovery_required_rotation = 0.0
            recovery_accumulated_rotation = 0.0
            recovery_success_streak = 0
            pitch_over_limit_streak = 0
            pitch_control_active = False
            nav_target: dict[str, float] | None = None
            position_history: list[dict[str, float]] = []
            for step_id in range(cfg.max_steps_per_episode):
                sample_index = episode_id * cfg.max_steps_per_episode + step_id
                center_depth = step_result.metadata.get("center_depth_m")
                agent_horizon = float(step_result.metadata.get("agent_horizon", 0.0))
                curr_pos = step_result.metadata.get("agent_position", {})
                curr_yaw = float(step_result.metadata.get("agent_rotation", {}).get("y", 0.0))
                if isinstance(curr_pos, dict):
                    position_history.append(
                        {
                            "x": float(curr_pos.get("x", 0.0)),
                            "z": float(curr_pos.get("z", 0.0)),
                        }
                    )
                if len(position_history) > 8:
                    position_history = position_history[-8:]
                is_near_wall = isinstance(center_depth, (int, float)) and center_depth < cfg.depth_threshold_m
                near_wall_streak = near_wall_streak + 1 if is_near_wall else 0
                if cfg.pitch_control_enable and abs(agent_horizon) > cfg.pitch_control_threshold_deg:
                    pitch_over_limit_streak += 1
                else:
                    pitch_over_limit_streak = 0
                mode_id = int(episode_rng.choice(len(ACTION_SPACE), p=action_probs))
                if float(episode_rng.random()) < float(cfg.navmesh_rollout_ratio):
                    mode_id = 1
                if failed_actions >= cfg.max_failed_actions:
                    mode_id = 2
                mode = ACTION_SPACE[mode_id]
                recovery_active = False
                recovery_stage = "none"

                if cfg.recovery_enable and recovery_state == "Normal":
                    if (
                        failed_actions >= cfg.recovery_enter_failed_actions
                        or near_wall_streak >= cfg.recovery_enter_low_depth_steps
                    ):
                        recovery_state = "Recover"
                        recovery_step = 0
                        low = int(min(cfg.recovery_small_turn_steps_range))
                        high = int(max(cfg.recovery_small_turn_steps_range))
                        sampled_steps = int(episode_rng.integers(low, high + 1))
                        incident = _estimate_incident_angle_deg(position_history=position_history, current_yaw=curr_yaw)
                        same_dir_need = max(0.0, incident - 10.0)
                        opposite_dir_need = max(0.0, 180.0 - incident + 10.0)
                        if same_dir_need <= opposite_dir_need:
                            recovery_turn_sign = 1.0
                            recovery_required_rotation = same_dir_need
                        else:
                            recovery_turn_sign = -1.0
                            recovery_required_rotation = opposite_dir_need
                        max_small_turn = max(
                            abs(float(cfg.recovery_small_turn_range[0])),
                            abs(float(cfg.recovery_small_turn_range[1])),
                            1.0,
                        )
                        required_steps = int(np.ceil(recovery_required_rotation / max_small_turn))
                        recovery_turn_steps_total = max(sampled_steps, required_steps)
                        recovery_accumulated_rotation = 0.0
                        recovery_success_streak = 0

                if cfg.recovery_enable and recovery_state == "Recover":
                    recovery_active = True
                    recovery_step += 1
                    recovery_stage = "backoff"
                    move = _uniform_in_range(episode_rng, cfg.recovery_backoff_distance_range)
                    yaw = 0.0
                    pitch = 0.0
                    mode = "ForcedRotate"
                    mode_id = ACTION_SPACE.index(mode)
                    action = {
                        "move_ahead_distance": float(move),
                        "delta_yaw": 0.0,
                        "delta_pitch": 0.0,
                        "move_back": True,
                    }
                    if recovery_step == 2:
                        recovery_stage = "rotate_small"
                        rotate = _uniform_in_range(episode_rng, cfg.recovery_small_turn_range)
                        rotate *= recovery_turn_sign
                        recovery_accumulated_rotation += abs(rotate)
                        action = {
                            "move_ahead_distance": 0.0,
                            "delta_yaw": float(rotate),
                            "delta_pitch": 0.0,
                        }
                    elif recovery_step <= 1 + max(1, recovery_turn_steps_total):
                        recovery_stage = "rotate_small"
                        rotate = _uniform_in_range(episode_rng, cfg.recovery_small_turn_range)
                        rotate *= recovery_turn_sign
                        recovery_accumulated_rotation += abs(rotate)
                        action = {
                            "move_ahead_distance": 0.0,
                            "delta_yaw": float(rotate),
                            "delta_pitch": 0.0,
                        }
                        if (
                            recovery_accumulated_rotation < recovery_required_rotation
                            and recovery_step >= 1 + max(1, recovery_turn_steps_total)
                        ):
                            recovery_turn_steps_total += 1
                    elif recovery_step >= 2 + max(1, recovery_turn_steps_total):
                        recovery_stage = "assist_navmesh"
                        mode = "NavMesh"
                        mode_id = ACTION_SPACE.index(mode)
                        nav_assist_limit = max(1, cfg.recovery_navmesh_assist_steps)
                        if recovery_step > 2 + nav_assist_limit:
                            recovery_stage = "forward_probe"
                        action = {
                            "move_ahead_distance": float(cfg.move_ahead_range[0]),
                            "delta_yaw": 0.0,
                            "delta_pitch": 0.0,
                        }
                        if hasattr(env, "get_reachable_positions"):
                            reachable = env.get_reachable_positions()
                            if reachable:
                                pos = step_result.metadata.get("agent_position", {})
                                if nav_target is None or step_id % max(1, cfg.navmesh_target_refresh_steps) == 0:
                                    nav_target = reachable[int(episode_rng.integers(len(reachable)))]
                                if nav_target is not None:
                                    dx = float(nav_target.get("x", 0.0)) - float(pos.get("x", 0.0))
                                    dz = float(nav_target.get("z", 0.0)) - float(pos.get("z", 0.0))
                                    action["delta_yaw"] = _clip_range(
                                        float(np.degrees(np.arctan2(dx, dz))),
                                        cfg.yaw_delta_range[0],
                                        cfg.yaw_delta_range[1],
                                    )
                                    action["move_ahead_distance"] = _clip_range(
                                        float(np.hypot(dx, dz)),
                                        cfg.move_ahead_range[0],
                                        cfg.move_ahead_range[1],
                                    )
                    if cfg.pitch_control_enable and pitch_over_limit_streak >= cfg.pitch_control_over_limit_steps:
                        pitch_control_active = True
                        recenter_delta = _uniform_in_range(episode_rng, cfg.pitch_control_recenter_step_deg)
                        action["delta_pitch"] = -recenter_delta if agent_horizon > 0 else recenter_delta
                    elif cfg.pitch_control_enable and abs(agent_horizon) <= cfg.pitch_control_safe_band_deg:
                        pitch_control_active = False
                    if sample_index < existing_count:
                        step_result = env.step(action=action, step_id=step_id)
                        continue
                    step_result = env.step(action=action, step_id=step_id)
                    metadata = step_result.metadata
                    metadata["scene"] = scene
                    metadata["action_mode"] = mode
                    metadata["recovery_active"] = True
                    metadata["recovery_stage"] = recovery_stage
                    metadata["near_wall"] = bool(is_near_wall)
                    metadata["pitch_control_active"] = bool(pitch_control_active)
                    metadata["pitch_over_limit_streak"] = int(pitch_over_limit_streak)
                    image_name = f"{scene_tag}_ep{episode_id:04d}_step{step_id:04d}.png"
                    image_path = img_dir / image_name
                    Image.fromarray(step_result.frame).save(image_path)
                    if bool(metadata.get("collided", False)):
                        failed_actions += 1
                        recovery_success_streak = 0
                    else:
                        failed_actions = 0
                        center_depth_after = metadata.get("center_depth_m")
                        is_safe_depth = isinstance(center_depth_after, (int, float)) and (
                            center_depth_after >= cfg.recovery_safe_depth_threshold_m
                        )
                        recovery_success_streak = recovery_success_streak + 1 if is_safe_depth else 0
                    if recovery_step >= cfg.recovery_steps or recovery_success_streak >= cfg.recovery_success_exit_steps:
                        recovery_state = "Normal"
                        recovery_step = 0
                        recovery_success_streak = 0
                    sample = FrameState(
                        episode_id=episode_id,
                        step_id=step_id,
                        image_path=str(image_path),
                        action="continuous",
                        action_id=mode_id,
                        move_ahead_distance=float(action["move_ahead_distance"]),
                        delta_yaw=float(action["delta_yaw"]),
                        delta_pitch=float(action["delta_pitch"]),
                        recovery_active=True,
                        recovery_stage=str(recovery_stage),
                        near_wall=bool(is_near_wall),
                        pitch_control_active=bool(pitch_control_active),
                        pitch_over_limit_streak=int(pitch_over_limit_streak),
                        agent_horizon=float(agent_horizon),
                        metadata=metadata,
                    )
                    append_jsonl(manifest, sample.to_dict())
                    sample_count += 1
                    _report_progress(1)
                    continue
                if is_near_wall and mode == "RandomWalk":
                    avoid_draw = float(episode_rng.random())
                    mode = "ForcedRotate" if avoid_draw < cfg.avoid_probability else "RandomWalk"
                if mode == "RandomWalk":
                    if (
                        abs(rw_remaining_move) <= 1e-6
                        and abs(rw_remaining_yaw) <= 1e-6
                        and abs(rw_remaining_pitch) <= 1e-6
                    ):
                        move_target = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_move_target_range),
                            cfg.rw_target_jitter_ratio,
                            episode_rng,
                        )
                        yaw_target = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_yaw_target_range),
                            cfg.rw_target_jitter_ratio,
                            episode_rng,
                        )
                        yaw_target *= 1.0 if float(episode_rng.random()) > 0.5 else -1.0
                        pitch_target = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_pitch_target_range),
                            cfg.rw_target_jitter_ratio,
                            episode_rng,
                        )
                        pitch_target *= 1.0 if float(episode_rng.random()) > 0.5 else -1.0

                        move_speed = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_move_speed_range),
                            cfg.rw_speed_jitter_ratio,
                            episode_rng,
                        )
                        yaw_speed = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_yaw_speed_range),
                            cfg.rw_speed_jitter_ratio,
                            episode_rng,
                        )
                        pitch_speed = _apply_jitter(
                            _uniform_in_range(episode_rng, cfg.rw_pitch_speed_range),
                            cfg.rw_speed_jitter_ratio,
                            episode_rng,
                        )
                        rw_remaining_move = float(max(0.0, move_target))
                        rw_remaining_yaw = float(yaw_target)
                        rw_remaining_pitch = float(pitch_target)
                        rw_move_speed = float(max(1e-6, move_speed))
                        rw_yaw_speed = float(max(1e-6, yaw_speed))
                        rw_pitch_speed = float(max(1e-6, pitch_speed))

                        if float(episode_rng.random()) < float(cfg.rw_single_axis_prob):
                            axis = int(episode_rng.integers(3))
                            if axis != 0:
                                rw_remaining_move = 0.0
                            if axis != 1:
                                rw_remaining_yaw = 0.0
                            if axis != 2:
                                rw_remaining_pitch = 0.0

                    move, rw_remaining_move = _consume_axis(rw_remaining_move, rw_move_speed)
                    yaw, rw_remaining_yaw = _consume_axis(rw_remaining_yaw, rw_yaw_speed)
                    pitch, rw_remaining_pitch = _consume_axis(rw_remaining_pitch, rw_pitch_speed)
                else:
                    move = _uniform_in_range(episode_rng, cfg.move_ahead_range)
                    yaw = _uniform_in_range(episode_rng, cfg.yaw_delta_range)
                    pitch = _uniform_in_range(episode_rng, cfg.pitch_delta_range)

                if is_near_wall:
                    move_scale = max(0.0, float(cfg.min_move_scale_when_near_wall))
                    move *= move_scale
                    yaw += float(np.sign(yaw if abs(yaw) > 1e-6 else (episode_rng.random() - 0.5))) * float(
                        cfg.rotate_boost_when_near_wall
                    ) * float(cfg.yaw_delta_range[1])

                if mode == "NavMesh" and hasattr(env, "get_reachable_positions"):
                    reachable = env.get_reachable_positions()
                    if (
                        nav_target is None
                        or step_id % max(1, cfg.navmesh_target_refresh_steps) == 0
                        or _distance_sq_2d(step_result.metadata.get("agent_position", {}), nav_target)
                        < cfg.navmesh_neighbor_radius * cfg.navmesh_neighbor_radius
                    ):
                        if reachable:
                            nav_target = reachable[int(episode_rng.integers(len(reachable)))]
                    if nav_target is not None:
                        pos = step_result.metadata.get("agent_position", {})
                        dx = float(nav_target.get("x", 0.0)) - float(pos.get("x", 0.0))
                        dz = float(nav_target.get("z", 0.0)) - float(pos.get("z", 0.0))
                        yaw = _clip_range(float(np.degrees(np.arctan2(dx, dz))), cfg.yaw_delta_range[0], cfg.yaw_delta_range[1])
                        move = max(float(cfg.move_ahead_range[0]), min(float(cfg.move_ahead_range[1]), float(np.hypot(dx, dz))))
                elif mode == "ForcedRotate":
                    move = 0.0
                    yaw = float(cfg.yaw_delta_range[1]) * (1.0 if float(episode_rng.random()) > 0.5 else -1.0)
                    pitch = 0.0
                elif mode == "Done":
                    move = 0.0
                    yaw = 0.0
                    pitch = 0.0

                if is_near_wall and float(episode_rng.random()) < float(cfg.wall_hit_probability):
                    move = float(cfg.move_ahead_range[1])
                    yaw = 0.0
                    pitch = 0.0

                action = {
                    "move_ahead_distance": float(abs(move)),
                    "delta_yaw": float(yaw),
                    "delta_pitch": float(pitch),
                }
                if cfg.pitch_control_enable and pitch_over_limit_streak >= cfg.pitch_control_over_limit_steps:
                    pitch_control_active = True
                    recenter_delta = _uniform_in_range(episode_rng, cfg.pitch_control_recenter_step_deg)
                    action["delta_pitch"] = -recenter_delta if agent_horizon > 0 else recenter_delta
                elif cfg.pitch_control_enable and abs(agent_horizon) <= cfg.pitch_control_safe_band_deg:
                    pitch_control_active = False
                if sample_index < existing_count:
                    step_result = env.step(action=action, step_id=step_id)
                    continue
                step_result = env.step(action=action, step_id=step_id)
                metadata: dict[str, Any] = step_result.metadata
                metadata["scene"] = scene
                metadata["action_mode"] = mode
                metadata["recovery_active"] = bool(recovery_active)
                metadata["recovery_stage"] = recovery_stage
                metadata["near_wall"] = bool(is_near_wall)
                metadata["pitch_control_active"] = bool(pitch_control_active)
                metadata["pitch_over_limit_streak"] = int(pitch_over_limit_streak)
                image_name = f"{scene_tag}_ep{episode_id:04d}_step{step_id:04d}.png"
                image_path = img_dir / image_name
                Image.fromarray(step_result.frame).save(image_path)
                if bool(metadata.get("collided", False)):
                    failed_actions += 1
                else:
                    failed_actions = 0
                sample = FrameState(
                    episode_id=episode_id,
                    step_id=step_id,
                    image_path=str(image_path),
                    action="continuous",
                    action_id=mode_id,
                    move_ahead_distance=float(action["move_ahead_distance"]),
                    delta_yaw=float(action["delta_yaw"]),
                    delta_pitch=float(action["delta_pitch"]),
                    recovery_active=bool(recovery_active),
                    recovery_stage=str(recovery_stage),
                    near_wall=bool(is_near_wall),
                    pitch_control_active=bool(pitch_control_active),
                    pitch_over_limit_streak=int(pitch_over_limit_streak),
                    agent_horizon=float(agent_horizon),
                    metadata=metadata,
                )
                append_jsonl(manifest, sample.to_dict())
                sample_count += 1
                _report_progress(1)
            # 中间合并：每完成 N 个 episode 追加到最终 manifest。
            if cfg.merge_interval_episodes > 0 and (episode_id + 1) % cfg.merge_interval_episodes == 0:
                out_dir = ensure_dir(cfg.output_dir)
                _append_worker_manifest_to_final(out_dir, manifest, worker_id, scene)
    finally:
        if progress_queue is not None and pending_progress > 0:
            progress_queue.put(pending_progress)
        flush_all_buffers()
        env.close()
    return manifest, min(existing_count + sample_count, total_samples)


def count_existing_samples(cfg: CollectConfig) -> int:
    out_dir = ensure_dir(cfg.output_dir)
    total = 0
    for idx, scene in enumerate(cfg.scenes):
        manifest = out_dir / f"manifest_worker_{idx}_{scene}.jsonl"
        total += min(
            _count_jsonl_lines(manifest),
            cfg.num_episodes_per_scene * cfg.max_steps_per_episode,
        )
    return total


def _append_worker_manifest_to_final(out_dir: Path, worker_manifest: Path, worker_id: int, scene: str) -> None:
    """将 worker manifest 中未合并的 episode 追加到最终 manifest 并更新跟踪文件。"""
    if not worker_manifest.exists():
        return
    track_path = out_dir / f"manifest_merge_track_{worker_id}_{scene}.json"
    last_ep = -1
    if track_path.exists():
        try:
            last_ep = json.loads(track_path.read_text(encoding="utf-8")).get("last_episode", -1)
        except Exception:
            last_ep = -1

    final_manifest = out_dir / "manifest.jsonl"
    lines_to_append = []
    max_ep_this_pass = last_ep
    for line in worker_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        ep = int(record.get("episode_id", -1))
        if ep > last_ep:
            lines_to_append.append(line)
            if ep > max_ep_this_pass:
                max_ep_this_pass = ep

    if lines_to_append:
        with final_manifest.open("a", encoding="utf-8") as fout:
            for ln in lines_to_append:
                fout.write(ln + "\n")
        track_path.write_text(
            json.dumps({"last_episode": max_ep_this_pass}, ensure_ascii=False),
            encoding="utf-8",
        )


def _collect_scene_for_process(
    scene_cfg: CollectConfig,
    scene: str,
    worker_id: int,
    progress_queue: Queue | None = None,
) -> tuple[Path, int]:
    """独立进程调用的采集入口（顶层函数，可 pickle）。"""
    return _collect_scene(scene_cfg, scene, worker_id, progress_queue=progress_queue)


def run_collection(
    cfg: CollectConfig,
    progress_queue: Queue | None = None,
) -> Path:
    """执行多场景采集并输出合并 manifest。

    进程池方案：每个 worker 是独立进程，无 GIL 争用。
    progress_queue 用于向主进程推送采样进度（跨进程安全）。
    主进程轮询该队列，实时更新进度条。
    """
    if not cfg.scenes:
        raise RuntimeError("未配置 scene 列表：data.env.scenes 不能为空。")
    _ = _validate_action_weights(cfg.action_weights)
    out_dir = ensure_dir(cfg.output_dir)
    final_manifest = out_dir / "manifest.jsonl"
    if final_manifest.exists():
        final_manifest.unlink()

    if cfg.num_workers <= 1 or len(cfg.scenes) == 1:
        partials = [
            _collect_scene(cfg, scene, idx, progress_queue=progress_queue)
            for idx, scene in enumerate(cfg.scenes)
        ]
    else:
        # 构建每个场景的子配置（排除已覆盖字段）
        excluded = ("scenes", "seed", "num_workers")
        fields = {
            f.name: getattr(cfg, f.name)
            for f in dc.fields(cfg)
            if f.name not in excluded
        }
        scene_cfgs = [
            CollectConfig(**fields, scenes=[scene], seed=cfg.seed + idx, num_workers=1)
            for idx, scene in enumerate(cfg.scenes)
        ]

        partials = []
        with ProcessPoolExecutor(max_workers=cfg.num_workers) as pool:
            futures = {
                pool.submit(_collect_scene_for_process, scene_cfgs[i], scene, i, progress_queue): (i, scene)
                for i, scene in enumerate(cfg.scenes)
            }
            for future in as_completed(futures):
                result = future.result()
                partials.append(result)
                del futures[future]

    interval = cfg.merge_interval_episodes
    if interval > 0:
        for partial_manifest, _ in partials:
            partial_manifest.unlink(missing_ok=True)
    else:
        for partial_manifest, _ in partials:
            for line in partial_manifest.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    append_jsonl(final_manifest, data=json.loads(line))
            partial_manifest.unlink(missing_ok=True)
        flush_all_buffers()
    return final_manifest