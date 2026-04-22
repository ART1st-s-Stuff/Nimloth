"""数据采集入口。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.data.collector import CollectConfig, count_existing_samples, run_collection
from src.utils.console import info, progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    dataset_cfg = cfg.dataset
    collect_cfg_node = dataset_cfg.collect
    run_dir = build_run_output_dir(
        path_segments=[
            str(collect_cfg_node.operation.outputs_root),
            str(dataset_cfg.name),
        ],
        resume=bool(collect_cfg_node.operation.resume),
    )
    tracker = init_tracker(task_name="collect_data", config={"backend": str(dataset_cfg.backend), "dataset": str(dataset_cfg.name)})
    collect_cfg = CollectConfig(
        backend=str(dataset_cfg.backend),
        scenes=list(dataset_cfg.scenes),
        seed=int(cfg.project.seed),
        output_dir=str(run_dir),
        num_episodes_per_scene=int(collect_cfg_node.num_episodes_per_scene),
        max_steps_per_episode=int(collect_cfg_node.max_steps_per_episode),
        num_workers=int(collect_cfg_node.num_workers),
        image_height=int(collect_cfg_node.image_height),
        image_width=int(collect_cfg_node.image_width),
        action_weights=[float(w) for w in collect_cfg_node.action_weights],
        move_ahead_range=[float(v) for v in collect_cfg_node.continuous_action.move_ahead_range],
        yaw_delta_range=[float(v) for v in collect_cfg_node.continuous_action.yaw_delta_range],
        pitch_delta_range=[float(v) for v in collect_cfg_node.continuous_action.pitch_delta_range],
        rw_move_target_range=[float(v) for v in collect_cfg_node.random_walk.move_target_range],
        rw_yaw_target_range=[float(v) for v in collect_cfg_node.random_walk.yaw_target_range],
        rw_pitch_target_range=[float(v) for v in collect_cfg_node.random_walk.pitch_target_range],
        rw_move_speed_range=[float(v) for v in collect_cfg_node.random_walk.move_speed_range],
        rw_yaw_speed_range=[float(v) for v in collect_cfg_node.random_walk.yaw_speed_range],
        rw_pitch_speed_range=[float(v) for v in collect_cfg_node.random_walk.pitch_speed_range],
        rw_target_jitter_ratio=float(collect_cfg_node.random_walk.target_jitter_ratio),
        rw_speed_jitter_ratio=float(collect_cfg_node.random_walk.speed_jitter_ratio),
        rw_single_axis_prob=float(collect_cfg_node.random_walk.single_axis_prob),
        depth_threshold_m=float(collect_cfg_node.anti_collision.depth_threshold_m),
        avoid_probability=float(collect_cfg_node.anti_collision.avoid_probability),
        wall_hit_probability=float(collect_cfg_node.anti_collision.wall_hit_probability),
        max_failed_actions=int(collect_cfg_node.anti_collision.max_failed_actions),
        min_move_scale_when_near_wall=float(collect_cfg_node.anti_collision.min_move_scale_when_near_wall),
        rotate_boost_when_near_wall=float(collect_cfg_node.anti_collision.rotate_boost_when_near_wall),
        recovery_enable=bool(collect_cfg_node.recovery.enable),
        recovery_enter_failed_actions=int(collect_cfg_node.recovery.enter_failed_actions),
        recovery_enter_low_depth_steps=int(collect_cfg_node.recovery.enter_low_depth_steps),
        recovery_steps=int(collect_cfg_node.recovery.recover_steps),
        recovery_backoff_distance_range=[float(v) for v in collect_cfg_node.recovery.backoff_distance_range],
        recovery_yaw_range=[float(v) for v in collect_cfg_node.recovery.recover_yaw_range],
        recovery_small_turn_range=[float(v) for v in collect_cfg_node.recovery.small_turn_range],
        recovery_small_turn_steps_range=[int(v) for v in collect_cfg_node.recovery.small_turn_steps_range],
        recovery_safe_depth_threshold_m=float(collect_cfg_node.recovery.safe_depth_threshold_m),
        recovery_success_exit_steps=int(collect_cfg_node.recovery.success_exit_steps),
        recovery_navmesh_assist_steps=int(collect_cfg_node.recovery.navmesh_assist_steps),
        pitch_control_enable=bool(collect_cfg_node.pitch_control.enable),
        pitch_control_threshold_deg=float(collect_cfg_node.pitch_control.threshold_deg),
        pitch_control_over_limit_steps=int(collect_cfg_node.pitch_control.over_limit_steps),
        pitch_control_recenter_step_deg=[float(v) for v in collect_cfg_node.pitch_control.recenter_step_deg],
        pitch_control_safe_band_deg=float(collect_cfg_node.pitch_control.safe_band_deg),
        navmesh_rollout_ratio=float(collect_cfg_node.navmesh.rollout_ratio),
        navmesh_target_refresh_steps=int(collect_cfg_node.navmesh.target_refresh_steps),
        navmesh_neighbor_radius=float(collect_cfg_node.navmesh.neighbor_radius),
        visibility_distance=float(dataset_cfg.visibility_distance),
        grid_size=float(dataset_cfg.grid_size),
        render_depth_image=bool(dataset_cfg.render_depth_image),
        render_instance_segmentation=bool(dataset_cfg.render_instance_segmentation),
        ai2thor_platform=str(dataset_cfg.ai2thor_platform),
        ai2thor_cache_dir=str(dataset_cfg.ai2thor_cache_dir),
        resume=bool(collect_cfg_node.operation.resume),
    )
    show_kv_table(
        "Collect Config",
        [
            ("backend", collect_cfg.backend),
            ("num_scenes", str(len(collect_cfg.scenes))),
            ("episodes_per_scene", str(collect_cfg.num_episodes_per_scene)),
            ("steps_per_episode", str(collect_cfg.max_steps_per_episode)),
            ("num_workers", str(collect_cfg.num_workers)),
            ("navmesh_ratio", f"{collect_cfg.navmesh_rollout_ratio:.2f}"),
            ("depth_threshold_m", f"{collect_cfg.depth_threshold_m:.2f}"),
        ],
    )
    total_steps = (
        len(collect_cfg.scenes)
        * collect_cfg.num_episodes_per_scene
        * collect_cfg.max_steps_per_episode
    )
    existing_steps = count_existing_samples(collect_cfg) if collect_cfg.resume else 0
    existing_steps = min(existing_steps, total_steps)
    with progress_context() as progress:
        task = progress.add_task("collecting", total=total_steps)
        if existing_steps > 0:
            progress.update(task, completed=existing_steps)
            info(f"检测到断点数据，已跳过 {existing_steps}/{total_steps} 条样本。")

        completed_steps = existing_steps

        def _on_sample() -> None:
            nonlocal completed_steps
            completed_steps += 1
            progress.update(task, completed=completed_steps)

        manifest = run_collection(collect_cfg, on_sample=_on_sample)
        progress.update(task, completed=total_steps)
    tracker.log_metrics(
        {
            "collect/scenes": len(collect_cfg.scenes),
            "collect/episodes_per_scene": collect_cfg.num_episodes_per_scene,
            "collect/steps": total_steps,
        }
    )
    tracker.log_artifact_path("collect-manifest", manifest, artifact_type="dataset")
    tracker.finish()
    info(f"manifest={manifest}")
    info(f"output_dir={run_dir}")
    success(f"采集完成 backend={collect_cfg.backend}")


if __name__ == "__main__":
    main()

