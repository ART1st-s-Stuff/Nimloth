"""数据采集入口。"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import queue
import threading

import hydra
from omegaconf import DictConfig, ListConfig

from src.data.collector import CollectConfig, count_existing_samples, run_collection
from src.utils.console import info, progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker


def _build_collect_cfg(
    dataset_cfg: DictConfig,
    collect_cfg_node: DictConfig,
    scenes: list[str],
    split: str,
) -> CollectConfig:
    return CollectConfig(
        backend=str(dataset_cfg.backend),
        scenes=list(scenes),
        seed=int(dataset_cfg.get("seed", 114514)),
        output_dir=str(build_run_output_dir(
            path_segments=[
                str(collect_cfg_node.operation.outputs_root),
                str(dataset_cfg.name),
                split,
            ],
            resume=bool(collect_cfg_node.operation.resume),
        )),
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
        ai2thor_gpu_device=int(dataset_cfg.ai2thor_gpu_device) if dataset_cfg.get("ai2thor_gpu_device") is not None else None,
        resume=bool(collect_cfg_node.operation.resume),
        merge_interval_episodes=int(collect_cfg_node.operation.get("merge_interval_episodes", 0)),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    dataset_cfg = cfg.dataset
    collect_cfg_node = dataset_cfg.collect

    splits = [
        ("train", dataset_cfg.train_scenes),
        ("val", dataset_cfg.val_scenes),
        ("test", dataset_cfg.test_scenes),
    ]

    manifests: dict[str, Path] = {}

    tracker = init_tracker(
        task_name="collect_data",
        config={"backend": str(dataset_cfg.backend), "dataset": str(dataset_cfg.name)},
    )

    for split, scenes in splits:
        if not scenes:
            info(f"skipping {split}: no scenes configured")
            continue

        info(f"=== Collecting {split} ({len(scenes)} scenes) ===")
        collect_cfg = _build_collect_cfg(dataset_cfg, collect_cfg_node, scenes, split)

        show_kv_table(
            f"Collect Config [{split}]",
            [
                ("backend", collect_cfg.backend),
                ("num_scenes", str(len(collect_cfg.scenes))),
                ("episodes_per_scene", str(collect_cfg.num_episodes_per_scene)),
                ("steps_per_episode", str(collect_cfg.max_steps_per_episode)),
                ("num_workers", str(collect_cfg.num_workers)),
                ("output_dir", collect_cfg.output_dir),
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
            task = progress.add_task(f"collect[{split}]", total=total_steps)
            if existing_steps > 0:
                progress.update(task, completed=existing_steps)
                info(f"  {split}: 断点续跑，已跳过 {existing_steps}/{total_steps} 条样本")

            completed_steps = existing_steps
            manager = mp.Manager()
            progress_queue = manager.Queue()
            stop_event = threading.Event()

            def _progress_watcher() -> None:
                nonlocal completed_steps
                while not stop_event.is_set():
                    try:
                        delta = progress_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if isinstance(delta, int) and delta > 0:
                        completed_steps = min(total_steps, completed_steps + delta)
                        progress.update(task, completed=completed_steps)

            watcher = threading.Thread(target=_progress_watcher, daemon=True)
            watcher.start()

            try:
                manifest = run_collection(
                    collect_cfg,
                    progress_queue=progress_queue,
                )
            finally:
                stop_event.set()
                watcher.join(timeout=1.0)

            while True:
                try:
                    delta = progress_queue.get_nowait()
                except queue.Empty:
                    break
                if isinstance(delta, int) and delta > 0:
                    completed_steps = min(total_steps, completed_steps + delta)
                    progress.update(task, completed=completed_steps)

            manager.shutdown()
            if not isinstance(manifest, Path):
                raise RuntimeError("run_collection 返回值异常：manifest 不是 Path 类型。")
            progress.update(task, completed=total_steps)

        manifests[split] = manifest

        tracker.log_metrics(
            {
                f"collect/{split}/scenes": len(collect_cfg.scenes),
                f"collect/{split}/episodes_per_scene": collect_cfg.num_episodes_per_scene,
                f"collect/{split}/steps": total_steps,
            }
        )

    tracker.finish()

    for split, manifest in manifests.items():
        info(f"  {split}: manifest={manifest}")
    success(f"采集完成，共 {len(manifests)} 个 split")


if __name__ == "__main__":
    main()

