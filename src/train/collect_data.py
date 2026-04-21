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
    collect_pipeline = cfg.pipeline.collect
    dataset_cfg = cfg.dataset
    run_dir = build_run_output_dir(
        path_segments=[
            str(collect_pipeline.collect.operation.outputs_root),
            str(dataset_cfg.name),
        ],
        resume=bool(collect_pipeline.collect.operation.resume),
    )
    tracker = init_tracker(task_name="collect_data", config={"backend": str(dataset_cfg.backend), "dataset": str(dataset_cfg.name)})
    collect_cfg = CollectConfig(
        backend=str(dataset_cfg.backend),
        scenes=list(dataset_cfg.scenes),
        seed=int(cfg.project.seed),
        output_dir=str(run_dir),
        num_episodes_per_scene=int(collect_pipeline.collect.num_episodes_per_scene),
        max_steps_per_episode=int(collect_pipeline.collect.max_steps_per_episode),
        num_workers=int(collect_pipeline.collect.num_workers),
        image_height=int(collect_pipeline.collect.image_height),
        image_width=int(collect_pipeline.collect.image_width),
        action_weights=[float(w) for w in collect_pipeline.collect.action_weights],
        visibility_distance=float(dataset_cfg.visibility_distance),
        grid_size=float(dataset_cfg.grid_size),
        render_depth_image=bool(dataset_cfg.render_depth_image),
        render_instance_segmentation=bool(dataset_cfg.render_instance_segmentation),
        ai2thor_platform=str(dataset_cfg.ai2thor_platform),
        ai2thor_cache_dir=str(dataset_cfg.ai2thor_cache_dir),
        resume=bool(collect_pipeline.collect.operation.resume),
    )
    show_kv_table(
        "Collect Config",
        [
            ("backend", collect_cfg.backend),
            ("num_scenes", str(len(collect_cfg.scenes))),
            ("episodes_per_scene", str(collect_cfg.num_episodes_per_scene)),
            ("steps_per_episode", str(collect_cfg.max_steps_per_episode)),
            ("num_workers", str(collect_cfg.num_workers)),
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

