"""数据采集入口。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.data.collector import CollectConfig, run_collection
from src.utils.console import info, progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    run_dir = build_run_output_dir(
        outputs_root=str(cfg.data.collect.operation.outputs_root),
        phase=str(cfg.data.collect.operation.phase),
        task=str(cfg.data.collect.operation.task),
    )
    tracker = init_tracker(task_name="collect_data", config={"backend": str(cfg.data.env.backend)})
    collect_cfg = CollectConfig(
        backend=str(cfg.data.env.backend),
        scenes=list(cfg.data.env.scenes),
        seed=int(cfg.project.seed),
        output_dir=str(run_dir),
        num_episodes_per_scene=int(cfg.data.collect.num_episodes_per_scene),
        max_steps_per_episode=int(cfg.data.collect.max_steps_per_episode),
        num_workers=int(cfg.data.collect.num_workers),
        image_height=int(cfg.data.collect.image_height),
        image_width=int(cfg.data.collect.image_width),
        visibility_distance=float(cfg.data.env.visibility_distance),
        grid_size=float(cfg.data.env.grid_size),
        render_depth_image=bool(cfg.data.env.render_depth_image),
        render_instance_segmentation=bool(cfg.data.env.render_instance_segmentation),
        ai2thor_platform=str(cfg.data.env.ai2thor_platform),
        ai2thor_cache_dir=str(cfg.data.env.ai2thor_cache_dir),
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
    with progress_context() as progress:
        task = progress.add_task("collecting", total=total_steps)
        manifest = run_collection(collect_cfg)
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

