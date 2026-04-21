"""数据采集入口。"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.data.collector import CollectConfig, run_collection
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.project.seed))
    collect_cfg = CollectConfig(
        backend=str(cfg.data.env.backend),
        scene=str(cfg.data.env.scene),
        seed=int(cfg.project.seed),
        output_dir=str(cfg.data.collect.output_dir),
        num_episodes=int(cfg.data.collect.num_episodes),
        max_steps_per_episode=int(cfg.data.collect.max_steps_per_episode),
        image_height=int(cfg.data.collect.image_height),
        image_width=int(cfg.data.collect.image_width),
        visibility_distance=float(cfg.data.env.visibility_distance),
        grid_size=float(cfg.data.env.grid_size),
        render_depth_image=bool(cfg.data.env.render_depth_image),
        render_instance_segmentation=bool(cfg.data.env.render_instance_segmentation),
        ai2thor_platform=str(cfg.data.env.ai2thor_platform),
        ai2thor_cache_dir=str(cfg.data.env.ai2thor_cache_dir),
        fallback_to_mock_on_error=bool(cfg.data.env.fallback_to_mock_on_error),
    )
    manifest = run_collection(collect_cfg)
    print(f"[collect] done backend={collect_cfg.backend}, manifest={manifest}")


if __name__ == "__main__":
    main()

