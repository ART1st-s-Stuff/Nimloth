"""WM latent 预编码入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.train.latent_cache import build_latent_cache_path, build_wm_dataset_with_cache
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.wm import resolve_patch_layout
from src.wm.encoder import build_wm_image_encoder


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    num_patches, token_dim = resolve_patch_layout(wm_cfg=wm_cfg, allow_zero=True)
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)
    precompute_split = str(train_cfg.get("precompute_split", "train"))

    def _resolve_precompute_manifest_path(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=str(train_cfg.operation.outputs_root),
            dataset_name=str(dataset_cfg.name),
        )

    resolved_manifest_path = _resolve_precompute_manifest_path(precompute_split)
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if image_encoder is None:
        raise RuntimeError("当前 WM 配置未启用图像编码器，无需执行 latent 预编码。")
    latent_cache_path = build_latent_cache_path(resolved_manifest_path, str(wm_cfg.name))
    show_kv_table(
        "WM Latent Precompute",
        [
            ("manifest", str(resolved_manifest_path)),
            ("wm", str(wm_cfg.name)),
            ("encoder_workers", str(int(train_cfg.encoder_num_workers))),
            ("latent_cache", str(latent_cache_path)),
        ],
    )
    with progress_context() as progress:
        task = progress.add_task("precompute_latents", total=1)

        def _on_latent_progress(done: int, total: int) -> None:
            if total <= 0:
                progress.update(task, total=1, completed=1, description="latent_cache 已命中")
                return
            progress.update(
                task,
                total=total,
                completed=done,
                description=f"precompute_latents {done}/{total}",
            )

        dataset, latent_cache_path = build_wm_dataset_with_cache(
            run_dir=resolved_manifest_path,
            wm_name=str(wm_cfg.name),
            latent_dim=int(wm_cfg.latent_dim),
            action_dim=int(dataset_cfg.action_dim),
            history_len=int(wm_cfg.history_len),
            temporal_stride=1,
            image_encoder=image_encoder,
            encoder_num_workers=int(train_cfg.encoder_num_workers),
            encoder_batch_size=int(train_cfg.encoder_batch_size),
            expected_num_patches=num_patches,
            expected_token_dim=token_dim,
            on_latent_progress=_on_latent_progress,
        )
    success(f"latent 预编码完成，缓存文件: {latent_cache_path}，样本数: {len(dataset)}")


if __name__ == "__main__":
    main()

