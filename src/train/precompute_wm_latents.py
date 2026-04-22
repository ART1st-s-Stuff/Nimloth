"""WM latent 预编码入口。"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.data.dataset import WMDataset
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.wm.encoders import build_wm_image_encoder


def _resolve_manifest_path(manifest_path: str) -> Path:
    candidate = Path(manifest_path)
    if candidate.exists():
        return candidate
    parts = candidate.parts
    if len(parts) >= 3 and parts[-2] == "latest":
        group_dir = Path(*parts[:-2])
        meta_path = group_dir / "metadata.json"
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            latest = metadata.get("latest")
            if isinstance(latest, str):
                latest_path = group_dir / latest / parts[-1]
                if latest_path.exists():
                    return latest_path
    return candidate


def _build_latent_cache_path(manifest_path: Path, wm_name: str) -> Path:
    stem = manifest_path.stem
    return manifest_path.parent / f"{stem}.latents.{wm_name}.pt"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    resolved_manifest_path = _resolve_manifest_path(str(dataset_cfg.manifest_path))
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if image_encoder is None:
        raise RuntimeError("当前 WM 配置未启用图像编码器，无需执行 latent 预编码。")
    latent_cache_path = _build_latent_cache_path(resolved_manifest_path, str(wm_cfg.name))
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

        dataset = WMDataset(
            manifest_path=str(resolved_manifest_path),
            latent_dim=int(dataset_cfg.latent_dim),
            action_dim=int(dataset_cfg.action_dim),
            history_len=int(wm_cfg.history_len),
            image_encoder=image_encoder,
            latent_cache_path=str(latent_cache_path),
            encoder_num_workers=int(train_cfg.encoder_num_workers),
            encoder_batch_size=int(train_cfg.encoder_batch_size),
            on_latent_progress=_on_latent_progress,
        )
    success(f"latent 预编码完成，缓存文件: {latent_cache_path}，样本数: {len(dataset)}")


if __name__ == "__main__":
    main()

