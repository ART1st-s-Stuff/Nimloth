"""Phase 3 语义对齐训练入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_align import DeltaProjector, SemanticAlignModel
from src.vlm.semantic_state import SemanticStateGenerator
from src.wm.encoders import build_wm_image_encoder


def _collate_semantic_batch(batch: list[dict]) -> dict:
    return {
        "z_t": torch.stack([item["z_t"] for item in batch], dim=0),
        "z_t_pos": torch.stack([item["z_t_pos"] for item in batch], dim=0),
        "z_t_neg": torch.stack([item["z_t_neg"] for item in batch], dim=0),
        "image_path": [str(item["image_path"]) for item in batch],
        "pos_image_path": [str(item["pos_image_path"]) for item in batch],
        "task_text": [str(item["task_text"]) for item in batch],
        "env_context": [str(item["env_context"]) for item in batch],
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train.semantic_align
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    vlm_cfg = cfg.vlm
    device = torch.device(str(train_cfg.device))
    path_segments = [
        str(train_cfg.operation.outputs_root),
        "semantic_align",
        str(vlm_cfg.name),
    ]
    force_new_run = bool(getattr(train_cfg.operation, "force_new_run", False))
    from src.core import FileSystemModelProvider
    model_provider = FileSystemModelProvider(path_segments=path_segments)
    run_dir, resumed = model_provider.resolve_run_dir(force_new=force_new_run)
    tracker = init_tracker(
        task_name="train_semantic_align",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": float(train_cfg.lr),
            "positive_k": int(train_cfg.positive_k),
            "negative_gap": int(train_cfg.negative_gap),
            "temperature": float(train_cfg.temperature),
            "temporal_weight": float(train_cfg.temporal_weight),
            "use_vlm_for_st": bool(train_cfg.use_vlm_for_st),
            "run_resumed": resumed,
            "force_new_run": force_new_run,
        },
    )
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if image_encoder is None:
        raise RuntimeError("未启用 WM 图像编码器，无法构建 Phase 3 对齐数据。")
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)
    semantic_align_split = str(train_cfg.get("split", "train"))

    def _resolve_semantic_manifest_path(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=str(train_cfg.operation.outputs_root),
            dataset_name=str(dataset_cfg.name),
        )

    resolved_manifest = _resolve_semantic_manifest_path(semantic_align_split)
    from src.train.latent_cache import infer_latent_cache_path_from_manifest
    resolved_cache = infer_latent_cache_path_from_manifest(str(resolved_manifest), str(wm_cfg.name))
    dataset = SemanticAlignDataset(
        manifest_path=str(resolved_manifest),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=image_encoder,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        enable_cot_target=bool(train_cfg.enable_cot_target),
        latent_cache_path=str(resolved_cache) if resolved_cache else None,
    )
    if len(dataset) == 0:
        raise RuntimeError("SemanticAlignDataset 为空，请先完成数据采集。")
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=int(train_cfg.num_workers),
        collate_fn=_collate_semantic_batch,
    )
    vlm_model_cfg = vlm_cfg.get("model", None) or {}
    vlm_model_name = str(OmegaConf.select(vlm_cfg, "model.hf_model_name", default="Qwen/Qwen2.5-VL-7B-Instruct"))
    vlm_max_new_tokens = int(OmegaConf.select(vlm_cfg, "model.max_new_tokens", default=128))
    vlm_adapter = QwenVLMAdapter(
        model_name=vlm_model_name,
        latent_dim=int(wm_cfg.latent_dim),
        enabled=bool(vlm_cfg.get("enabled", False) and train_cfg.use_vlm_for_st),
        fallback_enabled=bool(vlm_cfg.get("fallback_enabled", True)),
        max_new_tokens=vlm_max_new_tokens,
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=vlm_adapter)
    projector = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=float(train_cfg.lr))
    align_model = SemanticAlignModel(
        projector=projector,
        semantic_generator=semantic_generator,
        optimizer=optimizer,
        device=device,
        temporal_weight=float(train_cfg.temporal_weight),
    )
    align_model._temperature = float(train_cfg.temperature)
    show_kv_table(
        "Train Semantic Align",
        [
            ("device", str(device)),
            ("dataset_size", str(len(dataset))),
            ("batch_size", str(train_cfg.batch_size)),
            ("epochs", str(train_cfg.epochs)),
            ("vlm_enabled", str(vlm_cfg.enabled and train_cfg.use_vlm_for_st)),
        ],
    )
    total_epochs = int(train_cfg.epochs)
    start_epoch = 0
    if resumed:
        checkpoint_state = model_provider.load_checkpoint(run_dir)
        if checkpoint_state is not None:
            align_model.load_state(checkpoint_state)
            start_epoch = int(checkpoint_state.get("epoch", -1)) + 1
    model_provider.mark_running(run_dir)
    try:
        with progress_context() as progress:
            task = progress.add_task("training_semantic_align", total=total_epochs * len(loader))
            for epoch in range(start_epoch, total_epochs):
                running_loss = 0.0
                running_nce = 0.0
                running_temporal = 0.0
                for batch_idx, batch in enumerate(loader, start=1):
                    step_metrics = align_model.train_step(batch)
                    running_loss += step_metrics["loss"]
                    running_nce += step_metrics["loss_nce"]
                    running_temporal += step_metrics["loss_temporal"]
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"epoch={epoch + 1}/{total_epochs} "
                            f"batch={batch_idx}/{len(loader)} loss={step_metrics['loss']:.6f}"
                        ),
                    )
                avg_loss = running_loss / max(1, len(loader))
                avg_nce = running_nce / max(1, len(loader))
                avg_temporal = running_temporal / max(1, len(loader))
                tracker.log_metrics(
                    {
                        "train/loss": avg_loss,
                        "train/loss_nce": avg_nce,
                        "train/loss_temporal": avg_temporal,
                        "train/epoch": epoch,
                    },
                    step=epoch,
                )
                model_provider.save_checkpoint(
                    run_dir=run_dir,
                    state={"epoch": epoch, **align_model.get_state()},
                )
        epoch_loss = avg_loss
        epoch_nce = avg_nce
        epoch_temporal = avg_temporal
        model_provider.mark_completed(run_dir)
    except Exception as exc:
        model_provider.mark_failed(run_dir, error=str(exc))
        raise

    out_dir = ensure_dir(run_dir)
    ckpt_path = Path(out_dir) / "semantic_projector.pt"
    torch.save(projector.state_dict(), ckpt_path)
    metrics_path = Path(out_dir) / "semantic_align_metrics.json"
    write_json(
        metrics_path,
        {
            "loss": epoch_loss,
            "loss_nce": epoch_nce,
            "loss_temporal": epoch_temporal,
            "vlm_init_error": vlm_adapter.init_error,
        },
    )
    tracker.log_artifact_path("semantic-align-projector", ckpt_path, artifact_type="model")
    tracker.log_artifact_path("semantic-align-metrics", metrics_path, artifact_type="metrics")
    tracker.finish()
    success(f"Phase3 训练完成: checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
