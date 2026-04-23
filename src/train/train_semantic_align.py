"""Phase 3 语义对齐训练入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from src.data.semantic_dataset import SemanticAlignDataset
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker
from src.vlm.losses import info_nce_loss, temporal_consistency_loss
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.vlm.semantic_state import SemanticStateGenerator
from src.wm.encoders import build_wm_image_encoder


class DeltaProjector(nn.Module):
    """h(z_t, z_t+k) 轻量投影器。"""

    def __init__(self, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tp: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_tp], dim=-1))


def _collate_semantic_batch(batch: list[dict]) -> dict:
    z_t = torch.stack([item["z_t"] for item in batch], dim=0)
    z_t_pos = torch.stack([item["z_t_pos"] for item in batch], dim=0)
    z_t_neg = torch.stack([item["z_t_neg"] for item in batch], dim=0)
    return {
        "z_t": z_t,
        "z_t_pos": z_t_pos,
        "z_t_neg": z_t_neg,
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
    run_dir = build_run_output_dir(
        path_segments=[
            str(train_cfg.operation.outputs_root),
            "semantic_align",
            str(vlm_cfg.name),
        ],
    )
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
        },
    )
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if image_encoder is None:
        raise RuntimeError("未启用 WM 图像编码器，无法构建 Phase 3 对齐数据。")
    dataset = SemanticAlignDataset(
        manifest_path=str(dataset_cfg.manifest_path),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        image_encoder=image_encoder,
        positive_k=int(train_cfg.positive_k),
        negative_gap=int(train_cfg.negative_gap),
        enable_cot_target=bool(train_cfg.enable_cot_target),
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
    adapter = QwenVLMAdapter(
        model_name=str(vlm_cfg.model.hf_model_name),
        latent_dim=int(wm_cfg.latent_dim),
        enabled=bool(vlm_cfg.enabled and train_cfg.use_vlm_for_st),
        fallback_enabled=bool(vlm_cfg.fallback_enabled),
        max_new_tokens=int(vlm_cfg.model.max_new_tokens),
    )
    semantic_generator = SemanticStateGenerator(vlm_adapter=adapter)
    projector = DeltaProjector(latent_dim=int(wm_cfg.latent_dim), hidden_dim=int(wm_cfg.hidden_dim)).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=float(train_cfg.lr))
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
    projector.train()
    epoch_loss = 0.0
    epoch_nce = 0.0
    epoch_temporal = 0.0
    total_steps = max(1, int(train_cfg.epochs) * len(loader))
    with progress_context() as progress:
        task = progress.add_task("training_semantic_align", total=total_steps)
        for epoch in range(int(train_cfg.epochs)):
            running_loss = 0.0
            running_nce = 0.0
            running_temporal = 0.0
            for batch_idx, batch in enumerate(loader, start=1):
                z_t = batch["z_t"].to(device)
                z_t_pos = batch["z_t_pos"].to(device)
                z_t_neg = batch["z_t_neg"].to(device)
                s_t_list: list[torch.Tensor] = []
                s_tp1_list: list[torch.Tensor] = []
                for i in range(z_t.size(0)):
                    output = semantic_generator.infer(
                        image_path=batch["image_path"][i],
                        history_image_paths=[batch["image_path"][i]],
                        task_text=batch["task_text"][i],
                        env_context=batch["env_context"][i],
                    )
                    next_output = semantic_generator.infer(
                        image_path=batch["pos_image_path"][i],
                        history_image_paths=[batch["image_path"][i], batch["pos_image_path"][i]],
                        task_text=batch["task_text"][i],
                        env_context=batch["env_context"][i],
                    )
                    s_t_list.append(output.s_t)
                    s_tp1_list.append(next_output.s_t)
                s_t = torch.stack(s_t_list, dim=0).to(device)
                s_tp1 = torch.stack(s_tp1_list, dim=0).to(device)
                pred_positive = projector(z_t=z_t, z_tp=z_t_pos)
                pred_negative = projector(z_t=z_t, z_tp=z_t_neg)
                loss_nce = info_nce_loss(
                    anchor=s_t,
                    positive=pred_positive,
                    negatives=pred_negative,
                    temperature=float(train_cfg.temperature),
                )
                loss_temporal = temporal_consistency_loss(s_t=s_t, s_tp1=s_tp1)
                loss = loss_nce + float(train_cfg.temporal_weight) * loss_temporal
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                running_loss += float(loss.item())
                running_nce += float(loss_nce.item())
                running_temporal += float(loss_temporal.item())
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"epoch={epoch + 1}/{int(train_cfg.epochs)} "
                        f"batch={batch_idx}/{len(loader)} loss={float(loss.item()):.6f}"
                    ),
                )
            epoch_loss = running_loss / max(1, len(loader))
            epoch_nce = running_nce / max(1, len(loader))
            epoch_temporal = running_temporal / max(1, len(loader))
            tracker.log_metrics(
                {
                    "train/loss": epoch_loss,
                    "train/loss_nce": epoch_nce,
                    "train/loss_temporal": epoch_temporal,
                    "train/epoch": epoch,
                },
                step=epoch,
            )
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
            "vlm_init_error": adapter.init_error,
        },
    )
    tracker.log_artifact_path("semantic-align-projector", ckpt_path, artifact_type="model")
    tracker.log_artifact_path("semantic-align-metrics", metrics_path, artifact_type="metrics")
    tracker.finish()
    success(f"Phase3 训练完成: checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
