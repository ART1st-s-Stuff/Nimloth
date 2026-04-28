"""WM + Vision Encoder 联合训练入口。

支持 Vision Encoder 在线编码，通过 LLM backbone 获取 hidden state。
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from src.data.eb_nav_dataset import EBNavSequenceDataset
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.wm.encoder.qwen import QwenLLMLatentEncoder
from src.wm.predictor.lewm import LeWMWorldModel
from src.utils.console import show_kv_table, success
from src.utils.env import load_project_env
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))

    train_cfg = cfg.pipeline.train
    wm_cfg = cfg.wm

    device = torch.device(str(train_cfg.device))

    # 构建 Vision Encoder
    model_name = str(getattr(wm_cfg.encoder, "model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))
    latent_dim = int(wm_cfg.latent_dim)
    num_patches = int(getattr(wm_cfg, "num_patches", 1))
    token_dim = int(getattr(wm_cfg, "token_dim", latent_dim))

    # 创建 Qwen Adapter
    qwen_adapter = QwenVLMAdapter(
        model_name=model_name,
        latent_dim=latent_dim,
        enabled=True,
        fallback_enabled=False,
    )
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")

    # 设置 Vision Encoder 可训练，LLM backbone 冻结
    qwen_adapter._set_llm_backbone_trainable(trainable=False)
    qwen_adapter._model.train()  # 不调用 .to(device)，accelerate 已处理

    # 创建 Vision Encoder wrapper
    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=latent_dim,
        qwen_adapter=qwen_adapter,
        use_vision_only=False,
        llm_backbone_trainable=False,
    )

    # 构建 LeWM World Model
    wm_model = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=3,
        hidden_dim=int(getattr(wm_cfg, "hidden_dim", 512)),
        history_len=int(getattr(wm_cfg, "history_len", 4)),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(getattr(wm_cfg.transformer, "num_layers", 6)),
        num_heads=int(getattr(wm_cfg.transformer, "num_heads", 16)),
        dim_head=int(getattr(wm_cfg.transformer, "dim_head", 64)),
        mlp_ratio=float(getattr(wm_cfg.transformer, "mlp_ratio", 4.0)),
        dropout=float(getattr(wm_cfg.transformer, "dropout", 0.1)),
        emb_dropout=float(getattr(wm_cfg.lewm, "emb_dropout", 0.0)),
        sigreg_enabled=bool(getattr(wm_cfg.lewm, "sigreg_enabled", False)),
        sigreg_latent_dim=int(getattr(wm_cfg.lewm, "sigreg_latent_dim", latent_dim)),
        sigreg_num_proj=int(getattr(wm_cfg.lewm, "sigreg_num_proj", 256)),
    )
    wm_model = wm_model.to(device)
    wm_model.train()

    # 优化器：Vision Encoder + WM
    lr = float(train_cfg.lr)
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 1000))

    # Vision Encoder 参数在 adapter._model 中
    trainable_params = list(qwen_adapter._model.parameters()) + list(wm_model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    if warmup_steps > 0:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))
    else:
        scheduler = None

    # 打印参数数量
    # Vision Encoder 参数在 adapter._model 中
    vision_params = sum(p.numel() for p in qwen_adapter._model.parameters() if p.requires_grad)
    wm_params = sum(p.numel() for p in wm_model.parameters() if p.requires_grad)

    show_kv_table("Joint Training Config", [
        ("model", model_name),
        ("latent_dim", str(latent_dim)),
        ("batch_size", str(int(train_cfg.batch_size))),
        ("epochs", str(int(train_cfg.epochs))),
        ("device", str(device)),
        ("vision_params", f"{vision_params:,}"),
        ("wm_params", f"{wm_params:,}"),
        ("total_params", f"{vision_params + wm_params:,}"),
    ])

    # 初始化 tracker
    tracker = init_tracker(
        task_name="train_wm_joint",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": lr,
            "model": model_name,
            "vision_params": vision_params,
            "wm_params": wm_params,
        },
    )

    # 数据集
    json_path = "datasets/EB-Nav/eb-nav_dataset_single_step.json"
    images_base_dir = "datasets/EB-Nav/images"

    dataset = EBNavSequenceDataset(
        json_path=json_path,
        images_base_dir=images_base_dir,
        latent_dim=latent_dim,
        action_dim=3,
        history_len=int(wm_cfg.history_len),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=0,  # Qwen 模型不支持多进程
    )

    show_kv_table("Dataset", [
        ("samples", str(len(dataset))),
        ("batch_size", str(int(train_cfg.batch_size))),
        ("steps_per_epoch", str(len(dataloader))),
    ])

    # 训练循环
    epochs = int(train_cfg.epochs)
    global_step = 0
    grad_clip = float(train_cfg.get("grad_clip_norm", 5.0))

    for epoch in range(epochs):
        wm_model.train()

        for batch_idx, batch in enumerate(dataloader):
            # Collate returns: [H, B] for images, [H, action_dim, B] for actions
            # Reorganize to [B, H] for images, [B, H, action_dim] for actions
            history_images_list = batch["history_images"]  # [H, B] each is list of paths
            batch_size = len(history_images_list[0])
            history_images = [[history_images_list[h][b] for h in range(len(history_images_list))] for b in range(batch_size)]

            # Actions: [H, action_dim, B] -> [B, H, action_dim]
            # Use float() to convert from float64 to float32
            history_actions = torch.stack([
                torch.stack(ha_step).float() for ha_step in batch["history_actions"]
            ]).permute(2, 0, 1).to(device)
            future_images_list = batch["future_images"]  # [T, B]
            future_images = [[future_images_list[t][b] for t in range(len(future_images_list))] for b in range(batch_size)]
            future_actions = torch.stack([
                torch.stack(fa_step).float() for fa_step in batch["future_actions"]
            ]).permute(2, 0, 1).to(device)

            optimizer.zero_grad()

            # 编码历史图像
            z_history_list = []
            for img_paths in history_images:
                batch_latents = []
                for path in img_paths:
                    if path and Path(path).exists():
                        latent = vision_encoder.encode_image_path(path).z.to(device)
                    else:
                        latent = torch.zeros(latent_dim, device=device)
                    batch_latents.append(latent)
                z_history_list.append(torch.stack(batch_latents))
            z_history = torch.stack(z_history_list)  # [B, H, D]

            # 编码未来图像
            z_future_list = []
            for img_paths in future_images:
                batch_latents = []
                for path in img_paths:
                    if path and Path(path).exists():
                        latent = vision_encoder.encode_image_path(path).z.to(device)
                    else:
                        latent = torch.zeros(latent_dim, device=device)
                    batch_latents.append(latent)
                z_future_list.append(torch.stack(batch_latents))
            z_future = torch.stack(z_future_list)  # [B, T, D]

            # Reshape: [B, H, D] → [B, H, 1, D] (num_patches=1)
            z_history = z_history.unsqueeze(2)
            z_future = z_future.unsqueeze(2)

            # WM 前向
            pred_z_next = wm_model(z_history, history_actions)

            # 简单 MSE loss: pred_z_next is [B, P, D], target is [B, P, D]
            target = z_future[:, -1, :, :]  # [B, P, D]
            loss = torch.nn.functional.mse_loss(pred_z_next, target)

            # 反传
            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # 记录
            if global_step % int(train_cfg.get("log_every_n_steps", 10)) == 0:
                log_dict = {
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch + 1,
                }
                tracker.log_metrics(log_dict, step=global_step)

            global_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        success(f"Epoch {epoch+1}/{epochs} 完成")

    tracker.finish()
    success("训练完成")


if __name__ == "__main__":
    main()
