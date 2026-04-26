#!/usr/bin/env python3
"""LeWM Phase2 测试脚本 - 复用 src/train 训练流程。

用法：
    # 完整运行（训练 + 可视化）
    python dev/test_lewm_phase2.py --run-all --models dinov2

    # 仅训练
    python dev/test_lewm_phase2.py --train-only --models dinov2 --num-rollouts 100 --train-epochs 2

    # 仅可视化（使用已训练的模型）
    python dev/test_lewm_phase2.py --visualize-only --model-dir outputs/dev/wm/lewm_dinov2m/2026-05-01_12-00-00

可选参数：
    --models: 模型列表，默认 dinov2
    --num-rollouts: 训练使用的 rollout 数量，默认 1000
    --train-epochs: 训练 epoch 数，默认 1
    --num-test-rollouts: 测试 rollout 数量，默认 3
    --num-steps: 测试轨迹步数，默认 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataset import Dataset

import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import wandb

# 设置项目根目录
project_root = Path(__file__).parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.data.dataset import WMDataset, read_worker_manifests
from src.train.latent_cache import build_wm_dataset_with_cache
from src.wm.encoder import build_wm_image_encoder
from src.wm.predictor.lewm import LeWMModel, LeWMWorldModel
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.action_mapper import build_action_mapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUTS_ROOT = "outputs/dev"


class SubsetSampler(Sampler[int]):
    """从 WMDataset 中采样指定数量的 episode/rollout。

    WMDataset 的每个训练样本对应 (episode, history_position) 组合。
    我们需要先识别所有唯一的 episode，然后采样指定数量的 episode，
    最后返回属于这些 episode 的所有训练样本索引。
    """

    def __init__(
        self,
        dataset: WMDataset,
        max_rollouts: int,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.seed = seed

        # 检查必要属性
        if not hasattr(dataset, '_sample_episode_map'):
            raise TypeError(f"dataset 必须有 _sample_episode_map 属性，类型是 {type(dataset).__name__}")
        if not hasattr(dataset, '_training_indices'):
            raise TypeError(f"dataset 必须有 _training_indices 属性，类型是 {type(dataset).__name__}")

        # 构建 episode_key -> 样本索引集合 的映射
        # _sample_episode_map 现在存储完整的 episode_key（字符串格式）
        episode_to_sample_indices: dict[str, list[int]] = {}
        for idx in range(len(dataset.samples)):
            episode_key = dataset._sample_episode_map[idx]
            if episode_key not in episode_to_sample_indices:
                episode_to_sample_indices[episode_key] = []
            episode_to_sample_indices[episode_key].append(idx)

        # 获取所有唯一的 episode
        all_episodes = sorted(episode_to_sample_indices.keys())
        rng = random.Random(seed)

        # 采样 episode
        if max_rollouts <= 0 or max_rollouts >= len(all_episodes):
            selected_episodes = all_episodes
        else:
            selected_episodes = rng.sample(all_episodes, max_rollouts)

        # 收集选中 episode 的所有样本索引
        selected_sample_indices: set[int] = set()
        for ep_id in selected_episodes:
            selected_sample_indices.update(episode_to_sample_indices[ep_id])

        # 构建 sample_index -> training_indices 的映射
        sample_to_training_map: dict[int, list[int]] = {}
        for ti, idx_data in enumerate(dataset._training_indices):
            repr_idx = idx_data["history_indices"][0]
            if repr_idx not in sample_to_training_map:
                sample_to_training_map[repr_idx] = []
            sample_to_training_map[repr_idx].append(ti)

        # 选择属于选中 episode 的 training 样本
        selected_training_indices: list[int] = []
        for repr_idx, training_indices in sample_to_training_map.items():
            if repr_idx in selected_sample_indices:
                selected_training_indices.extend(training_indices)

        selected_training_indices.sort()
        self.indices = selected_training_indices

        logger.info(
            f"SubsetSampler: 从 {len(all_episodes)} 个 episode 中选取 {len(selected_episodes)} 个，"
            f"共 {len(self.indices)} 个训练样本"
        )

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def load_wm_dataset(
    run_dir: Path,
    wm_cfg: Any,
    dataset_cfg: Any,
    temporal_stride: int = 1,
    num_workers: int = 0,
) -> tuple[WMDataset, Path]:
    """加载 WM 数据集。"""
    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    dataset, cache_dir = build_wm_dataset_with_cache(
        run_dir=run_dir,
        wm_name=str(wm_cfg.name),
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        temporal_stride=temporal_stride,
        image_encoder=encoder,
        encoder_num_workers=num_workers,
        encoder_batch_size=16,
        expected_num_patches=int(getattr(wm_cfg.encoder, "num_patches", 0)),
        expected_token_dim=(
            int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 1))
            if int(getattr(wm_cfg.encoder, "num_patches", 0)) > 0
            else 0
        ),
        lazy_mode=False,  # 预编码模式
    )
    dataset.disable_encoder_after_warmup()
    return dataset, cache_dir


def build_lewm_model(
    wm_cfg: Any,
    dataset_cfg: Any,
    train_cfg: Any | None = None,
    device: torch.device | None = None,
) -> LeWMModel:
    """构建 LeWM 模型。

    Args:
        wm_cfg: WM 配置
        dataset_cfg: 数据集配置
        train_cfg: 训练配置（从 configs/pipeline/train/default.yaml 读取）
        device: 设备
    """
    num_patches = int(getattr(wm_cfg.encoder, "num_patches", 16))
    latent_dim = int(wm_cfg.latent_dim)
    token_dim = latent_dim // num_patches

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 从训练配置读取参数
    training_mode = "unsupervised"
    reconstruction_weight = 1.0
    semi_supervised_weight = 0.0
    grad_clip_norm = 5.0
    ema_decay = 0.999
    detach_idm_in_wm = True
    sigreg_enabled = False
    sigreg_target_weight = 0.0
    sigreg_warmup_steps = 0
    wm_lr = 1e-4
    idm_lr = 1e-4

    if train_cfg is not None:
        training_mode = str(getattr(train_cfg, "training_mode", "unsupervised"))
        reconstruction_weight = float(getattr(train_cfg, "reconstruction_weight", 1.0))
        semi_supervised_weight = float(getattr(train_cfg, "semi_supervised_weight", 0.0))
        grad_clip_norm = float(getattr(train_cfg, "grad_clip_norm", 5.0))
        ema_decay = float(getattr(train_cfg, "ema", {}).get("decay", 0.999))
        detach_idm_in_wm = bool(getattr(train_cfg, "detach_idm_in_wm", True))
        sigreg_enabled = bool(getattr(train_cfg, "sigreg", {}).get("enabled", False))
        sigreg_target_weight = float(getattr(train_cfg, "sigreg", {}).get("weight", 0.0))
        sigreg_warmup_steps = int(getattr(train_cfg, "sigreg", {}).get("warmup_steps", 0))
        wm_lr = float(getattr(train_cfg, "wm_lr", None) or getattr(train_cfg, "lr", 1e-4))
        idm_lr = float(getattr(train_cfg, "idm_lr", None) or getattr(train_cfg, "lr", 1e-4))

    # 构建 LeWM World Model
    wm_module = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.transformer.num_layers),
        num_heads=int(wm_cfg.transformer.num_heads),
        dim_head=int(wm_cfg.transformer.get("dim_head", 64)),
        mlp_ratio=float(wm_cfg.transformer.get("mlp_ratio", 4.0)),
        dropout=float(wm_cfg.transformer.get("dropout", 0.1)),
        emb_dropout=float(wm_cfg.lewm.get("emb_dropout", 0.0)),
        sigreg_enabled=sigreg_enabled,
    )

    inverse_dynamics = InverseDynamicsModel(
        latent_dim=latent_dim,
        action_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.inverse_dynamics.num_layers),
        num_heads=int(wm_cfg.inverse_dynamics.num_heads),
        dropout=float(wm_cfg.inverse_dynamics.dropout),
    )

    action_mapper = build_action_mapper(
        input_dim=int(dataset_cfg.action_dim),
        output_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
    )

    wm_optimizer = torch.optim.AdamW(wm_module.parameters(), lr=wm_lr, weight_decay=0.01)
    idm_optimizer = torch.optim.AdamW(
        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
        lr=idm_lr,
        weight_decay=0.01,
    )
    from torch.optim.lr_scheduler import LambdaLR
    wm_scheduler = LambdaLR(wm_optimizer, lr_lambda=lambda step: 1.0)
    idm_scheduler = LambdaLR(idm_optimizer, lr_lambda=lambda step: 1.0)

    wm_model = LeWMModel(
        wm=wm_module,
        inverse_dynamics=inverse_dynamics,
        action_mapper=action_mapper,
        wm_optimizer=wm_optimizer,
        idm_optimizer=idm_optimizer,
        wm_scheduler=wm_scheduler,
        idm_scheduler=idm_scheduler,
        device=device,
        training_mode=training_mode,
        reconstruction_weight=reconstruction_weight,
        semi_supervised_weight=semi_supervised_weight,
        grad_clip_norm=grad_clip_norm,
        ema_decay=ema_decay,
        detach_idm_in_wm=detach_idm_in_wm,
        sigreg_enabled=sigreg_enabled,
        sigreg_target_weight=sigreg_target_weight,
        sigreg_warmup_steps=sigreg_warmup_steps,
    )
    return wm_model


def train_model(
    model: LeWMModel,
    train_loader: DataLoader,
    num_epochs: int = 1,
    log_every_n_steps: int = 10,
    wandb_run=None,
) -> dict[str, list[float]]:
    """训练模型。"""
    metrics = {
        "loss": [],
        "loss_recon": [],
        "loss_action": [],
        "loss_sigreg": [],
    }

    model.wm.train()
    model.idm.train()
    model.action_mapper.train()

    total_batches = len(train_loader)
    device = model.device
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            batch_device = {
                "z_history": batch["z_history"].to(device),
                "action_history": batch["action_history"].to(device),
                "z_future": batch["z_future"].to(device),
                "gt_action_future": batch["gt_action_future"].to(device),
            }

            step_metrics = model.train_step(batch_device)

            batch_loss = step_metrics.get("loss", 0.0)
            batch_recon = step_metrics.get("loss_recon", 0.0)

            epoch_loss += batch_loss
            epoch_recon += batch_recon
            batch_count += 1
            global_step += 1

            if log_every_n_steps > 0 and batch_idx % log_every_n_steps == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{total_batches} | "
                    f"loss={batch_loss:.6f} recon={batch_recon:.6f}"
                )

                # 记录到 wandb
                if wandb_run is not None:
                    wandb_run.log({
                        "train/step": global_step,
                        "train/loss": batch_loss,
                        "train/loss_recon": batch_recon,
                        "train/epoch": epoch + 1,
                        "train/batch_idx": batch_idx + 1,
                    })

            metrics["loss"].append(batch_loss)
            metrics["loss_recon"].append(batch_recon)
            metrics["loss_action"].append(step_metrics.get("loss_action", 0.0))
            metrics["loss_sigreg"].append(step_metrics.get("loss_sigreg", 0.0))

        avg_loss = epoch_loss / max(1, batch_count)
        avg_recon = epoch_recon / max(1, batch_count)
        logger.info(f"Epoch {epoch+1} 完成: avg_loss={avg_loss:.6f} recon={avg_recon:.6f}")

    return metrics


def predict_trajectories(
    wm_module: torch.nn.Module,
    latents: list[torch.Tensor],
    actions: list[torch.Tensor],
    history_len: int,
    num_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """使用模型预测轨迹。"""
    real_trajectory: list[torch.Tensor] = []
    pred_trajectory: list[torch.Tensor] = []

    if len(latents) < history_len + 1:
        logger.warning("latent 序列太短，无法进行轨迹预测")
        return [], []

    # 初始化 history window
    history_z = torch.stack(latents[:history_len], dim=0).unsqueeze(0).to(device)  # [1, H, P, D]
    history_action = torch.zeros(1, history_len, actions[0].size(-1), dtype=torch.float32, device=device)

    wm_module.eval()
    with torch.no_grad():
        for step_idx in range(num_steps):
            state_index = history_len + step_idx
            if state_index >= len(latents):
                break

            # 真实 latent
            real_z = latents[state_index].to(device)
            real_trajectory.append(real_z.cpu())

            # 更新 action history（使用真实 action）
            if step_idx < len(actions):
                new_action = actions[step_idx].to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, A]
                history_action = torch.cat([history_action[:, 1:, :], new_action], dim=1)

            # 预测
            pred_z = wm_module.predict_next(history_z, history_action).squeeze(0)
            pred_trajectory.append(pred_z.cpu())

            # 更新 history window（使用真实 latent 校准）
            if state_index < len(latents) - 1:
                next_z = latents[state_index + 1] if state_index + 1 < len(latents) else latents[-1]
                history_z = torch.cat(
                    [history_z[:, 1:, ...], next_z.to(device).unsqueeze(0).unsqueeze(1)],
                    dim=1
                )

    return real_trajectory, pred_trajectory


def compute_umap_3d(points: list[torch.Tensor]) -> list[list[float]]:
    """使用 UMAP 将高维 latent 降维到 3D。"""
    try:
        import umap
    except ImportError:
        logger.warning("UMAP 未安装，跳过降维")
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points[:100]]]

    if len(points) < 3:
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points]]

    arr = torch.stack([p.flatten() for p in points], dim=0).numpy()
    n_neighbors = min(15, max(2, len(points) - 1))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
    try:
        embedded = reducer.fit_transform(arr.astype(np.float32))
        return embedded.tolist()
    except Exception as e:
        logger.warning(f"UMAP 计算失败: {e}")
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points]]


def generate_single_rollout_figure(
    real_coords: np.ndarray,
    pred_coords: np.ndarray,
    real_trajectory: list[torch.Tensor],
    pred_trajectory: list[torch.Tensor],
    scene: str,
    rollout_idx: int,
    avg_mse: float,
    output_path: Path,
) -> None:
    """生成单条 rollout 的可视化图。"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 计算每一步的 MSE
    mse_per_step = []
    for rz, pz in zip(real_trajectory, pred_trajectory):
        mse = torch.mean((rz - pz) ** 2).item()
        mse_per_step.append(mse)

    # 找出差距最大的 5 个索引
    top5_indices = sorted(range(len(mse_per_step)), key=lambda i: mse_per_step[i], reverse=True)[:5]

    # 绘制 ground truth 轨迹
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.scatter(real_coords[:1, 0], real_coords[:1, 1], real_coords[:1, 2],
                   c='blue', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(real_coords[-1:, 0], real_coords[-1:, 1], real_coords[-1:, 2],
                   c='blue', s=150, marker='^', label='End', zorder=5)
        # 标注时间步
        for i in range(0, len(real_coords), max(1, len(real_coords) // 8)):
            ax.text(real_coords[i, 0], real_coords[i, 1], real_coords[i, 2],
                    f'T{i}', fontsize=8, color='blue', fontweight='bold')

    # 绘制预测轨迹
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter(pred_coords[:1, 0], pred_coords[:1, 1], pred_coords[:1, 2],
                   c='red', s=150, marker='o', label='Pred Start', zorder=5)
        ax.scatter(pred_coords[-1:, 0], pred_coords[-1:, 1], pred_coords[-1:, 2],
                   c='red', s=150, marker='^', label='Pred End', zorder=5)
        # 标注时间步
        for i in range(0, len(pred_coords), max(1, len(pred_coords) // 8)):
            ax.text(pred_coords[i, 0], pred_coords[i, 1], pred_coords[i, 2],
                    f'T{i}', fontsize=8, color='red', fontweight='bold')

        # 绘制差距最大的 5 组连接线（灰色细线）
        for idx in top5_indices:
            if idx < len(real_coords) and idx < len(pred_coords):
                ax.plot([real_coords[idx, 0], pred_coords[idx, 0]],
                        [real_coords[idx, 1], pred_coords[idx, 1]],
                        [real_coords[idx, 2], pred_coords[idx, 2]],
                        'gray', linewidth=1, linestyle=':', alpha=0.8)
                mid_x = (real_coords[idx, 0] + pred_coords[idx, 0]) / 2
                mid_y = (real_coords[idx, 1] + pred_coords[idx, 1]) / 2
                mid_z = (real_coords[idx, 2] + pred_coords[idx, 2]) / 2
                ax.text(mid_x, mid_y, mid_z, f'Δ{mse_per_step[idx]:.2f}',
                        fontsize=7, color='gray', fontweight='bold')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title(f'Rollout {rollout_idx + 1} - {scene} (MSE: {avg_mse:.4f}) | Gray lines = top 5 errors')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_visualization_images(
    real_trajectory: list[torch.Tensor],
    pred_trajectory: list[torch.Tensor],
    metadata_list: list[dict[str, Any]],
    avg_mse: float,
    output_dir: Path,
) -> dict[str, Path]:
    """生成轨迹可视化图片。兼容单条和多条 rollout。"""

    images = {}

    if len(real_trajectory) == 0:
        logger.warning("轨迹为空，跳过可视化图片生成")
        return images

    # 计算 UMAP 降维
    all_points = real_trajectory + pred_trajectory
    embedded = compute_umap_3d(all_points)
    mid = len(real_trajectory)

    # ========== 3D 轨迹对比图 ==========
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    real_coords = np.array(embedded[:mid])
    pred_coords = np.array(embedded[mid:])

    # 计算每一步的差距，找出差距最大的5组
    mse_per_step = []
    for rz, pz in zip(real_trajectory, pred_trajectory):
        mse = torch.mean((rz - pz) ** 2).item()
        mse_per_step.append(mse)

    # 找出差距最大的5个索引
    top5_indices = sorted(range(len(mse_per_step)), key=lambda i: mse_per_step[i], reverse=True)[:5]

    # 绘制 ground truth 轨迹
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.scatter(real_coords[:1, 0], real_coords[:1, 1], real_coords[:1, 2],
                   c='blue', s=150, marker='o', label='Start', zorder=5)
        ax.scatter(real_coords[-1:, 0], real_coords[-1:, 1], real_coords[-1:, 2],
                   c='blue', s=150, marker='^', label='End', zorder=5)
        # 标注时间步
        for i in range(0, len(real_coords), max(1, len(real_coords) // 10)):
            ax.text(real_coords[i, 0], real_coords[i, 1], real_coords[i, 2],
                    f'T{i}', fontsize=8, color='blue', fontweight='bold')

    # 绘制预测轨迹
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter(pred_coords[:1, 0], pred_coords[:1, 1], pred_coords[:1, 2],
                   c='red', s=150, marker='o', label='Pred Start', zorder=5)
        ax.scatter(pred_coords[-1:, 0], pred_coords[-1:, 1], pred_coords[-1:, 2],
                   c='red', s=150, marker='^', label='Pred End', zorder=5)
        # 标注时间步（预测的每一步对应真实轨迹的同一步）
        for i in range(0, len(pred_coords), max(1, len(pred_coords) // 10)):
            ax.text(pred_coords[i, 0], pred_coords[i, 1], pred_coords[i, 2],
                    f'T{i}', fontsize=8, color='red', fontweight='bold')

        # 绘制差距最大的5组连接线（灰色细线）
        for idx in top5_indices:
            if idx < len(real_coords) and idx < len(pred_coords):
                ax.plot([real_coords[idx, 0], pred_coords[idx, 0]],
                        [real_coords[idx, 1], pred_coords[idx, 1]],
                        [real_coords[idx, 2], pred_coords[idx, 2]],
                        'gray', linewidth=1, linestyle=':', alpha=0.8)
                # 在中间位置标注误差值
                mid_x = (real_coords[idx, 0] + pred_coords[idx, 0]) / 2
                mid_y = (real_coords[idx, 1] + pred_coords[idx, 1]) / 2
                mid_z = (real_coords[idx, 2] + pred_coords[idx, 2]) / 2
                ax.text(mid_x, mid_y, mid_z, f'Δ{mse_per_step[idx]:.2f}',
                        fontsize=7, color='gray', fontweight='bold')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title(f'Trajectory Comparison (MSE: {avg_mse:.4f}) | Gray lines = top 5 errors')
    ax.legend(loc='upper left')

    trajectory_3d_path = output_dir / "trajectory_3d.png"
    plt.tight_layout()
    plt.savefig(trajectory_3d_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    images["trajectory_3d"] = trajectory_3d_path

    # ========== 2D 轨迹投影图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # XY 投影
    ax = axes[0]
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 0], real_coords[:, 1], 'b-', linewidth=2, label='GT', alpha=0.8)
        for i in range(0, len(real_coords), max(1, len(real_coords) // 8)):
            ax.annotate(f'T{i}', (real_coords[i, 0], real_coords[i, 1]), fontsize=7, color='blue')
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], 'r--', linewidth=2, label='Pred', alpha=0.8)
        for i in range(0, len(pred_coords), max(1, len(pred_coords) // 8)):
            ax.annotate(f'T{i}', (pred_coords[i, 0], pred_coords[i, 1]), fontsize=7, color='red')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('XY Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # XZ 投影
    ax = axes[1]
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 0], real_coords[:, 2], 'b-', linewidth=2, label='GT', alpha=0.8)
        for i in range(0, len(real_coords), max(1, len(real_coords) // 8)):
            ax.annotate(f'T{i}', (real_coords[i, 0], real_coords[i, 2]), fontsize=7, color='blue')
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 0], pred_coords[:, 2], 'r--', linewidth=2, label='Pred', alpha=0.8)
        for i in range(0, len(pred_coords), max(1, len(pred_coords) // 8)):
            ax.annotate(f'T{i}', (pred_coords[i, 0], pred_coords[i, 2]), fontsize=7, color='red')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 3')
    ax.set_title('XZ Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # YZ 投影
    ax = axes[2]
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 1], real_coords[:, 2], 'b-', linewidth=2, label='GT', alpha=0.8)
        for i in range(0, len(real_coords), max(1, len(real_coords) // 8)):
            ax.annotate(f'T{i}', (real_coords[i, 1], real_coords[i, 2]), fontsize=7, color='blue')
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 1], pred_coords[:, 2], 'r--', linewidth=2, label='Pred', alpha=0.8)
        for i in range(0, len(pred_coords), max(1, len(pred_coords) // 8)):
            ax.annotate(f'T{i}', (pred_coords[i, 1], pred_coords[i, 2]), fontsize=7, color='red')
    ax.set_xlabel('UMAP 2')
    ax.set_ylabel('UMAP 3')
    ax.set_title('YZ Projection')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Trajectory 2D Projections (MSE: {avg_mse:.4f})')
    trajectory_2d_path = output_dir / "trajectory_2d.png"
    plt.tight_layout()
    plt.savefig(trajectory_2d_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    images["trajectory_2d"] = trajectory_2d_path

    # ========== 逐步 MSE 曲线 ==========
    if len(real_trajectory) > 0 and len(pred_trajectory) > 0:
        mse_per_step = []
        for rz, pz in zip(real_trajectory, pred_trajectory):
            mse = torch.mean((rz - pz) ** 2).item()
            mse_per_step.append(mse)

        fig, ax = plt.subplots(figsize=(10, 5))
        steps = range(len(mse_per_step))
        ax.plot(steps, mse_per_step, 'b-', linewidth=2, label='Step MSE')
        ax.axhline(y=avg_mse, color='r', linestyle='--', linewidth=2, label=f'Avg MSE: {avg_mse:.4f}')
        ax.fill_between(steps, mse_per_step, alpha=0.3)
        ax.set_xlabel('Step')
        ax.set_ylabel('MSE')
        ax.set_title('Prediction Error per Step')
        ax.legend()
        ax.grid(True, alpha=0.3)

        loss_curve_path = output_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        images["loss_curve"] = loss_curve_path

    # ========== Ground Truth vs Predicted 对比图 ==========
    if len(embedded) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for i, coords in enumerate(embedded):
            color = 'blue' if i < mid else 'red'
            alpha = 0.6 if i < mid else 0.4
            ax.scatter(coords[0], coords[1], coords[2], c=color, alpha=alpha, s=50)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3', labelpad=10)
        ax.set_title('Ground Truth (blue) vs Predicted (red)')
        ax.view_init(elev=30, azim=45)

        comparison_path = output_dir / "trajectory_comparison.png"
        plt.tight_layout()
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        images["comparison"] = comparison_path

    return images


def load_test_data(
    run_dir: Path,
    wm_cfg: Any,
    num_rollouts: int = 3,
    num_steps: int = 50,
    seed: int = 42,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[dict[str, Any]]]:
    """加载测试数据并编码为 latents。

    Returns:
        (latents, actions, metadata_list)
        - latents: [T, P, D] 张量列表
        - actions: [T, A] 张量列表
        - metadata_list: 每步的元数据
    """
    import re
    from src.wm.encoder import build_wm_image_encoder

    # 读取所有 worker manifest 文件
    worker_files: list[Path] = []
    for p in run_dir.iterdir():
        if p.is_file() and p.suffix == ".jsonl" and re.match(r"^manifest_worker_\d+_.+\.jsonl$", p.name):
            worker_files.append(p)

    all_samples: list[dict[str, Any]] = []
    for worker_path in sorted(worker_files):
        for line in worker_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                all_samples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # 按 episode 分组
    episode_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for idx, sample in enumerate(all_samples):
        metadata = sample.get("metadata", {})
        scene = metadata.get("scene", "unknown")
        episode_id = int(sample.get("episode_id", -1))
        key = f"{scene}_{episode_id}"
        episode_groups.setdefault(key, []).append((idx, sample))

    # 每个 episode 按 step_id 排序
    for key in episode_groups:
        episode_groups[key].sort(key=lambda x: int(x[1].get("step_id", -1)))

    # 随机选择指定数量的 episode
    rng = random.Random(seed)
    episode_keys = sorted(episode_groups.keys())
    selected_keys = rng.sample(episode_keys, min(num_rollouts, len(episode_keys)))

    # 构建 encoder
    encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    if encoder is None:
        raise RuntimeError("无法构建 WM encoder")

    # 构建 latent 和 action 序列
    latents: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    metadata_list: list[dict[str, Any]] = []

    with torch.no_grad():
        for episode_key in selected_keys:
            episode_samples = episode_groups[episode_key]
            for step_idx in range(min(num_steps, len(episode_samples))):
                _, sample = episode_samples[step_idx]
                image_path = str(sample.get("image_path", ""))
                if not image_path:
                    continue

                # 编码图像
                try:
                    enc_output = encoder.encode_image_path(image_path)
                    z = enc_output.z.float().cpu()
                except Exception as e:
                    logger.warning(f"编码失败: {image_path}, error={e}")
                    continue

                # 构建 action 向量
                move = float(sample.get("move_ahead_distance", 0.0))
                yaw = float(sample.get("delta_yaw", 0.0))
                pitch = float(sample.get("delta_pitch", 0.0))
                action = torch.tensor([move, yaw, pitch], dtype=torch.float32)

                latents.append(z)
                actions.append(action)
                metadata_list.append({
                    "episode_key": episode_key,
                    "step_id": step_idx,
                    "scene": sample.get("metadata", {}).get("scene", "unknown"),
                    "episode_id": sample.get("episode_id", -1),
                })

    logger.info(f"加载测试数据: {len(selected_keys)} 个 episode, {len(latents)} 步")
    return latents, actions, metadata_list


def run_training(
    wm_cfg: Any,
    dataset_cfg: Any,
    num_rollouts: int,
    train_epochs: int,
    outputs_root: str = OUTPUTS_ROOT,
    use_wandb: bool = True,
) -> tuple[Path | None, dict[str, Any]]:
    """运行训练。"""
    from src.utils.env import get_env, load_project_env

    # 加载 .env 配置
    load_project_env()

    # 加载训练配置
    train_cfg = OmegaConf.load(Path("configs/pipeline/train/default.yaml"))

    wm_name = str(wm_cfg.name)

    logger.info(f"=" * 60)
    logger.info(f"开始训练: {wm_name}")
    logger.info(f"训练模式: {train_cfg.training_mode}")
    logger.info(f"=" * 60)

    # 创建输出目录
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(outputs_root) / "wm" / wm_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 wandb（从 .env 读取配置）
    wandb_run = None
    if use_wandb:
        wandb_project = get_env("WANDB_PROJECT", "flower")
        wandb_entity = get_env("WANDB_ENTITY", None)
        wandb_mode = get_env("WANDB_MODE", "online")
        run_prefix = get_env("WANDB_RUN_PREFIX", "exp")

        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"{run_prefix}_{wm_name}_{ts}",
            config={
                "wm_name": wm_name,
                "num_rollouts": num_rollouts,
                "train_epochs": train_epochs,
                "history_len": int(wm_cfg.history_len),
                "latent_dim": int(wm_cfg.latent_dim),
                "num_patches": int(getattr(wm_cfg.encoder, "num_patches", 16)),
                "token_dim": int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 16)),
                "training_mode": str(train_cfg.training_mode),
            },
            mode=wandb_mode,
        )
        wandb_run = wandb.run

    # 解析训练数据路径 - 使用 test 数据集
    train_run_dir = Path("datasets/ai2thor/test/2026-04-24_14-47-16")
    if not train_run_dir.exists():
        train_run_dir = Path("datasets/ai2thor/test")
        candidates = [p for p in train_run_dir.iterdir() if p.is_dir() and p.name.startswith("2026")]
        if candidates:
            train_run_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    logger.info(f"训练数据路径: {train_run_dir}")

    # 加载数据集
    dataset, cache_dir = load_wm_dataset(
        run_dir=train_run_dir,
        wm_cfg=wm_cfg,
        dataset_cfg=dataset_cfg,
        temporal_stride=1,
        num_workers=2,
    )

    # 创建子集采样器
    sampler = SubsetSampler(dataset, max_rollouts=num_rollouts)
    num_training_samples = len(sampler)

    # 使用训练配置中的 batch_size
    batch_size = int(train_cfg.batch_size)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        persistent_workers=False,
    )

    # 构建设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    logger.info(f"batch_size: {batch_size}")

    # 构建模型
    wm_model = build_lewm_model(wm_cfg, dataset_cfg, train_cfg, device)

    # 训练
    start_time = time.time()
    metrics = train_model(
        model=wm_model,
        train_loader=train_loader,
        num_epochs=train_epochs,
        log_every_n_steps=20,
        wandb_run=wandb_run,
    )
    elapsed = time.time() - start_time

    logger.info(f"训练完成，耗时: {elapsed:.1f}秒")

    # 保存模型
    wm_ckpt_path = run_dir / "wm_ema.pt"
    torch.save(wm_model.wm.state_dict(), wm_ckpt_path)
    logger.info(f"模型保存至: {wm_ckpt_path}")

    # 保存训练指标
    train_metrics = {
        "wm_name": wm_name,
        "num_rollouts": num_rollouts,
        "train_epochs": train_epochs,
        "train_time_sec": elapsed,
        "num_training_samples": num_training_samples,
        "final_loss": metrics["loss"][-1] if metrics["loss"] else 0.0,
        "final_recon_loss": metrics["loss_recon"][-1] if metrics["loss_recon"] else 0.0,
        "loss_history": metrics["loss"][::max(1, len(metrics["loss"]) // 100)],
    }
    with open(run_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    # 记录到 wandb
    if wandb_run is not None:
        wandb_run.log({
            "train/final_loss": metrics["loss"][-1] if metrics["loss"] else 0.0,
            "train/final_recon_loss": metrics["loss_recon"][-1] if metrics["loss_recon"] else 0.0,
            "train/elapsed_sec": elapsed,
        })
        wandb_run.finish()

    return run_dir, train_metrics


def run_visualization(
    model_run_dir: Path,
    wm_cfg: Any,
    dataset_cfg: Any,
    num_test_rollouts: int,
    num_steps: int,
) -> dict[str, Any]:
    """对已训练模型运行可视化。"""
    from omegaconf import OmegaConf

    # 找到模型文件
    wm_ckpt_path = model_run_dir / "wm_ema.pt"
    if not wm_ckpt_path.exists():
        for name in ["checkpoint_final.pt", "wm.pt"]:
            candidate = model_run_dir / name
            if candidate.exists():
                wm_ckpt_path = candidate
                break
        if not wm_ckpt_path.exists():
            raise RuntimeError(f"未找到模型文件: {model_run_dir}")

    # 解析 wm_name
    wm_name = model_run_dir.parent.name
    wm_cfg_path = f"configs/wm/{wm_name}.yaml"
    if not Path(wm_cfg_path).exists():
        for cfg_name in ["lewm_dinov2m", "lewm_qwen25vl_8b", "cfm_dinov2m"]:
            if cfg_name in wm_name:
                wm_cfg_path = f"configs/wm/{cfg_name}.yaml"
                break

    wm_cfg = OmegaConf.load(Path(wm_cfg_path)) if wm_cfg is None else wm_cfg
    dataset_cfg = OmegaConf.load(Path("configs/dataset/ai2thor.yaml")) if dataset_cfg is None else dataset_cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"可视化设备: {device}")

    # 构建模型
    num_patches = int(getattr(wm_cfg.encoder, "num_patches", 16))
    latent_dim = int(wm_cfg.latent_dim)
    token_dim = latent_dim // num_patches

    wm_module = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.transformer.num_layers),
        num_heads=int(wm_cfg.transformer.num_heads),
        dim_head=int(wm_cfg.transformer.get("dim_head", 64)),
        mlp_ratio=float(wm_cfg.transformer.get("mlp_ratio", 4.0)),
        dropout=float(wm_cfg.transformer.get("dropout", 0.1)),
        emb_dropout=float(wm_cfg.lewm.get("emb_dropout", 0.0)),
    )

    # 加载权重
    state = torch.load(wm_ckpt_path, map_location=device, weights_only=False)
    wm_module.load_state_dict(state, strict=False)
    wm_module.eval()
    wm_module.to(device)

    # 加载测试数据
    test_run_dir = Path("datasets/ai2thor/test/2026-04-24_14-47-16")
    if not test_run_dir.exists():
        # 回退到 latest 链接
        latest = Path("datasets/ai2thor/test/latest")
        if latest.exists() and latest.is_symlink():
            test_run_dir = latest.resolve()

    if not test_run_dir.exists():
        raise RuntimeError(f"测试数据目录不存在: {test_run_dir}")

    logger.info(f"加载测试数据 from: {test_run_dir}")

    # 获取 history_len
    history_len = int(wm_cfg.history_len)

    # 加载 3 条独立的 rollout（使用不同的 seed 来确保独立性）
    rollout_data = []
    for rollout_idx in range(num_test_rollouts):
        latents, actions, metadata_list = load_test_data(
            run_dir=test_run_dir,
            wm_cfg=wm_cfg,
            num_rollouts=1,  # 每次只加载 1 条
            num_steps=num_steps,
            seed=42 + rollout_idx,  # 不同的 seed 确保选到不同的 rollout
        )
        if len(latents) > 0:
            # 预测轨迹
            rollout_real, rollout_pred = predict_trajectories(
                wm_module=wm_module,
                latents=latents,
                actions=actions,
                history_len=history_len,
                num_steps=min(num_steps, len(latents) - history_len),
                device=device,
            )
            rollout_data.append({
                "real": rollout_real,
                "pred": rollout_pred,
                "metadata": metadata_list[:len(rollout_real)],
                "scene": metadata_list[0].get("scene", "unknown") if metadata_list else "unknown",
            })

    if len(rollout_data) == 0:
        raise RuntimeError("测试数据为空")

    # 使用统一的 UMAP 映射（对所有 rollout 的 latents 一起降维）
    all_latents = []
    rollout_offsets = []  # 记录每个 rollout 在 embedded 数组中的起始位置
    for rd in rollout_data:
        rollout_offsets.append(len(all_latents))
        all_latents.extend(rd["real"])
        all_latents.extend(rd["pred"])

    embedded_all = compute_umap_3d(all_latents)

    # 计算总体 MSE
    total_mse = 0.0
    total_count = 0
    for rd in rollout_data:
        for rz, pz in zip(rd["real"], rd["pred"]):
            mse = torch.mean((rz - pz) ** 2).item()
            total_mse += mse
            total_count += 1
    avg_mse = total_mse / max(1, total_count)

    # 保存结果
    vis_dir = model_run_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_images = {}

    # 生成每条 rollout 的可视化图片
    for rollout_idx, rd in enumerate(rollout_data):
        start_offset = rollout_offsets[rollout_idx]
        num_real = len(rd["real"])
        num_pred = len(rd["pred"])

        rollout_embedded = embedded_all[start_offset:start_offset + num_real + num_pred]
        rollout_real_coords = np.array(rollout_embedded[:num_real])
        rollout_pred_coords = np.array(rollout_embedded[num_real:num_real + num_pred])

        # 计算这条 rollout 的 MSE
        rollout_mse = 0.0
        for rz, pz in zip(rd["real"], rd["pred"]):
            rollout_mse += torch.mean((rz - pz) ** 2).item()
        rollout_mse = rollout_mse / max(1, len(rd["pred"]))

        # 生成这张 rollout 的图
        rollout_fig_path = vis_dir / f"rollout_{rollout_idx + 1}_{rd['scene']}.png"
        generate_single_rollout_figure(
            real_coords=rollout_real_coords,
            pred_coords=rollout_pred_coords,
            real_trajectory=rd["real"],
            pred_trajectory=rd["pred"],
            scene=rd["scene"],
            rollout_idx=rollout_idx,
            avg_mse=rollout_mse,
            output_path=rollout_fig_path,
        )
        vis_images[f"rollout_{rollout_idx + 1}"] = rollout_fig_path

    result = {
        "model_run_dir": str(model_run_dir),
        "wm_ckpt_path": str(wm_ckpt_path),
        "num_test_rollouts": len(rollout_data),
        "num_steps": num_steps,
        "avg_mse": avg_mse,
        "avg_prediction_mse": avg_mse,
    }

    # 保存 JSON 结果（简化版，不含详细点数据）
    with open(vis_dir / "result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"可视化结果保存至: {vis_dir}")
    logger.info(f"生成图片: {list(vis_images.values())}")

    # 上传到 wandb
    wandb_run = wandb.run
    if wandb_run is not None:
        for name, img_path in vis_images.items():
            wandb_run.log({f"visualization/{name}": wandb.Image(str(img_path))})

    return result


def main() -> None:
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="LeWM Phase2 测试脚本")
    parser.add_argument("--train-only", action="store_true", help="仅运行训练")
    parser.add_argument("--visualize-only", action="store_true", help="仅运行可视化")
    parser.add_argument("--run-all", action="store_true", help="运行完整流程")
    parser.add_argument(
        "--models",
        type=str,
        default="dinov2",
        help="要测试的模型列表，逗号分隔 (dinov2, qwen)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="已有模型的目录（用于 visualize-only 模式）",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=1000,
        help="训练使用的 rollout 数量",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="训练 epoch 数",
    )
    parser.add_argument(
        "--num-test-rollouts",
        type=int,
        default=3,
        help="测试 rollout 数量",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="测试轨迹步数",
    )
    parser.add_argument(
        "--outputs-root",
        type=str,
        default=OUTPUTS_ROOT,
        help="输出根目录",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="禁用 wandb 日志",
    )

    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    use_wandb = not args.no_wandb

    MODEL_CONFIGS = {
        "dinov2": {
            "wm_name": "lewm_dinov2m",
            "config": "configs/wm/lewm_dinov2m.yaml",
        },
        "qwen": {
            "wm_name": "lewm_qwen25vl_8b",
            "config": "configs/wm/lewm_qwen25vl_8b.yaml",
        },
    }

    if args.visualize_only:
        if not args.model_dir:
            logger.error("--visualize-only 需要 --model-dir 参数")
            return
        run_dir = Path(args.model_dir)

        # 加载配置
        from omegaconf import OmegaConf
        wm_name = run_dir.parent.name
        wm_cfg_path = f"configs/wm/{wm_name}.yaml"
        if not Path(wm_cfg_path).exists():
            for cfg_name in ["lewm_dinov2m", "lewm_qwen25vl_8b", "cfm_dinov2m"]:
                if cfg_name in wm_name:
                    wm_cfg_path = f"configs/wm/{cfg_name}.yaml"
                    break
        wm_cfg = OmegaConf.load(Path(wm_cfg_path))
        dataset_cfg = OmegaConf.load(Path("configs/dataset/ai2thor.yaml"))

        try:
            run_visualization(
                model_run_dir=run_dir,
                wm_cfg=wm_cfg,
                dataset_cfg=dataset_cfg,
                num_test_rollouts=args.num_test_rollouts,
                num_steps=args.num_steps,
            )
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            import traceback
            traceback.print_exc()
    elif args.train_only or args.run_all:
        model_keys = [k.strip() for k in args.models.split(",")]
        results = {}

        for model_key in model_keys:
            if model_key not in MODEL_CONFIGS:
                logger.warning(f"未知模型: {model_key}，跳过")
                continue

            config = MODEL_CONFIGS[model_key]
            wm_cfg = OmegaConf.load(Path(config["config"]))
            dataset_cfg = OmegaConf.load(Path("configs/dataset/ai2thor.yaml"))

            try:
                run_dir, train_metrics = run_training(
                    wm_cfg=wm_cfg,
                    dataset_cfg=dataset_cfg,
                    num_rollouts=args.num_rollouts,
                    train_epochs=args.train_epochs,
                    outputs_root=args.outputs_root,
                    use_wandb=use_wandb,
                )
                results[model_key] = {"train_metrics": train_metrics}

                if args.run_all:
                    vis_result = run_visualization(
                        model_run_dir=run_dir,
                        wm_cfg=wm_cfg,
                        dataset_cfg=dataset_cfg,
                        num_test_rollouts=args.num_test_rollouts,
                        num_steps=args.num_steps,
                    )
                    results[model_key]["visualization"] = vis_result

            except Exception as e:
                logger.error(f"处理模型 {model_key} 失败: {e}")
                import traceback
                traceback.print_exc()
                results[model_key] = {"error": str(e)}

        # 保存最终汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": list(results.keys()),
            "config": {
                "num_rollouts": args.num_rollouts,
                "train_epochs": args.train_epochs,
                "num_test_rollouts": args.num_test_rollouts,
                "num_steps": args.num_steps,
            },
            "results": results,
        }

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        summary_path = Path(args.outputs_root) / "lewm_phase2_test" / f"summary_{ts}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"测试完成，汇总保存至: {summary_path}")
    else:
        parser.print_help()
        logger.info("\n请使用 --train-only, --visualize-only 或 --run-all")


if __name__ == "__main__":
    main()