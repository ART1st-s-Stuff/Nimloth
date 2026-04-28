"""Joint Training Rollout 可视化 - UMAP 降维对比."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_umap_3d(points: list[torch.Tensor], n_neighbors: int = 15) -> list[list[float]]:
    """使用 UMAP 将高维 latent 降维到 3D。"""
    try:
        import umap
    except ImportError:
        logger.warning("UMAP 未安装，跳过降维")
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points[:100]]]

    if len(points) < 3:
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points]]

    arr = torch.stack([p.flatten() for p in points], dim=0).numpy()
    n_neighbors = min(n_neighbors, max(2, len(points) - 1))
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, random_state=42)
    try:
        embedded = reducer.fit_transform(arr.astype(np.float32))
        return embedded.tolist()
    except Exception as e:
        logger.warning(f"UMAP 计算失败: {e}")
        return [[float(p.flatten()[0]) if p.numel() > 0 else 0.0 for p in points]]


def generate_rollout_visualization(
    real_latents: list[torch.Tensor],
    pred_latents: list[torch.Tensor],
    rollout_idx: int,
    avg_mse: float,
    output_path: Path,
) -> None:
    """生成 rollout 可视化图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
        return

    # UMAP 降维
    all_points = real_latents + pred_latents
    embedded = compute_umap_3d(all_points)
    mid = len(real_latents)

    real_coords = np.array(embedded[:mid])
    pred_coords = np.array(embedded[mid:])

    # 处理降维结果维度
    n_dims = real_coords.shape[1] if len(real_coords) > 0 else 2
    if n_dims < 3:
        # 如果 UMAP 降维到 2D，用 0 填充第三维
        if len(real_coords) > 0:
            real_coords = np.pad(real_coords, ((0, 0), (0, 3 - n_dims)), mode='constant', constant_values=0)
        if len(pred_coords) > 0:
            pred_coords = np.pad(pred_coords, ((0, 0), (0, 3 - n_dims)), mode='constant', constant_values=0)

    # 计算每步 MSE
    mse_per_step = []
    for rz, pz in zip(real_latents, pred_latents):
        mse = torch.mean((rz - pz) ** 2).item()
        mse_per_step.append(mse)

    # 找出最大误差的 5 步
    top5_indices = sorted(range(len(mse_per_step)), key=lambda i: mse_per_step[i], reverse=True)[:5]

    # 3D 图
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Ground Truth 轨迹 (蓝色)
    if len(real_coords) > 0:
        ax.plot(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2],
                'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        ax.scatter(real_coords[:1, 0], real_coords[:1, 1], real_coords[:1, 2],
                   c='blue', s=150, marker='o', label='GT Start', zorder=5)
        ax.scatter(real_coords[-1:, 0], real_coords[-1:, 1], real_coords[-1:, 2],
                   c='blue', s=150, marker='^', label='GT End', zorder=5)
        for i in range(0, len(real_coords), max(1, len(real_coords) // 8)):
            ax.text(real_coords[i, 0], real_coords[i, 1], real_coords[i, 2],
                    f'T{i}', fontsize=8, color='blue', fontweight='bold')

    # Predicted 轨迹 (红色)
    if len(pred_coords) > 0:
        ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2],
                'r--', linewidth=2, label='Predicted', alpha=0.8)
        ax.scatter(pred_coords[:1, 0], pred_coords[:1, 1], pred_coords[:1, 2],
                   c='red', s=150, marker='o', label='Pred Start', zorder=5)
        ax.scatter(pred_coords[-1:, 0], pred_coords[-1:, 1], pred_coords[-1:, 2],
                   c='red', s=150, marker='^', label='Pred End', zorder=5)

        # 误差连线
        for idx in top5_indices:
            if idx < len(real_coords) and idx < len(pred_coords):
                ax.plot([real_coords[idx, 0], pred_coords[idx, 0]],
                        [real_coords[idx, 1], pred_coords[idx, 1]],
                        [real_coords[idx, 2], pred_coords[idx, 2]],
                        'gray', linewidth=1, linestyle=':', alpha=0.8)
                mid_x = (real_coords[idx, 0] + pred_coords[idx, 0]) / 2
                mid_y = (real_coords[idx, 1] + pred_coords[idx, 1]) / 2
                mid_z = (real_coords[idx, 2] + pred_coords[idx, 2]) / 2
                ax.text(mid_x, mid_y, mid_z, f'{mse_per_step[idx]:.3f}',
                        fontsize=7, color='gray', fontweight='bold')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.set_title(f'Rollout {rollout_idx} - UMAP Trajectory (MSE: {avg_mse:.6f})')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved visualization to {output_path}")


def main():
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/dev/visualization")
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir

    # 加载配置 - 使用项目根目录
    project_root = Path(__file__).parent.parent.parent.parent
    config_dir = project_root / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=["wm=lewm_qwen_llm_joint"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 加载 Qwen Adapter
    from src.vlm.qwen_adapter import QwenVLMAdapter
    from src.wm.encoder.qwen import QwenLLMLatentEncoder
    from src.wm.predictor.lewm import LeWMWorldModel

    latent_dim = int(cfg.wm.latent_dim)
    model_name = str(getattr(cfg.wm.encoder, "model_name", "Qwen/Qwen2.5-VL-7B-Instruct"))

    qwen_adapter = QwenVLMAdapter(
        model_name=model_name,
        latent_dim=latent_dim,
        enabled=True,
        fallback_enabled=False,
    )
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")
    qwen_adapter._set_llm_backbone_trainable(trainable=False)

    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=latent_dim,
        qwen_adapter=qwen_adapter,
        use_vision_only=False,
        llm_backbone_trainable=False,
    )

    # 加载 WM
    wm_model = LeWMWorldModel(
        latent_dim=latent_dim,
        action_dim=3,
        hidden_dim=int(getattr(cfg.wm, "hidden_dim", 512)),
        history_len=args.history_len,
        num_patches=1,
        token_dim=latent_dim,
        num_layers=int(getattr(cfg.wm.transformer, "num_layers", 6)),
        num_heads=int(getattr(cfg.wm.transformer, "num_heads", 16)),
        dim_head=int(getattr(cfg.wm.transformer, "dim_head", 64)),
        mlp_ratio=float(getattr(cfg.wm.transformer, "mlp_ratio", 4.0)),
        dropout=float(getattr(cfg.wm.transformer, "dropout", 0.1)),
        emb_dropout=0.0,
    ).to(device)
    wm_model.eval()

    # 加载数据集
    from src.data.eb_nav_dataset import EBNavSequenceDataset

    json_path = "datasets/EB-Nav/eb-nav_dataset_single_step.json"
    # json 中的路径是相对于 datasets/EB-Nav 的，所以 base_dir 是 datasets/EB-Nav
    images_base_dir = "datasets/EB-Nav"

    dataset = EBNavSequenceDataset(
        json_path=json_path,
        images_base_dir=images_base_dir,
        latent_dim=latent_dim,
        action_dim=3,
        history_len=args.history_len,
    )

    # 随机选择 episode 进行 rollout
    import random
    rng = random.Random(42)
    episode_indices = rng.sample(range(len(dataset)), min(args.num_rollouts, len(dataset)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for rollout_idx, ep_idx in enumerate(tqdm(episode_indices, desc="Rollout")):
        sample = dataset[ep_idx]

        # 编码历史图像
        z_history_list = []
        for img_path in sample["history_images"]:
            if img_path and Path(img_path).exists():
                latent = vision_encoder.encode_image_path(img_path).z.to("cpu")
            else:
                latent = torch.zeros(latent_dim, device="cpu")
            z_history_list.append(latent)
        z_history = torch.stack(z_history_list).unsqueeze(0).unsqueeze(2)  # [1, H, 1, D]

        # 编码未来图像 (ground truth)
        z_future_list = []
        for img_path in sample["future_images"][:args.rollout_steps]:
            if img_path and Path(img_path).exists():
                latent = vision_encoder.encode_image_path(img_path).z.to("cpu")
            else:
                latent = torch.zeros(latent_dim, device="cpu")
            z_future_list.append(latent)

        # Rollout 预测
        real_latents = z_history.squeeze(0).cpu()  # [H, D]
        pred_latents = []
        actions = torch.tensor(sample["history_actions"][:args.history_len], dtype=torch.float32).unsqueeze(0).to(device)

        current_z = z_history.to(device)  # [1, H, 1, D] on GPU

        for step_idx in range(len(z_future_list)):
            # WM 预测
            pred_z = wm_model(current_z, actions)  # [1, P, D] on GPU
            pred_latents.append(pred_z.squeeze(0).squeeze(0).cpu())

            # 更新历史 (滑动窗口)
            if step_idx < len(z_future_list) - 1:
                new_z = torch.cat([current_z[:, 1:, :, :], pred_z.unsqueeze(1)], dim=1)
                current_z = new_z

        # 计算 MSE
        mse_values = []
        for rz, pz in zip(z_future_list, pred_latents):
            mse = torch.mean((rz - pz) ** 2).item()
            mse_values.append(mse)
        avg_mse = sum(mse_values) / len(mse_values) if mse_values else 0.0

        # 生成可视化
        vis_path = output_dir / f"rollout_{rollout_idx:03d}_mse{avg_mse:.4f}.png"
        generate_rollout_visualization(
            real_latents=[z.squeeze(0) for z in z_future_list],
            pred_latents=pred_latents,
            rollout_idx=rollout_idx,
            avg_mse=avg_mse,
            output_path=vis_path,
        )

        all_results.append({
            "rollout_idx": rollout_idx,
            "episode_idx": ep_idx,
            "avg_mse": avg_mse,
            "mse_per_step": mse_values,
            "vis_path": str(vis_path),
        })

    # 保存结果
    results_path = output_dir / "rollout_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # 汇总统计
    avg_mse_all = sum(r["avg_mse"] for r in all_results) / len(all_results) if all_results else 0.0
    logger.info(f"\n=== Rollout Summary ===")
    logger.info(f"Total rollouts: {len(all_results)}")
    logger.info(f"Average MSE: {avg_mse_all:.6f}")
    logger.info(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
