"""Generate EB-Nav rollout visualizations from a joint WM checkpoint."""

from __future__ import annotations

import argparse
import gc
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from src.data.eb_nav_dataset import ACTION_MAP, ACTION_NAMES, EBNavSequenceDataset
from src.train.train_wm_joint import (
    _compute_umap_3d,
    _encode_joint_batch,
    _joint_collate_fn,
    _normalize_patch_latent,
    _resolve_joint_resume_checkpoint,
    _save_rollout_figure,
)
from src.utils.env import load_project_env
from src.visualize.wandb_tracker import init_tracker
from src.vlm.qwen_adapter import QwenVLMAdapter
from src.wm.encoder.qwen import QwenLLMLatentEncoder
from src.wm.predictor.lewm import LeWMWorldModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint path or 'latest'.")
    parser.add_argument("--dataset", default="datasets/EB-Nav/eb-nav_dataset_single_step.json")
    parser.add_argument("--images-base-dir", default="datasets/EB-Nav")
    parser.add_argument("--reward-cache", default="datasets/EB-Nav/phase2_reward_cache.jsonl")
    parser.add_argument("--planner-lora", default="models/qwen_planner_lora")
    parser.add_argument("--output-dir", default="outputs/dev/visualization/eb_nav_rollout")
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--temporal-stride",
        type=int,
        default=None,
        help="Number of future states loaded per sequence. Defaults to --num-steps.",
    )
    parser.add_argument("--device", default="cuda", help="Device for WM and for Qwen when --qwen-device-map=none.")
    parser.add_argument(
        "--qwen-device-map",
        default="auto",
        choices=["auto", "none"],
        help="auto: shard Qwen with Accelerate across visible GPUs (avoids one huge .to(cuda:0) on a busy card). "
        "none: load then move whole model to --device (needs one GPU with enough free VRAM).",
    )
    parser.add_argument("--qwen-dtype", default="bfloat16")
    parser.add_argument("--encode-micro-batch-size", type=int, default=1)
    parser.add_argument("--projection", choices=["pca", "umap"], default="pca")
    parser.add_argument(
        "--disable-low-memory-planner",
        action="store_true",
        help="Use the regular planner extraction path with full hidden_states/logits.",
    )
    parser.add_argument(
        "--skip-vision-state",
        action="store_true",
        help="Do not load fine-tuned Qwen vision encoder state from checkpoint.",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--action-branching",
        action="store_true",
        help="Per-step WM predictions for multiple discrete actions (ACTION_MAP); GT trajectory + colored branches.",
    )
    parser.add_argument(
        "--branch-action-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated EB-Nav action ids to evaluate when --action-branching is set.",
    )
    return parser.parse_args()


def _parse_branch_action_ids(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty --branch-action-ids")
    out: list[int] = []
    for p in parts:
        aid = int(p)
        if aid not in ACTION_MAP:
            raise ValueError(f"unknown action id {aid}; valid keys: {sorted(ACTION_MAP.keys())}")
        out.append(aid)
    return out


def _warn_or_exit_if_qwen_vram_tight(
    *,
    qwen_device_map: str,
    device: torch.device,
    min_free_gib: float = 2.0,
) -> None:
    """Qwen in 'none' mode needs a lot of free memory on a single device; 'auto' can use multiple GPUs."""
    if not torch.cuda.is_available() or device.type != "cuda":
        return
    if qwen_device_map == "auto":
        return
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
    except Exception:
        return
    free_gib = free_b / (1024**3)
    if free_gib < min_free_gib:
        total_gib = total_b / (1024**3)
        print(
            f"[error] GPU {device} has only {free_gib:.2f} GiB free of {total_gib:.1f} GiB. "
            f"With --qwen-device-map=none the script moves the full Qwen model to this device and will likely OOM.\n"
            f"  Fix: use another GPU, e.g. CUDA_VISIBLE_DEVICES=<id with enough free VRAM>\n"
            f"  Or:   remove --qwen-device-map (default is auto) to shard Qwen across visible GPUs.\n"
            f"  Check: nvidia-smi"
        )
        raise SystemExit(1)


def _nearest_action_id(gt_vec: torch.Tensor, *, action_dim: int = 3) -> int:
    """Match GT action vector to a discrete EB-Nav action id."""
    gt = gt_vec.detach().cpu().float().view(-1)
    if gt.numel() >= len(ACTION_MAP):
        return int(torch.argmax(gt[: len(ACTION_MAP)]).item())
    gt = gt[:action_dim]
    best_id = 0
    best_d = float("inf")
    for aid, vec in ACTION_MAP.items():
        v = torch.tensor(vec[:action_dim], dtype=torch.float32)
        d = float(torch.sum((gt - v) ** 2).item())
        if d < best_d:
            best_d = d
            best_id = int(aid)
    return best_id


def _action_vec_tensor(
    action_id: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    action_dim: int,
) -> torch.Tensor:
    if action_dim >= len(ACTION_MAP):
        vec = torch.zeros(action_dim, device=device, dtype=dtype)
        vec[int(action_id)] = 1.0
    else:
        vec = torch.tensor(ACTION_MAP[int(action_id)][:action_dim], device=device, dtype=dtype)
    return vec.view(1, 1, action_dim)


def _load_cfg():
    config_dir = str(Path.cwd() / "configs")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        return hydra.compose(config_name="config", overrides=["wm=lewm_qwen_llm_joint"])


def _build_wm(cfg, checkpoint_config: dict, device: torch.device) -> LeWMWorldModel:
    wm_cfg = cfg.wm
    sigreg_cfg = cfg.pipeline.train.sigreg
    ckpt_reward = checkpoint_config.get("reward", {}) if isinstance(checkpoint_config, dict) else {}
    ckpt_perceptual = checkpoint_config.get("perceptual", {}) if isinstance(checkpoint_config, dict) else {}
    reward_enabled = bool(ckpt_reward.get("enabled", False))
    perceptual_enabled = bool(ckpt_perceptual.get("enabled", False))

    model = LeWMWorldModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(checkpoint_config.get("action_dim", getattr(wm_cfg, "action_dim", 3))),
        hidden_dim=int(getattr(wm_cfg, "hidden_dim", 512)),
        history_len=int(getattr(wm_cfg, "history_len", 4)),
        num_patches=int(getattr(wm_cfg, "num_patches", 1)),
        token_dim=int(getattr(wm_cfg, "token_dim", wm_cfg.latent_dim)),
        num_layers=int(getattr(wm_cfg.transformer, "num_layers", 6)),
        num_heads=int(getattr(wm_cfg.transformer, "num_heads", 16)),
        dim_head=int(getattr(wm_cfg.transformer, "dim_head", 64)),
        mlp_ratio=float(getattr(wm_cfg.transformer, "mlp_ratio", 4.0)),
        dropout=float(getattr(wm_cfg.transformer, "dropout", 0.1)),
        emb_dropout=float(getattr(wm_cfg.lewm, "emb_dropout", 0.0)),
        sigreg_enabled=bool(getattr(wm_cfg.lewm, "sigreg_enabled", False)),
        sigreg_latent_dim=int(getattr(wm_cfg.lewm, "sigreg_latent_dim", wm_cfg.latent_dim)),
        sigreg_num_proj=int(getattr(wm_cfg.lewm, "sigreg_num_proj", 256)),
        sigreg_num_quadrature_points=int(getattr(sigreg_cfg, "num_quadrature_points", 17)),
        sigreg_t_min=float(getattr(sigreg_cfg, "t_min", 0.2)),
        sigreg_t_max=float(getattr(sigreg_cfg, "t_max", 4.0)),
        sigreg_kernel_sigma=float(getattr(sigreg_cfg, "kernel_sigma", 1.0)),
        reward_enabled=reward_enabled,
        reward_hidden_dim=int(ckpt_reward.get("hidden_dim", 512)),
        image_decoder_enabled=perceptual_enabled,
        image_decoder_hidden_channels=int(ckpt_perceptual.get("decoder_hidden_channels", 128)),
        image_size=int(ckpt_perceptual.get("image_size", 128)),
    ).to(device)
    return model


def _plot_one_rollout(
    *,
    wm_model: LeWMWorldModel,
    batch_device: dict[str, torch.Tensor],
    output_path: Path,
    rollout_idx: int,
    device: torch.device,
    projection: str,
) -> float:
    z_history = batch_device["z_history"][:1].to(device)
    action_history = batch_device["action_history"][:1].to(device)
    z_future = batch_device["z_future"][:1].to(device)
    future_actions = batch_device["gt_action_future"][:1].to(device)

    real_traj: list[torch.Tensor] = []
    pred_traj: list[torch.Tensor] = []
    wm_model.eval()
    with torch.no_grad():
        teacher_action = action_history.clone()
        for step_idx in range(int(z_future.size(1))):
            teacher_action[:, -1, :] = future_actions[:, step_idx, :]
            pred_z = wm_model.predict_next(z_history, teacher_action)
            real_z = z_future[:, step_idx, ...]
            pred_traj.append(_normalize_patch_latent(pred_z.squeeze(0).detach().cpu(), z_future.size(2), z_future.size(3)))
            real_traj.append(_normalize_patch_latent(real_z.squeeze(0).detach().cpu(), z_future.size(2), z_future.size(3)))
            z_history = torch.cat([z_history[:, 1:, ...], real_z.unsqueeze(1)], dim=1)
            teacher_action = torch.cat([teacher_action[:, 1:, :], future_actions[:, step_idx, :].unsqueeze(1)], dim=1)
    if projection == "umap":
        return _save_rollout_figure(
            real_traj=real_traj,
            pred_traj=pred_traj,
            scene="EB-Nav",
            rollout_idx=rollout_idx,
            output_path=output_path,
        )
    return _save_pca_rollout_figure(real_traj, pred_traj, rollout_idx, output_path)


def _save_pca_rollout_figure(
    real_traj: list[torch.Tensor],
    pred_traj: list[torch.Tensor],
    rollout_idx: int,
    output_path: Path,
) -> float:
    if not real_traj or not pred_traj:
        return 0.0
    real = torch.stack([x.flatten().float() for x in real_traj], dim=0)
    pred = torch.stack([x.flatten().float() for x in pred_traj], dim=0)
    mse = torch.mean((real - pred) ** 2, dim=1)
    avg_mse = float(mse.mean().item())
    points = torch.cat([real, pred], dim=0)
    centered = points - points.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(centered, q=3, center=False)
        coords = centered @ v[:, :3]
    except Exception:
        coords = centered[:, :3]
    coords_np = coords.detach().cpu().numpy().astype(np.float32)
    n = int(real.size(0))
    real_coords = coords_np[:n]
    pred_coords = coords_np[n:]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(real_coords[:, 0], real_coords[:, 1], real_coords[:, 2], "b-", linewidth=2, label="Ground Truth")
    ax.plot(pred_coords[:, 0], pred_coords[:, 1], pred_coords[:, 2], "r--", linewidth=2, label="Predicted")
    for idx in range(n):
        ax.plot(
            [real_coords[idx, 0], pred_coords[idx, 0]],
            [real_coords[idx, 1], pred_coords[idx, 1]],
            [real_coords[idx, 2], pred_coords[idx, 2]],
            color="gray",
            alpha=0.35,
            linewidth=1,
        )
    ax.set_title(f"EB-Nav Rollout {rollout_idx + 1} - PCA (MSE: {avg_mse:.4f})")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    ax.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return avg_mse


def _plot_one_rollout_action_branching(
    *,
    wm_model: LeWMWorldModel,
    batch_device: dict[str, torch.Tensor],
    output_path: Path,
    rollout_idx: int,
    device: torch.device,
    projection: str,
    branch_action_ids: list[int],
    future_action_ids: torch.Tensor | None,
) -> float:
    """Teacher-forced rollout; each step predicts next latent under each candidate discrete action."""
    z_history = batch_device["z_history"][:1].to(device)
    action_history = batch_device["action_history"][:1].to(device)
    z_future = batch_device["z_future"][:1].to(device)
    future_actions = batch_device["gt_action_future"][:1].to(device)

    np_p = int(z_future.size(2))
    tok_d = int(z_future.size(3))
    dtype = action_history.dtype

    real_traj: list[torch.Tensor] = []
    branch_preds: list[list[torch.Tensor]] = []
    mse_gt_continuous: list[float] = []

    wm_model.eval()
    with torch.no_grad():
        teacher_action = action_history.clone()
        for step_idx in range(int(z_future.size(1))):
            real_z = z_future[:, step_idx, ...]
            real_norm = _normalize_patch_latent(
                real_z.squeeze(0).detach().cpu(), np_p, tok_d
            )
            real_traj.append(real_norm)

            teacher_action[:, -1, :] = future_actions[:, step_idx, :]
            pred_gt = wm_model.predict_next(z_history, teacher_action)
            pred_gt_n = _normalize_patch_latent(
                pred_gt.squeeze(0).detach().cpu(), np_p, tok_d
            )
            mse_gt_continuous.append(
                float(torch.mean((real_norm.float() - pred_gt_n.float()) ** 2).item())
            )

            step_branch: list[torch.Tensor] = []
            base_hist = action_history.clone()
            for aid in branch_action_ids:
                ta = base_hist.clone()
                av = _action_vec_tensor(aid, device=device, dtype=dtype, action_dim=int(action_history.size(-1)))
                ta[:, -1, :] = av.squeeze(1)
                pred_z = wm_model.predict_next(z_history, ta)
                step_branch.append(
                    _normalize_patch_latent(
                        pred_z.squeeze(0).detach().cpu(), np_p, tok_d
                    )
                )
            branch_preds.append(step_branch)

            z_history = torch.cat([z_history[:, 1:, ...], real_z.unsqueeze(1)], dim=1)
            teacher_action = torch.cat(
                [teacher_action[:, 1:, :], future_actions[:, step_idx, :].unsqueeze(1)],
                dim=1,
            )
            action_history = torch.cat(
                [action_history[:, 1:, :], future_actions[:, step_idx, :].unsqueeze(1)],
                dim=1,
            )

    avg_mse = float(sum(mse_gt_continuous) / max(1, len(mse_gt_continuous)))

    gt_action_ids_per_step: list[int] = []
    for step_idx in range(len(real_traj)):
        if future_action_ids is not None and step_idx < int(future_action_ids.numel()):
            gt_action_ids_per_step.append(int(future_action_ids[step_idx].item()))
        else:
            gt_action_ids_per_step.append(
                _nearest_action_id(future_actions[0, step_idx], action_dim=int(future_actions.size(-1)))
            )

    if projection == "umap":
        return _save_action_branch_umap_figure(
            real_traj=real_traj,
            branch_preds=branch_preds,
            branch_action_ids=branch_action_ids,
            gt_action_ids_per_step=gt_action_ids_per_step,
            rollout_idx=rollout_idx,
            output_path=output_path,
            avg_mse=avg_mse,
        )
    return _save_action_branch_pca_figure(
        real_traj=real_traj,
        branch_preds=branch_preds,
        branch_action_ids=branch_action_ids,
        gt_action_ids_per_step=gt_action_ids_per_step,
        rollout_idx=rollout_idx,
        output_path=output_path,
        avg_mse=avg_mse,
    )


def _color_for_step(step_idx: int, n_steps: int) -> tuple[float, float, float, float]:
    """Stable color per rollout step."""
    try:
        cmap = plt.colormaps["viridis"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("viridis")
    denom = max(1, n_steps - 1)
    return cmap(step_idx / denom)  # type: ignore[return-value]


def _marker_for_action(action_id: int) -> str:
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    return markers[action_id % len(markers)]


def _short_action_name(action_id: int, max_len: int = 34) -> str:
    short = ACTION_NAMES.get(action_id, str(action_id))
    if len(short) > max_len:
        short = short[: max_len - 3] + "..."
    return short


def _plot_action_branch_points(
    *,
    ax,
    real_coords: np.ndarray,
    branch_coords: np.ndarray,
    branch_action_ids: list[int],
    gt_action_ids_per_step: list[int],
) -> None:
    from matplotlib.lines import Line2D

    n_steps = int(branch_coords.shape[0])
    true_pred_coords: list[np.ndarray] = []
    true_pred_labels: list[str] = []

    ax.plot(
        real_coords[:, 0],
        real_coords[:, 1],
        real_coords[:, 2],
        color="0.25",
        linewidth=2,
        label="GT state trajectory",
    )
    for s in range(n_steps):
        ax.scatter(
            real_coords[s, 0],
            real_coords[s, 1],
            real_coords[s, 2],
            color="0.15",
            marker=".",
            s=42,
            depthshade=True,
        )
        ax.text(
            real_coords[s, 0],
            real_coords[s, 1],
            real_coords[s, 2],
            f"GT{s}",
            fontsize=7,
            color="0.20",
        )

    for s in range(n_steps):
        step_color = _color_for_step(s, n_steps)
        gt_aid = gt_action_ids_per_step[s]
        for j, aid in enumerate(branch_action_ids):
            coord = branch_coords[s, j]
            is_true_action = aid == gt_aid
            marker = _marker_for_action(aid)
            ax.scatter(
                coord[0],
                coord[1],
                coord[2],
                color="red" if is_true_action else step_color,
                marker=marker,
                s=92 if is_true_action else 48,
                depthshade=True,
                edgecolors="black" if is_true_action else "0.35",
                linewidths=0.8 if is_true_action else 0.35,
            )
            ax.text(
                coord[0],
                coord[1],
                coord[2],
                f"t{s}/a{aid}",
                fontsize=6,
                color="red" if is_true_action else "0.15",
            )
            if is_true_action:
                true_pred_coords.append(coord)
                true_pred_labels.append(f"t{s}: a{aid}")

    if true_pred_coords:
        arr = np.stack(true_pred_coords, axis=0)
        ax.plot(
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            color="red",
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="Predicted trajectory using GT actions",
        )

    action_handles = [
        Line2D(
            [0],
            [0],
            marker=_marker_for_action(aid),
            color="w",
            markerfacecolor="0.60",
            markeredgecolor="0.25",
            markersize=8,
            linestyle="None",
            label=f"a{aid}: {_short_action_name(aid)}",
        )
        for aid in branch_action_ids
    ]
    step_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_color_for_step(s, n_steps),
            markeredgecolor="0.35",
            markersize=7,
            linestyle="None",
            label=f"step {s}",
        )
        for s in range(n_steps)
    ]
    first_legend = ax.legend(handles=action_handles, title="Action marker", loc="upper left", fontsize=7)
    ax.add_artist(first_legend)
    second_legend = ax.legend(handles=step_handles, title="Step color", loc="lower left", fontsize=7)
    ax.add_artist(second_legend)
    ax.legend(loc="upper right", fontsize=8)


def _save_action_branch_pca_figure(
    *,
    real_traj: list[torch.Tensor],
    branch_preds: list[list[torch.Tensor]],
    branch_action_ids: list[int],
    gt_action_ids_per_step: list[int],
    rollout_idx: int,
    output_path: Path,
    avg_mse: float,
) -> float:
    if not real_traj or not branch_preds:
        return 0.0
    n_steps = len(real_traj)
    k = len(branch_action_ids)
    flat_real = [x.flatten().float() for x in real_traj]
    flat_branch: list[torch.Tensor] = []
    for step_preds in branch_preds:
        for p in step_preds:
            flat_branch.append(p.flatten().float())
    points = torch.stack(flat_real + flat_branch, dim=0)
    centered = points - points.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(centered, q=3, center=False)
        coords = centered @ v[:, :3]
    except Exception:
        coords = centered[:, :3]
    coords_np = coords.detach().cpu().numpy().astype(np.float32)
    real_coords = coords_np[:n_steps]
    branch_coords = coords_np[n_steps:].reshape(n_steps, k, 3)

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection="3d")
    _plot_action_branch_points(
        ax=ax,
        real_coords=real_coords,
        branch_coords=branch_coords,
        branch_action_ids=branch_action_ids,
        gt_action_ids_per_step=gt_action_ids_per_step,
    )
    ax.set_title(
        f"EB-Nav Rollout {rollout_idx + 1} — action branching (PCA), "
        f"MSE@GT continuous action: {avg_mse:.4f}"
    )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return avg_mse


def _save_action_branch_umap_figure(
    *,
    real_traj: list[torch.Tensor],
    branch_preds: list[list[torch.Tensor]],
    branch_action_ids: list[int],
    gt_action_ids_per_step: list[int],
    rollout_idx: int,
    output_path: Path,
    avg_mse: float,
) -> float:
    if not real_traj or not branch_preds:
        return 0.0
    n_steps = len(real_traj)
    k = len(branch_action_ids)
    all_points: list[torch.Tensor] = list(real_traj)
    for step_preds in branch_preds:
        all_points.extend(step_preds)
    emb = _compute_umap_3d(all_points)
    real_coords = emb[:n_steps]
    branch_coords = emb[n_steps:].reshape(n_steps, k, 3)

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection="3d")
    _plot_action_branch_points(
        ax=ax,
        real_coords=real_coords,
        branch_coords=branch_coords,
        branch_action_ids=branch_action_ids,
        gt_action_ids_per_step=gt_action_ids_per_step,
    )
    ax.set_title(
        f"EB-Nav Rollout {rollout_idx + 1} — action branching (UMAP), "
        f"MSE@GT continuous action: {avg_mse:.4f}"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return avg_mse


def main() -> None:
    load_project_env()
    args = _parse_args()
    branch_action_ids = _parse_branch_action_ids(args.branch_action_ids)
    if args.temporal_stride is None:
        args.temporal_stride = int(args.num_steps)
    args.temporal_stride = max(1, int(args.temporal_stride))
    args.num_steps = max(1, int(args.num_steps))
    if args.temporal_stride < args.num_steps:
        print(
            "[warn] --temporal-stride is smaller than --num-steps; "
            f"only {args.temporal_stride} future states are available per rollout."
        )
    cfg = _load_cfg()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    qwen_dm: str | None = None if args.qwen_device_map == "none" else "auto"
    _warn_or_exit_if_qwen_vram_tight(qwen_device_map=args.qwen_device_map, device=device)
    checkpoint_path = _resolve_joint_resume_checkpoint(args.checkpoint, Path("models/wm/joint_qwen"))
    if checkpoint_path is None:
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    # Keep the full training checkpoint on CPU. It contains optimizer states and
    # Qwen weights; mapping the whole file to CUDA can consume tens of GB before
    # visualization even starts.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint.get("config", {})

    wm_state = checkpoint["wm_state"]
    checkpoint.pop("wm_optimizer_state", None)
    checkpoint.pop("idm_optimizer_state", None)
    checkpoint.pop("wm_scheduler_state", None)
    checkpoint.pop("idm_scheduler_state", None)
    checkpoint.pop("idm_state", None)
    checkpoint.pop("action_mapper_state", None)

    qwen_adapter = QwenVLMAdapter(
        model_name=str(cfg.wm.encoder.model_name),
        latent_dim=int(cfg.wm.latent_dim),
        enabled=True,
        fallback_enabled=False,
        model_dtype=args.qwen_dtype,
        device_map=qwen_dm,
    )
    qwen_adapter.planner_low_memory = not bool(args.disable_low_memory_planner)
    qwen_adapter.planner_inference_mode = True
    qwen_adapter._ensure_model()
    if qwen_adapter._model is None:
        raise RuntimeError(f"Failed to load Qwen model: {qwen_adapter.init_error}")
    qwen_adapter.load_lora_adapter(args.planner_lora, trainable=False)
    if not args.skip_vision_state:
        qwen_adapter._model.load_state_dict(checkpoint["vision_encoder_state"], strict=False)
    checkpoint.pop("vision_encoder_state", None)
    checkpoint.pop("vision_encoder_ema_state", None)
    gc.collect()
    # With device_map="auto", Accelerate already placed modules; do not move full model to one GPU.
    if qwen_dm is None:
        qwen_adapter._model.to(device)
    qwen_adapter._model.eval()
    for param in qwen_adapter._model.parameters():
        param.requires_grad = False

    vision_encoder = QwenLLMLatentEncoder(
        latent_dim=int(cfg.wm.latent_dim),
        qwen_adapter=qwen_adapter,
        use_vision_only=False,
        llm_backbone_trainable=False,
        latent_anchor_mode="planner_special",
    )

    dataset = EBNavSequenceDataset(
        json_path=args.dataset,
        images_base_dir=args.images_base_dir,
        latent_dim=int(cfg.wm.latent_dim),
        action_dim=int(checkpoint.get("config", {}).get("action_dim", getattr(cfg.wm, "action_dim", 3))),
        history_len=int(cfg.wm.history_len),
        temporal_stride=int(args.temporal_stride),
        reward_cache_path=args.reward_cache,
    )
    subset = torch.utils.data.Subset(
        dataset,
        list(range(int(args.start_index), min(len(dataset), int(args.start_index) + int(args.num_rollouts)))),
    )
    if len(subset) == 0:
        raise ValueError(
            f"No rollout samples selected: dataset_len={len(dataset)}, "
            f"start_index={args.start_index}, num_rollouts={args.num_rollouts}"
        )
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_joint_collate_fn)

    encoded_batches: list[dict[str, torch.Tensor]] = []
    future_action_id_rows: list[torch.Tensor | None] = []
    for rollout_idx, batch in enumerate(loader):
        batch_device = _encode_joint_batch(
            batch=batch,
            vision_encoder=vision_encoder,
            device=device,
            num_patches=int(cfg.wm.num_patches),
            token_dim=int(cfg.wm.token_dim),
            planner_lora_enabled=True,
            planner_response_mode="dataset",
            planner_anchor_response=None,
            encoder_micro_batch_size=int(args.encode_micro_batch_size),
            perceptual_enabled=False,
            perceptual_image_size=128,
        )
        if int(batch_device["z_future"].size(1)) > int(args.num_steps):
            batch_device["z_future"] = batch_device["z_future"][:, : int(args.num_steps)]
            batch_device["gt_action_future"] = batch_device["gt_action_future"][:, : int(args.num_steps)]
        t_steps = int(batch_device["z_future"].size(1))
        fid_row: torch.Tensor | None = None
        if "future_action_ids" in batch:
            fid_row = batch["future_action_ids"][:, :t_steps].detach().cpu()
        encoded_batches.append({key: value.detach().cpu() for key, value in batch_device.items()})
        future_action_id_rows.append(fid_row[0] if fid_row is not None else None)
        print(
            f"encoded rollout {rollout_idx + 1}: available_future_steps="
            f"{int(batch_device['z_future'].size(1))}, requested_num_steps={args.num_steps}"
        )

    del vision_encoder
    del qwen_adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wm_model = _build_wm(cfg, checkpoint_config, device)
    wm_model.load_state_dict(wm_state, strict=False)
    wm_model.eval()
    del wm_state
    gc.collect()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    if not args.disable_wandb:
        tracker = init_tracker(
            task_name=f"visualize_eb_nav_rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset,
                "num_rollouts": args.num_rollouts,
                "num_steps": args.num_steps,
                "temporal_stride": args.temporal_stride,
                "projection": args.projection,
                "action_branching": bool(args.action_branching),
                "branch_action_ids": branch_action_ids,
            },
        )
        tracker.log_metrics(
            {
                "rollout/status": 1,
                "rollout/checkpoint_global_step": int(checkpoint.get("global_step", 0)),
                "rollout/selected_samples": len(subset),
            },
            step=0,
        )

    mse_values: list[float] = []
    for rollout_idx, batch_device in enumerate(encoded_batches):
        print(
            f"rollout {rollout_idx + 1}: available_future_steps="
            f"{int(batch_device['z_future'].size(1))}, requested_num_steps={args.num_steps}"
        )
        if args.action_branching:
            fig_path = output_dir / f"eb_nav_rollout_{rollout_idx + 1:03d}_action_branch.png"
            mse = _plot_one_rollout_action_branching(
                wm_model=wm_model,
                batch_device=batch_device,
                output_path=fig_path,
                rollout_idx=rollout_idx,
                device=device,
                projection=args.projection,
                branch_action_ids=branch_action_ids,
                future_action_ids=future_action_id_rows[rollout_idx],
            )
            wandb_image_key = "eb_nav_visualization/rollout_action_branch"
        else:
            fig_path = output_dir / f"eb_nav_rollout_{rollout_idx + 1:03d}.png"
            mse = _plot_one_rollout(
                wm_model=wm_model,
                batch_device=batch_device,
                output_path=fig_path,
                rollout_idx=rollout_idx,
                device=device,
                projection=args.projection,
            )
            wandb_image_key = "eb_nav_visualization/rollout"
        mse_values.append(float(mse))
        print(f"saved {fig_path} mse={mse:.6f}")
        if tracker is not None:
            log_payload: dict = {
                "rollout/index": rollout_idx + 1,
                "rollout/mse": mse,
                "rollout/available_future_steps": int(batch_device["z_future"].size(1)),
                wandb_image_key: wandb.Image(str(fig_path)),
            }
            if args.action_branching:
                log_payload["rollout/mse_gt_continuous_action"] = mse
            wandb.log(log_payload, step=rollout_idx + 1)
    if tracker is not None:
        if mse_values:
            wandb.log(
                {
                    "rollout/mse_mean": float(sum(mse_values) / len(mse_values)),
                    "rollout/mse_min": float(min(mse_values)),
                    "rollout/mse_max": float(max(mse_values)),
                },
                step=len(mse_values) + 1,
            )
            tracker.run.summary["rollout_mse_mean"] = float(sum(mse_values) / len(mse_values))
            tracker.run.summary["rollout_count"] = len(mse_values)
            tracker.run.summary["checkpoint_global_step"] = int(checkpoint.get("global_step", 0))
        tracker.log_artifact_path("eb-nav-rollout-figures", output_dir, artifact_type="visualization")
        tracker.finish()


if __name__ == "__main__":
    main()
