"""WM 训练入口。"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.train.latent_cache import build_wm_dataset_with_cache, resolve_manifest_path
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.run_output import build_run_output_dir
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker
from src.wm.encoders import build_wm_image_encoder
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.losses import action_supervision_loss, sigreg_loss, wm_reconstruction_loss
from src.wm.model import CFMWorldModel


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _resolve_patch_layout(wm_cfg: DictConfig) -> tuple[int, int]:
    num_patches = int(getattr(wm_cfg.encoder, "num_patches", 0))
    latent_dim = int(wm_cfg.latent_dim)
    if num_patches <= 0:
        raise ValueError("wm.encoder.num_patches 必须为正整数。")
    if latent_dim % num_patches != 0:
        raise ValueError(f"wm.latent_dim 必须能被 num_patches 整除: {latent_dim} / {num_patches}")
    return num_patches, latent_dim // num_patches


def _parse_temporal_stride(value: object) -> int | tuple[int, int]:
    if isinstance(value, int):
        return max(1, int(value))
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError("pipeline.train.temporal_stride 区间必须包含两个整数 [min, max]。")
        low = max(1, int(value[0]))
        high = max(low, int(value[1]))
        return (low, high)
    return 1


def _sigreg_weight(step: int, target_weight: float, warmup_steps: int) -> float:
    if target_weight <= 0.0:
        return 0.0
    if warmup_steps <= 0:
        return target_weight
    ratio = min(1.0, max(0.0, float(step) / float(warmup_steps)))
    return target_weight * ratio


def _latent_batch_stats(latents: torch.Tensor) -> tuple[float, float, float]:
    feature_dim = int(latents.size(-1))
    flat = latents.reshape(-1, feature_dim)
    if flat.size(0) <= 1:
        return 0.0, 0.0, 0.0
    var = flat.var(dim=0, unbiased=False)
    var_min = float(var.min().item())
    mean_norm = float(flat.mean(dim=0).norm().item())
    cov_trace = float(var.sum().item())
    return var_min, mean_norm, cov_trace


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    temporal_stride = _parse_temporal_stride(train_cfg.get("temporal_stride", 1))
    sigreg_cfg = train_cfg.get("sigreg", {})
    sigreg_enabled = bool(getattr(sigreg_cfg, "enabled", False))
    sigreg_target_weight = float(getattr(sigreg_cfg, "weight", 0.0))
    sigreg_warmup_steps = int(getattr(sigreg_cfg, "warmup_steps", 0))
    sigreg_num_projections = int(getattr(sigreg_cfg, "num_projections", 256))
    sigreg_num_quadrature_points = int(getattr(sigreg_cfg, "num_quadrature_points", 16))
    sigreg_t_min = float(getattr(sigreg_cfg, "t_min", 0.2))
    sigreg_t_max = float(getattr(sigreg_cfg, "t_max", 4.0))
    sigreg_kernel_sigma = float(getattr(sigreg_cfg, "kernel_sigma", 1.0))
    run_dir = build_run_output_dir(
        path_segments=[
            str(train_cfg.operation.outputs_root),
            "wm",
            str(wm_cfg.name),
        ],
    )
    tracker = init_tracker(
        task_name="train_wm",
        config={
            "batch_size": int(train_cfg.batch_size),
            "epochs": int(train_cfg.epochs),
            "lr": float(train_cfg.lr),
            "dataset": str(cfg.dataset.get("name", "ai2thor")),
            "wm": str(cfg.wm.get("name", "cfm")),
            "pm": str(cfg.pm.name),
            "vlm": str(cfg.vlm.name),
            "temporal_stride": str(temporal_stride),
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_target_weight,
            "log_every_n_steps": int(train_cfg.get("log_every_n_steps", 0)),
        },
    )
    device = torch.device(str(train_cfg.device))
    resolved_manifest_path = resolve_manifest_path(str(dataset_cfg.manifest_path))
    image_encoder = build_wm_image_encoder(wm_cfg=wm_cfg)
    with progress_context() as warmup_progress:
        warmup_task = warmup_progress.add_task("warmup_latent_cache", total=1)

        def _on_latent_progress(done: int, total: int) -> None:
            if total <= 0:
                warmup_progress.update(
                    warmup_task,
                    total=1,
                    completed=1,
                    description="warmup_latent_cache 已命中",
                )
                return
            warmup_progress.update(
                warmup_task,
                total=total,
                completed=done,
                description=f"warmup_latent_cache {done}/{total}",
            )

        dataset, latent_cache_path = build_wm_dataset_with_cache(
            manifest_path=resolved_manifest_path,
            wm_name=str(wm_cfg.name),
            latent_dim=int(dataset_cfg.latent_dim),
            action_dim=int(dataset_cfg.action_dim),
            history_len=int(wm_cfg.history_len),
            rollout_steps=int(train_cfg.rollout_steps),
            temporal_stride=temporal_stride,
            image_encoder=image_encoder,
            encoder_num_workers=int(train_cfg.encoder_num_workers),
            encoder_batch_size=int(train_cfg.encoder_batch_size),
            expected_num_patches=int(getattr(wm_cfg.encoder, "num_patches", 0)),
            expected_token_dim=(
                int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 1))
                if int(getattr(wm_cfg.encoder, "num_patches", 0)) > 0
                else 0
            ),
            on_latent_progress=_on_latent_progress,
        )
    # 预编码完成后，训练阶段仅从缓存读取 latent，支持 DataLoader 多进程并行。
    dataset.disable_encoder_after_warmup()
    if len(dataset) == 0:
        raise RuntimeError("数据集为空，请先执行 collect_data。")
    loader_num_workers = int(train_cfg.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.batch_size),
        shuffle=True,
        num_workers=loader_num_workers,
        persistent_workers=loader_num_workers > 0,
    )
    num_patches, token_dim = _resolve_patch_layout(wm_cfg=wm_cfg)
    model = CFMWorldModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(wm_cfg.action_dim),
        hidden_dim=int(wm_cfg.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.transformer.num_layers),
        num_heads=int(wm_cfg.transformer.num_heads),
        dropout=float(wm_cfg.transformer.dropout),
        conditioning_mode=str(getattr(wm_cfg.conditioning, "mode", "adaln")),
    ).to(device)
    inverse_dynamics = InverseDynamicsModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(wm_cfg.action_dim),
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.inverse_dynamics.num_layers),
        num_heads=int(wm_cfg.inverse_dynamics.num_heads),
        dropout=float(wm_cfg.inverse_dynamics.dropout),
    ).to(device)
    action_mapper = torch.nn.Linear(int(wm_cfg.action_dim), int(dataset_cfg.action_dim)).to(device)
    wm_lr = float(train_cfg.wm_lr) if train_cfg.get("wm_lr") is not None else float(train_cfg.lr)
    idm_lr = float(train_cfg.idm_lr) if train_cfg.get("idm_lr") is not None else float(train_cfg.lr)
    wm_optimizer = torch.optim.Adam(model.parameters(), lr=wm_lr)
    idm_optimizer = torch.optim.Adam(
        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
        lr=idm_lr,
    )
    wm_params = _count_trainable_params(model)
    idm_params = _count_trainable_params(inverse_dynamics)
    mapper_params = _count_trainable_params(action_mapper)
    show_kv_table(
        "Train WM",
        [
            ("device", str(device)),
            ("dataset_size", str(len(dataset))),
            ("epochs", str(train_cfg.epochs)),
            ("batch_size", str(train_cfg.batch_size)),
            ("rollout_steps", str(train_cfg.rollout_steps)),
            ("temporal_stride", str(temporal_stride)),
            ("wm_params", f"{wm_params:,}"),
            ("idm_params", f"{idm_params:,}"),
            ("action_mapper_params", f"{mapper_params:,}"),
        ],
    )

    mode = str(train_cfg.training_mode).strip().lower()
    if mode not in {"unsupervised", "semi_supervised", "fully_supervised"}:
        raise ValueError(f"不支持的 training_mode={train_cfg.training_mode}")
    model.train()
    inverse_dynamics.train()
    action_mapper.train()
    last_loss = None
    last_recon_loss = 0.0
    last_action_loss = 0.0
    last_sigreg_loss = 0.0
    last_sigreg_weight = 0.0
    last_latent_var_min = 0.0
    last_latent_mean_norm = 0.0
    last_latent_cov_trace = 0.0
    step_loss_keys = [f"train/loss_recon_step_{i}" for i in range(1, int(train_cfg.rollout_steps) + 1)]
    last_step_losses = {key: 0.0 for key in step_loss_keys}
    total_epochs = int(train_cfg.epochs)
    total_batches = max(1, len(loader))
    global_step = 0
    log_every_n_steps = int(train_cfg.get("log_every_n_steps", 0))
    detach_idm_in_wm = bool(train_cfg.get("detach_idm_in_wm", True))
    grad_clip_norm = float(train_cfg.grad_clip_norm)
    rollout_steps = int(train_cfg.rollout_steps)
    with progress_context() as progress:
        task = progress.add_task("training_wm", total=total_epochs * total_batches)
        for epoch in range(int(train_cfg.epochs)):
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_action_loss = 0.0
            epoch_sigreg_loss = 0.0
            epoch_latent_var_min = 0.0
            epoch_latent_mean_norm = 0.0
            epoch_latent_cov_trace = 0.0
            epoch_sigreg_weight = 0.0
            epoch_step_losses = [0.0 for _ in range(rollout_steps)]
            for batch_idx, batch in enumerate(loader, start=1):
                global_step += 1
                z_history = batch["z_history"].to(device)
                action_history = batch["action_history"].to(device)
                z_future = batch["z_future"].to(device)
                gt_action_future = batch["gt_action_future"].to(device)
                pred_action = None
                pred_action_for_wm = None
                if mode in {"unsupervised", "semi_supervised"}:
                    pred_action = inverse_dynamics(z_history.detach() if mode == "semi_supervised" else z_history)
                    pred_action_for_wm = (
                        pred_action.detach() if (mode == "semi_supervised" and detach_idm_in_wm) else pred_action
                    )
                rollout_z_history = z_history
                rollout_action_history = action_history.clone()
                loss_recon_steps: list[torch.Tensor] = []
                for step_idx in range(rollout_steps):
                    if mode == "fully_supervised":
                        rollout_action_history[:, -1, :] = gt_action_future[:, step_idx, :]
                    elif step_idx == 0:
                        rollout_action_history[:, -1, :] = pred_action_for_wm
                    pred_z_step = model(rollout_z_history, rollout_action_history)
                    step_target = z_future[:, step_idx, :]
                    step_loss = wm_reconstruction_loss(pred_z_step, step_target)
                    loss_recon_steps.append(step_loss)
                    rollout_z_history = torch.cat([rollout_z_history[:, 1:, ...], pred_z_step.unsqueeze(1)], dim=1)
                    if step_idx < rollout_steps - 1:
                        rollout_action_history = torch.cat(
                            [rollout_action_history[:, 1:, :], gt_action_future[:, step_idx, :].unsqueeze(1)],
                            dim=1,
                        )
                loss_recon = torch.stack(loss_recon_steps).mean()
                loss_recon_weighted = float(train_cfg.reconstruction_weight) * loss_recon
                latent_for_reg = torch.cat([z_history, z_future], dim=1)
                latent_var_min, latent_mean_norm, latent_cov_trace = _latent_batch_stats(latent_for_reg.detach())
                loss_sigreg = torch.tensor(0.0, device=device)
                current_sigreg_weight = 0.0
                if sigreg_enabled:
                    loss_sigreg = sigreg_loss(
                        latent_for_reg,
                        num_projections=sigreg_num_projections,
                        num_quadrature_points=sigreg_num_quadrature_points,
                        t_min=sigreg_t_min,
                        t_max=sigreg_t_max,
                        kernel_sigma=sigreg_kernel_sigma,
                    )
                    current_sigreg_weight = _sigreg_weight(
                        step=global_step,
                        target_weight=sigreg_target_weight,
                        warmup_steps=sigreg_warmup_steps,
                    )
                loss_action = torch.tensor(0.0, device=device)
                if mode == "semi_supervised":
                    idm_optimizer.zero_grad(set_to_none=True)
                    mapped_action = action_mapper(pred_action)
                    loss_action = action_supervision_loss(mapped_action, gt_action_future[:, 0, :])
                    loss_action_weighted = float(train_cfg.semi_supervised_weight) * loss_action
                    loss_action_weighted.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
                        grad_clip_norm,
                    )
                    idm_optimizer.step()
                elif mode == "unsupervised":
                    idm_optimizer.zero_grad(set_to_none=True)
                wm_optimizer.zero_grad(set_to_none=True)
                loss_wm_total = loss_recon_weighted + current_sigreg_weight * loss_sigreg
                loss_wm_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                wm_optimizer.step()
                if mode == "unsupervised":
                    torch.nn.utils.clip_grad_norm_(
                        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
                        grad_clip_norm,
                    )
                    idm_optimizer.step()
                batch_loss = float(loss_wm_total.item()) + (
                    float(train_cfg.semi_supervised_weight) * float(loss_action.item()) if mode == "semi_supervised" else 0.0
                )
                # step 级指标用于观察 batch 内优化动态；可通过 log_every_n_steps 降采样。
                if log_every_n_steps > 0 and (global_step == 1 or global_step % log_every_n_steps == 0):
                    tracker.log_metrics(
                        {
                            "train_step/loss": batch_loss,
                            "train_step/loss_recon": float(loss_recon.item()),
                            "train_step/loss_action": float(loss_action.item()),
                            "train_step/loss_sigreg": float(loss_sigreg.item()),
                            "train_step/sigreg_weight": current_sigreg_weight,
                            "train_step/global_step": global_step,
                            "train_step/epoch": epoch,
                        },
                        step=global_step,
                    )
                epoch_loss += batch_loss
                epoch_recon_loss += float(loss_recon.item())
                epoch_action_loss += float(loss_action.item())
                epoch_sigreg_loss += float(loss_sigreg.item())
                epoch_latent_var_min += latent_var_min
                epoch_latent_mean_norm += latent_mean_norm
                epoch_latent_cov_trace += latent_cov_trace
                epoch_sigreg_weight += current_sigreg_weight
                for step_idx, step_loss in enumerate(loss_recon_steps):
                    epoch_step_losses[step_idx] += float(step_loss.item())
                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"epoch={epoch + 1}/{total_epochs} "
                        f"batch={batch_idx}/{total_batches} mode={mode} loss={batch_loss:.6f}"
                    ),
                )
            last_loss = epoch_loss / max(1, len(loader))
            last_recon_loss = epoch_recon_loss / max(1, len(loader))
            last_action_loss = epoch_action_loss / max(1, len(loader))
            last_sigreg_loss = epoch_sigreg_loss / max(1, len(loader))
            last_latent_var_min = epoch_latent_var_min / max(1, len(loader))
            last_latent_mean_norm = epoch_latent_mean_norm / max(1, len(loader))
            last_latent_cov_trace = epoch_latent_cov_trace / max(1, len(loader))
            last_sigreg_weight = epoch_sigreg_weight / max(1, len(loader))
            last_step_losses = {
                f"train/loss_recon_step_{i + 1}": epoch_step_losses[i] / max(1, len(loader))
                for i in range(rollout_steps)
            }
            tracker.log_metrics(
                {
                    "train/loss": last_loss,
                    "train/loss_recon": last_recon_loss,
                    "train/loss_action": last_action_loss,
                    "train/loss_sigreg": last_sigreg_loss,
                    "train/sigreg_weight": last_sigreg_weight,
                    "train/latent_var_min": last_latent_var_min,
                    "train/latent_mean_norm": last_latent_mean_norm,
                    "train/latent_cov_trace": last_latent_cov_trace,
                    "train/epoch": epoch,
                    **last_step_losses,
                },
                step=epoch,
            )

    out_dir = ensure_dir(run_dir)
    ckpt_path = Path(out_dir) / "wm.pt"
    idm_ckpt_path = Path(out_dir) / "inverse_dynamics.pt"
    mapper_ckpt_path = Path(out_dir) / "action_mapper.pt"
    torch.save(model.state_dict(), ckpt_path)
    torch.save(inverse_dynamics.state_dict(), idm_ckpt_path)
    torch.save(action_mapper.state_dict(), mapper_ckpt_path)
    write_json(
        Path(out_dir) / "train_metrics.json",
        {
            "last_loss": last_loss,
            "last_loss_recon": last_recon_loss,
            "last_loss_action": last_action_loss,
            "last_loss_sigreg": last_sigreg_loss,
            "training_mode": mode,
            "rollout_steps": rollout_steps,
            "temporal_stride": temporal_stride,
            "detach_idm_in_wm": detach_idm_in_wm,
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_target_weight,
            "sigreg_warmup_steps": sigreg_warmup_steps,
            "log_every_n_steps": log_every_n_steps,
            "last_sigreg_weight": last_sigreg_weight,
            "last_latent_var_min": last_latent_var_min,
            "last_latent_mean_norm": last_latent_mean_norm,
            "last_latent_cov_trace": last_latent_cov_trace,
            "step_losses": last_step_losses,
        },
    )
    tracker.log_artifact_path("wm-checkpoint", ckpt_path, artifact_type="model")
    tracker.log_artifact_path("idm-checkpoint", idm_ckpt_path, artifact_type="model")
    tracker.log_artifact_path("action-mapper-checkpoint", mapper_ckpt_path, artifact_type="model")
    tracker.log_artifact_path("wm-train-metrics", Path(out_dir) / "train_metrics.json", artifact_type="metrics")
    tracker.finish()
    success(f"训练完成 checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()

