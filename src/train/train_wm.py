"""WM 训练入口。"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from typing import Any
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)

from src.core import FileSystemModelProvider, WMDataProvider, WMModelAdapter
from src.train.latent_cache import build_wm_dataset_with_cache
from src.train.manifest_resolver import resolve_manifest_for_split
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed
from src.visualize.wandb_tracker import init_tracker
from src.wm.encoder import build_wm_image_encoder, build_trainable_image_encoder
from src.wm.encoder.dino import TrainableDinoV2Encoder
from src.wm.encoder.qwen import TrainableQwenLatentAdapter
from src.wm.predictor import WMModel, LeWMModel
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.action_mapper import build_action_mapper
from src.wm.factory import resolve_patch_layout


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


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


def _linear_warmup_lambda(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step + 1) / float(warmup_steps)))


def _safe_cosine_mean(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = x.reshape(-1, x.shape[-1]).float()
    y_flat = y.reshape(-1, y.shape[-1]).float()
    if x_flat.numel() == 0 or y_flat.numel() == 0:
        return 0.0
    cos = torch.nn.functional.cosine_similarity(x_flat, y_flat, dim=-1)
    return float(cos.mean().item())


def _compute_qwen_aux_losses(
    *,
    trainable_encoder: TrainableQwenLatentAdapter,
    z_history: torch.Tensor,
    z_future: torch.Tensor,
    action_history: torch.Tensor,
    physics_weight: float,
    distill_weight: float,
    temporal_weight: float,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 Qwen latent adapter 的辅助损失并返回变换后的 latent。"""
    z_history_student = trainable_encoder(z_history)
    z_future_student = trainable_encoder(z_future)
    z_history_teacher = trainable_encoder.teacher_forward(z_history)
    z_future_teacher = trainable_encoder.teacher_forward(z_future)
    distill_loss = torch.tensor(0.0, device=z_history.device)
    if distill_weight > 0.0:
        distill_loss = (
            torch.nn.functional.mse_loss(z_history_student, z_history_teacher)
            + torch.nn.functional.mse_loss(z_future_student, z_future_teacher)
        ) * 0.5
    physics_loss = torch.tensor(0.0, device=z_history.device)
    if physics_weight > 0.0 and z_history_student.size(1) > 0 and z_future_student.size(1) > 0:
        delta_pred = z_future_student[:, 0] - z_history_student[:, -1]
        action_signal = action_history[:, -1].float().norm(dim=-1, keepdim=True)
        while action_signal.dim() < delta_pred.dim():
            action_signal = action_signal.unsqueeze(-1)
        target_delta = action_signal * torch.tanh(z_history_student[:, -1])
        physics_loss = torch.nn.functional.mse_loss(delta_pred, target_delta)
    temporal_loss = torch.tensor(0.0, device=z_history.device)
    if temporal_weight > 0.0:
        if z_history_student.size(1) > 1:
            temporal_loss = temporal_loss + torch.nn.functional.mse_loss(
                z_history_student[:, 1:], z_history_student[:, :-1]
            )
        if z_future_student.size(1) > 1:
            temporal_loss = temporal_loss + torch.nn.functional.mse_loss(
                z_future_student[:, 1:], z_future_student[:, :-1]
            )
    aux_loss = physics_weight * physics_loss + distill_weight * distill_loss + temporal_weight * temporal_loss
    aux_metrics = {
        "loss_distill": float(distill_loss.item()),
        "loss_physics": float(physics_loss.item()),
        "loss_temporal": float(temporal_loss.item()),
        "embedding_cosine_to_teacher": _safe_cosine_mean(
            torch.cat([z_history_student, z_future_student], dim=1),
            torch.cat([z_history_teacher, z_future_teacher], dim=1),
        ),
    }
    return aux_metrics, aux_loss, z_history_student, z_future_student


def _train_step_with_encoder(
    batch: Any,
    wm_model: Any,
    trainable_encoder: TrainableDinoV2Encoder | None,
    encoder_optimizer: torch.optim.Optimizer | None,
    encoder_scheduler: Any | None,
    device: torch.device,
    sigreg_weight: float,
    physics_weight: float,
    distill_weight: float,
    temporal_weight: float,
    global_step: int,
    sigreg_warmup_steps: int,
    grad_clip_norm: float,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """执行带有可微调 encoder 的训练步骤（方案2）。

    Args:
        batch: 训练数据批次
        wm_model: WM 模型
        trainable_encoder: 可微调 encoder（方案2）
        encoder_optimizer: encoder 优化器
        encoder_scheduler: encoder 学习率调度器
        device: 计算设备
        sigreg_weight: SIGReg 权重
        global_step: 全局步数
        sigreg_warmup_steps: SIGReg warmup 步数
        grad_clip_norm: 梯度裁剪范数

    Returns:
        (step_metrics, encoder_loss): 训练指标和 encoder 损失
    """
    if trainable_encoder is None or encoder_optimizer is None:
        prepared = {
            "z_history": batch["z_history"].to(device),
            "action_history": batch["action_history"].to(device),
            "z_future": batch["z_future"].to(device),
            "gt_action_future": batch["gt_action_future"].to(device),
        }
        return wm_model.train_step(prepared), None

    # 方案2：需要处理图像 encoder 的前向和反向传播
    z_history = batch["z_history"].to(device)
    action_history = batch["action_history"].to(device)
    z_future = batch["z_future"].to(device)
    gt_action_future = batch["gt_action_future"].to(device)
    aux_metrics = {
        "loss_distill": 0.0,
        "loss_physics": 0.0,
        "loss_temporal": 0.0,
        "embedding_cosine_to_teacher": 0.0,
    }

    # 计算 SIGReg 权重（带 warmup）
    current_encoder_sigreg_weight = 0.0
    if sigreg_weight > 0.0 and sigreg_warmup_steps > 0:
        current_encoder_sigreg_weight = sigreg_weight * min(
            1.0, float(global_step + 1) / float(sigreg_warmup_steps)
        )

    # Encoder 前向：计算 encoder 输出的 SIGReg 损失
    encoder_sigreg_loss = torch.tensor(0.0, device=device)
    if isinstance(trainable_encoder, TrainableDinoV2Encoder) and trainable_encoder.sigreg_enabled and current_encoder_sigreg_weight > 0.0:
        # 对 z_history 和 z_future 应用 SIGReg
        latent_seq = torch.cat([z_history, z_future], dim=1)  # [B, T, P, D]
        encoder_sigreg_loss = trainable_encoder.compute_sigreg(latent_seq)
    encoder_aux_loss = torch.tensor(0.0, device=device)
    if isinstance(trainable_encoder, TrainableQwenLatentAdapter):
        aux_metrics, encoder_aux_loss, z_history, z_future = _compute_qwen_aux_losses(
            trainable_encoder=trainable_encoder,
            z_history=z_history,
            z_future=z_future,
            action_history=action_history,
            physics_weight=physics_weight,
            distill_weight=distill_weight,
            temporal_weight=temporal_weight,
        )

    # 正常的 WM 训练步骤（使用缓存的 latent，不需要 encoder 前向）
    prepared = {
        "z_history": z_history,
        "action_history": action_history,
        "z_future": z_future,
        "gt_action_future": gt_action_future,
    }
    wm_metrics = wm_model.train_step(prepared)

    # 如果有 encoder SIGReg 损失，需要单独更新 encoder
    total_encoder_loss = current_encoder_sigreg_weight * encoder_sigreg_loss + encoder_aux_loss
    if total_encoder_loss > 0.0:
        encoder_optimizer.zero_grad(set_to_none=True)
        total_encoder_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_encoder.parameters(), grad_clip_norm)
        encoder_optimizer.step()
        # 更新 encoder scheduler（在 optimizer.step() 之后）
        if encoder_scheduler is not None:
            encoder_scheduler.step()

        # 更新 encoder SIGReg 损失到 metrics
        wm_metrics["loss_encoder_sigreg"] = float(encoder_sigreg_loss.item())
        wm_metrics["encoder_sigreg_weight"] = current_encoder_sigreg_weight
    else:
        wm_metrics["loss_encoder_sigreg"] = 0.0
        wm_metrics["encoder_sigreg_weight"] = 0.0
    wm_metrics.update(aux_metrics)

    return wm_metrics, encoder_sigreg_loss


def _cosine_annealing_lambda(
    step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """warmup + cosine annealing 调度器。

    Args:
        step: 当前步数
        warmup_steps: warmup 总步数（warmup 期间 lr 从 0 线性上升到 base_lr）
        total_steps: 总步数（warmup + annealing）
        min_lr_ratio: 最终学习率相对于 base_lr 的比例
    """
    if step < warmup_steps:
        # warmup 阶段
        return float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        # 没有 annealing，退回到恒定 lr
        return 1.0
    # cosine annealing 阶段
    progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0


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
    loss_cfg = train_cfg.get("loss", {})
    physics_weight = float(getattr(loss_cfg, "physics_weight", 0.0))
    distill_weight = float(getattr(loss_cfg, "distill_weight", 0.0))
    temporal_weight = float(getattr(loss_cfg, "temporal_weight", 0.0))
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    cos_annealing_steps = int(train_cfg.get("cos_annealing_steps", 0))  # 0 表示自动推断
    cos_min_lr_ratio = float(train_cfg.get("cos_min_lr_ratio", 0.1))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema_decay = float(getattr(ema_cfg, "decay", 0.999))
    path_segments = [str(train_cfg.operation.outputs_root), "wm", str(wm_cfg.name)]
    force_new_run = bool(getattr(train_cfg.operation, "force_new_run", False))
    model_provider = FileSystemModelProvider(path_segments=path_segments)
    run_dir, resumed = model_provider.resolve_run_dir(force_new=force_new_run)
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
            "physics_weight": physics_weight,
            "distill_weight": distill_weight,
            "temporal_weight": temporal_weight,
            "log_every_n_steps": int(train_cfg.get("log_every_n_steps", 0)),
            "lr_warmup_steps": warmup_steps,
            "cos_annealing_steps": cos_annealing_steps,
            "cos_min_lr_ratio": cos_min_lr_ratio,
            "weight_decay": weight_decay,
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay,
            "run_resumed": resumed,
            "force_new_run": force_new_run,
        },
    )
    device = torch.device(str(train_cfg.device))
    manifests_cfg = dataset_cfg.get("manifests", {})
    manifests_cfg = dict(manifests_cfg)
    outputs_root = str(train_cfg.operation.outputs_root)

    def _build_split_manifest_path(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=outputs_root,
            dataset_name=str(dataset_cfg.name),
        )

    def _build_split_loader(split: str, encoder_for_split: Any = None) -> DataLoader | None:
        manifest_path = _build_split_manifest_path(split)
        if not manifest_path.exists():
            return None
        encoder = encoder_for_split if encoder_for_split is not None else build_wm_image_encoder(wm_cfg=wm_cfg)
        dataset, _ = build_wm_dataset_with_cache(
            run_dir=manifest_path,
            wm_name=str(wm_cfg.name),
            latent_dim=int(wm_cfg.latent_dim),
            action_dim=int(dataset_cfg.action_dim),
            history_len=int(wm_cfg.history_len),
            temporal_stride=temporal_stride,
            image_encoder=encoder,
            encoder_num_workers=int(train_cfg.encoder_num_workers),
            encoder_batch_size=int(train_cfg.encoder_batch_size),
            expected_num_patches=int(getattr(wm_cfg.encoder, "num_patches", 0)),
            expected_token_dim=(
                int(wm_cfg.latent_dim) // int(getattr(wm_cfg.encoder, "num_patches", 1))
                if int(getattr(wm_cfg.encoder, "num_patches", 0)) > 0
                else 0
            ),
        )
        dataset.disable_encoder_after_warmup()
        num_workers = int(train_cfg.num_workers)
        return DataLoader(
            dataset,
            batch_size=int(train_cfg.batch_size),
            shuffle=(split == "train"),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    train_loader = _build_split_loader("train")
    val_loader = _build_split_loader("val")
    test_loader = _build_split_loader("test")

    if train_loader is None:
        raise RuntimeError("训练集 manifest 不存在，请先执行 collect_data。")

    # 方案2：创建可微调 encoder（用于图像 encoder 微调 + SIGReg）
    trainable_encoder = None
    encoder_optimizer = None
    encoder_scheduler = None
    encoder_lr = float(train_cfg.get("encoder_lr", train_cfg.lr))
    encoder_sigreg_weight = sigreg_target_weight
    encoder_finetune_cfg = train_cfg.get("encoder_finetune", {})
    encoder_finetune_enabled = bool(getattr(encoder_finetune_cfg, "enabled", False))
    if sigreg_enabled or encoder_finetune_enabled:
        trainable_encoder = build_trainable_image_encoder(wm_cfg=wm_cfg, train_cfg=train_cfg)
        if trainable_encoder is not None:
            logger.info(
                "方案2：启用可微调图像 encoder，encoder_lr=%.6f, sigreg_weight=%.4f, encoder_finetune=%s",
                encoder_lr,
                encoder_sigreg_weight,
                str(encoder_finetune_enabled),
            )
            trainable_encoder = trainable_encoder.to(device)
            encoder_optimizer = torch.optim.AdamW(
                trainable_encoder.parameters(), lr=encoder_lr, weight_decay=weight_decay
            )
            if cos_annealing_steps > 0:
                encoder_scheduler = LambdaLR(
                    encoder_optimizer,
                    lr_lambda=lambda step: _cosine_annealing_lambda(
                        step, warmup_steps, cos_annealing_steps, cos_min_lr_ratio
                    ),
                )
            else:
                encoder_scheduler = LambdaLR(
                    encoder_optimizer, lr_lambda=lambda step: _linear_warmup_lambda(step, warmup_steps)
                )

    data_provider = WMDataProvider(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        path_segments=path_segments,
    )
    num_patches, token_dim = resolve_patch_layout(wm_cfg=wm_cfg)
    wm_type = resolve_wm_type(wm_cfg)
    wm_module = build_world_model(
        wm_cfg=wm_cfg,
        train_cfg=train_cfg,
        action_dim=int(dataset_cfg.action_dim),
        device=device,
    )
    inverse_dynamics = InverseDynamicsModel(
        latent_dim=int(wm_cfg.latent_dim),
        action_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
        history_len=int(wm_cfg.history_len),
        num_patches=num_patches,
        token_dim=token_dim,
        num_layers=int(wm_cfg.inverse_dynamics.num_layers),
        num_heads=int(wm_cfg.inverse_dynamics.num_heads),
        dropout=float(wm_cfg.inverse_dynamics.dropout),
    ).to(device)
    action_mapper = build_action_mapper(
        input_dim=int(dataset_cfg.action_dim),
        output_dim=int(dataset_cfg.action_dim),
        hidden_dim=int(wm_cfg.inverse_dynamics.hidden_dim),
    ).to(device)
    wm_lr = float(train_cfg.wm_lr) if train_cfg.get("wm_lr") is not None else float(train_cfg.lr)
    idm_lr = float(train_cfg.idm_lr) if train_cfg.get("idm_lr") is not None else float(train_cfg.lr)
    wm_optimizer = torch.optim.AdamW(wm_module.parameters(), lr=wm_lr, weight_decay=weight_decay)
    idm_optimizer = torch.optim.AdamW(
        list(inverse_dynamics.parameters()) + list(action_mapper.parameters()),
        lr=idm_lr,
        weight_decay=weight_decay,
    )
    # 计算总训练步数（用于自动推断 cos_annealing_steps）
    total_epochs_cfg = int(train_cfg.epochs)
    total_batches_per_epoch = max(1, len(train_loader))
    total_training_steps = total_epochs_cfg * total_batches_per_epoch
    # 自动推断：若 cos_annealing_steps=0，则设为总训练步数
    if cos_annealing_steps <= 0:
        cos_annealing_steps = total_training_steps
    if cos_annealing_steps > 0:
        logger.info(
            "使用 Cosine Annealing 调度器: warmup=%d, total=%d (auto=%s), min_ratio=%.2f",
            warmup_steps,
            cos_annealing_steps,
            "yes" if train_cfg.get("cos_annealing_steps", 0) == 0 else "no",
            cos_min_lr_ratio,
        )
        wm_scheduler = LambdaLR(
            wm_optimizer,
            lr_lambda=lambda step: _cosine_annealing_lambda(
                step, warmup_steps, cos_annealing_steps, cos_min_lr_ratio
            ),
        )
        idm_scheduler = LambdaLR(
            idm_optimizer,
            lr_lambda=lambda step: _cosine_annealing_lambda(
                step, warmup_steps, cos_annealing_steps, cos_min_lr_ratio
            ),
        )
    else:
        logger.info("使用线性 warmup 调度器: warmup=%d", warmup_steps)
        wm_scheduler = LambdaLR(wm_optimizer, lr_lambda=lambda step: _linear_warmup_lambda(step, warmup_steps))
        idm_scheduler = LambdaLR(idm_optimizer, lr_lambda=lambda step: _linear_warmup_lambda(step, warmup_steps))
    mode = str(train_cfg.training_mode).strip().lower()
    if mode not in {"unsupervised", "semi_supervised", "fully_supervised"}:
        raise ValueError(f"不支持的 training_mode={train_cfg.training_mode}")
    wm_module.train()
    inverse_dynamics.train()
    action_mapper.train()
    grad_clip_norm = float(train_cfg.grad_clip_norm)
    detach_idm_in_wm = bool(train_cfg.get("detach_idm_in_wm", True))
    log_every_n_steps = int(train_cfg.get("log_every_n_steps", 0))
    if wm_type == "lewm":
        wm_model = LeWMModel(
            wm=wm_module,
            inverse_dynamics=inverse_dynamics,
            action_mapper=action_mapper,
            wm_optimizer=wm_optimizer,
            idm_optimizer=idm_optimizer,
            wm_scheduler=wm_scheduler,
            idm_scheduler=idm_scheduler,
            device=device,
            training_mode=mode,
            reconstruction_weight=float(train_cfg.reconstruction_weight),
            semi_supervised_weight=float(train_cfg.semi_supervised_weight),
            grad_clip_norm=grad_clip_norm,
            ema_decay=ema_decay,
            detach_idm_in_wm=detach_idm_in_wm,
            sigreg_enabled=sigreg_enabled,
            sigreg_target_weight=sigreg_target_weight,
            sigreg_warmup_steps=sigreg_warmup_steps,
        )
    else:
        wm_model = WMModel(
            wm=wm_module,
            inverse_dynamics=inverse_dynamics,
            action_mapper=action_mapper,
            wm_optimizer=wm_optimizer,
            idm_optimizer=idm_optimizer,
            wm_scheduler=wm_scheduler,
            idm_scheduler=idm_scheduler,
            device=device,
            training_mode=mode,
            reconstruction_weight=float(train_cfg.reconstruction_weight),
            semi_supervised_weight=float(train_cfg.semi_supervised_weight),
            grad_clip_norm=grad_clip_norm,
            ema_decay=ema_decay,
            detach_idm_in_wm=detach_idm_in_wm,
            sigreg_enabled=sigreg_enabled,
            sigreg_target_weight=sigreg_target_weight,
            sigreg_warmup_steps=sigreg_warmup_steps,
            sigreg_num_projections=sigreg_num_projections,
            sigreg_num_quadrature_points=sigreg_num_quadrature_points,
            sigreg_t_min=sigreg_t_min,
            sigreg_t_max=sigreg_t_max,
            sigreg_kernel_sigma=sigreg_kernel_sigma,
        )
    wm_adapter = WMModelAdapter(
        train_step_fn=lambda batch: {
            "z_history": batch["z_history"].to(device),
            "action_history": batch["action_history"].to(device),
            "z_future": batch["z_future"].to(device),
            "gt_action_future": batch["gt_action_future"].to(device),
        }
    )
    total_epochs = int(train_cfg.epochs)
    total_batches = max(1, len(train_loader))
    global_step = 0
    start_epoch = 0
    wm_params = _count_trainable_params(wm_module)
    idm_params = _count_trainable_params(inverse_dynamics)
    mapper_params = _count_trainable_params(action_mapper)
    show_kv_table(
        "Train WM",
        [
            ("device", str(device)),
            ("dataset_size", str(len(train_loader))),
            ("epochs", str(total_epochs)),
            ("batch_size", str(train_cfg.batch_size)),
            ("temporal_stride", str(temporal_stride)),
            ("lr_warmup_steps", str(warmup_steps)),
            ("cos_annealing_steps", str(cos_annealing_steps)),
            ("cos_min_lr_ratio", str(cos_min_lr_ratio)),
            ("weight_decay", str(weight_decay)),
            ("ema_enabled", str(ema_enabled)),
            ("ema_decay", str(ema_decay)),
            ("wm_params", f"{wm_params:,}"),
            ("idm_params", f"{idm_params:,}"),
            ("action_mapper_params", f"{mapper_params:,}"),
        ],
    )
    run_status_extra = {"resumed": resumed, "force_new_run": force_new_run}
    if resumed:
        checkpoint_state = model_provider.load_checkpoint(run_dir)
        if checkpoint_state is not None:
            wm_model.load_state(
                checkpoint_state,
                start_epoch=int(checkpoint_state.get("epoch", -1)) + 1,
                global_step=int(checkpoint_state.get("global_step", 0)),
            )
            start_epoch = wm_model._epoch
            global_step = wm_model._global_step
            run_status_extra["resume_epoch"] = max(0, start_epoch)
    model_provider.mark_running(run_dir, **run_status_extra)
    try:
        with progress_context() as progress:
            task = progress.add_task("training_wm", total=total_epochs * total_batches)
            for epoch in range(start_epoch, total_epochs):
                epoch_loss = 0.0
                epoch_recon_loss = 0.0
                epoch_action_loss = 0.0
                epoch_sigreg_loss = 0.0
                epoch_encoder_sigreg_loss = 0.0
                epoch_distill_loss = 0.0
                epoch_physics_loss = 0.0
                epoch_temporal_loss = 0.0
                epoch_embedding_cosine_to_teacher = 0.0
                for batch_idx, batch in enumerate(data_provider.train(), start=1):
                    global_step += 1
                    wm_model._global_step = global_step

                    # 方案2：使用辅助函数处理 encoder 训练
                    step_metrics, _ = _train_step_with_encoder(
                        batch=batch,
                        wm_model=wm_model,
                        trainable_encoder=trainable_encoder,
                        encoder_optimizer=encoder_optimizer,
                        encoder_scheduler=encoder_scheduler,
                        device=device,
                        sigreg_weight=sigreg_target_weight,
                        physics_weight=physics_weight,
                        distill_weight=distill_weight,
                        temporal_weight=temporal_weight,
                        global_step=global_step,
                        sigreg_warmup_steps=sigreg_warmup_steps,
                        grad_clip_norm=grad_clip_norm,
                    )
                    batch_loss = step_metrics["loss"]
                    if log_every_n_steps > 0 and (global_step == 1 or global_step % log_every_n_steps == 0):
                        log_data = {
                            "train_step/loss": batch_loss,
                            "train_step/loss_recon": step_metrics["loss_recon"],
                            "train_step/loss_action": step_metrics["loss_action"],
                            "train_step/loss_sigreg": step_metrics["loss_sigreg"],
                            "train_step/sigreg_weight": step_metrics["sigreg_weight"],
                            "train_step/global_step": global_step,
                            "train_step/epoch": epoch,
                            "train_step/lr_wm": step_metrics["lr_wm"],
                            "train_step/lr_idm": step_metrics["lr_idm"],
                            "train_step/loss_distill": step_metrics.get("loss_distill", 0.0),
                            "train_step/loss_physics": step_metrics.get("loss_physics", 0.0),
                            "train_step/loss_temporal": step_metrics.get("loss_temporal", 0.0),
                            "train_step/embedding_cosine_to_teacher": step_metrics.get(
                                "embedding_cosine_to_teacher", 0.0
                            ),
                        }
                        # 方案2：添加 encoder 相关指标
                        if trainable_encoder is not None:
                            log_data["train_step/loss_encoder_sigreg"] = step_metrics.get("loss_encoder_sigreg", 0.0)
                            log_data["train_step/encoder_sigreg_weight"] = step_metrics.get("encoder_sigreg_weight", 0.0)
                            if encoder_scheduler is not None:
                                log_data["train_step/lr_encoder"] = float(encoder_scheduler.get_last_lr()[0])
                        tracker.log_metrics(log_data, step=global_step)
                    epoch_loss += batch_loss
                    epoch_recon_loss += step_metrics["loss_recon"]
                    epoch_action_loss += step_metrics["loss_action"]
                    epoch_sigreg_loss += step_metrics["loss_sigreg"]
                    epoch_encoder_sigreg_loss += step_metrics.get("loss_encoder_sigreg", 0.0)
                    epoch_distill_loss += step_metrics.get("loss_distill", 0.0)
                    epoch_physics_loss += step_metrics.get("loss_physics", 0.0)
                    epoch_temporal_loss += step_metrics.get("loss_temporal", 0.0)
                    epoch_embedding_cosine_to_teacher += step_metrics.get(
                        "embedding_cosine_to_teacher", 0.0
                    )
                    progress.update(
                        task,
                        advance=1,
                        description=(
                            f"epoch={epoch + 1}/{total_epochs} "
                            f"batch={batch_idx}/{total_batches} mode={mode} loss={batch_loss:.6f}"
                        ),
                    )
                avg_loss = epoch_loss / max(1, len(train_loader))
                avg_recon = epoch_recon_loss / max(1, len(train_loader))
                avg_action = epoch_action_loss / max(1, len(train_loader))
                avg_sigreg = epoch_sigreg_loss / max(1, len(train_loader))
                avg_encoder_sigreg = epoch_encoder_sigreg_loss / max(1, len(train_loader))
                avg_distill = epoch_distill_loss / max(1, len(train_loader))
                avg_physics = epoch_physics_loss / max(1, len(train_loader))
                avg_temporal = epoch_temporal_loss / max(1, len(train_loader))
                avg_embed_cos = epoch_embedding_cosine_to_teacher / max(1, len(train_loader))
                epoch_metrics = {
                    "train/loss": avg_loss,
                    "train/loss_recon": avg_recon,
                    "train/loss_action": avg_action,
                    "train/loss_sigreg": avg_sigreg,
                    "train/epoch": epoch,
                    "train/lr_wm": float(wm_scheduler.get_last_lr()[0]),
                    "train/lr_idm": float(idm_scheduler.get_last_lr()[0]),
                    "train/loss_distill": avg_distill,
                    "train/loss_physics": avg_physics,
                    "train/loss_temporal": avg_temporal,
                    "train/embedding_cosine_to_teacher": avg_embed_cos,
                }
                # 方案2：添加 encoder 相关指标
                if trainable_encoder is not None:
                    epoch_metrics["train/loss_encoder_sigreg"] = avg_encoder_sigreg
                    if encoder_scheduler is not None:
                        epoch_metrics["train/lr_encoder"] = float(encoder_scheduler.get_last_lr()[0])
                tracker.log_metrics(epoch_metrics, step=epoch)
                model_provider.save_checkpoint(
                    run_dir=run_dir,
                    state={
                        "epoch": epoch,
                        "global_step": global_step,
                        **wm_model.get_state(),
                    },
                )
        last_loss = avg_loss
        last_recon_loss = avg_recon
        last_action_loss = avg_action
        last_sigreg_loss = avg_sigreg
        model_provider.mark_completed(run_dir, global_step=global_step, epochs=total_epochs)
    except Exception as exc:
        model_provider.mark_failed(run_dir, error=str(exc), global_step=global_step)
        raise

    out_dir = ensure_dir(run_dir)
    ckpt_path = Path(out_dir) / "wm.pt"
    ema_ckpt_path = Path(out_dir) / "wm_ema.pt"
    idm_ckpt_path = Path(out_dir) / "inverse_dynamics.pt"
    mapper_ckpt_path = Path(out_dir) / "action_mapper.pt"
    encoder_ckpt_path = Path(out_dir) / "encoder.pt"
    torch.save(wm_module.state_dict(), ckpt_path)
    if wm_model._ema_model is not None:
        torch.save(wm_model._ema_model.state_dict(), ema_ckpt_path)
    torch.save(inverse_dynamics.state_dict(), idm_ckpt_path)
    torch.save(action_mapper.state_dict(), mapper_ckpt_path)
    # 方案2：保存 encoder
    if trainable_encoder is not None:
        torch.save(trainable_encoder.state_dict(), encoder_ckpt_path)
    model_provider.save_checkpoint(
        run_dir=run_dir,
        state={
            "epoch": total_epochs - 1,
            "global_step": global_step,
            **wm_model.get_state(),
        },
        is_final=True,
    )
    write_json(
        Path(out_dir) / "train_metrics.json",
        {
            "last_loss": last_loss,
            "last_loss_recon": last_recon_loss,
            "last_loss_action": last_action_loss,
            "last_loss_sigreg": last_sigreg_loss,
            "training_mode": mode,
            "temporal_stride": str(temporal_stride),
            "detach_idm_in_wm": detach_idm_in_wm,
            "sigreg_enabled": sigreg_enabled,
            "sigreg_weight": sigreg_target_weight,
            "physics_weight": physics_weight,
            "distill_weight": distill_weight,
            "temporal_weight": temporal_weight,
            "sigreg_warmup_steps": sigreg_warmup_steps,
            "lr_warmup_steps": warmup_steps,
            "cos_annealing_steps": cos_annealing_steps,
            "cos_min_lr_ratio": cos_min_lr_ratio,
            "weight_decay": weight_decay,
            "ema_enabled": ema_enabled,
            "ema_decay": ema_decay,
            "log_every_n_steps": log_every_n_steps,
        },
    )
    tracker.log_artifact_path("wm-checkpoint", ckpt_path, artifact_type="model")
    if wm_model._ema_model is not None:
        tracker.log_artifact_path("wm-ema-checkpoint", ema_ckpt_path, artifact_type="model")
    tracker.log_artifact_path("idm-checkpoint", idm_ckpt_path, artifact_type="model")
    tracker.log_artifact_path("action-mapper-checkpoint", mapper_ckpt_path, artifact_type="model")
    # 方案2：保存 encoder checkpoint
    if trainable_encoder is not None:
        tracker.log_artifact_path("encoder-checkpoint", encoder_ckpt_path, artifact_type="model")
    tracker.log_artifact_path("wm-train-metrics", Path(out_dir) / "train_metrics.json", artifact_type="metrics")
    tracker.finish()
    success(f"训练完成 checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
