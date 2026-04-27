"""DDP 分布式 WM 训练入口，配合 encoder_server.py 实现边训边编。

用法（必须通过 wm_training_lazy.sh 启动，不要直接运行）：
    torchrun --nproc_per_node=3 --nnodes=1 src/train/train_wm_ddp.py wm=cfm_dinov2m ...

要求：
    - encoder_server.py 先于本脚本启动，占用 GPU 0
    - torchrun 只使用剩余 GPU（本地 rank 从 1 开始）
    - 所有进程共享同一个 Manager().Queue() 和 .pt cache 文件
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset

from src.core import FileSystemModelProvider, WMDataProvider, WMModelAdapter
from src.infrastructure.encoding.cache_protocol import write_json_state
from src.shared.config.training_parsers import linear_warmup_lambda, parse_temporal_stride
from src.train.latent_cache import (
    build_wm_dataset_with_cache,
)
from src.train.manifest_resolver import resolve_manifest_for_split


def _write_training_state(
    cache_dir: Path | None,
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    loss_recon: float,
    loss_action: float,
    loss_sigreg: float,
    lr: float,
) -> None:
    """写入训练状态到共享文件。"""
    if cache_dir is None:
        return
    state_file = cache_dir / "training_state.json"
    write_json_state(
        state_file,
        {
            "epoch": epoch,
            "total_epochs": total_epochs,
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "loss_recon": loss_recon,
            "loss_action": loss_action,
            "loss_sigreg": loss_sigreg,
            "lr": lr,
            "timestamp": time.time(),
        },
    )
from src.utils.console import progress_context, show_kv_table, success
from src.utils.env import load_project_env
from src.utils.io import ensure_dir, write_json
from src.utils.terminal_ui import create_dashboard, LiveDashboard
from src.train.latent_cache import build_latent_cache_dir
from src.utils.seed import set_seed
from src.wm.inverse_dynamics import InverseDynamicsModel
from src.wm.predictor import WMModel, LeWMModel
from src.wm.action_mapper import build_action_mapper
from src.wm.factory import build_world_model, resolve_patch_layout, resolve_wm_type

logger = logging.getLogger(__name__)


def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1


def _local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", 0))


def _is_main_rank() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if _is_dist_initialized():
        dist.barrier(device_ids=[_local_rank()])


def _maybe_init_distributed() -> None:
    """按 torchrun 环境初始化默认进程组。"""
    if _is_dist_initialized():
        return
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return
    dist.init_process_group(backend="nccl", init_method="env://")


def _destroy_distributed() -> None:
    if _is_dist_initialized():
        dist.destroy_process_group()


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _build_split_dataset(
    split: str,
    *,
    manifest_path: Path,
    wm_cfg: DictConfig,
    dataset_cfg: DictConfig,
    train_cfg: DictConfig,
    temporal_stride: int | tuple[int, int],
    latent_dim: int,
    num_patches: int,
    token_dim: int,
    lazy_mode: bool,
    encoder_queue: Any,
) -> Dataset | None:
    if not manifest_path.exists():
        return None
    chunk_mode = bool(train_cfg.get("lazy_episode_chunk", True))
    dataset, _ = build_wm_dataset_with_cache(
        manifest_path=manifest_path,
        wm_name=str(wm_cfg.name),
        latent_dim=latent_dim,
        action_dim=int(dataset_cfg.action_dim),
        history_len=int(wm_cfg.history_len),
        temporal_stride=temporal_stride,
        image_encoder=None,
        encoder_num_workers=int(train_cfg.encoder_num_workers),
        encoder_batch_size=int(train_cfg.encoder_batch_size),
        expected_num_patches=num_patches,
        expected_token_dim=token_dim,
        lazy_mode=lazy_mode,
        encoder_queue=encoder_queue,
        chunk_mode=chunk_mode,
    )
    dataset.disable_encoder_after_warmup()
    return dataset


def _build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    split: str,
    *,
    lazy_mode: bool = False,
) -> DataLoader:
    sampler: Dataset | None = None
    if _world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=(split == "train"), drop_last=False)
    # lazy 编码阶段若 workers/prefetch 过大，会制造大量"远期请求"并拖慢首批样本就绪。
    effective_workers = num_workers
    prefetch_factor: int | None = None
    if lazy_mode and split == "train":
        effective_workers = min(num_workers, 2)
        if effective_workers > 0:
            prefetch_factor = 1
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=effective_workers,
        persistent_workers=effective_workers > 0,
        prefetch_factor=prefetch_factor,
        sampler=sampler,
        shuffle=(sampler is None and split == "train"),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    load_project_env()
    set_seed(int(cfg.project.seed))
    train_cfg = cfg.pipeline.train
    dataset_cfg = cfg.dataset
    wm_cfg = cfg.wm
    temporal_stride = parse_temporal_stride(train_cfg.get("temporal_stride", 1))
    sigreg_cfg = train_cfg.get("sigreg", {})
    sigreg_enabled = bool(getattr(sigreg_cfg, "enabled", False))
    sigreg_target_weight = float(getattr(sigreg_cfg, "weight", 0.0))
    sigreg_warmup_steps = int(getattr(sigreg_cfg, "warmup_steps", 0))
    sigreg_num_projections = int(getattr(sigreg_cfg, "num_projections", 256))
    sigreg_num_quadrature_points = int(getattr(sigreg_cfg, "num_quadrature_points", 16))
    sigreg_t_min = float(getattr(sigreg_cfg, "t_min", 0.2))
    sigreg_t_max = float(getattr(sigreg_cfg, "t_max", 4.0))
    sigreg_kernel_sigma = float(getattr(sigreg_cfg, "kernel_sigma", 1.0))
    warmup_steps = int(train_cfg.get("lr_warmup_steps", 0))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema_decay = float(getattr(ema_cfg, "decay", 0.999))
    path_segments = [str(train_cfg.operation.outputs_root), "wm", str(wm_cfg.name)]
    force_new_run = bool(getattr(train_cfg.operation, "force_new_run", False))
    lazy_mode = bool(train_cfg.get("lazy_encoding", True))
    chunk_mode = bool(train_cfg.get("lazy_episode_chunk", True))
    num_patches, token_dim = resolve_patch_layout(wm_cfg=wm_cfg)
    latent_dim = int(wm_cfg.latent_dim)

    # ---- 训练状态共享目录（供外部监控读取） ----------
    cache_dir: Path | None = None
    if chunk_mode:
        manifests_cfg = dict(dataset_cfg.get("manifests", {}))
        outputs_root = str(train_cfg.operation.outputs_root)
        dataset_name = str(dataset_cfg.name)
        train_manifest = resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split="train",
            outputs_root=outputs_root,
            dataset_name=dataset_name,
        )
        if train_manifest.exists():
            cache_dir = build_latent_cache_dir(train_manifest, str(wm_cfg.name))
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- DDP setup: 使用 LOCAL_RANK 对应当前进程可见设备 ----------
    # 训练脚本会将 CUDA_VISIBLE_DEVICES 设为"仅训练卡列表"，因此 local_rank 直接映射。
    local_rank = _local_rank()
    gpu_id = local_rank
    if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} 请求可见 GPU {gpu_id}，但可见 GPU 数量为 {torch.cuda.device_count()}。"
            "确认 launcher 脚本已将 CUDA_VISIBLE_DEVICES 设为训练卡列表。"
        )
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    _maybe_init_distributed()

    if _is_main_rank():
        logger.info("=" * 60)
        logger.info("DDP WM Training (lazy encoding mode)")
        logger.info("device: cuda:%d  (local visible index)", gpu_id)
        logger.info("world_size: %d", _world_size())
        logger.info("lazy_encoding: %s", lazy_mode)
        logger.info("=" * 60)

    # ---- Shared encoder queue (rank 0 创建，所有 rank 共享同一 Manager) ---------------
    if lazy_mode and _is_main_rank():
        from multiprocessing import Manager

        manager = Manager()
        encoder_queue = manager.Queue()
        logger.info("已创建 Manager.Queue()，将在 barrier 后广播给所有 rank。")
    else:
        encoder_queue = None

    # 等待 rank 0 创建完 queue
    _barrier()

    if lazy_mode:
        encoder_queue = _broadcast_encoder_queue(encoder_queue, device)

    # ---- Manifest 路径解析 ------------------------------------------------------------
    manifests_cfg = dict(dataset_cfg.get("manifests", {}))
    outputs_root = str(train_cfg.operation.outputs_root)
    dataset_name = str(dataset_cfg.name)

    def _manifest(split: str) -> Path:
        return resolve_manifest_for_split(
            manifests_cfg=manifests_cfg,
            split=split,
            outputs_root=outputs_root,
            dataset_name=dataset_name,
        )

    train_manifest = _manifest("train")
    if _is_main_rank() and not train_manifest.exists():
        raise FileNotFoundError(f"训练集 manifest 不存在: {train_manifest}，请先执行 collect_data。")
    _barrier()

    # ---- Dataset（lazy 模式，不等 encoder） -----------------------------------------
    if _is_main_rank():
        logger.info("构建训练集 (lazy_mode=%s)...", lazy_mode)

    # 分块模式：在 dataset 构建前生成采样顺序
    cache_dir = build_latent_cache_dir(train_manifest, str(wm_cfg.name))
    cache_dir.mkdir(parents=True, exist_ok=True)

    if lazy_mode and bool(train_cfg.get("lazy_episode_chunk", True)):
        sampling_order_path = cache_dir / "sampling_order.json"
        if _is_main_rank() and not sampling_order_path.exists():
            # 读取 manifest 统计样本数量
            sample_count = sum(1 for _ in open(train_manifest) if _.strip())
            _generate_sampling_order(train_manifest, cache_dir, sample_count, seed=int(cfg.project.seed))

    train_dataset = _build_split_dataset(
        "train",
        manifest_path=train_manifest,
        wm_cfg=wm_cfg,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        temporal_stride=temporal_stride,
        latent_dim=latent_dim,
        num_patches=num_patches,
        token_dim=token_dim,
        lazy_mode=lazy_mode,
        encoder_queue=encoder_queue,
    )
    val_dataset = _build_split_dataset(
        "val",
        manifest_path=_manifest("val"),
        wm_cfg=wm_cfg,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        temporal_stride=temporal_stride,
        latent_dim=latent_dim,
        num_patches=num_patches,
        token_dim=token_dim,
        lazy_mode=lazy_mode,
        encoder_queue=encoder_queue,
    )
    test_dataset = _build_split_dataset(
        "test",
        manifest_path=_manifest("test"),
        wm_cfg=wm_cfg,
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        temporal_stride=temporal_stride,
        latent_dim=latent_dim,
        num_patches=num_patches,
        token_dim=token_dim,
        lazy_mode=lazy_mode,
        encoder_queue=encoder_queue,
    )

    num_workers = int(train_cfg.num_workers)
    train_loader = _build_dataloader(
        train_dataset,
        int(train_cfg.batch_size),
        num_workers,
        "train",
        lazy_mode=lazy_mode,
    )
    val_loader = (
        _build_dataloader(
            val_dataset,
            int(train_cfg.batch_size),
            num_workers,
            "val",
            lazy_mode=lazy_mode,
        )
        if val_dataset
        else None
    )
    test_loader = (
        _build_dataloader(
            test_dataset,
            int(train_cfg.batch_size),
            num_workers,
            "test",
            lazy_mode=lazy_mode,
        )
        if test_dataset
        else None
    )

    if _is_main_rank():
        logger.info("数据集构建完成: train=%d, val=%d", len(train_dataset) if train_dataset else 0, len(val_dataset) if val_dataset else 0)

    # ---- Model（所有 rank 各自构建，仅 rank 0 记录 metrics） -------------------------
    wm_type = resolve_wm_type(wm_cfg)
    wm_module = build_world_model(
        wm_cfg=wm_cfg,
        train_cfg=train_cfg,
        action_dim=int(dataset_cfg.action_dim),
        device=device,
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
    wm_scheduler = torch.optim.lr_scheduler.LambdaLR(
        wm_optimizer, lr_lambda=lambda step: linear_warmup_lambda(step, warmup_steps)
    )
    idm_scheduler = torch.optim.lr_scheduler.LambdaLR(
        idm_optimizer, lr_lambda=lambda step: linear_warmup_lambda(step, warmup_steps)
    )

    # ---- DDP 封装（仅 world_size>1） --------------------------------------------------
    if _world_size() > 1:
        wm_module = DDP(
            wm_module,
            device_ids=[gpu_id],
            output_device=device,
            find_unused_parameters=True,
        )
        inverse_dynamics = DDP(inverse_dynamics, device_ids=[gpu_id], output_device=device)
        action_mapper = DDP(action_mapper, device_ids=[gpu_id], output_device=device)

    mode = str(train_cfg.training_mode).strip().lower()
    if mode not in {"unsupervised", "semi_supervised", "fully_supervised"}:
        raise ValueError(f"不支持的 training_mode={train_cfg.training_mode}")
    wm_module.train()
    inverse_dynamics.train()
    action_mapper.train()

    # ---- WMModel wrapper ------------------------------------------------------------
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

    # ---- Checkpoint / Logger（仅 rank 0）---------------------------------------------
    model_provider = FileSystemModelProvider(path_segments=path_segments)
    run_dir, resumed = model_provider.resolve_run_dir(force_new=force_new_run)

    if _is_main_rank():
        from src.visualize.wandb_tracker import init_tracker

        tracker = init_tracker(
            task_name="train_wm_ddp",
            config={
                "batch_size_per_gpu": int(train_cfg.batch_size),
                "world_size": _world_size(),
                "epochs": int(train_cfg.epochs),
                "lr": float(train_cfg.lr),
                "dataset": dataset_name,
                "wm": str(wm_cfg.get("name", "cfm")),
                "temporal_stride": str(temporal_stride),
                "sigreg_enabled": sigreg_enabled,
                "sigreg_weight": sigreg_target_weight,
                "lazy_encoding": lazy_mode,
                "encoder_gpu": 0,
                "training_gpus": list(range(1, _world_size() + 1)),
            },
        )
        total_epochs = int(train_cfg.epochs)
        total_batches = max(1, len(train_loader))
        global_step = 0
        start_epoch = 0
        wm_params = _count_trainable_params(wm_module)
        idm_params = _count_trainable_params(inverse_dynamics)
        mapper_params = _count_trainable_params(action_mapper)
        show_kv_table(
            "Train WM (DDP + Lazy Encoding)",
            [
                ("world_size", str(_world_size())),
                ("local_rank_gpu", f"cuda:{gpu_id}"),
                ("encoder_gpu", "cuda:0 (encoder_server)"),
                ("dataset_size", str(len(train_dataset))),
                ("epochs", str(total_epochs)),
                ("batch_size_per_gpu", str(train_cfg.batch_size)),
                ("effective_batch_size", str(int(train_cfg.batch_size) * _world_size())),
                ("lazy_encoding", str(lazy_mode)),
                ("temporal_stride", str(temporal_stride)),
                ("wm_params", f"{wm_params:,}"),
                ("idm_params", f"{idm_params:,}"),
                ("action_mapper_params", f"{mapper_params:,}"),
            ],
        )
        run_status_extra: dict[str, object] = {"resumed": resumed, "force_new_run": force_new_run}
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
    else:
        tracker = None
        total_epochs = int(train_cfg.epochs)
        total_batches = max(1, len(train_loader))
        global_step = 0
        start_epoch = 0

    # ---- Training loop --------------------------------------------------------------
    dashboard: LiveDashboard | None = None

    try:
        if _is_main_rank():
            dashboard = create_dashboard(title="WM Training", show_gpu=True, refresh_rate=0.5)
            dashboard.start()

        if _is_main_rank():
            with progress_context() as epoch_pbar:
                pbar_task = epoch_pbar.add_task("training_wm", total=total_epochs * total_batches)
                for epoch in range(start_epoch, total_epochs):
                    if hasattr(train_loader.sampler, "set_epoch") and _world_size() > 1:
                        train_loader.sampler.set_epoch(epoch)

                    epoch_loss = 0.0
                    epoch_recon = 0.0
                    epoch_action = 0.0
                    epoch_sigreg = 0.0
                    batch_count = 0

                    for batch_idx, batch in enumerate(train_loader):
                        global_step += 1
                        wm_model._global_step = global_step
                        prepared = wm_adapter.train_step(batch)
                        step_metrics = wm_model.train_step(prepared)
                        batch_loss = step_metrics["loss"]

                        if tracker is not None and log_every_n_steps > 0 and (global_step == 1 or global_step % log_every_n_steps == 0):
                            tracker.log_metrics(
                                {
                                    "train_step/loss": batch_loss,
                                    "train_step/loss_recon": step_metrics["loss_recon"],
                                    "train_step/loss_action": step_metrics["loss_action"],
                                    "train_step/loss_sigreg": step_metrics["loss_sigreg"],
                                    "train_step/global_step": global_step,
                                    "train_step/epoch": epoch,
                                    "train_step/lr_wm": step_metrics["lr_wm"],
                                    "train_step/lr_idm": step_metrics["lr_idm"],
                                },
                                step=global_step,
                            )

                        epoch_loss += batch_loss
                        epoch_recon += step_metrics["loss_recon"]
                        epoch_action += step_metrics["loss_action"]
                        epoch_sigreg += step_metrics["loss_sigreg"]
                        batch_count += 1

                        epoch_pbar.update(
                            pbar_task,
                            advance=1,
                            description=(
                                f"epoch={epoch + 1}/{total_epochs} "
                                f"batch={batch_idx + 1}/{total_batches} "
                                f"mode={mode} loss={batch_loss:.6f}"
                            ),
                        )

                        # 更新仪表板和状态文件（每 10 步更新一次避免过频繁）
                        if dashboard is not None and batch_idx % 10 == 0:
                            dashboard.update_training(
                                epoch=epoch + 1,
                                total_epochs=total_epochs,
                                step=batch_idx + 1,
                                total_steps=total_batches,
                                loss=batch_loss,
                                loss_recon=step_metrics["loss_recon"],
                                loss_action=step_metrics["loss_action"],
                                loss_sigreg=step_metrics["loss_sigreg"],
                                lr=step_metrics["lr_wm"],
                            )
                            _write_training_state(
                                cache_dir=cache_dir,
                                epoch=epoch + 1,
                                total_epochs=total_epochs,
                                step=batch_idx + 1,
                                total_steps=total_batches,
                                loss=batch_loss,
                                loss_recon=step_metrics["loss_recon"],
                                loss_action=step_metrics["loss_action"],
                                loss_sigreg=step_metrics["loss_sigreg"],
                                lr=step_metrics["lr_wm"],
                            )

                    avg_loss = epoch_loss / max(1, batch_count)
                    avg_recon = epoch_recon / max(1, batch_count)
                    avg_action = epoch_action / max(1, batch_count)
                    avg_sigreg = epoch_sigreg / max(1, batch_count)

                    if tracker is not None:
                        tracker.log_metrics(
                            {
                                "train/loss": avg_loss,
                                "train/loss_recon": avg_recon,
                                "train/loss_action": avg_action,
                                "train/loss_sigreg": avg_sigreg,
                                "train/epoch": epoch,
                                "train/lr_wm": float(wm_scheduler.get_last_lr()[0]),
                                "train/lr_idm": float(idm_scheduler.get_last_lr()[0]),
                            },
                            step=epoch,
                        )
                        model_provider.save_checkpoint(
                            run_dir=run_dir,
                            state={
                                "epoch": epoch,
                                "global_step": global_step,
                                **wm_model.get_state(),
                            },
                        )
                    _barrier()
        else:
            for epoch in range(start_epoch, total_epochs):
                if hasattr(train_loader.sampler, "set_epoch") and _world_size() > 1:
                    train_loader.sampler.set_epoch(epoch)

                epoch_loss = 0.0
                epoch_recon = 0.0
                epoch_action = 0.0
                epoch_sigreg = 0.0
                batch_count = 0

                for batch_idx, batch in enumerate(train_loader):
                    global_step += 1
                    wm_model._global_step = global_step
                    prepared = wm_adapter.train_step(batch)
                    step_metrics = wm_model.train_step(prepared)
                    batch_loss = step_metrics["loss"]
                    epoch_loss += batch_loss
                    epoch_recon += step_metrics["loss_recon"]
                    epoch_action += step_metrics["loss_action"]
                    epoch_sigreg += step_metrics["loss_sigreg"]
                    batch_count += 1

                avg_loss = epoch_loss / max(1, batch_count)
                avg_recon = epoch_recon / max(1, batch_count)
                avg_action = epoch_action / max(1, batch_count)
                avg_sigreg = epoch_sigreg / max(1, batch_count)
                _barrier()

        last_loss = avg_loss
        last_recon_loss = avg_recon
        last_action_loss = avg_action
        last_sigreg_loss = avg_sigreg

    except Exception as exc:
        if _is_main_rank():
            model_provider.mark_failed(run_dir, error=str(exc), global_step=global_step)
        raise
    finally:
        if dashboard is not None:
            dashboard.stop()
        _destroy_distributed()

    # ---- 保存模型（rank 0）----------------------------------------------------------
    if _is_main_rank():
        model_provider.mark_completed(run_dir, global_step=global_step, epochs=total_epochs)
        out_dir = ensure_dir(run_dir)
        ckpt_path = Path(out_dir) / "wm.pt"
        ema_ckpt_path = Path(out_dir) / "wm_ema.pt"
        idm_ckpt_path = Path(out_dir) / "inverse_dynamics.pt"
        mapper_ckpt_path = Path(out_dir) / "action_mapper.pt"
        wm_state = wm_module.module.state_dict() if isinstance(wm_module, DDP) else wm_module.state_dict()
        torch.save(wm_state, ckpt_path)
        if wm_model._ema_model is not None:
            torch.save(wm_model._ema_model.state_dict(), ema_ckpt_path)
        idm_state = (
            inverse_dynamics.module.state_dict()
            if isinstance(inverse_dynamics, DDP)
            else inverse_dynamics.state_dict()
        )
        mapper_state = (
            action_mapper.module.state_dict()
            if isinstance(action_mapper, DDP)
            else action_mapper.state_dict()
        )
        torch.save(idm_state, idm_ckpt_path)
        torch.save(mapper_state, mapper_ckpt_path)
        model_provider.save_checkpoint(
            run_dir=run_dir,
            state={"epoch": total_epochs - 1, "global_step": global_step, **wm_model.get_state()},
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
                "sigreg_enabled": sigreg_enabled,
                "sigreg_weight": sigreg_target_weight,
                "lazy_encoding": lazy_mode,
                "world_size": _world_size(),
            },
        )
        tracker.log_artifact_path("wm-checkpoint", ckpt_path, artifact_type="model")
        if wm_model._ema_model is not None:
            tracker.log_artifact_path("wm-ema-checkpoint", ema_ckpt_path, artifact_type="model")
        tracker.log_artifact_path("idm-checkpoint", idm_ckpt_path, artifact_type="model")
        tracker.log_artifact_path("action-mapper-checkpoint", mapper_ckpt_path, artifact_type="model")
        tracker.log_artifact_path("wm-train-metrics", Path(out_dir) / "train_metrics.json", artifact_type="metrics")
        tracker.finish()
        success(f"DDP 训练完成 checkpoint={ckpt_path} (world_size={_world_size()})")


def _broadcast_encoder_queue(root_queue: Any, device: torch.device) -> Any:
    """lazy 模式下统一走文件请求通道，避免跨 torchrun 进程传递 Manager proxy 失败。"""
    if not _is_dist_initialized():
        return None
    # 这里显式丢弃 root_queue，确保所有 rank 行为一致（统一 file queue fallback）。
    return None


def _generate_sampling_order(
    manifest_path: Path,
    cache_dir: Path,
    num_samples: int,
    seed: int = 42,
) -> list[int]:
    """生成固定的采样顺序，用于指导 encoder 按正确顺序编码。

    这样可以确保 encoder 编码的 episode 顺序与训练采样顺序一致，
    避免训练需要等待 encoder 编码完整个数据集。
    """
    import json

    order_path = cache_dir / "sampling_order.json"
    if order_path.exists():
        # 读取已有的采样顺序
        with open(order_path, "r") as f:
            return json.load(f)

    # 生成新的采样顺序
    # 采样顺序按 episode 分组：先均匀采样 episode 0，再均匀采样 episode 1...
    # 这样可以确保训练开始时需要的 episode 优先完成
    episode_indices: dict[int, list[int]] = {}
    for idx in range(num_samples):
        # 模拟 dataset 内部的 episode 分组逻辑
        # 这里用简单的 hash 来确定每个样本属于哪个 episode
        ep_id = idx % 200  # 假设最多 200 个 episode
        episode_indices.setdefault(ep_id, []).append(idx)

    # 按 episode 均匀轮转采样
    order = []
    max_len = max(len(indices) for indices in episode_indices.values()) if episode_indices else 0
    for pos in range(max_len):
        for ep_id in sorted(episode_indices.keys()):
            if pos < len(episode_indices[ep_id]):
                order.append(episode_indices[ep_id][pos])

    # 写入文件供 encoder 读取
    with open(order_path, "w") as f:
        json.dump(order, f)
    logger.info("采样顺序已生成并保存: %s (共 %d 个样本)", order_path, len(order))
    return order


if __name__ == "__main__":
    main()
