"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from nimloth.agent import Agent
from nimloth.backbone import (
    build_input_builder,
    build_vision_ema,
    load_backbone,
    resolve_tune_modes,
    resolve_vision_ema,
    uses_lora,
)
from nimloth.config.sft2 import SFT2LoopConfig
from nimloth.latent import (
    query_labels_are_masked,
    resolve_latent_query_mode,
)
from nimloth.training.sft2.checkpoint import (
    SFT2CheckpointManager,
    SFT2CheckpointRuntime,
    load_aux_checkpoint,
    resolve_resume_checkpoint_dir,
)
from nimloth.training.sft2.batch import SFT2BatchAssembler
from nimloth.training.sft2.cli import parse_sft2_args
from nimloth.training.sft2.data.factory import build_data_bundle
from nimloth.training.sft2.algorithm import (
    SFT2Algorithm,
    require_sft2_wm_history,
)
from nimloth.training.sft2.loop import (
    SFT2TrainingLoop,
    load_sft2_loop_state,
)
from nimloth.training.sft2.runtime import (
    SFT2ModelRuntime,
    SFT2OptimizationRuntime,
)
from nimloth.training.sft2.reporting import SFT2Reporter
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.wandb import init_wandb_run
from nimloth.util.optim import OptimizationRuntime
from nimloth.wm import (
    LeWMConfig,
    LatentWMPredictor,
    SequenceSIGReg,
    StateProjector,
    ValueHead,
    WorldModel,
)


def _build_world_model(
    args: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    pair_parallel: bool,
    resume_ckpt_dir: Path | None,
    train_wm_predictor: bool,
) -> tuple[WorldModel, torch.device]:
    """构造并恢复 SFT2 的三个 world-model 子模块。"""

    if args.wm_predictor_checkpoint is not None:
        wm_predictor = LatentWMPredictor.load_checkpoint(
            args.wm_predictor_checkpoint,
            map_location=device,
        ).to(device)
        require_sft2_wm_history(
            wm_predictor,
            history_size=args.history_size,
            source=args.wm_predictor_checkpoint,
        )
    else:
        wm_predictor = LatentWMPredictor.create(
            LeWMConfig(
                emb_dim=args.emb_dim,
                history_size=args.history_size,
            )
        ).to(device)
    if not train_wm_predictor:
        for parameter in wm_predictor.parameters():
            parameter.requires_grad = False

    aux_device = device
    if pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get(
            "model.language_model.norm"
        )
        if mapped is not None:
            aux_device = torch.device(f"cuda:{mapped}")
        wm_predictor = wm_predictor.to(aux_device)

    model_dtype = next(model.parameters()).dtype
    world_model = WorldModel(
        state_proj=StateProjector(
            model.config.hidden_size,
            wm_predictor.emb_dim,
            latent_token_count=args.latent_token_count,
        ).to(device=aux_device, dtype=model_dtype),
        wm_predictor=wm_predictor,
        value_head=ValueHead(wm_predictor.emb_dim).to(
            device=aux_device,
            dtype=model_dtype,
        ),
    )
    resume_state = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    if args.resume and resume_state is not None and resume_state.exists():
        load_aux_checkpoint(
            resume_ckpt_dir,
            world_model,
            device,
            latent_query_mode=args.latent_query_mode,
            query_tune=args.query_tune,
        )
    return world_model, aux_device


def _wrap_sft2_agent(
    loaded,
    world_model: WorldModel,
    *,
    device: torch.device,
    aux_device: torch.device,
    world_size: int,
    train_wm_predictor: bool,
) -> tuple[Agent, bool]:
    """按现有多卡语义包装模型，再组成唯一的神经网络 Agent。"""

    model = loaded.backbone.model
    state_proj = world_model.state_proj
    wm_predictor = world_model.wm_predictor
    value_head = world_model.value_head
    static_graph = world_size > 1
    if world_size > 1:
        if loaded.pair_parallel:
            model = DDP(
                model,
                device_ids=None,
                output_device=None,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        else:
            device_index = int(str(device).split(":")[-1])
            model = DDP(
                model,
                device_ids=[device_index],
                output_device=device_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )
        aux_index = int(str(aux_device).split(":")[-1])
        state_proj = DDP(
            state_proj,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        value_head = DDP(
            value_head,
            device_ids=[aux_index],
            output_device=aux_index,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
        if train_wm_predictor:
            wm_predictor = DDP(
                wm_predictor,
                device_ids=[aux_index],
                output_device=aux_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )

    return (
        Agent(
            backbone=loaded.backbone.with_model(model),
            wm=WorldModel(
                state_proj=state_proj,
                wm_predictor=wm_predictor,
                value_head=value_head,
            ),
        ),
        static_graph,
    )


def _build_optimizer(
    args: Any,
    *,
    agent: Agent,
    query_adapter: Any,
    train_wm_predictor: bool,
) -> torch.optim.Optimizer:
    """按模块名称建立可审计的 SFT2 参数组。"""

    query_parameter = query_adapter.delta if query_adapter is not None else None
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in agent.backbone.model.parameters()
                if parameter.requires_grad and parameter is not query_parameter
            ],
            "lr": args.lr_qwen_start,
            "name": "qwen",
        },
        {
            "params": agent.wm.state_proj.parameters(),
            "lr": args.state_proj_lr,
            "name": "state_proj",
        },
        {
            "params": agent.wm.value_head.parameters(),
            "lr": args.value_head_lr,
            "name": "value_head",
        },
    ]
    if query_parameter is not None:
        parameter_groups.append(
            {
                "params": [query_parameter],
                "lr": args.query_lr,
                "weight_decay": 0.0,
                "name": "query_adapter",
            }
        )
    if train_wm_predictor:
        predictor = agent.wm.wm_predictor
        predictor_parameters = (
            predictor.module.parameters()
            if hasattr(predictor, "module")
            else predictor.parameters()
        )
        parameter_groups.append(
            {
                "params": list(predictor_parameters),
                "lr": args.wm_predictor_lr,
                "name": "wm_predictor",
            }
        )
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def train_sft2(args=None) -> int:
    if args is None:
        args = parse_sft2_args()
    args.latent_token_count = int(getattr(args, "latent_token_count", 1))
    args.latent_query_mode = resolve_latent_query_mode(
        getattr(args, "latent_query_mode", None),
        getattr(args, "mask_latent_query_labels", None),
        default="inject",
    )
    args.mask_latent_query_labels = query_labels_are_masked(args.latent_query_mode)
    args.query_tune = str(getattr(args, "query_tune", "freeze"))
    args.query_lr = float(getattr(args, "query_lr", 5e-5))
    if args.query_tune not in {"freeze", "adapter"}:
        raise ValueError(f"query_tune must be freeze or adapter, got {args.query_tune!r}")
    if args.latent_token_count < 1:
        raise ValueError(f"--latent-token-count must be >= 1, got {args.latent_token_count}")
    args.history_size = int(getattr(args, "history_size", 4))
    if args.history_size < 1:
        raise ValueError(f"--history-size must be >= 1, got {args.history_size}")
    if args.backbone_rows_per_forward is not None and args.backbone_rows_per_forward < 1:
        raise ValueError("--backbone-rows-per-forward must be >= 1")

    llm_tune, vision_tune = resolve_tune_modes(args)
    if args.query_tune == "adapter" and uses_lora(args):
        raise ValueError("query_tune=adapter is not supported with LoRA tuning")
    vision_ema_enabled = resolve_vision_ema(args, vision_tune)
    train_wm_predictor = args.train_wm_predictor and not args.freeze_wm_predictor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rank, world, local_rank, device = setup_dist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = os.environ.get("WANDB_RUN_PREFIX", "")
    wandb_run = init_wandb_run(
        rank=rank,
        output_dir=args.output_dir,
        enabled=not args.no_wandb,
        default_project="nimloth",
        run_name=args.wandb_run_name or f"{prefix}sft2-latentwm-value",
        config=vars(args),
        metric_definitions=(
            ("global_step", None),
            ("train/*", "global_step"),
            ("epoch", None),
            ("val/*", "epoch"),
        ),
    )

    resume_ckpt_dir: Path | None = None
    if args.resume:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(args.output_dir, args.resume_from)
    resume_state_path = (
        resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir is not None else None
    )
    if is_main():
        print(
            json.dumps(
                {
                    "llm_tune": llm_tune,
                    "vision_tune": vision_tune,
                    "vision_ema": vision_ema_enabled,
                    "vision_ema_decay": args.vision_ema_decay,
                    "train_wm_predictor": train_wm_predictor,
                    "resume": args.resume,
                    "resume_from": str(resume_ckpt_dir) if resume_ckpt_dir is not None else None,
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                    "batch_mode": args.batch_mode,
                    "history_size": args.history_size,
                    "backbone_rows_per_forward": args.backbone_rows_per_forward,
                    "latent_token_count": args.latent_token_count,
                    "latent_query_mode": args.latent_query_mode,
                    "query_tune": args.query_tune,
                    "mask_latent_query_labels": args.mask_latent_query_labels,
                    "preprocess_cache_format": args.preprocess_cache_format,
                    "preprocess_cache_image_dtype": args.preprocess_cache_image_dtype,
                    "require_prebuilt_cache": args.require_prebuilt_cache,
                }
            )
        )

    loaded = load_backbone(
        args,
        device=device,
        latent_token_count=args.latent_token_count,
        resume_dir=resume_ckpt_dir,
        resume_state_path=resume_state_path,
    )
    world_model, aux_device = _build_world_model(
        args,
        model=loaded.backbone.model,
        device=device,
        pair_parallel=loaded.pair_parallel,
        resume_ckpt_dir=resume_ckpt_dir,
        train_wm_predictor=train_wm_predictor,
    )
    agent, ddp_static_graph = _wrap_sft2_agent(
        loaded,
        world_model,
        device=device,
        aux_device=aux_device,
        world_size=world,
        train_wm_predictor=train_wm_predictor,
    )
    vision_ema = build_vision_ema(
        enabled=vision_ema_enabled,
        decay=args.vision_ema_decay,
        llm=agent.backbone.model,
        resume_path=(resume_ckpt_dir / "vision_ema.pt") if resume_ckpt_dir else None,
        device=device,
    )
    input_builder = build_input_builder(
        loaded,
        max_length=args.max_length,
        latent_token_count=args.latent_token_count,
        mask_latent_query_labels=args.mask_latent_query_labels,
    )
    batch_builder = SFT2BatchAssembler(
        input_builder=input_builder,
        device=aux_device,
        history_size=args.history_size,
    )
    model_runtime = SFT2ModelRuntime(
        agent=agent,
        backbone_ema=vision_ema,
    )
    optimizer = _build_optimizer(
        args,
        agent=agent,
        query_adapter=loaded.query_adapter,
        train_wm_predictor=train_wm_predictor,
    )
    use_ddp_no_sync = (
        world > 1
        and not loaded.pair_parallel
        and not ddp_static_graph
    )
    if (
        is_main()
        and world > 1
        and args.grad_accum > 1
        and not use_ddp_no_sync
    ):
        print(
            json.dumps(
                {
                    "ddp_gradient_accumulation": "sync_each_microbatch",
                    "reason": "torch_2_8_static_graph_no_sync_regression",
                }
            )
        )
    data = build_data_bundle(
        args,
        batch_builder,
        rank=rank,
        world_size=world,
    )
    train_loader = data.train_loader
    val_loader = data.val_loader
    train_batch_sampler = data.train_batch_sampler

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))
    optimization_runtime = SFT2OptimizationRuntime(
        optimization=OptimizationRuntime(
            optimizer=optimizer,
            synchronized_modules=agent.synchronized_modules,
            enable_no_sync=use_ddp_no_sync,
            after_step=(
                lambda: vision_ema.update(agent.backbone.model)
                if vision_ema is not None
                else None
            ),
        ),
        qwen_warmup_steps=qwen_warmup_steps,
        total_steps=total_steps,
        qwen_start_lr=args.lr_qwen_start,
        qwen_peak_lr=args.lr_qwen_peak,
    )
    checkpoint_invariants = {
        "seed": int(args.seed),
        "world_size": int(world),
        "grad_accum": int(args.grad_accum),
        "latent_query_mode": args.latent_query_mode,
        "query_tune": args.query_tune,
        "history_size": int(args.history_size),
        "backbone_rows_per_forward": args.backbone_rows_per_forward,
        "train_micro_batches": int(len(train_loader)),
        "rng_schedule_version": "epoch_micro_rank_v1",
    }
    checkpoint_manager = SFT2CheckpointManager(
        output_dir=args.output_dir,
        agent=agent,
        processor=loaded.processor,
        vision_ema=vision_ema,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
        lora=uses_lora(args),
        base_model_path=Path(loaded.base_model_path),
        llm_tune=llm_tune,
        vision_tune=vision_tune,
        latent_query_mode=args.latent_query_mode,
        query_tune=args.query_tune,
    )
    checkpoint_runtime = SFT2CheckpointRuntime(
        manager=checkpoint_manager,
        device=device,
        interval_steps=int(args.checkpoint_interval_steps or 0),
        interval_minutes=float(args.checkpoint_interval_minutes),
        keep_last=int(args.checkpoint_keep_last or 0),
    )

    log_writer = CSVRecordWriter(
        args.output_dir / "train_step_log.csv",
        (
            "time",
            "epoch",
            "global_step",
            "total_loss",
            "wm_mse",
            "sigreg_loss",
            "value_total",
            "value_reg",
            "value_rank",
            "lm_ce",
            "lambda_wm",
            "lambda_sigreg",
            "qwen_lr",
            "val_wm_mse",
        ),
    )
    if is_main():
        log_writer.ensure_header()
    reporter = SFT2Reporter(
        log_writer=log_writer,
        wandb_run=wandb_run,
        llm_tune=llm_tune,
        vision_tune=vision_tune,
    )

    algorithm = SFT2Algorithm(
        history_size=args.history_size,
        sigreg=(
            SequenceSIGReg(
                knots=args.sigreg_knots,
                num_proj=args.sigreg_num_proj,
            ).to(device=aux_device)
            if args.lambda_sigreg > 0.0
            else None
        ),
        sigreg_weight=args.lambda_sigreg,
        value_weight=args.lambda_value,
        ce_weight=args.lambda_ce,
        value_rank_margin=args.value_rank_margin,
        value_rank_weight=args.value_rank_lambda,
        wm_weight_start=args.lambda_wm_start,
        wm_weight_end=args.lambda_wm_end,
        backbone_rows_per_forward=args.backbone_rows_per_forward,
    )

    loop_state = load_sft2_loop_state(
        resume=args.resume,
        resume_state_path=resume_state_path,
        resume_checkpoint_dir=resume_ckpt_dir,
        optimizer=optimizer,
        training_invariants=checkpoint_invariants,
    )
    training_loop = SFT2TrainingLoop(
        config=SFT2LoopConfig.from_namespace(args),
        rank=rank,
        train_loader=train_loader,
        val_loader=val_loader,
        train_batch_sampler=train_batch_sampler,
        algorithm=algorithm,
        model_runtime=model_runtime,
        optimization_runtime=optimization_runtime,
        batch_builder=batch_builder,
        checkpoint_runtime=checkpoint_runtime,
        reporter=reporter,
        state=loop_state,
        total_steps=total_steps,
    )
    training_loop.run()
    if wandb_run is not None:
        wandb_run.finish()
    cleanup_dist()
    return 0


def main() -> int:
    return train_sft2()
