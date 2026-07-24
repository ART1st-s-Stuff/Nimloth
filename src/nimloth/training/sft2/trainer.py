"""SFT2 training loop (Qwen latent WM + value head)."""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
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
from nimloth.training.sft2.variant import (
    SFT2Variant,
    SFT2VariantBuildContext,
    resolve_sft2_variant,
)
from nimloth.training.sft2.loop import (
    SFT2TrainingLoop,
    load_sft2_loop_state,
)
from nimloth.training.sft2.history_cache import OnlineHistoryStateCache
from nimloth.training.sft2.runtime import (
    SFT2ModelRuntime,
    SFT2OptimizationRuntime,
)
from nimloth.training.sft2.reporting import SFT2Reporter
from nimloth.util.distributed import cleanup_dist, is_main, setup_dist
from nimloth.util.csv_log import CSVRecordWriter
from nimloth.util.wandb import init_wandb_run
from nimloth.util.optim import OptimizationRuntime
from nimloth.wm import SequenceSIGReg, WorldModel


def _build_world_model(
    args: Any,
    *,
    model: torch.nn.Module,
    device: torch.device,
    pair_parallel: bool,
    resume_ckpt_dir: Path | None,
    train_wm_predictor: bool,
    variant: SFT2Variant | None = None,
) -> tuple[WorldModel, torch.device]:
    """由 objective variant 构造模型；公共 trainer 只处理设备与恢复。"""

    aux_device = device
    if pair_parallel:
        device_map = getattr(model, "hf_device_map", {}) or {}
        mapped = device_map.get("lm_head") or device_map.get(
            "model.language_model.norm"
        )
        if mapped is not None:
            aux_device = torch.device(f"cuda:{mapped}")

    selected = variant or resolve_sft2_variant(args.objective)
    world_model = selected.build_world_model(
        SFT2VariantBuildContext(
            args=args,
            model=model,
            aux_device=aux_device,
            model_dtype=next(model.parameters()).dtype,
        )
    )
    if not train_wm_predictor:
        world_model.wm_predictor.requires_grad_(False)
    resume_state = resume_ckpt_dir / "training_state.pt" if resume_ckpt_dir else None
    if args.resume and resume_state is not None and resume_state.exists():
        load_aux_checkpoint(
            resume_ckpt_dir,
            world_model,
            aux_device,
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

        def wrap_component(_name: str, module: torch.nn.Module) -> torch.nn.Module:
            return DDP(
                module,
                device_ids=[aux_index],
                output_device=aux_index,
                find_unused_parameters=False,
                static_graph=static_graph,
            )

        world_model.wrap_distributed_components(wrap_component)

    return (
        Agent(
            backbone=loaded.backbone.with_model(model),
            wm=world_model,
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
    """按 WorldModel 声明的具名组件建立参数组。"""

    del train_wm_predictor  # predictor 是否参与已由 requires_grad 唯一决定。
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
        }
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
    for name, module in agent.wm.optimization_components:
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if not parameters:
            continue
        learning_rate = getattr(args, f"{name}_lr")
        parameter_groups.append(
            {"params": parameters, "lr": learning_rate, "name": name}
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
    args.objective = str(getattr(args, "objective", "latent"))
    variant = resolve_sft2_variant(args.objective)
    if args.query_tune not in {"freeze", "adapter"}:
        raise ValueError(f"query_tune must be freeze or adapter, got {args.query_tune!r}")
    if args.latent_token_count < 1:
        raise ValueError(f"--latent-token-count must be >= 1, got {args.latent_token_count}")
    args.history_size = int(getattr(args, "history_size", 4))
    if args.history_size < 1:
        raise ValueError(f"--history-size must be >= 1, got {args.history_size}")
    variant.validate_args(args)

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
        run_name=args.wandb_run_name or f"{prefix}sft2-{args.objective}-value",
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
                    "objective": args.objective,
                    "resume": args.resume,
                    "resume_from": str(resume_ckpt_dir) if resume_ckpt_dir is not None else None,
                    "init_model": str(args.model),
                    "wm_predictor_checkpoint": str(args.wm_predictor_checkpoint) if args.wm_predictor_checkpoint else None,
                    "output_dir": str(args.output_dir),
                    "batch_mode": args.batch_mode,
                    "history_size": args.history_size,
                    "history_state_cache": "online_detached_state_v1",
                    "latent_token_count": args.latent_token_count,
                    "latent_query_mode": args.latent_query_mode,
                    "query_tune": args.query_tune,
                    "mask_latent_query_labels": args.mask_latent_query_labels,
                    "preprocess_cache_format": args.preprocess_cache_format,
                    "preprocess_cache_image_dtype": args.preprocess_cache_image_dtype,
                    "require_prebuilt_cache": args.require_prebuilt_cache,
                    **variant.runtime_metadata(args),
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
        variant=variant,
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
    base_batch_builder = SFT2BatchAssembler(
        input_builder=input_builder,
        device=aux_device,
        history_size=args.history_size,
    )
    batch_builder = variant.build_batch_builder(args, base_batch_builder)
    history_cache = OnlineHistoryStateCache()
    if resume_ckpt_dir is not None:
        history_cache_path = resume_ckpt_dir / f"history_cache_rank_{rank:03d}.pt"
        if not history_cache_path.is_file():
            raise FileNotFoundError(
                f"resume checkpoint is missing rank history cache: {history_cache_path}"
            )
        history_cache.load(history_cache_path)
    model_runtime = SFT2ModelRuntime(
        agent=agent,
        history_cache=history_cache,
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
    local_batch_histogram = Counter(train_batch_sampler.current_steps_per_batch)
    batch_histograms: list[dict[int, int] | None] = [None] * world
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(
            batch_histograms,
            dict(local_batch_histogram),
        )
    else:
        batch_histograms[0] = dict(local_batch_histogram)
    global_batch_histogram: Counter[int] = Counter()
    for histogram in batch_histograms:
        if histogram is not None:
            global_batch_histogram.update(histogram)
    owned_current_steps = sum(
        batch_size * count
        for batch_size, count in global_batch_histogram.items()
    )
    if owned_current_steps != train_batch_sampler.window_count:
        raise RuntimeError(
            "online history sampler current-step ownership mismatch: "
            f"expected={train_batch_sampler.window_count}, actual={owned_current_steps}"
        )
    if is_main():
        print(
            json.dumps(
                {
                    "sft2_step_ownership": "current_step_once_v2_online_cache",
                    "train_current_steps": train_batch_sampler.window_count,
                    "actual_current_steps_per_microbatch": dict(
                        sorted(global_batch_histogram.items())
                    ),
                }
            )
        )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    qwen_warmup_steps = max(1, int(total_steps * args.qwen_lr_warmup_ratio))

    def after_optimizer_step() -> None:
        if vision_ema is not None:
            vision_ema.update(agent.backbone.model)
        agent.wm.after_optimizer_step()

    optimization_runtime = SFT2OptimizationRuntime(
        optimization=OptimizationRuntime(
            optimizer=optimizer,
            synchronized_modules=agent.synchronized_modules,
            enable_no_sync=use_ddp_no_sync,
            after_step=after_optimizer_step,
        ),
        qwen_warmup_steps=qwen_warmup_steps,
        total_steps=total_steps,
        qwen_start_lr=args.lr_qwen_start,
        qwen_peak_lr=args.lr_qwen_peak,
    )
    checkpoint_invariants = {
        "objective": args.objective,
        "seed": int(args.seed),
        "world_size": int(world),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "latent_query_mode": args.latent_query_mode,
        "query_tune": args.query_tune,
        "history_size": int(args.history_size),
        "history_state_cache": "online_detached_state_v1",
        "sigreg_batch_scope": "global_valid_states_v1",
        "sample_ownership_version": "current_step_once_v2_online_cache",
        "train_micro_batches": int(len(train_loader)),
        "rng_schedule_version": "epoch_micro_rank_v1",
    }
    checkpoint_invariants.update(variant.checkpoint_invariants(args))
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
        history_cache=history_cache,
        rank=rank,
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
            "sigreg_global_batch_size",
            "value_total",
            "value_reg",
            "value_rank",
            "lm_ce",
            "lambda_wm",
            "lambda_sigreg",
            "qwen_lr",
            "context_length",
            "current_batch_size",
            "history_cache_entries",
            "val_wm_mse",
            *variant.metric_fields,
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

    algorithm_kwargs = dict(
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
    )
    algorithm = variant.build_algorithm(args, **algorithm_kwargs)

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
